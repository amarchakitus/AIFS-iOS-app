import Foundation
import MLX

/// AIFS Single v2: graph-transformer encoder (N320 grid -> o96 mesh), 16-layer
/// sliding-window transformer processor, graph-transformer decoder (mesh -> grid).
///
/// A line-by-line port of anemoi-models 0.9.3's `AnemoiModelEncProcDec.predict_step`,
/// validated against it: fp32 agrees with PyTorch fp32 to ~3e-6 relative on every output;
/// fp16 is as close to fp32 as PyTorch's own fp16 autocast is.
///
/// Numerics follow torch autocast: linear layers run in the compute dtype; layer norm,
/// attention scores and softmax accumulate in fp32; pre/post-processing is fp32.
public final class AIFSModel: @unchecked Sendable {
    public let manifest: Manifest
    public let dtype: DType
    public let latitudes: [Float]
    public let longitudes: [Float]

    /// Hidden-mesh chunks for the encoder and grid chunks for the decoder. Chunking bounds
    /// peak memory only; results do not depend on it. Fewer, larger chunks are faster: on
    /// an M1 iPad a fp16 matmul reaches ~2.2 TFLOPS from ~17k rows up but only ~1.1 below
    /// that, so 16 decoder chunks (34k grid points each) run twice as fast as 128.
    public var encoderChunks = 16
    public var decoderChunks = 16
    /// Query block of the banded processor attention.
    public var attentionBlock = 1024

    /// Called as a step advances: a phase name and the step's overall fraction done.
    public var onProgress: (@Sendable (String, Double) -> Void)?

    /// Wall-clock seconds per phase of the last `predict` (each phase ends in an eval, so
    /// these are GPU times, not graph-building times).
    public private(set) var lastTimings: [(phase: String, seconds: Double)] = []

    /// MLX copies of the weights. With the Neural Engine active, those it runs are left
    /// out (~420 MB, which matters on a phone) and reloaded only if the GPU fallback runs.
    private var weights: [String: MLXArray]
    private let bundleURL: URL
    private var hasAllWeights: Bool
    private let dataAttr: MLXArray, hiddenAttr: MLXArray
    private let encSrc: MLXArray, encAttr: MLXArray, encPtr: [Int]
    private let decSrc: MLXArray, decAttr: MLXArray

    private let inMul: MLXArray, inAdd: MLXArray, outMul: MLXArray, outAdd: MLXArray
    private let imputeMask: MLXArray, imputeValue: MLXArray
    private let progInput: MLXArray, progMask: MLXArray
    private let restoreNaNSource: MLXArray, restoreNaNMask: MLXArray

    private var bandMasks: [String: MLXArray] = [:]

    /// The processor on the Neural Engine, if the bundle has `coreml/` (see
    /// `tools/export_coreml.py`) and `useNeuralEngine` was set when loading.
    private let neuralEngine: NeuralEngineProcessor?
    private let neuralEngineDecoder: NeuralEngineDecoder?
    public var usesNeuralEngine: Bool { neuralEngine != nil }

    /// Load a bundle directory (`manifest.json`, `weights/graph/grid.safetensors`).
    /// `dtype` is the compute dtype; fp16 is what the checkpoint's autocast uses.
    public init(bundle: URL, dtype: DType = .float16, useNeuralEngine: Bool = true) throws {
        let manifest = try Manifest.load(from: bundle.appending(path: "manifest.json"))
        self.manifest = manifest
        self.dtype = dtype

        func load(_ name: String) throws -> [String: MLXArray] {
            try loadArrays(url: bundle.appending(path: name))
        }
        func cast(_ a: MLXArray) -> MLXArray {
            a.dtype.isFloatingPoint ? a.asType(dtype) : a
        }
        bundleURL = bundle
        weights = try load("weights.safetensors").mapValues(cast)
        hasAllWeights = true
        let graph = try load("graph.safetensors")
        dataAttr = cast(graph["data_attr"]!)
        hiddenAttr = cast(graph["hidden_attr"]!)
        encSrc = graph["enc_src"]!
        encAttr = cast(graph["enc_attr"]!)
        encPtr = graph["enc_ptr"]!.asArray(Int32.self).map(Int.init)
        decSrc = graph["dec_src"]!
        decAttr = cast(graph["dec_attr"]!)
        let grid = try load("grid.safetensors")
        latitudes = grid["latitudes"]!.asType(.float32).asArray(Float.self)
        longitudes = grid["longitudes"]!.asType(.float32).asArray(Float.self)
        guard latitudes.count == manifest.numDataNodes, encPtr.count == manifest.numHiddenNodes + 1
        else { throw AIFSError.invalidBundle("graph does not match manifest") }

        inMul = MLXArray(manifest.inputNormMul)
        inAdd = MLXArray(manifest.inputNormAdd)
        outMul = MLXArray(manifest.outputNormMul)
        outAdd = MLXArray(manifest.outputNormAdd)

        let nIn = manifest.inputVars.count, nOut = manifest.outputVars.count
        var impMask = [Bool](repeating: false, count: nIn), impValue = [Float](repeating: 0, count: nIn)
        var nanSource = [Int32](repeating: 0, count: nOut), nanMask = [Bool](repeating: false, count: nOut)
        for imp in manifest.imputation {
            impMask[imp.input] = true
            impValue[imp.input] = imp.value
            if let o = imp.output {
                nanSource[o] = Int32(imp.input)
                nanMask[o] = true
            }
        }
        imputeMask = MLXArray(impMask)
        imputeValue = MLXArray(impValue)
        restoreNaNSource = MLXArray(nanSource)
        restoreNaNMask = MLXArray(nanMask)

        var pin = [Int32](repeating: 0, count: nOut), pmask = [Bool](repeating: false, count: nOut)
        for (i, o) in zip(manifest.prognosticInput, manifest.prognosticOutput) {
            pin[o] = Int32(i)
            pmask[o] = true
        }
        progInput = MLXArray(pin)
        progMask = MLXArray(pmask)

        let coreml = bundle.appending(path: "coreml")
        if useNeuralEngine, dtype == .float16,
            FileManager.default.fileExists(atPath: coreml.appending(path: "coreml.json").path)
        {
            let processor = try NeuralEngineProcessor(directory: coreml, channels: manifest.numChannels)
            neuralEngine = processor
            neuralEngineDecoder = try processor.config.decoder.map {
                try NeuralEngineDecoder(directory: coreml, config: $0, channels: manifest.numChannels)
            }
            let onDecoder = neuralEngineDecoder != nil
            weights = weights.filter { !Self.runsOnNeuralEngine($0.key, decoder: onDecoder) }
            hasAllWeights = false
        } else {
            neuralEngine = nil
            neuralEngineDecoder = nil
        }
    }

    // MARK: - Layers

    /// Weights the Neural Engine models replace on the GPU.
    static func runsOnNeuralEngine(_ name: String, decoder: Bool) -> Bool {
        if name.hasPrefix("processor.") { return true }
        guard decoder else { return false }
        return ["decoder.proc.projection.", "decoder.proc.layer_norm_mlp_dst.", "decoder.proc.node_dst_mlp.",
                "decoder.node_data_extractor."].contains { name.hasPrefix($0) }
    }

    /// Reload the weights the Neural Engine path dropped, for the GPU fallback.
    private func ensureAllWeights() {
        guard !hasAllWeights else { return }
        AIFSRuntime.log("[AIFS] reloading GPU weights for the fallback path")
        if let all = try? loadArrays(url: bundleURL.appending(path: "weights.safetensors")) {
            weights = all.mapValues { $0.dtype.isFloatingPoint ? $0.asType(dtype) : $0 }
            hasAllWeights = true
        }
    }

    private func w(_ name: String) -> MLXArray {
        if weights[name] == nil { ensureAllWeights() }
        guard let a = weights[name] else { fatalError("bundle lacks weight \(name)") }
        return a
    }

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let y = matmul(x, w(name + ".weight").T)
        if let b = weights[name + ".bias"] { return y + b }
        return y
    }

    private func layerNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        MLXFast.layerNorm(x, weight: w(name + ".weight"), bias: w(name + ".bias"), eps: 1e-5)
    }

    /// Exact (erf) GELU, as torch.nn.GELU() defaults to. Compiled so the four elementwise
    /// ops fuse: uncompiled, each materialises a full (40320, 4096) temporary.
    private let gelu: (MLXArray) -> MLXArray = compile(shapeless: true) { x in
        x * 0.5 * (1 + erf(x * Float(1 / 2.0.squareRoot())))
    }

    private func mlp(_ x: MLXArray, _ name: String) -> MLXArray {
        linear(gelu(linear(x, name + ".0")), name + ".2")
    }

    // MARK: - Pre/post-processing

    /// Physical `(2, N, V)` fp32 -> normalised, NaNs imputed.
    func preprocess(_ x: MLXArray) -> MLXArray {
        let xn = x * inMul + inAdd
        return which(isNaN(xn) .&& imputeMask, imputeValue, xn)
    }

    /// Normalised `(N, 120)` (residual already added) -> physical, with the pipeline's
    /// boundings and NaN masks. `firstInput` is the physical input at the first time slot,
    /// whose NaNs mark where imputed outputs are undefined.
    func postprocess(_ yIn: MLXArray, firstInput: MLXArray) -> MLXArray {
        var y = yIn
        let nOut = manifest.outputVars.count
        for b in manifest.boundings {
            var mask = [Bool](repeating: false, count: nOut)
            for i in b.indices { mask[i] = true }
            let m = MLXArray(mask)
            switch b.type {
            case "relu":
                y = which(m, maximum(y, 0), y)
            case "hardtanh":
                y = which(m, clip(y, min: b.min!, max: b.max!), y)
            case "fraction":
                let total = y[0..., b.total!].expandedDimensions(axis: 1)
                y = which(m, clip(y, min: b.min!, max: b.max!) * total, y)
            default:
                fatalError("unknown bounding \(b.type)")
            }
        }
        let inputNaN = isNaN(take(firstInput, restoreNaNSource, axis: 1))
        y = which(restoreNaNMask .&& inputNaN, Float.nan, y)
        y = (y - outAdd) / outMul
        var targets = [Bool](repeating: false, count: nOut)
        for t in manifest.conditionalNan.targets { targets[t] = true }
        let masked = isNaN(y[0..., manifest.conditionalNan.mask]).expandedDimensions(axis: 1)
        return which(masked .&& MLXArray(targets), Float.nan, y)
    }

    // MARK: - Graph transformer attention

    /// Graph-transformer message passing (anemoi `GraphTransformerConv`) for one chunk of
    /// `n` destinations with up to `K` incoming edges each.
    ///
    /// anemoi computes, per edge and head, `softmax(q . (k_src + e)) / sqrt(d)` and sums
    /// `(v_src + e)` with those weights, where `e = W_e a + b_e` embeds the edge's 11
    /// attributes `a` into 1024 dims. Because `e` is linear in `a`, both uses fold into
    /// 11-dim terms -- `q . e = (W_e^T q) . a + q . b_e` and `sum w e = W_e (sum w a) + b_e`
    /// (weights sum to 1) -- so no per-edge 1024-wide edge embedding is ever built. What is
    /// left per edge is a gather of fp16 key/value rows feeding batched matmuls; on iPad's
    /// 68 GB/s memory that traffic, not arithmetic, was the mappers' cost.
    ///
    /// - nodeKeys/nodeValues: `lin_key`/`lin_value` of the source nodes, `(S, 1024)`.
    /// - index: `(n, K)` rows into them; padded slots may point anywhere (they are masked).
    /// - attr: `(n, K, 11)` edge attributes; zeros at padded slots.
    /// - mask: additive `(n, 1, 1, K)`, `-inf` at padded slots; nil when none are padded.
    /// - dst: the chunk's layer-normed destination features, `(n, 1024)`.
    private func graphAttention(
        _ p: String, nodeKeys: MLXArray, nodeValues: MLXArray, index: MLXArray, attr: MLXArray,
        mask: MLXArray?, dst: MLXArray
    ) -> MLXArray {
        let heads = manifest.mapperHeads, c = manifest.numChannels, d = c / heads
        let n = dst.dim(0), k = index.dim(1), edgeDim = manifest.edgeDim
        let q = linear(dst, p + ".proc.lin_query").reshaped(n, heads, 1, d)
        let flat = index.reshaped(n * k)
        func gather(_ t: MLXArray) -> MLXArray {
            t.take(flat, axis: 0).reshaped(n, k, heads, d).transposed(0, 2, 1, 3)  // (n, h, K, d)
        }
        let keys = gather(nodeKeys), values = gather(nodeValues)
        let we = w(p + ".proc.lin_edge.weight").asType(.float32).reshaped(heads, d, edgeDim)
        let be = w(p + ".proc.lin_edge.bias").asType(.float32).reshaped(1, heads, 1, d)
        let q32 = q.asType(.float32)
        let a = attr.asType(.float32).reshaped(n, 1, k, edgeDim)

        var logits = matmul(q, keys.transposed(0, 1, 3, 2)).asType(.float32)  // q . k_src
        logits = logits + matmul(matmul(q32, we), a.transposed(0, 1, 3, 2))  // (W_e^T q) . a
        logits = logits + (q32 * be).sum(axis: -1, keepDims: true)  // q . b_e
        logits = logits * Float(1 / Double(d).squareRoot())
        if let mask { logits = logits + mask }
        let weights = softmax(logits, axis: -1, precise: true)  // (n, h, 1, K)

        let fromValues = matmul(weights.asType(values.dtype), values).asType(.float32)
        let fromEdges = matmul(matmul(weights, a), we.transposed(0, 2, 1))  // W_e (sum w a)
        return (fromValues + fromEdges + be).reshaped(n, c)
    }

    /// The rest of anemoi's `GraphTransformerMapperBlock` after the attention.
    private func mapperBlockTail(_ p: String, attention: MLXArray, dstNormed: MLXArray, dst: MLXArray)
        -> MLXArray
    {
        let selfTerm = linear(dstNormed, p + ".proc.lin_self")
        var out = linear(attention.asType(dtype) + selfTerm, p + ".proc.projection") + dst
        out = mlp(layerNorm(out, p + ".proc.layer_norm_mlp_dst"), p + ".proc.node_dst_mlp") + out
        return out
    }

    private func chunkBounds(_ total: Int, _ chunks: Int) -> [(Int, Int)] {
        let c = max(1, min(chunks, total))
        return (0 ..< c).map { (total * $0 / c, total * ($0 + 1) / c) }
    }

    // MARK: - Network

    /// One encoder chunk: a range of hidden nodes, their incoming edges, and the distinct
    /// grid points feeding them (a grid point feeds ~1.4 edges; within a chunk, projecting
    /// each distinct source once instead of once per edge saves ~25% of the encoder).
    private struct EncoderChunk {
        let nodes: Range<Int>
        let sources: MLXArray  // distinct grid points
        let slotSource: MLXArray  // (n, K) -> index into `sources`; 0 at padded slots
        let slotAttr: MLXArray  // (n, K, 11) edge attributes; 0 at padded slots
        let mask: MLXArray  // (n, 1, 1, K) additive, -inf at padded slots
    }

    private var encoderPlan: (chunks: Int, plan: [EncoderChunk])?

    private func encoderChunkPlan() -> [EncoderChunk] {
        if let cached = encoderPlan, cached.chunks == encoderChunks { return cached.plan }
        let src = encSrc.asArray(Int32.self)
        let maxDegree = manifest.encoderMaxInDegree
        let plan = chunkBounds(manifest.numHiddenNodes, encoderChunks).map { a, b -> EncoderChunk in
            let ea = encPtr[a], eb = encPtr[b], nEdges = eb - ea, n = b - a
            var position: [Int32: Int32] = [:], sources: [Int32] = []
            var slotSource = [Int32](repeating: 0, count: n * maxDegree)
            var slotEdge = [Int32](repeating: Int32(nEdges), count: n * maxDegree)
            var mask = [Float](repeating: -.infinity, count: n * maxDegree)
            for node in 0 ..< n {
                for (rank, e) in (encPtr[a + node] ..< encPtr[a + node + 1]).enumerated() {
                    let s = src[e]
                    let i = position[s] ?? {
                        position[s] = Int32(sources.count)
                        sources.append(s)
                        return Int32(sources.count - 1)
                    }()
                    let slot = node * maxDegree + rank
                    slotSource[slot] = i
                    slotEdge[slot] = Int32(e - ea)
                    mask[slot] = 0
                }
            }
            // Padded slots read an appended zero row.
            let attr = concatenated([encAttr[ea ..< eb], MLXArray.zeros([1, manifest.edgeDim], dtype: encAttr.dtype)])
                .take(MLXArray(slotEdge), axis: 0).reshaped(n, maxDegree, manifest.edgeDim)
            let chunk = EncoderChunk(
                nodes: a ..< b, sources: MLXArray(sources),
                slotSource: MLXArray(slotSource, [n, maxDegree]), slotAttr: attr,
                mask: MLXArray(mask, [n, 1, 1, maxDegree]))
            eval(chunk.slotAttr)
            return chunk
        }
        encoderPlan = (encoderChunks, plan)
        return plan
    }

    /// Grid -> hidden mesh. `xData` is `(N, 224)`: both input times, then node attributes.
    func encode(_ xData: MLXArray) -> MLXArray {
        let hidden = linear(hiddenAttr, "encoder.emb_nodes_dst")
        var outs: [MLXArray] = []
        for chunk in encoderChunkPlan() {
            let src = layerNorm(
                linear(xData.take(chunk.sources, axis: 0), "encoder.emb_nodes_src"),
                "encoder.proc.layer_norm_attention_src")
            let dst = hidden[chunk.nodes.lowerBound ..< chunk.nodes.upperBound]
            let dstNormed = layerNorm(dst, "encoder.proc.layer_norm_attention")
            let att = graphAttention(
                "encoder", nodeKeys: linear(src, "encoder.proc.lin_key"),
                nodeValues: linear(src, "encoder.proc.lin_value"), index: chunk.slotSource,
                attr: chunk.slotAttr, mask: chunk.mask, dst: dstNormed)
            let out = mapperBlockTail("encoder", attention: att, dstNormed: dstNormed, dst: dst)
            eval(out)
            outs.append(out)
            onProgress?(
                "encoder", Phase.encoder.at(Double(chunk.nodes.upperBound) / Double(manifest.numHiddenNodes)))
        }
        return concatenated(outs)
    }

    /// Additive band mask `|i - j| <= window` for one query block, cached by geometry.
    private func bandMask(rows: Int, cols: Int, offset: Int, window: Int) -> MLXArray {
        let key = "\(rows)x\(cols)+\(offset)"
        if let m = bandMasks[key] { return m }
        var values = [Float](repeating: 0, count: rows * cols)
        for i in 0 ..< rows {
            for j in 0 ..< cols where abs(i + offset - j) > window {
                values[i * cols + j] = -.infinity
            }
        }
        let m = MLXArray(values, [rows, cols]).asType(dtype)
        bandMasks[key] = m
        return m
    }

    /// Sliding-window transformer over the hidden mesh (without the latent skip).
    /// Sliding-window self-attention, `|i - j| <= window`, in query blocks so memory is
    /// O(T * window). `q` is `(1, heads, Tq, d)` holding queries from global position
    /// `queryOffset`; `k`/`v` are `(1, heads, Tk, d)` from `keyOffset` and must cover every
    /// key within the window of those queries (or run to the sequence's ends).
    /// Returns `(1, heads, Tq, d)`.
    private func bandedAttention(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, queryOffset: Int = 0, keyOffset: Int = 0
    ) -> MLXArray {
        let p = manifest.processor, tq = q.dim(2), tk = k.dim(2), d = q.dim(3)
        let scale = Float(1 / Double(d).squareRoot())
        var blocks: [MLXArray] = []
        for s in stride(from: 0, to: tq, by: attentionBlock) {
            let e = min(s + attentionBlock, tq)
            let lo = max(0, queryOffset + s - p.window - keyOffset)
            let hi = min(tk, queryOffset + e + p.window - keyOffset)
            let mask = bandMask(
                rows: e - s, cols: hi - lo, offset: queryOffset + s - (keyOffset + lo), window: p.window)
            blocks.append(
                MLXFast.scaledDotProductAttention(
                    queries: q[0..., 0..., s ..< e], keys: k[0..., 0..., lo ..< hi],
                    values: v[0..., 0..., lo ..< hi], scale: scale, mask: mask))
        }
        return concatenated(blocks, axis: 2)
    }

    /// Sliding-window transformer over the hidden mesh (without the latent skip): on the
    /// Neural Engine when the bundle has Core ML models for it, otherwise on the GPU.
    func process(_ x0: MLXArray) -> MLXArray {
        let p = manifest.processor
        if let ane = neuralEngine, ane.tokens == x0.dim(0) {
            do {
                // One chunk's attention: queries of chunk i against keys of chunks i-1..i+1,
                // all `(1, heads, S, d)`, positioned by their global offsets.
                return try ane.process(x0, heads: p.heads, attend: { q, k, v, qOffset, kOffset in
                    let att = bandedAttention(q, k, v, queryOffset: qOffset, keyOffset: kOffset)
                    eval(att)
                    return att
                }, progress: { layer in
                    onProgress?("processor", Phase.processor.at(Double(layer) / Double(p.layers)))
                })
            } catch {
                // A Core ML failure mid-run is not recoverable for this step's state, but
                // the GPU path below computes the same thing from the same input.
                AIFSRuntime.log("[AIFS] Neural Engine processor failed (\(error)); using the GPU")
            }
        }
        return processOnGPU(x0)
    }

    func processOnGPU(_ x0: MLXArray) -> MLXArray {
        let p = manifest.processor, c = x0.dim(1), d = c / p.heads
        let h = x0.dim(0)
        var x = x0
        for layer in 0 ..< p.layers {
            let name = "processor.blocks.\(layer)"
            let normed = layerNorm(x, name + ".layer_norm_attention")
            func heads(_ t: MLXArray) -> MLXArray {
                t.reshaped(1, h, p.heads, d).transposed(0, 2, 1, 3)
            }
            let att = bandedAttention(
                heads(linear(normed, name + ".attention.lin_q")),
                heads(linear(normed, name + ".attention.lin_k")),
                heads(linear(normed, name + ".attention.lin_v"))
            ).transposed(0, 2, 1, 3).reshaped(h, c)
            x = x + linear(att, name + ".attention.projection")
            x = x + mlp(layerNorm(x, name + ".layer_norm_mlp"), name + ".mlp")
            eval(x)
            onProgress?("processor", Phase.processor.at(Double(layer + 1) / Double(p.layers)))
        }
        return x
    }

    /// Hidden mesh -> grid; returns the normalised `(N, 120)` output before the residual.
    func decode(_ latent: MLXArray, _ xData: MLXArray) -> MLXArray {
        let srcNormed = layerNorm(latent, "decoder.proc.layer_norm_attention_src")
        // Keys and values depend only on the 40k mesh nodes: project them once, not for
        // each of the 1.6M edges (that alone was ~15% of a step's arithmetic).
        let nodeKeys = linear(srcNormed, "decoder.proc.lin_key")
        let nodeValues = linear(srcNormed, "decoder.proc.lin_value")
        eval(nodeKeys, nodeValues)
        // In chunk order: finished GPU outputs, or Neural Engine work still in flight.
        enum Part { case done(MLXArray), pending(NeuralEngineDecoder.Pending) }
        var parts: [Part] = []
        // On the Neural Engine, chunk sizes must be multiples of its call size.
        let chunks = neuralEngineDecoder.map { ane -> [(Int, Int)] in
            let calls = manifest.numDataNodes / ane.config.chunk
            let per = max(1, calls / decoderChunks)
            return stride(from: 0, to: calls, by: per).map {
                ($0 * ane.config.chunk, min($0 + per, calls) * ane.config.chunk)
            }
        } ?? chunkBounds(manifest.numDataNodes, decoderChunks)
        var tGPU = 0.0, tSubmit = 0.0
        func since(_ t: ContinuousClock.Instant) -> Double {
            let d = (ContinuousClock.now - t).components
            return Double(d.seconds) + Double(d.attoseconds) / 1e18
        }
        for (a, b) in chunks {
            var t0 = ContinuousClock.now
            let dst = linear(xData[a ..< b], "decoder.emb_nodes_dst")
            let dstNormed = layerNorm(dst, "decoder.proc.layer_norm_attention")
            // Exactly `k` edges per grid point: nothing padded, no mask.
            let att = graphAttention(
                "decoder", nodeKeys: nodeKeys, nodeValues: nodeValues, index: decSrc[a ..< b],
                attr: decAttr[a ..< b], mask: nil, dst: dstNormed)
            if let ane = neuralEngineDecoder {
                // The rest is dense per grid point: hand it to the Neural Engine and move on
                // to the next chunk's attention while it runs.
                let selfTerm = linear(dstNormed, "decoder.proc.lin_self")
                let sum = att.asType(dtype) + selfTerm
                eval(sum, dst)
                tGPU += since(t0)
                t0 = ContinuousClock.now
                // At most three chunks in flight: the GPU outpaces the Neural Engine here, and
                // every pending chunk keeps its inputs alive.
                let inFlight = parts.indices.filter { if case .pending = parts[$0] { true } else { false } }
                if inFlight.count >= 3, case .pending(let oldest) = parts[inFlight[0]] {
                    do { parts[inFlight[0]] = .done(try oldest.wait()) } catch {
                        fatalError("Neural Engine decoder failed: \(error)")
                    }
                }
                do {
                    parts.append(.pending(try ane.submit(a: sum, dst: dst)))
                    tSubmit += since(t0)
                    onProgress?("decoder", Phase.decoder.at(Double(b) / Double(manifest.numDataNodes)))
                    continue
                } catch {
                    AIFSRuntime.log("[AIFS] Neural Engine decoder failed (\(error)); using the GPU for this chunk")
                }
            }
            let out = mapperBlockTail("decoder", attention: att, dstNormed: dstNormed, dst: dst)
            let y = linear(
                layerNorm(out, "decoder.node_data_extractor.0"), "decoder.node_data_extractor.1")
            eval(y)
            parts.append(.done(y))
            onProgress?("decoder", Phase.decoder.at(Double(b) / Double(manifest.numDataNodes)))
        }
        let tWait0 = ContinuousClock.now
        let outs = parts.map { part -> MLXArray in
            switch part {
            case .done(let y): return y
            case .pending(let p):
                do { return try p.wait() } catch {
                    // Too late to redo on the GPU without the chunk's inputs; fail loudly
                    // rather than return a plausible-looking partial forecast.
                    fatalError("Neural Engine decoder failed: \(error)")
                }
            }
        }
        if neuralEngineDecoder != nil {
            decoderTimings = [("dec.gpu", tGPU), ("dec.submit", tSubmit), ("dec.wait", since(tWait0))]
        }
        return concatenated(outs)
    }

    /// Sub-phase timings of the last Neural Engine decoder run (diagnostics).
    var decoderTimings: [(String, Double)] = []

    /// One 6-hour step. `x` is the physical input `(2, N, V)` fp32; returns the physical
    /// output `(N, 120)` fp32, NaN where a field is undefined (e.g. waves over land).
    public func predict(_ x: MLXArray) -> MLXArray {
        let n = manifest.numDataNodes
        var timings: [(String, Double)] = []
        var mark = ContinuousClock.now
        func lap(_ phase: String) {
            let now = ContinuousClock.now
            let d = (now - mark).components
            timings.append((phase, Double(d.seconds) + Double(d.attoseconds) / 1e18))
            mark = now
        }
        // xn is the largest transient (2 x N x 106 fp32); scope it so it is released
        // before the network runs, and recompute the slot the residual needs afterwards.
        let xData: MLXArray = {
            let xn = preprocess(x)
            let xData = concatenated(
                [xn.transposed(1, 0, 2).reshaped(n, -1).asType(dtype), dataAttr], axis: 1)
            eval(xData)
            return xData
        }()
        lap("preprocess")
        let latent = encode(xData)
        lap("encoder")
        let processed = process(latent) + latent
        eval(processed)
        lap("processor")
        var y = decode(processed, xData).asType(.float32)
        eval(y)
        lap("decoder")
        // Residual for prognostic variables, from the normalised latest input.
        y = y + which(progMask, take(preprocess(x[-1]), progInput, axis: 1), Float(0))
        let out = postprocess(y, firstInput: x[0])
        eval(out)
        lap("postprocess")
        lastTimings = timings + decoderTimings
        return out
    }
}

/// Rough share of a step's time per phase, for progress reporting (measured on M4 Max).
enum Phase {
    case encoder, processor, decoder

    var range: (Double, Double) {
        switch self {
        case .encoder: (0, 0.2)
        case .processor: (0.2, 0.7)
        case .decoder: (0.7, 1)
        }
    }

    func at(_ fraction: Double) -> Double {
        range.0 + (range.1 - range.0) * fraction
    }
}
