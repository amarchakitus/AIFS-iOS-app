import CoreML
import Foundation
import MLX

/// The processor's dense layers on the Neural Engine, with the sliding-window attention
/// between them left on the GPU (see `tools/export_coreml.py` for how the 16 layers are
/// cut into 17 Core ML models).
///
/// Measured on an M1 iPad Pro, the processor's MLP runs at ~6 TFLOPS on the Neural Engine
/// against ~2.2 for the same matmuls in MLX on the GPU.
///
/// The residual stream never leaves Core ML: each model's `x_out` chunks feed the next
/// model directly. Only q/k/v (out) and the attention result (back in) cross to MLX, once
/// per layer, as one memcpy per chunk.
final class NeuralEngineProcessor: @unchecked Sendable {
    struct Config: Decodable {
        let chunk: Int
        let chunks: Int
        let models: [String]
        let decoder: DecoderConfig?
    }

    struct DecoderConfig: Decodable {
        let model: String
        let chunk: Int
        let outputs: Int
        let output: String
    }

    let config: Config
    let channels: Int
    private let models: [MLModel]

    /// Loads (and on first use, lets the OS compile for the Neural Engine) every model in
    /// `directory`. Throws if any is missing; callers fall back to the GPU path.
    init(directory: URL, channels: Int) throws {
        config = try JSONDecoder().decode(
            Config.self, from: Data(contentsOf: directory.appending(path: "coreml.json")))
        self.channels = channels
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine
        models = try config.models.map {
            try MLModel(contentsOf: directory.appending(path: "\($0).mlmodelc"), configuration: mlConfig)
        }
    }

    var tokens: Int { config.chunk * config.chunks }

    // MARK: - Processor

    /// Runs all layers on `x0` `(T, C)`, with the Neural Engine and the GPU working at once.
    ///
    /// Model `l` turns chunk i's residual stream and layer l-1's attention into layer l's
    /// q/k/v. Attention for chunk i needs q/k/v of chunks i-1..i+1 only (the window is
    /// narrower than a chunk), so the two chips form a wavefront: while the GPU computes
    /// attention for chunk i+1 of layer l, the Neural Engine is already running layer l+1
    /// on chunk i. The Neural Engine runs on its own thread in (layer, chunk) order, each
    /// call waiting only for the attention it consumes; the GPU work stays on this thread.
    ///
    /// `attend(q, k, v, queryOffset, keyOffset)` takes `(1, heads, *, d)` MLX arrays and
    /// returns `(1, heads, S, d)`.
    func process(
        _ x0: MLXArray, heads: Int,
        attend: (MLXArray, MLXArray, MLXArray, Int, Int) -> MLXArray,
        progress: (Int) -> Void
    ) throws -> MLXArray {
        precondition(x0.dim(0) == tokens, "processor expects \(tokens) tokens")
        let n = config.chunks, s = config.chunk, c = channels, d = c / heads
        let stages = models.count  // layers + 1: proc_in, proc_01...proc_15, proc_out
        let pipe = Pipeline(stages: stages, chunks: n)
        let inputs = try split(x0.T)
        let models = models

        // Core ML may recycle a model's output buffers on its next call, and here outputs
        // outlive several calls. So every output goes to a buffer we own: q/k/v one set per
        // chunk (the GPU copies them out before the next layer rewrites them), and the
        // residual stream two per chunk, alternating, since each model reads x and writes it.
        func buffer() throws -> MLMultiArray {
            try MLMultiArray(shape: [1, c, 1, s].map { NSNumber(value: $0) }, dataType: .float16)
        }
        let qkvBuffers = try (0 ..< n).map { _ in try ["q": buffer(), "k": buffer(), "v": buffer()] }
        let residualBuffers = try (0 ..< n).map { _ in try [buffer(), buffer()] }

        // Neural Engine thread.
        let done = DispatchSemaphore(value: 0)
        Thread {
            defer { done.signal() }
            var x = inputs
            for l in 0 ..< stages {
                for i in 0 ..< n {
                    var features: [String: MLMultiArray] = ["x": x[i]]
                    if l > 0 {
                        guard let att = pipe.waitAttention(layer: l - 1, chunk: i) else { return }
                        features["att"] = att
                    }
                    var backings: [String: Any] = l < stages - 1 ? qkvBuffers[i] : [:]
                    if l > 0 { backings["x_out"] = residualBuffers[i][l % 2] }
                    let options = MLPredictionOptions()
                    options.outputBackings = backings
                    do {
                        let provider = try MLDictionaryFeatureProvider(
                            dictionary: features.mapValues { MLFeatureValue(multiArray: $0) })
                        let out = try models[l].prediction(from: provider, options: options)
                        if l > 0 { x[i] = out.featureValue(for: "x_out")!.multiArrayValue! }
                        pipe.publish(stage: l, chunk: i, out, residual: l == stages - 1 ? x[i] : nil)
                    } catch {
                        pipe.fail(error)
                        return
                    }
                }
            }
        }.start()

        // GPU: attention for every layer and chunk, in order.
        func toHeads(_ a: MLMultiArray) -> MLXArray {  // (1, C, 1, S) -> (1, h, S, d)
            join([a]).reshaped(heads, d, s).transposed(0, 2, 1).expandedDimensions(axis: 0)
        }
        do {
            for l in 0 ..< stages - 1 {
                var q = [MLXArray?](repeating: nil, count: n)
                var k = q, v = q
                func load(_ j: Int) throws {
                    guard q[j] == nil else { return }
                    let out = try pipe.waitOutput(stage: l, chunk: j)
                    q[j] = toHeads(out.featureValue(for: "q")!.multiArrayValue!)
                    k[j] = toHeads(out.featureValue(for: "k")!.multiArrayValue!)
                    v[j] = toHeads(out.featureValue(for: "v")!.multiArrayValue!)
                }
                for i in 0 ..< n {
                    let lo = max(0, i - 1), hi = min(n - 1, i + 1)
                    for j in lo ... hi { try load(j) }
                    let att = attend(
                        q[i]!, concatenated((lo ... hi).map { k[$0]! }, axis: 2),
                        concatenated((lo ... hi).map { v[$0]! }, axis: 2), i * s, lo * s)
                    // (1, h, S, d) -> (C, S) channel-major, the Neural Engine's layout.
                    let cm = att.squeezed(axis: 0).transposed(0, 2, 1).reshaped(c, s)
                    pipe.publishAttention(layer: l, chunk: i, try split(cm)[0])
                    // The next chunk reads chunks i..i+2: chunk i-1 is done with.
                    if i >= 1 { (q[i - 1], k[i - 1], v[i - 1]) = (nil, nil, nil) }
                }
                progress(l + 1)
            }
            let residual = try pipe.waitResiduals()
            done.wait()
            return join(residual).T
        } catch {
            pipe.fail(error)
            done.wait()
            throw error
        }
    }

    private func split(_ channelMajor: MLXArray) throws -> [MLMultiArray] {
        try CoreMLChunks.split(channelMajor: channelMajor, channels: channels, chunk: config.chunk)
    }

    private func join(_ arrays: [MLMultiArray]) -> MLXArray {
        CoreMLChunks.join(arrays, channels: channels, chunk: config.chunk)
    }

    /// Hand-off between the two threads: Neural Engine outputs per (stage, chunk),
    /// attention results per (layer, chunk), the final residual stream, and failure.
    private final class Pipeline: @unchecked Sendable {
        private let condition = NSCondition()
        private var outputs: [[MLFeatureProvider?]]
        private var attention: [[MLMultiArray?]]
        private var residuals: [MLMultiArray?]
        private var error: Error?

        init(stages: Int, chunks: Int) {
            outputs = Array(repeating: Array(repeating: nil, count: chunks), count: stages)
            attention = Array(repeating: Array(repeating: nil, count: chunks), count: stages)
            residuals = Array(repeating: nil, count: chunks)
        }

        private func locked<T>(_ body: () -> T) -> T {
            condition.lock()
            defer { condition.unlock() }
            return body()
        }

        func publish(stage: Int, chunk: Int, _ out: MLFeatureProvider, residual: MLMultiArray?) {
            locked {
                outputs[stage][chunk] = out
                if let residual { residuals[chunk] = residual }
                condition.broadcast()
            }
        }

        func publishAttention(layer: Int, chunk: Int, _ att: MLMultiArray) {
            locked {
                attention[layer][chunk] = att
                condition.broadcast()
            }
        }

        func fail(_ e: Error) {
            locked {
                if error == nil { error = e }
                condition.broadcast()
            }
        }

        /// Takes (and releases) the Neural Engine's output; throws if the other side failed.
        func waitOutput(stage: Int, chunk: Int) throws -> MLFeatureProvider {
            condition.lock()
            defer { condition.unlock() }
            while outputs[stage][chunk] == nil, error == nil { condition.wait() }
            if let error { throw error }
            let out = outputs[stage][chunk]!
            outputs[stage][chunk] = nil
            return out
        }

        /// Nil if the GPU side failed (the Neural Engine thread then just stops).
        func waitAttention(layer: Int, chunk: Int) -> MLMultiArray? {
            condition.lock()
            defer { condition.unlock() }
            while attention[layer][chunk] == nil, error == nil { condition.wait() }
            if error != nil { return nil }
            let att = attention[layer][chunk]
            attention[layer][chunk] = nil
            return att
        }

        func waitResiduals() throws -> [MLMultiArray] {
            condition.lock()
            defer { condition.unlock() }
            while residuals.contains(where: { $0 == nil }), error == nil { condition.wait() }
            if let error { throw error }
            return residuals.map { $0! }
        }
    }
}

/// Moving `(C, T)` fp16 data between MLX and `(1, C, 1, chunk)` Core ML inputs/outputs.
enum CoreMLChunks {
    /// `(C, T)` -> one `(1, C, 1, chunk)` MLMultiArray per `chunk` columns.
    ///
    /// The GPU writes the chunk-major layout into a fresh buffer, which is memcpy'd into
    /// arrays Core ML owns. (Copying a transposed view on the CPU instead was 90% of the
    /// Neural Engine decoder's time; wrapping MLX's memory in the MLMultiArrays directly
    /// handed Core ML stale bytes once arrays outlived the call that made them.)
    static func split(channelMajor x: MLXArray, channels c: Int, chunk s: Int) throws -> [MLMultiArray] {
        let n = x.dim(1) / s
        precondition(n * s == x.dim(1), "\(x.dim(1)) columns do not split into chunks of \(s)")
        // `+ 0` makes a fresh buffer (so `contiguous` can never return a view of someone
        // else's), but elementwise ops may keep a permuted layout; `contiguous` then makes
        // it row-major. Only a row-major buffer can be read in place: MLX's `.copy` access
        // walks permuted layouts element by element on the CPU.
        let blocks = contiguous(x.asType(.float16).reshaped(c, n, s).transposed(1, 0, 2) + Float16(0))
        eval(blocks)
        let data = blocks.asData(access: .noCopyIfContiguous)
        precondition(data.strides == [c * s, s, 1], "chunk layout is not contiguous")
        let bytes = c * s * 2
        precondition(data.data.count == n * bytes)
        return try withExtendedLifetime(blocks) { try data.data.withUnsafeBytes { raw in
            try (0 ..< n).map { i in
                let a = try MLMultiArray(shape: [1, c, 1, s].map { NSNumber(value: $0) }, dataType: .float16)
                a.withUnsafeMutableBytes { dst, strides in
                    precondition(strides == [c * s, s, s, 1], "unexpected MLMultiArray layout")
                    dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[(i * bytes) ..< ((i + 1) * bytes)]))
                }
                return a
            }
        } }
    }

    /// Chunks of `(1, C, 1, chunk)` -> `(C, T)` fp16, honouring Core ML's output strides
    /// (the Neural Engine may pad each channel's row).
    static func join(_ arrays: [MLMultiArray], channels c: Int, chunk s: Int) -> MLXArray {
        var out = [Float16](repeating: 0, count: c * s * arrays.count)
        out.withUnsafeMutableBufferPointer { dst in
            for (i, a) in arrays.enumerated() {
                a.withUnsafeBufferPointer(ofType: Float16.self) { src in
                    let strides = a.strides.map(\.intValue)
                    precondition(strides[3] == 1, "unexpected MLMultiArray layout")
                    let base = i * c * s
                    for ch in 0 ..< c {
                        let row = ch * strides[1]
                        _ = UnsafeMutableBufferPointer(rebasing: dst[(base + ch * s) ..< (base + (ch + 1) * s)])
                            .initialize(from: UnsafeBufferPointer(rebasing: src[row ..< row + s]))
                    }
                }
            }
        }
        return out.withUnsafeBufferPointer { MLXArray($0, [arrays.count, c, s]) }
            .transposed(1, 0, 2).reshaped(c, s * arrays.count)
    }

    /// `(1, C, 1, S)` arrays -> contiguous `(chunks, C, S)` values, honouring strides.
    static func copy(_ arrays: [MLMultiArray]) -> [Float16] {
        guard let first = arrays.first else { return [] }
        let c = first.shape[1].intValue, s = first.shape[3].intValue
        var out = [Float16](repeating: 0, count: c * s * arrays.count)
        out.withUnsafeMutableBufferPointer { dst in
            for (i, a) in arrays.enumerated() {
                a.withUnsafeBufferPointer(ofType: Float16.self) { src in
                    let rowStride = a.strides[1].intValue
                    for ch in 0 ..< c {
                        let row = ch * rowStride, base = (i * c + ch) * s
                        _ = UnsafeMutableBufferPointer(rebasing: dst[base ..< base + s])
                            .initialize(from: UnsafeBufferPointer(rebasing: src[row ..< row + s]))
                    }
                }
            }
        }
        return out
    }

    static func predict(_ model: MLModel, _ inputs: [[String: MLMultiArray]]) throws -> [MLFeatureProvider] {
        let batch = MLArrayBatchProvider(array: try inputs.map {
            try MLDictionaryFeatureProvider(dictionary: $0.mapValues { MLFeatureValue(multiArray: $0) })
        })
        let out = try model.predictions(fromBatch: batch)
        return (0 ..< out.count).map { out.features(at: $0) }
    }
}

/// The decoder's dense second half on the Neural Engine, run asynchronously so the GPU can
/// compute the next chunk's graph attention meanwhile.
final class NeuralEngineDecoder: @unchecked Sendable {
    let config: NeuralEngineProcessor.DecoderConfig
    let channels: Int
    private let model: MLModel
    private let queue = DispatchQueue(label: "aifs.ane.decoder")

    init(directory: URL, config: NeuralEngineProcessor.DecoderConfig, channels: Int) throws {
        self.config = config
        self.channels = channels
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: directory.appending(path: "\(config.model).mlmodelc"), configuration: mlConfig)
    }

    /// A submitted chunk; `wait()` returns its `(n, outputs)` normalised output.
    final class Pending: @unchecked Sendable {
        fileprivate let group = DispatchGroup()
        /// Copied out on the prediction queue itself: Core ML may recycle output buffers on
        /// the model's next call, which can run before `wait()`.
        fileprivate var result: Result<[Float16], Error> = .success([])
        fileprivate let decoder: NeuralEngineDecoder
        fileprivate init(_ d: NeuralEngineDecoder) { decoder = d }

        func wait() throws -> MLXArray {
            group.wait()
            let values = try result.get()
            let (c, s) = (decoder.config.outputs, decoder.config.chunk)
            let calls = values.count / (c * s)
            return values.withUnsafeBufferPointer { MLXArray($0, [calls, c, s]) }
                .transposed(1, 0, 2).reshaped(c, s * calls).T
        }
    }

    /// `a` = attention + lin_self, `dst` = embedded destinations, both `(n, C)` token-major
    /// MLX arrays. Copies them out now (cheap), predicts on a background queue.
    func submit(a: MLXArray, dst: MLXArray) throws -> Pending {
        let aChunks = try CoreMLChunks.split(channelMajor: a.T, channels: channels, chunk: config.chunk)
        let dChunks = try CoreMLChunks.split(channelMajor: dst.T, channels: channels, chunk: config.chunk)
        let pending = Pending(self)
        let model = model, name = config.output
        pending.group.enter()
        queue.async {
            pending.result = Result {
                let outputs = try CoreMLChunks.predict(model, zip(aChunks, dChunks).map { ["a": $0, "dst": $1] })
                    .map { $0.featureValue(for: name)!.multiArrayValue! }
                return CoreMLChunks.copy(outputs)
            }
            pending.group.leave()
        }
        return pending
    }
}
