import AuroraANE
import CoreML
import CoreVideo
import Foundation

/// GraphCast operational on the Neural Engine: drives the Core ML programs that
/// `graphcast-ane/graphcast_ane/ane_export.py` writes, one 6-hour step at a time.
///
/// A step takes the grid nodes' 184 normalised input features and returns GraphCast's 83
/// normalised outputs. Between the programs, data moves along the graphs' edges: a node's
/// projection is gathered to each of its edges (the processor's sender's and receiver's
/// summed), and each node's incoming messages are summed in float32. That is prepared for a
/// chunk, and finished for the chunk before, while the Neural Engine runs the chunk between.
///
/// Every 512-wide tensor lives in an IOSurface-backed buffer that the Neural Engine reads and
/// writes in place (Core ML copies other memory in and out, 15% of a call). The processor's
/// gathers run on the GPU when it will take them (`GraphCastGPU`), reading the same buffers;
/// the encoder's are copies of consecutive rows (its edges carry their receiver's term in
/// their coefficients), cheap for the CPU. (On an iPhone, the GPU reading the grid program's
/// output keeps the Neural Engine from writing it: that program then runs 50x slower.)
public final class GraphCastRuntime {
    public struct Metadata: Decodable {
        public struct Input: Decodable {
            public let name: String
            public let variable: String
            public let time: String?
            public let level: Int?
            public let mean: Double
            public let std: Double
        }
        public struct Output: Decodable {
            public let name: String
            public let variable: String
            public let level: Int?
            public let kind: String
            public let mean: Double?
            public let scale: Double
        }
        public struct Grid: Decodable {
            public let rows: Int
            public let cols: Int
            public let first_latitude: Double
            public let spacing: Double
        }
        public struct Programs: Decodable {
            public let grid_chunk: Int
            public let enc_edges: Int
            public let mesh_edges: Int
            public let mesh_nodes: Int
            public let rank: Int
            public let enc_rank: Int
        }
        public let model: String
        public let step_hours: Int
        public let grid: Grid
        public let levels: [Int]
        public let inputs: [Input]
        public let outputs: [Output]
        public let programs: Programs

        public static func load(_ directory: URL) throws -> Metadata {
            try JSONDecoder().decode(Metadata.self, from: Data(contentsOf: directory.appending(path: "graphcast.json")))
        }
    }

    public static let features = 184, outputs = 83, width = 512
    public static let programNames = ["grid", "enc_edges", "mesh_embed", "mesh_edges", "mesh_nodes", "decoder"]

    public let directory: URL
    public let metadata: Metadata
    public let gridNodes: Int
    let meshNodes: Int
    let p: Metadata.Programs
    let graph: SafeTensorsFile
    let tables: SafeTensorsFile
    let weights: SafeTensorsFile
    private var models: [String: MLModel] = [:]

    // Graph (mmap'd).
    let g2mSenders: UnsafePointer<Int32>, g2mReceivers: UnsafePointer<Int32>
    let meshSenders: UnsafePointer<Int32>, meshReceivers: UnsafePointer<Int32>
    let m2gSenders: UnsafePointer<Int32>
    let meshEdgeFeatures: UnsafePointer<Float16>
    let m0: UnsafePointer<Float16>
    let encCoef: UnsafePointer<Float16>, decCoef: UnsafePointer<Float16>
    let meshEdges: Int, encEdges: Int
    /// First encoder edge of each grid chunk (edges are in sender order), and each chunk's
    /// edges in receiver order (chunk-local positions) with their receivers, so threads can
    /// sum messages into disjoint sets of mesh nodes.
    private var encBounds: [Int] = []
    private var encOrder: [Int32] = []
    private var encOrderReceivers: [Int32] = []

    /// Seconds per phase over the last `step`.
    public private(set) var timings: [String: Double] = [:]
    /// Gathers the CPU did in the last `step` because the GPU would not (in the background).
    public private(set) var cpuGathers = 0
    /// Whether to gather on the GPU at all (else always the CPU).
    public var gathersOnGPU = true
    private lazy var gpu: GraphCastGPU? = gathersOnGPU ? GraphCastGPU(graph: graph, tables: tables) : nil
    private let offsets: [String: Int]
    /// Called as a step goes: ("encoder" | "processor" | "decoder", fraction of that phase),
    /// between Neural Engine calls.
    public var onProgress: ((String, Double) -> Void)?

    public init(directory: URL) throws {
        self.directory = directory
        metadata = try Metadata.load(directory)
        p = metadata.programs
        gridNodes = metadata.grid.rows * metadata.grid.cols
        graph = try SafeTensorsFile(directory.appending(path: "graph.safetensors"))
        tables = try SafeTensorsFile(directory.appending(path: "tables.safetensors"))
        weights = try SafeTensorsFile(directory.appending(path: "weights.safetensors"))
        g2mSenders = UnsafePointer(try graph.pointer("g2m_senders", as: Int32.self))
        g2mReceivers = UnsafePointer(try graph.pointer("g2m_receivers", as: Int32.self))
        meshSenders = UnsafePointer(try graph.pointer("mesh_senders", as: Int32.self))
        meshReceivers = UnsafePointer(try graph.pointer("mesh_receivers", as: Int32.self))
        m2gSenders = UnsafePointer(try graph.pointer("m2g_senders", as: Int32.self))
        meshEdgeFeatures = UnsafePointer(try graph.pointer("mesh_edge_features", as: Float16.self))
        m0 = UnsafePointer(try tables.pointer("m0", as: Float16.self))
        encCoef = UnsafePointer(try tables.pointer("enc_coef", as: Float16.self))
        decCoef = UnsafePointer(try tables.pointer("dec_coef", as: Float16.self))
        encEdges = try graph.tensor("g2m_senders").shape[0]
        meshEdges = try graph.tensor("mesh_senders").shape[0]
        meshNodes = try tables.tensor("m0").shape[0]
        offsets = try [
            "g2m_senders": graph.tensor("g2m_senders").offset, "g2m_receivers": graph.tensor("g2m_receivers").offset,
            "mesh_senders": graph.tensor("mesh_senders").offset, "mesh_receivers": graph.tensor("mesh_receivers").offset,
        ]
        indexEncoderEdges()
    }

    private func indexEncoderEdges() {
        let chunks = (gridNodes + p.grid_chunk - 1) / p.grid_chunk
        encBounds = [0]
        var e = 0
        for c in 0 ..< chunks {
            let hi = min((c + 1) * p.grid_chunk, gridNodes)
            while e < encEdges && Int(g2mSenders[e]) < hi { e += 1 }
            encBounds.append(e)
            precondition(e - encBounds[c] <= p.enc_edges, "a grid chunk has more encoder edges than enc_edges takes")
        }
        encOrder = [Int32](repeating: 0, count: encEdges)
        encOrderReceivers = [Int32](repeating: 0, count: encEdges)
        for c in 0 ..< chunks {
            let lo = encBounds[c], hi = encBounds[c + 1]
            let local = (0 ..< hi - lo).sorted { g2mReceivers[lo + $0] < g2mReceivers[lo + $1] }
            for (i, k) in local.enumerated() {
                encOrder[lo + i] = Int32(k)
                encOrderReceivers[lo + i] = g2mReceivers[lo + k]
            }
        }
    }

    // MARK: - Programs

    private var programURLs: [String: URL] = [:]

    private func model(_ name: String) throws -> MLModel {
        if let m = models[name] { return m }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let url = programURLs[name] ?? directory.appending(path: "\(name).mlmodelc")
        let m = try MLModel(contentsOf: url, configuration: config)
        models[name] = m
        return m
    }

    /// Loads the six programs. The first load on a device compiles each for the Neural Engine;
    /// later loads hit the system's cache. `progress(fraction)` may throw to stop. Then checks
    /// that each runs on the Neural Engine (`verify`).
    public func prepare(progress: (Double) throws -> Void = { _ in }) throws {
        try progress(0)
        for (i, name) in Self.programNames.enumerated() {
            _ = try model(name)
            try progress(Double(i + 1) / Double(Self.programNames.count))
        }
        try verify()
    }

    /// Each program's time per call on a Neural Engine relative to the others' (an M4 Max's,
    /// ms): any device runs them all within a factor of two of these proportions.
    static let referenceMilliseconds = ["grid": 4.0, "enc_edges": 7.1, "mesh_embed": 8.4, "mesh_edges": 16.3,
                                        "mesh_nodes": 19.0, "decoder": 30.3]
    /// Programs `verify` loaded again, and how many times.
    public private(set) var reloads: [String: Int] = [:]
    /// Receives what `verify` finds.
    public var log: ((String) -> Void)?

    /// Core ML has no "Neural Engine only": with `.cpuAndNeuralEngine` it may run a program on
    /// the CPU instead, silently, and keep doing so for that compiled copy (some installs on an
    /// iPhone 18 Pro Max do this to the grid program: 50x slower). So every program runs once
    /// here, timed against its share of the others' time; one that is over 5x slower than the
    /// rest is loaded again from a fresh copy of the program, which Core ML compiles anew, up
    /// to 3 times.
    func verify() throws {
        var ratio: [String: Double] = [:]
        for name in Self.programNames { ratio[name] = try timeOne(name) / Self.referenceMilliseconds[name]! }
        for name in Self.programNames {
            var attempt = 0
            while attempt < 3 {
                let others = ratio.filter { $0.key != name }.values.sorted()
                let typical = others[others.count / 2]
                guard ratio[name]! > 5 * typical else { break }
                attempt += 1
                log?(String(format: "%@ runs %.0fx slower than the other programs: compiling it again (%d)", name,
                            ratio[name]! / typical, attempt))
                let fresh = FileManager.default.temporaryDirectory.appending(path: "graphcast-\(name)-\(UUID().uuidString).mlmodelc")
                try FileManager.default.copyItem(at: directory.appending(path: "\(name).mlmodelc"), to: fresh)
                if let old = programURLs[name], old != directory.appending(path: "\(name).mlmodelc") {
                    try? FileManager.default.removeItem(at: old)
                }
                programURLs[name] = fresh
                models[name] = nil
                _ = try model(name)
                ratio[name] = try timeOne(name) / Self.referenceMilliseconds[name]!
                reloads[name, default: 0] += 1
            }
        }
    }

    /// Milliseconds for one call of `name` (after one to warm it up), on the buffers a step
    /// uses: Core ML keeps staging memory for any other buffer a program is given. Their
    /// contents don't matter here, and a step overwrites what this writes.
    private func timeOne(_ name: String) throws -> Double {
        let b = buffers
        let (inputs, outputs): ([String: MLMultiArray], [String: MLMultiArray]) = try {
            func weights(_ prefix: String, _ keys: [String]) throws -> [String: MLMultiArray] {
                try keys.reduce(into: [:]) { $0[$1] = try weight("\(prefix).\($1)") }
            }
            switch name {
            case "grid": return (["x": b.x[0]], ["q": b.q[0].array])
            case "enc_edges": return (["coef": b.encCoef[0], "s": b.encSums[0].array], ["msg": b.encMsg[0].array])
            case "mesh_embed": return (["f": try meshFeatures()[0]], ["e": b.e[0].array])
            case "mesh_edges":
                var i = try weights("edge0", ["we", "w1", "b1", "scale", "offset"])
                i["e"] = b.e[0].array
                i["s"] = b.meshSums[0].array
                return (i, ["msg": b.meshMsg[0].array, "e_new": b.e[0].array])
            case "mesh_nodes":
                var i = try weights("enc", ["wm", "wa", "b0", "w1", "b1", "scale", "offset", "ws", "wr", "br"])
                i["m"] = b.m[0].array
                i["agg"] = b.agg.array
                return (i, ["m_new": b.m[1].array, "ps": b.ps.array, "pr": b.pr.array])
            default:
                return (["x": b.x[0], "coef": b.decCoef[0], "s": b.decSums[0].array], ["out": b.out[0]])
            }
        }()
        try run(name, inputs, outputs)
        let t = Date()
        try run(name, inputs, outputs)
        return Date().timeIntervalSince(t) * 1000
    }

    /// Where Core ML places each program's operations ("NeuralEngine", "CPU", "GPU": count),
    /// and the bundle's compiled program it plans from.
    public func placement() async -> [String: [String: Int]] {
        var out: [String: [String: Int]] = [:]
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        for name in Self.programNames {
            guard let plan = try? await MLComputePlan.load(contentsOf: directory.appending(path: "\(name).mlmodelc"), configuration: config),
                  case .program(let program) = plan.modelStructure, let main = program.functions["main"] else { continue }
            var counts: [String: Int] = [:]
            for op in main.block.operations {
                guard let usage = plan.deviceUsage(for: op) else { continue }
                let device: String = switch usage.preferred {
                case .neuralEngine: "NeuralEngine"
                case .gpu: "GPU"
                case .cpu: "CPU"
                @unknown default: "other"
                }
                counts[device, default: 0] += 1
            }
            out[name] = counts
        }
        return out
    }

    private func run(_ name: String, _ inputs: [String: MLMultiArray], _ outputs: [String: MLMultiArray]) throws {
        let m = try model(name)
        let options = MLPredictionOptions()
        options.outputBackings = outputs
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
        let t = Date()
        let result = try m.prediction(from: provider, options: options)
        add(name, since: t)
        // Core ML writes into a backing it can use, else returns its own buffer. A pixel buffer
        // backing is always used (and must not be read through `dataPointer`, which locks it).
        for (k, backing) in outputs where backing.pixelBuffer == nil {
            guard let got = result.featureValue(for: k)?.multiArrayValue, got.dataPointer != backing.dataPointer else { continue }
            copy(got, into: backing)
        }
    }

    private let lock = NSLock()
    private func countCPUGather() {
        lock.lock()
        cpuGathers += 1
        lock.unlock()
    }

    private func add(_ key: String, since t: Date) {
        let dt = Date().timeIntervalSince(t)
        lock.lock()
        timings[key, default: 0] += dt
        lock.unlock()
    }

    // MARK: - Buffers

    private lazy var buffers = Buffers(self)

    final class Buffers {
        let q: [Surface], encSums: [Surface], encMsg: [Surface]
        let encCoef: [MLMultiArray]
        let x: [MLMultiArray]
        let m: [Surface], agg: Surface, ps: Surface, pr: Surface
        let e: [Surface], meshSums: [Surface], meshMsg: [Surface]
        let decSums: [Surface], decCoef: [MLMultiArray], out: [MLMultiArray]
        let acc: UnsafeMutablePointer<Float>
        let accCount: Int

        init(_ r: GraphCastRuntime) {
            let p = r.p, w = GraphCastRuntime.width
            let two = 0 ..< 2
            q = two.map { _ in Surface(rows: p.grid_chunk, cols: w) }
            encSums = two.map { _ in Surface(rows: p.enc_edges, cols: w) }
            encMsg = two.map { _ in Surface(rows: p.enc_edges, cols: w) }
            encCoef = two.map { _ in half([p.enc_edges, p.enc_rank]) }
            x = two.map { _ in half([p.grid_chunk, GraphCastRuntime.features]) }
            m = two.map { _ in Surface(rows: p.mesh_nodes, cols: w) }
            agg = Surface(rows: p.mesh_nodes, cols: w)
            ps = Surface(rows: p.mesh_nodes, cols: w)
            pr = Surface(rows: p.mesh_nodes, cols: w)
            e = (0 ..< r.meshEdges / p.mesh_edges).map { _ in Surface(rows: p.mesh_edges, cols: w) }
            meshSums = two.map { _ in Surface(rows: p.mesh_edges, cols: w) }
            meshMsg = two.map { _ in Surface(rows: p.mesh_edges, cols: w) }
            decSums = two.map { _ in Surface(rows: 3 * p.grid_chunk, cols: w) }
            decCoef = two.map { _ in half([3 * p.grid_chunk, p.rank]) }
            out = two.map { _ in half([p.grid_chunk, GraphCastRuntime.outputs]) }
            accCount = p.mesh_nodes * w
            acc = .allocate(capacity: accCount)
        }
        deinit { acc.deallocate() }
    }

    // MARK: - Step

    /// One step: `x` the grid nodes' features `(gridNodes, 184)` fp16. The 83 normalised
    /// outputs go to `output(first node, count, rows)` a decoder chunk at a time, `rows` being
    /// `count` x 83 fp16, on a background queue (one call at a time) while the Neural Engine
    /// runs the next chunk.
    public func step(features x: UnsafePointer<Float16>,
                     output: @escaping (Int, Int, UnsafePointer<Float16>) -> Void) throws {
        timings = [:]
        cpuGathers = 0
        let b = buffers
        let t0 = Date()
        try encode(x, b)
        add("encoder (all)", since: t0)
        let t1 = Date()
        try process(b)
        add("processor (all)", since: t1)
        let t2 = Date()
        try decode(x, output, b)
        add("decoder (all)", since: t2)
    }

    private var chunks: Int { (gridNodes + p.grid_chunk - 1) / p.grid_chunk }

    private func copyFeatures(_ x: UnsafePointer<Float16>, chunk c: Int, into a: MLMultiArray) {
        let lo = c * p.grid_chunk, n = min(p.grid_chunk, gridNodes - lo), f = Self.features
        let dst = a.dataPointer.assumingMemoryBound(to: Float16.self)
        dst.update(from: x + lo * f, count: n * f)
        if n < p.grid_chunk { (dst + n * f).update(repeating: 0, count: (p.grid_chunk - n) * f) }
    }

    /// Encoder: per grid chunk, the grid program (the edges' sender term), then that chunk's
    /// edges; their messages summed into the mesh nodes (float32), then the mesh update.
    private func encode(_ x: UnsafePointer<Float16>, _ b: Buffers) throws {
        b.acc.update(repeating: 0, count: b.accCount)
        let prep = DispatchQueue(label: "graphcast.encoder.gather"), fin = DispatchQueue(label: "graphcast.encoder.sum")
        let gathered = [DispatchGroup(), DispatchGroup()], summed = [DispatchGroup(), DispatchGroup()]
        let n = chunks
        let w = Self.width
        let acc = UnsafeSendable(b.acc)

        func gather(_ c: Int) {
            let s = c % 2, lo = encBounds[c], count = encBounds[c + 1] - lo, base = c * p.grid_chunk
            let q = UnsafeSendable(b.q[s]), sumsBuffer = UnsafeSendable(b.encSums[s])
            let coef = UnsafeSendable(b.encCoef[s].dataPointer.assumingMemoryBound(to: Float16.self))
            let me = UnsafeSendable(self)
            gathered[s].enter()
            prep.async {
                let t = Date()
                let r = me.value
                defer {
                    r.add("encoder gather", since: t)
                    gathered[s].leave()
                }
                let rank = r.p.enc_rank
                (coef.value).update(from: r.encCoef + lo * rank, count: count * rank)
                (coef.value + count * rank).update(repeating: 0, count: (r.p.enc_edges - count) * rank)
                q.value.rows(readOnly: true) { qr in
                    sumsBuffer.value.rows { sums in
                        parallel(count, grain: 1024) { i in
                            copyRow(sums.row(i), qr.row(Int(r.g2mSenders[lo + i]) - base))
                        }
                    }
                }
            }
        }

        func sum(_ c: Int) {
            let s = c % 2, lo = encBounds[c], count = encBounds[c + 1] - lo
            let msg = UnsafeSendable(b.encMsg[s]), me = UnsafeSendable(self)
            summed[s].enter()
            fin.async {
                let t = Date()
                let r = me.value
                msg.value.rows(readOnly: true) { m in
                    r.encOrder.withUnsafeBufferPointer { order in
                        r.encOrderReceivers.withUnsafeBufferPointer { recv in
                            let o = UnsafeSendable(order.baseAddress! + lo), rv = UnsafeSendable(recv.baseAddress! + lo)
                            sumRuns(count: count, receivers: rv.value) { i in
                                accumulate(acc.value + Int(rv.value[i]) * w, m.row(Int(o.value[i])))
                            }
                        }
                    }
                }
                r.add("encoder sum", since: t)
                summed[s].leave()
            }
        }

        for c in 0 ... n {
            if c < n {
                copyFeatures(x, chunk: c, into: b.x[c % 2])
                try run("grid", ["x": b.x[c % 2]], ["q": b.q[c % 2].array])
                gather(c)
            }
            if c >= 1 {
                let d = c - 1, s = d % 2
                gathered[s].wait()
                summed[s].wait()
                try run("enc_edges", ["coef": b.encCoef[s], "s": b.encSums[s].array], ["msg": b.encMsg[s].array])
                sum(d)
                onProgress?("encoder", Double(c) / Double(n))
            }
        }
        summed[0].wait()
        summed[1].wait()

        // The mesh update: m0 (constant) and the summed messages.
        let t = Date()
        b.m[0].fill(from: m0, rows: meshNodes)
        toHalf(b.acc, into: b.agg, rows: meshNodes)
        add("convert", since: t)
        try nodes("enc", m: b.m[0], newM: b.m[1], b)
    }

    private func nodes(_ prefix: String, m: Surface, newM: Surface, _ b: Buffers) throws {
        var inputs: [String: MLMultiArray] = ["m": m.array, "agg": b.agg.array]
        for k in ["wm", "wa", "b0", "w1", "b1", "scale", "offset", "ws", "wr", "br"] {
            inputs[k] = try weight("\(prefix).\(k)")
        }
        try run("mesh_nodes", inputs, ["m_new": newM.array, "ps": b.ps.array, "pr": b.pr.array])
    }

    private var weightArrays: [String: MLMultiArray] = [:]
    private func weight(_ name: String) throws -> MLMultiArray {
        if let a = weightArrays[name] { return a }
        let t = try weights.tensor(name)
        let a = try MLMultiArray(dataPointer: UnsafeMutableRawPointer(try weights.pointer(name, as: Float16.self)),
                                 shape: t.shape.map { NSNumber(value: $0) }, dataType: .float16,
                                 strides: [NSNumber(value: t.shape[1]), 1])
        weightArrays[name] = a
        return a
    }

    private var meshFeatureArrays: [MLMultiArray] = []

    /// The multimesh edges' features, a chunk per array (views of the mapped graph).
    private func meshFeatures() throws -> [MLMultiArray] {
        if meshFeatureArrays.isEmpty {
            let chunk = p.mesh_edges
            meshFeatureArrays = try (0 ..< meshEdges / chunk).map { j in
                try MLMultiArray(dataPointer: UnsafeMutableRawPointer(mutating: meshEdgeFeatures + j * chunk * 4),
                                 shape: [NSNumber(value: chunk), 4], dataType: .float16, strides: [4, 1])
            }
        }
        return meshFeatureArrays
    }

    /// Processor: the multimesh edges embedded (6 chunks), then 16 rounds of edge messages
    /// (summed per receiver, edges being in receiver order) and node updates.
    private func process(_ b: Buffers) throws {
        let chunk = p.mesh_edges, w = Self.width
        let nChunks = meshEdges / chunk
        let features = try meshFeatures()
        for j in 0 ..< nChunks {
            try run("mesh_embed", ["f": features[j]], ["e": b.e[j].array])
        }
        var m = 1
        let acc = UnsafeSendable(b.acc)
        for layer in 0 ..< 16 {
            b.acc.update(repeating: 0, count: b.accCount)
            let edgeWeights = try ["we", "w1", "b1", "scale", "offset"].reduce(into: [String: MLMultiArray]()) {
                $0[$1] = try weight("edge\(layer).\($1)")
            }
            let ps = UnsafeSendable(b.ps), pr = UnsafeSendable(b.pr), me = UnsafeSendable(self)
            try pipelined(count: nChunks, label: "graphcast.processor", prepare: { j, s in
                let t = Date()
                let r = me.value
                defer { r.add("processor gather", since: t) }
                if let g = r.gpu, let out = g.buffer(b.meshSums[s]), let psb = g.buffer(ps.value), let prb = g.buffer(pr.value),
                   g.gatherAdd(out: .init(buffer: out, offset: 0), a: .init(buffer: psb, offset: 0), aBase: 0,
                               b: .init(buffer: prb, offset: 0),
                               ia: .init(buffer: g.graph, offset: r.offsets["mesh_senders"]! + j * chunk * 4),
                               ib: .init(buffer: g.graph, offset: r.offsets["mesh_receivers"]! + j * chunk * 4), rows: chunk) {
                    return
                }
                r.countCPUGather()
                ps.value.rows(readOnly: true) { psr in
                    pr.value.rows(readOnly: true) { prr in
                        b.meshSums[s].rows { sums in
                            parallel(chunk, grain: 1024) { i in
                                let e = j * chunk + i
                                addRows(sums.row(i), psr.row(Int(r.meshSenders[e])), prr.row(Int(r.meshReceivers[e])))
                            }
                        }
                    }
                }
            }, run: { j, s in
                var inputs = edgeWeights
                inputs["e"] = b.e[j].array
                inputs["s"] = b.meshSums[s].array
                // The updated edge latents go back into the buffer they came from: each row of
                // e_new depends only on its own row of e (checked against separate buffers).
                try self.run("mesh_edges", inputs, ["msg": b.meshMsg[s].array, "e_new": b.e[j].array])
            }, finish: { j, s in
                let t = Date()
                let r = me.value
                let recv = UnsafeSendable(r.meshReceivers + j * chunk)
                b.meshMsg[s].rows(readOnly: true) { msg in
                    sumRuns(count: chunk, receivers: recv.value) { i in
                        accumulate(acc.value + Int(recv.value[i]) * w, msg.row(i))
                    }
                }
                r.add("processor sum", since: t)
            })
            let t = Date()
            toHalf(b.acc, into: b.agg, rows: meshNodes)
            add("convert", since: t)
            try nodes("node\(layer)", m: b.m[m], newM: b.m[1 - m], b)
            m = 1 - m
            onProgress?("processor", Double(layer + 1) / 16)
        }
    }

    /// Decoder: per grid chunk, each node's 3 edges stacked slot-major, their sender term
    /// gathered from the last node update's projection.
    private func decode(_ x: UnsafePointer<Float16>, _ output: @escaping (Int, Int, UnsafePointer<Float16>) -> Void,
                        _ b: Buffers) throws {
        let n = p.grid_chunk, rank = p.rank
        let ps = UnsafeSendable(b.ps), me = UnsafeSendable(self), xs = UnsafeSendable(x)
        try pipelined(count: chunks, label: "graphcast.decoder", prepare: { c, s in
            let t = Date()
            let r = me.value, lo = c * n, count = min(n, r.gridNodes - lo)
            let coef = UnsafeSendable(b.decCoef[s].dataPointer.assumingMemoryBound(to: Float16.self))
            ps.value.rows(readOnly: true) { psr in
                b.decSums[s].rows { sums in
                    parallel(count, grain: 512) { i in
                        let node = lo + i
                        for slot in 0 ..< 3 {
                            let row = slot * n + i
                            copyRow(sums.row(row), psr.row(Int(r.m2gSenders[node * 3 + slot])))
                            (coef.value + row * rank).update(from: r.decCoef + (node * 3 + slot) * rank, count: rank)
                        }
                    }
                }
            }
            r.copyFeatures(xs.value, chunk: c, into: b.x[s])
            r.add("decoder gather", since: t)
        }, run: { c, s in
            try self.run("decoder", ["x": b.x[s], "coef": b.decCoef[s], "s": b.decSums[s].array], ["out": b.out[s]])
            self.onProgress?("decoder", Double(c + 1) / Double(self.chunks))
        }, finish: { c, s in
            let r = me.value, lo = c * n, count = min(n, r.gridNodes - lo)
            let t = Date()
            output(lo, count, UnsafePointer(b.out[s].dataPointer.assumingMemoryBound(to: Float16.self)))
            r.add("outputs", since: t)
        })
    }

    /// Runs items through `prepare` (CPU, one item ahead), `run` (this thread: the Neural
    /// Engine) and `finish` (CPU, behind), with two sets of buffers (slot = item % 2).
    private func pipelined(count: Int, label: String, prepare: @escaping (Int, Int) -> Void,
                           run: (Int, Int) throws -> Void, finish: @escaping (Int, Int) -> Void) throws {
        let prep = DispatchQueue(label: "\(label).prepare"), fin = DispatchQueue(label: "\(label).finish")
        let prepared = [DispatchGroup(), DispatchGroup()], finished = [DispatchGroup(), DispatchGroup()]
        func start(_ i: Int) {
            prepared[i % 2].enter()
            prep.async { prepare(i, i % 2); prepared[i % 2].leave() }
        }
        start(0)
        for i in 0 ..< count {
            let s = i % 2
            if i + 1 < count { start(i + 1) }
            prepared[s].wait()
            finished[s].wait()
            try run(i, s)
            finished[s].enter()
            fin.async { finish(i, s); finished[s].leave() }
        }
        finished[0].wait()
        finished[1].wait()
    }
}

// MARK: - Memory

/// A (rows, cols) fp16 tensor in an IOSurface-backed pixel buffer, which the Neural Engine
/// reads and writes without copies. The CPU locks it around its own work (`rows()`), never
/// across a Neural Engine call: Core ML refuses locked buffers.
final class Surface: @unchecked Sendable {
    let rows: Int, cols: Int
    let buffer: CVPixelBuffer
    let array: MLMultiArray

    init(rows: Int, cols: Int) {
        var pb: CVPixelBuffer?
        let attrs: [CFString: Any] = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
                                      kCVPixelBufferMetalCompatibilityKey: true]
        let status = CVPixelBufferCreate(nil, cols, rows, kCVPixelFormatType_OneComponent16Half, attrs as CFDictionary, &pb)
        precondition(status == kCVReturnSuccess, "IOSurface \(rows)x\(cols): \(status)")
        buffer = pb!
        self.rows = rows
        self.cols = cols
        precondition(CVPixelBufferGetBytesPerRow(buffer) == cols * 2, "padded rows for \(cols) columns")
        array = MLMultiArray(pixelBuffer: buffer, shape: [NSNumber(value: rows), NSNumber(value: cols)])
    }

    /// Runs `body` with the rows locked for the CPU.
    func rows<T>(readOnly: Bool = false, _ body: (Rows) throws -> T) rethrows -> T {
        let flags: CVPixelBufferLockFlags = readOnly ? .readOnly : []
        CVPixelBufferLockBaseAddress(buffer, flags)
        defer { CVPixelBufferUnlockBaseAddress(buffer, flags) }
        return try body(Rows(base: CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: Float16.self), cols: cols))
    }

    /// The first `n` rows from `src` (contiguous), the rest zero.
    func fill(from src: UnsafePointer<Float16>, rows n: Int) {
        rows { r in
            r.base.update(from: src, count: n * cols)
            (r.base + n * cols).update(repeating: 0, count: (self.rows - n) * cols)
        }
    }
}

/// A locked Surface's rows.
struct Rows: @unchecked Sendable {
    let base: UnsafeMutablePointer<Float16>
    let cols: Int
    @inline(__always) func row(_ i: Int) -> UnsafeMutablePointer<Float16> { base + i * cols }
}

func half(_ shape: [Int]) -> MLMultiArray {
    try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
}

/// Copies `a` into `b` (fp16, same shape; either may have padded rows).
func copy(_ a: MLMultiArray, into b: MLMultiArray) {
    let shape = a.shape.map(\.intValue)
    let inner = shape.last!, outer = a.count / inner
    let sa = a.strides.map(\.intValue)[shape.count - 2], sb = b.strides.map(\.intValue)[shape.count - 2]
    let pa = a.dataPointer.assumingMemoryBound(to: Float16.self), pb = b.dataPointer.assumingMemoryBound(to: Float16.self)
    for o in 0 ..< outer { (pb + o * sb).update(from: pa + o * sa, count: inner) }
}

typealias Row = SIMD32<Float16>  // 512 = 16 x 32

@inline(__always) func copyRow(_ dst: UnsafeMutablePointer<Float16>, _ src: UnsafePointer<Float16>) {
    UnsafeMutableRawPointer(dst).copyMemory(from: src, byteCount: 1024)
}

/// dst = a + b, rows of 512 (b may be unaligned: mmap'd tables).
@inline(__always) func addRows(_ dst: UnsafeMutablePointer<Float16>, _ a: UnsafePointer<Float16>, _ b: UnsafePointer<Float16>) {
    let d = UnsafeMutableRawPointer(dst), x = UnsafeRawPointer(a), y = UnsafeRawPointer(b)
    for k in 0 ..< 16 {
        let o = k * 64
        d.storeBytes(of: x.loadUnaligned(fromByteOffset: o, as: Row.self) + y.loadUnaligned(fromByteOffset: o, as: Row.self),
                     toByteOffset: o, as: Row.self)
    }
}

@inline(__always) func addInPlace(_ dst: UnsafeMutablePointer<Float16>, _ a: UnsafePointer<Float16>) {
    addRows(dst, UnsafePointer(dst), a)
}

/// acc (512 float32) += row (512 fp16).
@inline(__always) func accumulate(_ acc: UnsafeMutablePointer<Float>, _ row: UnsafePointer<Float16>) {
    let a = UnsafeMutableRawPointer(acc), r = UnsafeRawPointer(row)
    for k in 0 ..< 32 {
        let v = SIMD16<Float>(r.loadUnaligned(fromByteOffset: k * 32, as: SIMD16<Float16>.self))
        a.storeBytes(of: a.loadUnaligned(fromByteOffset: k * 64, as: SIMD16<Float>.self) + v, toByteOffset: k * 64, as: SIMD16<Float>.self)
    }
}

/// The float32 sums into fp16 rows of `dst` (the rest of its rows zero).
func toHalf(_ acc: UnsafeMutablePointer<Float>, into dst: Surface, rows n: Int) {
    let a = UnsafeSendable(acc)
    dst.rows { d in
        parallel(n, grain: 512) { i in
            let src = UnsafeRawPointer(a.value + i * 512), out = UnsafeMutableRawPointer(d.row(i))
            for k in 0 ..< 32 {
                out.storeBytes(of: SIMD16<Float16>(src.loadUnaligned(fromByteOffset: k * 64, as: SIMD16<Float>.self)),
                               toByteOffset: k * 32, as: SIMD16<Float16>.self)
            }
        }
        (d.base + n * dst.cols).update(repeating: 0, count: (dst.rows - n) * dst.cols)
    }
}

/// `body(i)` for i in 0..<count across the cores, in blocks that never split a run of equal
/// receivers (sorted), so no two threads add into the same node.
func sumRuns(count: Int, receivers: UnsafePointer<Int32>, _ body: (Int) -> Void) {
    let blocks = 64
    var cuts = [0]
    for k in 1 ..< blocks {
        var c = max(count * k / blocks, cuts.last!)
        while c > 0 && c < count && receivers[c] == receivers[c - 1] { c += 1 }
        cuts.append(min(c, count))
    }
    cuts.append(count)
    cuts.withUnsafeBufferPointer { cut in
        let cs = UnsafeSendable(cut.baseAddress!)
        DispatchQueue.concurrentPerform(iterations: blocks) { k in
            for i in cs.value[k] ..< cs.value[k + 1] { body(i) }
        }
    }
}

/// `body(i)` for i in 0..<n across the cores, `grain` iterations per task.
func parallel(_ n: Int, grain: Int = 1, _ body: (Int) -> Void) {
    let tasks = (n + grain - 1) / grain
    DispatchQueue.concurrentPerform(iterations: tasks) { k in
        for i in k * grain ..< min(n, (k + 1) * grain) { body(i) }
    }
}

/// A value shared across threads where the code guarantees it is safe.
struct UnsafeSendable<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}
