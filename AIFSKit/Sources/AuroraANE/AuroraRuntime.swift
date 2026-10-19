import Accelerate
import CoreML
import Foundation

/// Aurora 1.5 on the Neural Engine: drives the Core ML programs that
/// `aurora-1p5/aurora_1p5/ane_export.py` writes, one 6-hour step at a time.
///
/// State lives in normalised fp16, which is what the programs consume: the two input times'
/// surface fields `(19, 2, 720, 1440)` and atmospheric fields `(5, 2, 13, 720, 1440)`, as
/// `ane_run.py` lays them out. A step writes the normalised prediction, surface
/// `(26, 720, 1440)` and atmospheric `(5, 13, 720, 1440)`.
///
/// The backbone's tokens are one fp16 stream in natural order, a row of D channels per
/// token. A Swin block's output for a window depends only on that window's input, and the
/// windows partition the tokens, so each call gathers its windows' rows, runs, and writes
/// the result back to the same rows: a stage updates its stream in place.
///
/// With `resident` false, each program is loaded while the ones before it run (`prefetch`
/// ahead) and released after its last call, so at most `prefetch` + 1 programs' weights are in
/// memory at a time.
///
/// With `gpuStages`, the GPU runs those stages' blocks beside the Neural Engine: a block's
/// chunks go to whichever is free, so the two share the work in proportion to their speeds.
/// The GPU holds its weights expanded, so stage 3 -- 64% of the weights, a quarter of the
/// time -- is best left to the Neural Engine alone.
public final class AuroraRuntime {
    public enum Unit: String, Sendable, CaseIterable { case neuralEngine = "ANE", gpu = "GPU" }

    public struct Options {
        public var resident: Bool
        /// Stages (1, 2, 3) whose blocks the GPU runs too.
        public var gpuStages: Set<Int>
        /// Streaming: how many programs ahead to load.
        public var prefetch: Int
        public init(resident: Bool = true, gpuStages: Set<Int> = [], prefetch: Int = 2) {
            self.resident = resident
            self.gpuStages = gpuStages
            self.prefetch = prefetch
        }
    }

    struct Manifest: Decodable {
        struct Stage: Decodable {
            let dim: Int
            let windows_per_call: Int
            let res: [Int]
            let slots: Int
        }
        struct Layer: Decodable {
            let prefix: String
            let stage: Int
            let blocks: [String]
        }
        let patch_chunk: Int
        let stages: [String: Stage]
        let layers: [Layer]
        let surf_vars: [String]
    }

    public static let height = 720, width = 1440
    static let hp = 180, wp = 360, patches = 180 * 360

    public let directory: URL
    public let options: Options
    let manifest: Manifest
    let tables: SafeTensorsFile
    private var models: [String: MLModel] = [:]
    private var pending: [String: DispatchGroup] = [:]  // loads in flight (streaming)
    private let lock = NSLock()
    /// Every program in the order a step runs them.
    private let order: [String]
    public var programNames: [String] { order }
    /// Each block's stage.
    private let stageOf: [String: Int]

    /// Seconds per phase over the last `step` (Neural Engine calls, data movement, loading).
    public private(set) var timings: [String: Double] = [:]
    /// Program loads in the last `step`, and their total time.
    public private(set) var loads = 0
    /// Block chunks each unit ran in the last `step`.
    public private(set) var chunksRun: [Unit: Int] = [:]
    /// Seconds each program load took in the last `step` (in the background or not).
    public private(set) var loadTimes: [String: Double] = [:]

    public var surfaceVariables: [String] { manifest.surf_vars }

    /// Called as a step goes: ("encoder" | "processor" | "decoder", fraction of that phase
    /// done), between Neural Engine calls -- where a backgrounded app can safely wait.
    public var onProgress: ((String, Double) -> Void)?
    private var blocksDone = 0

    public init(directory: URL, options: Options = Options()) throws {
        self.directory = directory
        self.options = options
        manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: directory.appending(path: "manifest.json")))
        tables = try SafeTensorsFile(directory.appending(path: "tables.safetensors"))
        let l = manifest.layers.map(\.blocks)
        stageOf = Dictionary(uniqueKeysWithValues: manifest.layers.flatMap { layer in layer.blocks.map { ($0, layer.stage) } })
        order = ["encoder"] + l[0] + ["merge_1"] + l[1] + ["merge_2"] + l[2] + l[3] + ["split_3"] + l[4]
            + ["split_2"] + l[5] + ["decoder"]
    }

    /// Loads every program, so that `step` does not (resident mode). The first load on a
    /// device compiles each for the Neural Engine, which takes a while; later loads hit the
    /// system's cache.
    public func preload() throws {
        for name in order {
            _ = try model(name, .neuralEngine)
            if units(for: name).contains(.gpu) { _ = try model(name, .gpu) }
        }
    }

    /// Loads every Neural Engine program once and lets it go again. The first time on a
    /// device (and after an app update) each is compiled for the Neural Engine -- minutes in
    /// all, on an iPhone -- and the system caches the result; after that this takes well
    /// under a second. `progress(fraction)`, which may throw to stop, counts programs by
    /// what compiling them costs: on an iPhone 18 Pro Max, 2.4 s each plus 0.086 s per MB
    /// of weights -- the size plus 27.5 MB.
    public func prepare(progress: (Double) throws -> Void) throws {
        let sizes = order.map { name in
            let weights = directory.appending(path: "\(name).mlmodelc/weights/weight.bin")
            let bytes = ((try? FileManager.default.attributesOfItem(atPath: weights.path))?[.size] as? Int) ?? 0
            return bytes + 27_500_000
        }
        let total = Double(sizes.reduce(0, +))
        var done = 0
        try progress(0)
        // One at a time: the system compiles one program at a time anyway (54 take 63 s on an
        // M4 Max whether 1, 4 or 8 are requested at once).
        for (name, size) in zip(order, sizes) {
            if options.resident { _ = try model(name, .neuralEngine) } else { _ = try load(name, .neuralEngine) }
            if units(for: name).contains(.gpu) { _ = try model(name, .gpu) }  // kept: see `release`
            done += size
            try progress(Double(done) / total)
        }
    }

    // MARK: - Programs

    private func key(_ name: String, _ unit: Unit) -> String { "\(name)@\(unit.rawValue)" }

    private func load(_ name: String, _ unit: Unit) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = unit == .gpu ? .cpuAndGPU : .cpuAndNeuralEngine
        let t = Date()
        let m = try MLModel(contentsOf: directory.appending(path: "\(name).mlmodelc"), configuration: config)
        let dt = Date().timeIntervalSince(t)
        lock.lock()
        loadTimes[key(name, unit)] = dt
        lock.unlock()
        return m
    }

    private func model(_ name: String, _ unit: Unit) throws -> MLModel {
        let k = key(name, unit)
        lock.lock()
        if let m = models[k] { lock.unlock(); return m }
        let inFlight = pending[k]
        lock.unlock()
        if let inFlight {
            let t = Date()
            inFlight.wait()
            add("load", since: t)  // the part of a background load not hidden
            lock.lock()
            defer { lock.unlock() }
            if let m = models[k] { return m }
        }
        let t = Date()
        let m = try load(name, unit)
        add("load", since: t)
        lock.lock()
        loads += 1
        models[k] = m
        lock.unlock()
        return m
    }

    /// The units that run `name`: blocks of the GPU's stages on both, the rest on the Neural Engine.
    private func units(for name: String) -> [Unit] {
        guard let s = stageOf[name], options.gpuStages.contains(s) else { return [.neuralEngine] }
        return [.neuralEngine, .gpu]
    }

    /// Streaming: start loading the programs after `name`, to be ready when they are needed.
    private func prefetch(after name: String) {
        guard !options.resident, let i = order.firstIndex(of: name) else { return }
        for next in order[(i + 1) ..< min(order.count, i + 1 + options.prefetch)] { prefetch(next, .neuralEngine) }
    }

    private func prefetch(_ next: String, _ unit: Unit) {
        let k = key(next, unit)
        lock.lock()
        defer { lock.unlock() }
        guard models[k] == nil, pending[k] == nil else { return }
        let group = DispatchGroup()
        pending[k] = group
        DispatchQueue.global(qos: .userInitiated).async(group: group) { [self] in
            let m = try? load(next, unit)  // on failure, `model` loads it again and reports the error
            lock.lock()
            if let m { models[k] = m; loads += 1 }
            pending[k] = nil
            lock.unlock()
        }
    }

    /// Streaming releases Neural Engine programs only: a GPU program recompiles on every load
    /// (seconds each), so those stay loaded once compiled.
    private func release(_ name: String) {
        guard !options.resident else { return }
        lock.lock()
        models[key(name, .neuralEngine)] = nil
        lock.unlock()
    }

    private func add(_ key: String, since t: Date) {
        let dt = Date().timeIntervalSince(t)
        lock.lock()
        timings[key, default: 0] += dt
        lock.unlock()
    }

    /// Runs `name`, writing its outputs into `outputs` (buffers we own).
    private func run(_ name: String, _ key: String, unit: Unit = .neuralEngine, inputs: [String: MLMultiArray],
                     outputs: [String: MLMultiArray]) throws {
        let m = try model(name, unit)
        prefetch(after: name)
        let options = MLPredictionOptions()
        options.outputBackings = outputs
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
        let t = Date()
        let result = try m.prediction(from: provider, options: options)
        add(unit == .gpu ? "\(key) (GPU)" : key, since: t)
        // Core ML writes into a backing it can use; copy if it chose its own buffer instead.
        for (k, backing) in outputs {
            guard let got = result.featureValue(for: k)?.multiArrayValue, got.dataPointer != backing.dataPointer else { continue }
            precondition(got.count == backing.count && got.dataType == .float16, "unexpected output \(k)")
            copyContiguous(got, into: backing)
        }
    }

    private func copyContiguous(_ a: MLMultiArray, into b: MLMultiArray) {
        // Both fp16 with the same shape; `a` may be strided (padded rows).
        let shape = a.shape.map(\.intValue)
        let sa = a.strides.map(\.intValue), sb = b.strides.map(\.intValue)
        let pa = a.dataPointer.assumingMemoryBound(to: Float16.self)
        let pb = b.dataPointer.assumingMemoryBound(to: Float16.self)
        let inner = shape.last!
        let outer = a.count / inner
        for o in 0 ..< outer {
            var rem = o, oa = 0, ob = 0
            for d in stride(from: shape.count - 2, through: 0, by: -1) {
                let i = rem % shape[d]
                rem /= shape[d]
                oa += i * sa[d]
                ob += i * sb[d]
            }
            if sa.last == 1 && sb.last == 1 {
                (pb + ob).update(from: pa + oa, count: inner)
            } else {
                for i in 0 ..< inner { pb[ob + i * sb.last!] = pa[oa + i * sa.last!] }
            }
        }
    }

    // MARK: - A step

    /// One 6-hour step from the state valid at `time` (the later input time).
    public func step(
        surface: UnsafePointer<Float16>, atmosphere: UnsafePointer<Float16>, time: Date,
        surfaceOut: UnsafeMutablePointer<Float16>, atmosphereOut: UnsafeMutablePointer<Float16>
    ) throws {
        timings = [:]
        loads = 0
        chunksRun = [:]
        loadTimes = [:]
        let L = Self.patches
        let stream = Buffer(count: 4 * L * 512)
        blocksDone = 0
        onProgress?("encoder", 0)
        try encode(surface: surface, atmosphere: atmosphere, time: time, into: stream.p)
        let (x, skip) = try backbone(stream)
        onProgress?("decoder", 0)
        try decode(x: x.p, skip: skip.p, surfaceOut: surfaceOut, atmosphereOut: atmosphereOut)
    }

    // MARK: Encoder

    private func encode(surface: UnsafePointer<Float16>, atmosphere: UnsafePointer<Float16>, time: Date,
                        into stream: UnsafeMutablePointer<Float16>) throws {
        let S = manifest.patch_chunk, L = Self.patches, W = Self.width, H = Self.height
        let rowsPerChunk = S / Self.wp  // patch rows per call
        let sv = Buffer(count: 608 * S), av = Buffer(count: 160 * 13 * S), att = Buffer(count: 48 * 13 * S)
        let tokens = Buffer(count: 4 * S * 512)
        let avf = UnsafeMutablePointer<Float>.allocate(capacity: 160 * 13 * S)
        let scores = UnsafeMutablePointer<Float>.allocate(capacity: 48 * 13 * S)
        defer { avf.deallocate(); scores.deallocate() }
        let staticT = try tables.pointer("static", as: Float16.self)
        let posc = try tables.pointer("posc", as: Float16.self)
        let scoreW = try tables.pointer("enc_score_w", as: Float.self)
        let scoreC = try tables.pointer("enc_score_c", as: Float.self)
        let timeEmbed = Buffer(count: 512)
        try timeEmbedding(time, into: timeEmbed.p)

        for j in 0 ..< L / S {
            var t = Date()
            let hp0 = j * rowsPerChunk
            // Patch vectors, rows (variable, time, py, px), one column per patch.
            parallel(608) { r in
                let vt = r / 16, py = (r / 4) % 4, px = r % 4
                let dst = sv.p + r * S
                for hr in 0 ..< rowsPerChunk {
                    let src = surface + (vt * H + (hp0 + hr) * 4 + py) * W + px
                    for w in 0 ..< Self.wp { dst[hr * Self.wp + w] = src[w * 4] }
                }
            }
            parallel(160 * 13) { rc in
                let r = rc / 13, c = rc % 13
                let vt = r / 16, py = (r / 4) % 4, px = r % 4
                let dst = av.p + (r * 13 + c) * S
                for hr in 0 ..< rowsPerChunk {
                    let src = atmosphere + ((vt * 13 + c) * H + (hp0 + hr) * 4 + py) * W + px
                    for w in 0 ..< Self.wp { dst[hr * Self.wp + w] = src[w * 4] }
                }
            }
            add("encoder data", since: t)

            // Level-aggregation attention in float32 (see ane_export). Per level c, its patch
            // values (160, S) contiguous, scores (48, S) = W (48, 160) x values: laid out
            // (level, score, patch), then softmax over the levels and transposed to the
            // program's (score, level, patch).
            t = Date()
            let f32 = UnsafeSendable(avf), sc = UnsafeSendable(scores), half = UnsafeSendable(av.p)
            parallel(13) { c in
                let values = f32.value + c * 160 * S
                for r in 0 ..< 160 { convert(half.value + (r * 13 + c) * S, to: values + r * S, count: S) }
                vDSP_mmul(scoreW, 1, values, 1, sc.value + c * 48 * S, 1, 48, vDSP_Length(S), 160)
            }
            parallel(48) { s in
                func row(_ c: Int) -> UnsafeMutablePointer<Float> { sc.value + (c * 48 + s) * S }
                for c in 0 ..< 13 {
                    var k = scoreC[s * 13 + c]
                    vDSP_vsadd(row(c), 1, &k, row(c), 1, vDSP_Length(S))
                }
                // Softmax over the 13 levels, per patch.
                let m = UnsafeMutablePointer<Float>.allocate(capacity: S)
                let sum = UnsafeMutablePointer<Float>.allocate(capacity: S)
                defer { m.deallocate(); sum.deallocate() }
                m.update(from: row(0), count: S)
                for c in 1 ..< 13 { vDSP_vmax(m, 1, row(c), 1, m, 1, vDSP_Length(S)) }
                var n = Int32(S)
                for c in 0 ..< 13 {
                    vDSP_vsub(m, 1, row(c), 1, row(c), 1, vDSP_Length(S))
                    vvexpf(row(c), row(c), &n)
                }
                sum.update(from: row(0), count: S)
                for c in 1 ..< 13 { vDSP_vadd(sum, 1, row(c), 1, sum, 1, vDSP_Length(S)) }
                for c in 0 ..< 13 {
                    vDSP_vdiv(sum, 1, row(c), 1, row(c), 1, vDSP_Length(S))
                    convert(row(c), to: att.p + (s * 13 + c) * S, count: S)
                }
            }
            add("encoder scores", since: t)

            let inputs: [String: MLMultiArray] = [
                "surf": try array(sv.p, [1, 608, 1, S]),
                "static": try array(staticT + j * S, [1, 512, 1, S], strides: [512 * L, L, S, 1]),
                "atmos": try array(av.p, [1, 160, 13, S]),
                "posc": try array(posc + j * S, [1, 512, 1, S], strides: [512 * L, L, S, 1]),
                "time": try array(timeEmbed.p, [1, 512, 1, 1]),
                "a": try array(att.p, [3, 16, 13, S]),
            ]
            try run("encoder", "encoder", inputs: inputs, outputs: ["tokens": try array(tokens.p, [4, S, 512])])
            t = Date()
            for c in 0 ..< 4 {
                (stream + (c * L + j * S) * 512).update(from: tokens.p + c * S * 512, count: S * 512)
            }
            add("encoder data", since: t)
        }
        release("encoder")
    }

    /// Aurora's absolute-time embedding: a Fourier expansion of the hours since 1970 (as a
    /// float32, as Aurora has it), computed in float64, then a linear layer.
    private func timeEmbedding(_ time: Date, into out: UnsafeMutablePointer<Float16>) throws {
        let hours = Double(Float(time.timeIntervalSince1970 / 3600))
        let half = 256
        let lo = log10(1.0), hi = log10(24 * 365.25)
        var e = [Float](repeating: 0, count: 512)
        for i in 0 ..< half {
            let wavelength = pow(10, lo + Double(i) * (hi - lo) / Double(half - 1))
            let x = hours * 2 * .pi / wavelength
            e[i] = Float(sin(x))
            e[half + i] = Float(cos(x))
        }
        let w = try tables.pointer("time_w", as: Float.self), b = try tables.pointer("time_b", as: Float.self)
        for o in 0 ..< 512 {
            var acc = b[o]
            vDSP_dotpr(w + o * 512, 1, e, 1, &acc, vDSP_Length(512))
            out[o] = Float16(acc + b[o])
        }
    }

    // MARK: Backbone

    private func stage(_ s: Int) -> Manifest.Stage { manifest.stages[String(s)]! }

    private func backbone(_ x0: Buffer) throws -> (Buffer, Buffer) {
        let layers = manifest.layers
        var x = x0
        try runStage(layers[0], x)
        let skip = x
        x = try merge(1, skip)
        try runStage(layers[1], x)
        x = try merge(2, x)
        try runStage(layers[2], x)
        try runStage(layers[3], x)
        x = try split(3, x)
        try runStage(layers[4], x)
        x = try split(2, x)
        let t = Date()
        add(skip.p, into: x.p, count: x.count)
        add("stream data", since: t)
        try runStage(layers[5], x)
        return (x, skip)
    }

    private func runStage(_ layer: Manifest.Layer, _ x: Buffer) throws {
        for (b, name) in layer.blocks.enumerated() {
            try block(name, x.p, stage: layer.stage, shifted: b % 2 == 1)
            release(name)
            blocksDone += 1
            onProgress?("processor", Double(blocksDone) / Double(manifest.layers.reduce(0) { $0 + $1.blocks.count }))
        }
    }

    /// One Swin block, in place on the natural-order stream `x` (N, D).
    ///
    /// Every unit runs a worker that claims the next chunk, runs it, and meanwhile -- on other
    /// cores -- gathers the rows of the chunk it claims next and writes back the one it ran
    /// before. Chunks hold disjoint windows, so none of this touches the same rows twice.
    private func block(_ name: String, _ x: UnsafeMutablePointer<Float16>, stage s: Int, shifted: Bool) throws {
        let info = stage(s)
        let D = info.dim, n = info.windows_per_call, per = n * 144
        let chunks = info.slots / per
        // Shared by the workers; each writes rows no other touches (see above).
        let src = UnsafeSendable(try tables.pointer(shifted ? "src_\(s)_shifted" : "src_\(s)", as: Int32.self))
        let stream = UnsafeSendable(x)
        let group = try tables.pointer("group_\(s)", as: Float16.self)
        let claim = Counter(limit: chunks)
        let failure = FirstError()
        let units = self.units(for: name)
        let done = DispatchGroup()
        for unit in units {
            DispatchQueue.global(qos: .userInitiated).async(group: done) { [self] in
                do {
                    let inputs = [Buffer(count: per * D), Buffer(count: per * D)]
                    let outputs = [Buffer(count: per * D), Buffer(count: per * D)]
                    let xin = try inputs.map { try array($0.p, [1, per, D]) }
                    let yout = try outputs.map { try array($0.p, [1, per, D]) }
                    func gather(_ j: Int, _ slot: Int) {
                        let idx = src.value + j * per, dst = inputs[slot].p, x = stream.value
                        parallel(per, grain: 144) { q in
                            let i = Int(idx[q])
                            if i < 0 {
                                (dst + q * D).update(repeating: 0, count: D)
                            } else {
                                (dst + q * D).update(from: x + i * D, count: D)
                            }
                        }
                    }
                    func scatter(_ j: Int, _ slot: Int) {
                        let idx = src.value + j * per, from = outputs[slot].p, x = stream.value
                        parallel(per, grain: 144) { q in
                            let i = Int(idx[q])
                            if i >= 0 { (x + i * D).update(from: from + q * D, count: D) }
                        }
                    }
                    guard var j = claim.next() else { return }
                    var t = Date()
                    gather(j, 0)
                    add("stream data", since: t)
                    var slot = 0, previous: (chunk: Int, slot: Int)? = nil, ran = 0
                    while true {
                        let next = claim.next()
                        let side = DispatchGroup()
                        let (p, other) = (previous, 1 - slot)
                        DispatchQueue.global(qos: .userInitiated).async(group: side) {
                            if let next { gather(next, other) }
                            if let p { scatter(p.chunk, p.slot) }
                        }
                        var features = ["x": xin[slot]]
                        if shifted { features["g"] = try array(group + j * per, [1, 1, n, 144]) }
                        try run(name, "stage \(s)", unit: unit, inputs: features, outputs: ["y": yout[slot]])
                        ran += 1
                        t = Date()
                        side.wait()  // data movement the units did not hide
                        add("stream data", since: t)
                        previous = (j, slot)
                        guard let n2 = next, failure.error == nil else { break }
                        (j, slot) = (n2, other)
                    }
                    if let p = previous {
                        t = Date()
                        scatter(p.chunk, p.slot)
                        add("stream data", since: t)
                    }
                    lock.lock()
                    chunksRun[unit, default: 0] += ran
                    lock.unlock()
                } catch {
                    failure.set(error)
                }
            }
        }
        done.wait()
        if let error = failure.error { throw error }
    }

    /// Patch merging: (C, H, W, D) -> (C, H/2, W/2, 2D).
    private func merge(_ s: Int, _ x: Buffer) throws -> Buffer {
        let r = stage(s).res, D = stage(s).dim
        let (C, H, W) = (r[0], r[1], r[2])
        let N = C * (H / 2) * (W / 2)
        let S = N / (s == 1 ? 15 : 5)
        let out = Buffer(count: N * 2 * D)
        let parts = (0 ..< 4).map { _ in Buffer(count: S * D) }
        let arrays = try parts.map { try array($0.p, [1, S, D]) }
        for j in 0 ..< N / S {
            var t = Date()
            parallel(S, grain: 90) { q in
                let o = j * S + q
                let c = o / ((H / 2) * (W / 2)), i = (o / (W / 2)) % (H / 2), w = o % (W / 2)
                for (k, (a, b)) in [(0, 0), (0, 1), (1, 0), (1, 1)].enumerated() {
                    (parts[k].p + q * D).update(from: x.p + ((c * H + 2 * i + a) * W + 2 * w + b) * D, count: D)
                }
            }
            add("stream data", since: t)
            t = Date()
            try run("merge_\(s)", "merge/split",
                    inputs: ["x00": arrays[0], "x01": arrays[1], "x10": arrays[2], "x11": arrays[3]],
                    outputs: ["y": try array(out.p + j * S * 2 * D, [1, S, 2 * D])])
        }
        release("merge_\(s)")
        return out
    }

    /// Patch splitting: (C, H, W, D) -> (C, 2H, 2W, D/2).
    private func split(_ s: Int, _ x: Buffer) throws -> Buffer {
        let r = stage(s).res, D = stage(s).dim
        let (C, H, W) = (r[0], r[1], r[2])
        let N = C * H * W, d = D / 2
        let S = N / (D == 2048 ? 4 : 15)
        let out = Buffer(count: 4 * N * d)
        let parts = (0 ..< 4).map { _ in Buffer(count: S * d) }
        let outputs = try Dictionary(uniqueKeysWithValues: zip(["y00", "y01", "y10", "y11"], parts).map { ($0, try array($1.p, [1, S, d])) })
        for j in 0 ..< N / S {
            try run("split_\(s)", "merge/split", inputs: ["x": try array(x.p + j * S * D, [1, S, D])], outputs: outputs)
            let t = Date()
            parallel(S, grain: 90) { q in
                let o = j * S + q
                let c = o / (H * W), h = (o / W) % H, w = o % W
                for (k, (a, b)) in [(0, 0), (0, 1), (1, 0), (1, 1)].enumerated() {
                    (out.p + ((c * 2 * H + 2 * h + a) * 2 * W + 2 * w + b) * d).update(from: parts[k].p + q * d, count: d)
                }
            }
            add("stream data", since: t)
        }
        release("split_\(s)")
        return out
    }

    // MARK: Decoder

    private func decode(x: UnsafeMutablePointer<Float16>, skip: UnsafeMutablePointer<Float16>,
                        surfaceOut: UnsafeMutablePointer<Float16>, atmosphereOut: UnsafeMutablePointer<Float16>) throws {
        let S = manifest.patch_chunk, L = Self.patches, W = Self.width, H = Self.height
        let rowsPerChunk = S / Self.wp
        let surf = Buffer(count: 416 * S), atmos = Buffer(count: 13 * 80 * S)
        let outputs = ["surf": try array(surf.p, [1, 416, 1, S]), "atmos": try array(atmos.p, [13, 80, 1, S])]
        for j in 0 ..< L / S {
            try run("decoder", "decoder", inputs: [
                "x": try array(x + j * S * 512, [4, S, 512], strides: [L * 512, 512, 1]),
                "skip": try array(skip + j * S * 512, [4, S, 512], strides: [L * 512, 512, 1]),
            ], outputs: outputs)
            let t = Date()
            let hp0 = j * rowsPerChunk
            parallel(416) { r in  // rows (variable, py, px)
                let v = r / 16, py = (r / 4) % 4, px = r % 4
                let src = surf.p + r * S
                for hr in 0 ..< rowsPerChunk {
                    let dst = surfaceOut + (v * H + (hp0 + hr) * 4 + py) * W + px
                    for w in 0 ..< Self.wp { dst[w * 4] = src[hr * Self.wp + w] }
                }
            }
            parallel(13 * 80) { cr in  // level, rows (variable, py, px)
                let c = cr / 80, r = cr % 80
                let v = r / 16, py = (r / 4) % 4, px = r % 4
                let src = atmos.p + cr * S
                for hr in 0 ..< rowsPerChunk {
                    let dst = atmosphereOut + ((v * 13 + c) * H + (hp0 + hr) * 4 + py) * W + px
                    for w in 0 ..< Self.wp { dst[w * 4] = src[hr * Self.wp + w] }
                }
            }
            add("decoder data", since: t)
        }
        release("decoder")
    }

    // MARK: - Helpers

    private func array(_ p: UnsafeMutablePointer<Float16>, _ shape: [Int], strides: [Int]? = nil) throws -> MLMultiArray {
        var st = strides ?? []
        if strides == nil {
            var acc = 1
            for d in shape.reversed() { st.insert(acc, at: 0); acc *= d }
        }
        return try MLMultiArray(dataPointer: p, shape: shape.map { NSNumber(value: $0) }, dataType: .float16,
                                strides: st.map { NSNumber(value: $0) }, deallocator: nil)
    }

    private func convert(_ a: UnsafePointer<Float16>, to b: UnsafeMutablePointer<Float>, count: Int) {
        var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: a), height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        var dst = vImage_Buffer(data: b, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
    }

    private func convert(_ a: UnsafePointer<Float>, to b: UnsafeMutablePointer<Float16>, count: Int) {
        var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: a), height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
        var dst = vImage_Buffer(data: b, height: 1, width: vImagePixelCount(count), rowBytes: count * 2)
        vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
    }

    private func add(_ a: UnsafePointer<Float16>, into b: UnsafeMutablePointer<Float16>, count: Int) {
        parallel(64) { k in
            let lo = count * k / 64, hi = count * (k + 1) / 64
            for i in lo ..< hi { b[i] += a[i] }
        }
    }
}

/// fp16 memory we own, freed with the last reference.
final class Buffer {
    let p: UnsafeMutablePointer<Float16>
    let count: Int
    init(count: Int) {
        self.count = count
        p = .allocate(capacity: count)
    }
    deinit { p.deallocate() }
}

/// `body(i)` for i in 0..<n across the CPU's cores, `grain` iterations per task.
func parallel(_ n: Int, grain: Int = 1, _ body: (Int) -> Void) {
    let tasks = (n + grain - 1) / grain
    DispatchQueue.concurrentPerform(iterations: tasks) { k in
        for i in k * grain ..< min(n, (k + 1) * grain) { body(i) }
    }
}

/// A value shared across threads where the code guarantees it is safe (pointers into
/// buffers whose rows each thread touches alone).
struct UnsafeSendable<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

/// Hands out 0..<limit, once each, to any thread.
final class Counter: @unchecked Sendable {
    private var value = 0
    private let limit: Int
    private let lock = NSLock()
    init(limit: Int) { self.limit = limit }
    func next() -> Int? {
        lock.lock()
        defer { lock.unlock() }
        guard value < limit else { return nil }
        value += 1
        return value - 1
    }
}

/// The first error any of several threads hit.
final class FirstError: @unchecked Sendable {
    private var stored: Error?
    private let lock = NSLock()
    var error: Error? { lock.lock(); defer { lock.unlock() }; return stored }
    func set(_ e: Error) { lock.lock(); if stored == nil { stored = e }; lock.unlock() }
}
