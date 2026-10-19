import Accelerate
import Foundation

/// `aurora.json` in a bundle: Aurora 1.5's variables, normalisation and between-step rules.
public struct AuroraMetadata: Decodable, Sendable {
    public struct Grid: Decodable, Sendable {
        public let rows: Int
        public let cols: Int
        public let first_latitude: Double
        public let spacing: Double
    }
    public struct Bounds: Decodable, Sendable {
        public let min: Float?
        public let max: Float?
    }
    struct Normalisation: Decodable, Sendable {
        let surface: [String: [Float]]
        let atmosphere: [String: [String: [Float]]]
    }

    public let model: String
    public let step_hours: Int
    public let grid: Grid
    public let levels: [Int]
    /// The 26 surface variables the model predicts, in its output order.
    public let surface: [String]
    /// The 19 of them that are inputs, in the state's order.
    public let inputs: [String]
    public let atmosphere: [String]
    let normalisation: Normalisation
    let log_transform_eps: Float
    public let clipping: [String: Bounds]

    public static func load(_ directory: URL) throws -> AuroraMetadata {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: directory.appending(path: "aurora.json")))
    }

    /// Location and scale of a surface variable, or an atmospheric one at `level`.
    func stats(_ name: String, level: Int? = nil) -> (Float, Float) {
        let s = level.map { normalisation.atmosphere[name]![String($0)]! } ?? normalisation.surface[name]!
        return (s[0], s[1])
    }

    /// Physical value -> normalised (log-transforming the `scaled_*` variables first).
    public func normalise(_ x: Float, _ name: String, level: Int? = nil) -> Float {
        let (loc, scale) = stats(name, level: level)
        let v = name.hasPrefix("scaled_") ? log(x + log_transform_eps) - log(log_transform_eps) : x
        return (v - loc) / scale
    }

    /// Normalised -> physical.
    public func physical(_ x: Float, _ name: String, level: Int? = nil) -> Float {
        let (loc, scale) = stats(name, level: level)
        let v = x * scale + loc
        return name.hasPrefix("scaled_") ? log_transform_eps * (exp(v) - 1) : v
    }
}

/// Aurora's top-of-atmosphere insolation (`aurora.insolation`, 1995 orbital elements), for
/// `date` on the model grid (720 x 1440, rows from 90 N, columns from 0 E).
public enum AuroraInsolation {
    public static func field(_ date: Date, rows: Int = 720, cols: Int = 1440, into out: UnsafeMutablePointer<Float>) {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let startOfYear = cal.date(from: cal.dateComponents([.year], from: date))!
        let days = Double(Float(date.timeIntervalSince(startOfYear) / 86400))  // float32, as Aurora has it
        let eps = 23.4441 * Double.pi / 180, ecc = 0.016715, om = 282.7 * Double.pi / 180
        let beta = (1 - ecc * ecc).squareRoot()
        let lambdaM0 = ecc * (1 + beta) * sin(om)
        let lambdaM = lambdaM0 + 2 * Double.pi * (days - 80.5) / 365
        let lambda = lambdaM + 2 * ecc * sin(lambdaM - om)
        let dec = Double(Float(asin(sin(eps) * sin(lambda))))
        let rho = (1 - ecc * ecc) / (1 + ecc * cos(lambda - om))
        let factor = pow(rho, -2)
        let sinDec = sin(dec), cosDec = cos(dec)
        var cosH = [Double](repeating: 0, count: cols)
        for c in 0 ..< cols {
            let lon = Double(c) * 360 / Double(cols)
            cosH[c] = cos(Double(Float(2 * Double.pi * (days + lon / 360))))
        }
        for r in 0 ..< rows {
            let lat = Double(Float(90 - Double(r) * 180 / Double(rows)))  // 90, 89.75, ...
            let a = sin(Double.pi / 180 * lat) * sinDec, b = cos(Double.pi / 180 * lat) * cosDec
            for c in 0 ..< cols { out[r * cols + c] = Float((a - b * cosH[c]) * factor) }
        }
    }
}

/// The model's state: the two input times' normalised fp16 fields, as the programs take
/// them -- surface `(19, 2, rows, cols)` and atmospheric `(5, 2, 13, rows, cols)`.
public final class AuroraState: @unchecked Sendable {
    public let surface: UnsafeMutablePointer<Float16>
    public let atmosphere: UnsafeMutablePointer<Float16>
    /// Valid time of the later input time.
    public var time: Date
    let plane: Int

    public init(time: Date, rows: Int = 720, cols: Int = 1440) {
        self.time = time
        plane = rows * cols
        surface = .allocate(capacity: 19 * 2 * plane)
        atmosphere = .allocate(capacity: 5 * 2 * 13 * plane)
    }

    deinit {
        surface.deallocate()
        atmosphere.deallocate()
    }

    public var surfaceCount: Int { 19 * 2 * plane }
    public var atmosphereCount: Int { 5 * 2 * 13 * plane }

    /// Input variable `v` at history slot `t` (0 the earlier time, 1 the later).
    public func surface(_ v: Int, _ t: Int) -> UnsafeMutablePointer<Float16> { surface + (v * 2 + t) * plane }
    public func atmosphere(_ v: Int, _ t: Int, _ level: Int) -> UnsafeMutablePointer<Float16> {
        atmosphere + ((v * 2 + t) * 13 + level) * plane
    }

    /// Writes the state as a safetensors file (`surf`, `atmos`, and the time).
    public func save(to url: URL) throws {
        try SafeTensorsWriter.write(url, metadata: ["date": ISO8601DateFormatter().string(from: time), "model": "aurora-1.5"], [
            .init(name: "surf", dtype: "F16", shape: [19, 2, plane / 1440, 1440], bytes: UnsafeRawPointer(surface), count: surfaceCount * 2),
            .init(name: "atmos", dtype: "F16", shape: [5, 2, 13, plane / 1440, 1440], bytes: UnsafeRawPointer(atmosphere), count: atmosphereCount * 2),
        ])
    }

    /// Reads a state `save` wrote.
    public static func load(_ url: URL) throws -> AuroraState {
        let file = try SafeTensorsFile(url)
        guard let iso = file.metadata["date"] ?? file.metadata["time"], let time = ISO8601DateFormatter().date(from: iso) else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [NSLocalizedDescriptionKey: "\(url.lastPathComponent) has no date"])
        }
        let state = AuroraState(time: time)
        state.surface.update(from: try file.pointer("surf", as: Float16.self), count: state.surfaceCount)
        state.atmosphere.update(from: try file.pointer("atmos", as: Float16.self), count: state.atmosphereCount)
        return state
    }
}

/// A forecast: the runtime and the state, one 6-hour step at a time.
public final class AuroraForecaster {
    public let runtime: AuroraRuntime
    public let metadata: AuroraMetadata
    public let state: AuroraState
    public private(set) var step = 0
    /// Seconds in the last `advance`: the programs, physical fields, the next input.
    public private(set) var timings: [String: Double] = [:]
    private let rows: Int, cols: Int
    private let surfaceOut: UnsafeMutablePointer<Float16>
    private let atmosphereOut: UnsafeMutablePointer<Float16>

    public init(directory: URL, state: AuroraState, options: AuroraRuntime.Options = .init(resident: false)) throws {
        runtime = try AuroraRuntime(directory: directory, options: options)
        metadata = try AuroraMetadata.load(directory)
        self.state = state
        rows = metadata.grid.rows
        cols = metadata.grid.cols
        surfaceOut = .allocate(capacity: metadata.surface.count * rows * cols)
        atmosphereOut = .allocate(capacity: 5 * 13 * rows * cols)
    }

    deinit {
        surfaceOut.deallocate()
        atmosphereOut.deallocate()
    }

    /// Valid time of the next step's prediction.
    public var nextDate: Date { state.time.addingTimeInterval(TimeInterval(metadata.step_hours * 3600)) }

    /// Runs one step and returns `wanted` fields of the prediction in physical units, each
    /// `rows * cols` floats. Names are surface variables (`2t`, `msl`, ...), `name_level`
    /// for atmospheric ones (`z_500`), and `tp` / `sf`: precipitation and snowfall over the
    /// step in metres, estimated as Aurora's last-hour amounts times the step's hours.
    public func advance(wanted: [String]) throws -> (date: Date, fields: [String: [Float]]) {
        let valid = nextDate
        var t = Date()
        try runtime.step(surface: state.surface, atmosphere: state.atmosphere, time: state.time,
                         surfaceOut: surfaceOut, atmosphereOut: atmosphereOut)
        timings = ["model": Date().timeIntervalSince(t)]
        t = Date()
        step += 1
        let n = rows * cols
        var fields: [String: [Float]] = [:]
        for name in wanted {
            var source = name, factor: Float = 1, level: Int?
            if name == "tp" { (source, factor) = ("scaled_tp_1h", Float(metadata.step_hours)) }
            if name == "sf" { (source, factor) = ("scaled_sf_1h", Float(metadata.step_hours)) }
            let parts = source.split(separator: "_")
            if parts.count == 2, let l = Int(parts[1]), metadata.atmosphere.contains(String(parts[0])) {
                source = String(parts[0])
                level = l
            }
            let base: UnsafeMutablePointer<Float16>
            if let level, let v = metadata.atmosphere.firstIndex(of: source), let c = metadata.levels.firstIndex(of: level) {
                base = atmosphereOut + (v * 13 + c) * n
            } else if let i = metadata.surface.firstIndex(of: source) {
                base = surfaceOut + i * n
            } else {
                continue
            }
            // x * scale + loc, then for the log-transformed variables eps * (exp(v) - 1); in
            // vector operations, 64 slices at once.
            var (loc, scale) = metadata.stats(source, level: level)
            let logged = source.hasPrefix("scaled_")
            var out = [Float](repeating: 0, count: n)
            out.withUnsafeMutableBufferPointer { o in
                let dst = UnsafeSendable(o.baseAddress!)
                parallel(64) { k in
                    let lo = n * k / 64, count = n * (k + 1) / 64 - lo
                    let p = dst.value + lo
                    var src = vImage_Buffer(data: UnsafeMutableRawPointer(base + lo), height: 1,
                                            width: vImagePixelCount(count), rowBytes: count * 2)
                    var f = vImage_Buffer(data: p, height: 1, width: vImagePixelCount(count), rowBytes: count * 4)
                    vImageConvert_Planar16FtoPlanarF(&src, &f, 0)
                    vDSP_vsmsa(p, 1, &scale, &loc, p, 1, vDSP_Length(count))
                    if logged {
                        var c = Int32(count), minusOne: Float = -1, gain = metadata.log_transform_eps * factor
                        vvexpf(p, p, &c)
                        vDSP_vsadd(p, 1, &minusOne, p, 1, vDSP_Length(count))
                        vDSP_vsmul(p, 1, &gain, p, 1, vDSP_Length(count))
                    } else if factor != 1 {
                        var gain = factor
                        vDSP_vsmul(p, 1, &gain, p, 1, vDSP_Length(count))
                    }
                }
            }
            fields[name] = out
        }
        timings["physical fields"] = Date().timeIntervalSince(t)
        t = Date()
        try advanceState(to: valid)
        timings["next input"] = Date().timeIntervalSince(t)
        return (valid, fields)
    }

    /// The prediction becomes the later input time, as `aurora.rollout` does it: clipped
    /// variables clipped, insolation recomputed for the new time; output-only variables
    /// dropped (the programs zero them).
    private func advanceState(to valid: Date) throws {
        let n = rows * cols
        let insolation = UnsafeMutablePointer<Float>.allocate(capacity: n)
        defer { insolation.deallocate() }
        AuroraInsolation.field(valid, rows: rows, cols: cols, into: insolation)
        for (v, name) in metadata.inputs.enumerated() {
            let older = state.surface(v, 0), newer = state.surface(v, 1)
            older.update(from: newer, count: n)
            if name == "insolation" {
                parallel(64) { k in
                    for j in n * k / 64 ..< n * (k + 1) / 64 { newer[j] = Float16(metadata.normalise(insolation[j], name)) }
                }
                continue
            }
            let src = surfaceOut + metadata.surface.firstIndex(of: name)! * n
            let bounds = metadata.clipping[name]
            let lo = bounds?.min.map { metadata.normalise($0, name) } ?? -.infinity
            let hi = bounds?.max.map { metadata.normalise($0, name) } ?? .infinity
            parallel(64) { k in
                for j in n * k / 64 ..< n * (k + 1) / 64 { newer[j] = Float16(min(max(Float(src[j]), lo), hi)) }
            }
        }
        for v in 0 ..< 5 {
            for c in 0 ..< 13 {
                state.atmosphere(v, 0, c).update(from: state.atmosphere(v, 1, c), count: n)
                state.atmosphere(v, 1, c).update(from: atmosphereOut + (v * 13 + c) * n, count: n)
            }
        }
        state.time = valid
    }
}

/// Writes safetensors files from raw buffers (no MLX).
public enum SafeTensorsWriter {
    public struct Entry {
        public let name: String
        public let dtype: String
        public let shape: [Int]
        public let bytes: UnsafeRawPointer
        public let count: Int
        public init(name: String, dtype: String, shape: [Int], bytes: UnsafeRawPointer, count: Int) {
            (self.name, self.dtype, self.shape, self.bytes, self.count) = (name, dtype, shape, bytes, count)
        }
    }

    /// Float32 fields, each a flat array (as the app's step files hold them).
    public static func write(_ url: URL, metadata: [String: String] = [:], fields: [String: [Float]]) throws {
        let names = fields.keys.sorted()
        try write(url, metadata: metadata, sizes: names.map { (name: $0, dtype: "F32", shape: [fields[$0]!.count], count: fields[$0]!.count * 4) }) { handle in
            for name in names {
                try fields[name]!.withUnsafeBytes { try handle.write(contentsOf: Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: $0.baseAddress!), count: $0.count, deallocator: .none)) }
            }
        }
    }

    public static func write(_ url: URL, metadata: [String: String] = [:], _ entries: [Entry]) throws {
        try write(url, metadata: metadata, sizes: entries.map { (name: $0.name, dtype: $0.dtype, shape: $0.shape, count: $0.count) }) { handle in
            for e in entries {
                var at = 0
                while at < e.count {  // in 64 MB pieces
                    let len = min(64 << 20, e.count - at)
                    try handle.write(contentsOf: Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: e.bytes + at), count: len, deallocator: .none))
                    at += len
                }
            }
        }
    }

    /// The header for `sizes`, then `body` writes the tensors' bytes in that order.
    private static func write(_ url: URL, metadata: [String: String],
                              sizes: [(name: String, dtype: String, shape: [Int], count: Int)],
                              body: (FileHandle) throws -> Void) throws {
        var header: [String: Any] = ["__metadata__": metadata]
        var offset = 0
        for e in sizes {
            header[e.name] = ["dtype": e.dtype, "shape": e.shape, "data_offsets": [offset, offset + e.count]]
            offset += e.count
        }
        var json = try JSONSerialization.data(withJSONObject: header)
        while (8 + json.count) % 8 != 0 { json.append(0x20) }  // pad with spaces: data 8-byte aligned
        let tmp = url.appendingPathExtension("part")
        FileManager.default.createFile(atPath: tmp.path, contents: nil)
        let handle = try FileHandle(forWritingTo: tmp)
        var n = UInt64(json.count)
        try handle.write(contentsOf: Data(bytes: &n, count: 8))
        try handle.write(contentsOf: json)
        try body(handle)
        try handle.close()
        try? FileManager.default.removeItem(at: url)
        try FileManager.default.moveItem(at: tmp, to: url)
    }
}
