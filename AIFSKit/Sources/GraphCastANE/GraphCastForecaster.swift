import Accelerate
import AuroraANE
import Foundation

/// GraphCast's inputs at two times, 6 hours apart, on its grid (721 x 1440, latitude -90
/// first, longitude 0 first). The later time is kept in physical units (float32): each
/// step's prediction is a residual added to it, and fp16 would round z500 by 3% of a step's
/// change. The earlier one only feeds the model's inputs, so it is kept normalised in fp16.
public final class GraphCastState {
    /// The later input time.
    public var time: Date
    /// `variables` (82: 4 surface, then 6 x 13 levels as GraphCast orders its channels).
    public let variables: [String]
    /// (82, 721, 1440) normalised, the earlier time.
    public let older: UnsafeMutablePointer<Float16>
    /// (82, 721, 1440) physical, the later time.
    public let newer: UnsafeMutablePointer<Float>
    public static let plane = 721 * 1440

    public init(time: Date, variables: [String]) {
        self.time = time
        self.variables = variables
        older = .allocate(capacity: variables.count * Self.plane)
        newer = .allocate(capacity: variables.count * Self.plane)
    }

    deinit {
        older.deallocate()
        newer.deallocate()
    }

    public func save(to url: URL) throws {
        let n = variables.count * Self.plane
        try SafeTensorsWriter.write(url, metadata: ["date": ISO8601DateFormatter().string(from: time), "model": "graphcast",
                                                    "variables": variables.joined(separator: ",")], [
            .init(name: "older", dtype: "F16", shape: [variables.count, 721, 1440], bytes: UnsafeRawPointer(older), count: n * 2),
            .init(name: "newer", dtype: "F32", shape: [variables.count, 721, 1440], bytes: UnsafeRawPointer(newer), count: n * 4),
        ])
    }

    public static func load(_ url: URL) throws -> GraphCastState {
        let f = try SafeTensorsFile(url)
        guard let iso = f.metadata["date"] ?? f.metadata["time"], let time = ISO8601DateFormatter().date(from: iso),
              let vars = f.metadata["variables"]?.split(separator: ",").map(String.init)
        else { throw CocoaError(.fileReadCorruptFile, userInfo: [NSLocalizedDescriptionKey: "not a GraphCast state"]) }
        let s = GraphCastState(time: time, variables: vars)
        let n = vars.count * plane
        s.older.update(from: try f.pointer("older", as: Float16.self), count: n)
        s.newer.update(from: try f.pointer("newer", as: Float.self), count: n)
        return s
    }
}

/// The variables a state holds, as `variable` or `variable:level`, in GraphCast's order
/// (output channels are named the same way).
extension GraphCastRuntime.Metadata {
    public var stateVariables: [String] {
        let predicted = Set(outputs.filter { $0.kind == "residual" }.map(\.name))
        var seen = Set<String>(), out: [String] = []
        for c in inputs where c.time == "1" {  // the inputs the model also predicts (not the forcings)
            let key = c.level.map { "\(c.variable):\($0)" } ?? c.variable
            if predicted.contains(key), seen.insert(key).inserted { out.append(key) }
        }
        return out
    }
}

/// Drives the runtime through a rollout: builds each step's input features (the state,
/// static fields, forcings), and turns its outputs into the next state.
public final class GraphCastForecaster {
    public let runtime: GraphCastRuntime
    public let state: GraphCastState
    public var metadata: GraphCastRuntime.Metadata { runtime.metadata }
    /// Seconds per part of the last `advance`.
    public private(set) var timings: [String: Double] = [:]

    let n = GraphCastState.plane
    public let features: UnsafeMutablePointer<Float16>
    let gridFeatures: UnsafePointer<Float16>
    let graph: SafeTensorsFile
    /// Normalised static fields, by name.
    private var statics: [String: [Float16]] = [:]
    private let stateIndex: [String: Int]

    public init(directory: URL, state: GraphCastState) throws {
        runtime = try GraphCastRuntime(directory: directory)
        self.state = state
        let want = runtime.metadata.stateVariables
        guard state.variables == want else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [NSLocalizedDescriptionKey: "state variables do not match the bundle"])
        }
        stateIndex = Dictionary(uniqueKeysWithValues: want.enumerated().map { ($1, $0) })
        graph = try SafeTensorsFile(directory.appending(path: "graph.safetensors"))
        gridFeatures = UnsafePointer(try graph.pointer("grid_features", as: Float16.self))
        features = .allocate(capacity: n * GraphCastRuntime.features)
        let st = try SafeTensorsFile(directory.appending(path: "static.safetensors"))
        for c in runtime.metadata.inputs where c.time == nil {
            let src = try st.pointer(c.variable, as: Float.self)
            let (m, s) = (Float(c.mean), Float(c.std))
            statics[c.name] = (0 ..< n).map { Float16((src[$0] - m) / s) }
        }
    }

    deinit {
        features.deallocate()
        scratch?.deallocate()
    }

    public var stepHours: Int { metadata.step_hours }
    public var nextDate: Date { state.time.addingTimeInterval(TimeInterval(stepHours * 3600)) }

    /// Whether GraphCast predicts the app field `name` (it has no clouds, dew point or snow).
    public static func predicts(_ name: String) -> Bool { source(name) != nil }

    /// GraphCast's (variable, level) for an app field name: `2t`, `msl`, `10u`, `10v`, `tp`
    /// (metres over the step), `name_level` for `z`, `t`, `u`, `v`, `q`, `w`.
    static func source(_ name: String) -> (String, Int?)? {
        let surface = ["2t": "2m_temperature", "msl": "mean_sea_level_pressure", "10u": "10m_u_component_of_wind",
                       "10v": "10m_v_component_of_wind", "tp": "total_precipitation_6hr"]
        if let v = surface[name] { return (v, nil) }
        let levels = ["z": "geopotential", "t": "temperature", "u": "u_component_of_wind", "v": "v_component_of_wind",
                      "q": "specific_humidity", "w": "vertical_velocity"]
        let parts = name.split(separator: "_")
        guard parts.count == 2, let v = levels[String(parts[0])], let l = Int(parts[1]) else { return nil }
        return (v, l)
    }

    /// Runs one step and returns the `wanted` fields GraphCast predicts (others are left
    /// out), each 721 x 1440 floats north to south (row 0 at 90 N), in physical units.
    public func advance(wanted: [String]) throws -> (date: Date, fields: [String: [Float]]) {
        let valid = nextDate
        var t = Date()
        buildFeatures(target: valid)
        timings = ["features": Date().timeIntervalSince(t)]

        // The later input becomes the earlier one (normalised); the prediction will be added
        // to the later one in place.
        t = Date()
        for (v, key) in state.variables.enumerated() {
            let c = channel(key, time: "1")
            let (m, s) = (Float(c.mean), Float(c.std))
            let src = UnsafeSendable(state.newer + v * n), dst = UnsafeSendable(state.older + v * n)
            parallel(64) { k in
                for j in self.n * k / 64 ..< self.n * (k + 1) / 64 { dst.value[j] = Float16((src.value[j] - m) / s) }
            }
        }
        timings["roll"] = Date().timeIntervalSince(t)

        // Outputs, a decoder chunk at a time: residuals onto the later input (in place: the
        // features are built), precipitation directly.
        t = Date()
        let k = GraphCastRuntime.outputs
        var targets = [(UnsafeMutablePointer<Float>, Float, Float, Bool)]()  // field, scale, mean, residual
        let precipitation = UnsafeMutablePointer<Float>.allocate(capacity: n)
        defer { precipitation.deallocate() }
        for o in metadata.outputs {
            let key = o.level.map { "\(o.variable):\($0)" } ?? o.variable
            if o.kind == "residual", let v = stateIndex[key] {
                targets.append((state.newer + v * n, Float(o.scale), 0, true))
            } else {
                targets.append((precipitation, Float(o.scale), Float(o.mean ?? 0), false))
            }
        }
        let tg = UnsafeSendable(targets)
        try runtime.step(features: features) { lo, count, rows in
            let r = UnsafeSendable(rows)
            parallel((count + 1023) / 1024) { b in
                for j in b * 1024 ..< min(count, (b + 1) * 1024) {
                    let row = r.value + j * k
                    for (c, (dst, scale, mean, residual)) in tg.value.enumerated() {
                        let v = Float(row[c]) * scale
                        dst[lo + j] = residual ? dst[lo + j] + v : max(0, mean + v)
                    }
                }
            }
        }
        state.time = valid
        timings["model"] = Date().timeIntervalSince(t)

        t = Date()
        var fields: [String: [Float]] = [:]
        for name in wanted {
            guard let (variable, level) = Self.source(name) else { continue }
            if variable == "total_precipitation_6hr" {
                fields[name] = flipped(precipitation)
            } else if let v = stateIndex[level.map { "\(variable):\($0)" } ?? variable] {
                fields[name] = flipped(state.newer + v * n)
            }
        }
        timings["fields"] = Date().timeIntervalSince(t)
        return (valid, fields)
    }

    /// The state's later time as fields, as `advance` returns them (for persistence).
    public func persistence(_ wanted: [String]) throws -> [String: [Float]] {
        try Self.northFirst(state, wanted)
    }

    /// A state's later time as `wanted` fields, north first.
    public static func northFirst(_ state: GraphCastState, _ wanted: [String]) throws -> [String: [Float]] {
        var out: [String: [Float]] = [:]
        for name in wanted {
            guard let (variable, level) = source(name),
                  let v = state.variables.firstIndex(of: level.map { "\(variable):\($0)" } ?? variable) else { continue }
            out[name] = flip(state.newer + v * GraphCastState.plane)
        }
        return out
    }

    private func flipped(_ src: UnsafePointer<Float>) -> [Float] { Self.flip(src) }

    /// GraphCast's rows (south first) north first.
    public static func flip(_ src: UnsafePointer<Float>) -> [Float] {
        let n = GraphCastState.plane
        let rows = 721, cols = 1440
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { o in
            for r in 0 ..< rows { (o.baseAddress! + r * cols).update(from: src + (rows - 1 - r) * cols, count: cols) }
        }
        return out
    }

    private func channel(_ key: String, time: String) -> GraphCastRuntime.Metadata.Input {
        let parts = key.split(separator: ":")
        let variable = String(parts[0]), level = parts.count > 1 ? Int(parts[1]) : nil
        return metadata.inputs.first { $0.variable == variable && $0.level == level && $0.time == time }!
    }

    /// The (721 x 1440, 184) grid features: each input channel normalised, then the nodes'
    /// 3 structural features.
    func buildFeatures(target: Date) {
        let f = GraphCastRuntime.features
        let dates = ["0": state.time.addingTimeInterval(-TimeInterval(stepHours * 3600)), "1": state.time, "target": target]
        // Channels read from a field: forcings (computed for their time) and statics.
        var fieldChannels: [Int] = []
        for (i, c) in metadata.inputs.enumerated() where stateIndex[c.level.map { "\(c.variable):\($0)" } ?? c.variable] == nil || c.time == nil {
            fieldChannels.append(i)
        }
        if scratch == nil { scratch = .allocate(capacity: fieldChannels.count * n) }
        let scratch = scratch!
        for (slot, i) in fieldChannels.enumerated() {
            let c = metadata.inputs[i], dst = scratch + slot * n
            if c.time == nil {
                dst.update(from: statics[c.name]!, count: n)
                continue
            }
            let date = dates[c.time!]!
            let (m, s) = (Float(c.mean), Float(c.std))
            switch c.variable {
            case "toa_incident_solar_radiation":
                let v = GraphCastForcings.toaRadiation(at: date)
                for j in 0 ..< n { dst[j] = Float16((v[j] - m) / s) }
            case "year_progress_sin", "year_progress_cos":
                let p = GraphCastForcings.yearProgress(date) * 2 * .pi
                dst.update(repeating: Float16((Float(c.variable.hasSuffix("sin") ? sin(p) : cos(p)) - m) / s), count: n)
            default:  // day progress
                let row = GraphCastForcings.dayProgress(date).map {
                    Float16((Float(c.variable.hasSuffix("sin") ? sin($0 * 2 * .pi) : cos($0 * 2 * .pi)) - m) / s)
                }
                for r in 0 ..< 721 { (dst + r * 1440).update(from: row, count: 1440) }
            }
        }
        // Per channel: an fp16 source taken as is, or a float32 one normalised.
        var half = [UnsafePointer<Float16>?](repeating: nil, count: metadata.inputs.count)
        var full = [UnsafePointer<Float>?](repeating: nil, count: metadata.inputs.count)
        var norm = [(Float, Float)](repeating: (0, 1), count: metadata.inputs.count)
        for (i, c) in metadata.inputs.enumerated() {
            let key = c.level.map { "\(c.variable):\($0)" } ?? c.variable
            if let slot = fieldChannels.firstIndex(of: i) {
                half[i] = UnsafePointer(scratch + slot * n)
            } else if c.time == "0" {
                half[i] = UnsafePointer(state.older + stateIndex[key]! * n)
            } else {
                full[i] = UnsafePointer(state.newer + stateIndex[key]! * n)
                norm[i] = (Float(c.mean), 1 / Float(c.std))
            }
        }
        let nIn = metadata.inputs.count, block = 1024
        let x = UnsafeSendable(features), grid = UnsafeSendable(gridFeatures)
        half.withUnsafeBufferPointer { hp in
            full.withUnsafeBufferPointer { fp in
                norm.withUnsafeBufferPointer { np in
                    let h = UnsafeSendable(hp.baseAddress!), fl = UnsafeSendable(fp.baseAddress!), no = UnsafeSendable(np.baseAddress!)
                    parallel((n + block - 1) / block) { b in
                        let lo = b * block, hi = min(lo + block, self.n)
                        for c in 0 ..< nIn {
                            if let src = h.value[c] {
                                for j in lo ..< hi { x.value[j * f + c] = src[j] }
                            } else {
                                let src = fl.value[c]!, (m, inv) = no.value[c]
                                for j in lo ..< hi { x.value[j * f + c] = Float16((src[j] - m) * inv) }
                            }
                        }
                        for j in lo ..< hi {
                            for k in 0 ..< 3 { x.value[j * f + nIn + k] = grid.value[j * 3 + k] }
                        }
                    }
                }
            }
        }
    }

    private var scratch: UnsafeMutablePointer<Float16>?
}

/// GraphCast's forcings, as Google computes them (weathernext/utils/solar_radiation.py,
/// data_utils.py), on its grid.
public enum GraphCastForcings {
    static let secondsPerDay = 86400.0

    /// Fraction of the (365.24219-day) year since the Unix epoch, in [0, 1).
    public static func yearProgress(_ date: Date) -> Double {
        let years = date.timeIntervalSince1970.rounded(.down) / secondsPerDay / 365.24219
        return years - years.rounded(.down)
    }

    /// Local solar day fraction at each of the 1440 longitudes (0, 0.25, ...), in [0, 1).
    public static func dayProgress(_ date: Date) -> [Double] {
        let s = date.timeIntervalSince1970.rounded(.down)
        let greenwich = s.truncatingRemainder(dividingBy: secondsPerDay) / secondsPerDay
        return (0 ..< 1440).map { j in
            let p = greenwich + Double(j) * 0.25 / 360
            return p - p.rounded(.down)
        }
    }

    /// ERA5's total solar irradiance by year (W/m2), interpolated.
    static func tsi(_ date: Date) -> Double {
        let table: [Double] = [
            1365.7765, 1365.7676, 1365.6284, 1365.6564, 1365.7773, 1366.3109, 1366.6681, 1366.6328, 1366.3828, 1366.2767,
            1365.9199, 1365.7484, 1365.6963, 1365.6976, 1365.7341, 1365.9178, 1366.1143, 1366.1644, 1366.2476, 1366.2426,
            1365.9580, 1366.0525, 1365.7991, 1365.7271, 1365.5345, 1365.6453, 1365.8331, 1366.2747, 1366.6348, 1366.6482,
            1366.6951, 1366.2859, 1366.1992, 1365.8103, 1365.6416, 1365.6379, 1365.7899, 1366.0826, 1366.6479, 1366.5533,
            1366.4457, 1366.3021, 1366.0286, 1365.7971, 1365.6996, 1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836,
            1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545, 1365.8107, 1365.7240, 1365.6918,
            1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836, 1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
            1365.8107, 1365.7240, 1365.6918,
            1365.6121, 1365.7399, 1366.1021, 1366.3851, 1366.6836, 1366.6022, 1366.6807, 1366.2300, 1366.0480, 1365.8545,
            1365.8107, 1365.7240, 1365.6918,
        ].map { $0 * 0.9965 }  // years 1951.5 ... 2034.5
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let year = cal.component(.year, from: date)
        let start = cal.date(from: DateComponents(year: year, month: 1, day: 1))!
        let length = Double(cal.range(of: .day, in: .year, for: date)!.count)
        let fractional = Double(year) + date.timeIntervalSince(start) / secondsPerDay / length
        let x = min(max(fractional - 1951.5, 0), Double(table.count - 1))
        let i = min(Int(x), table.count - 2)
        return table[i] + (table[i + 1] - table[i]) * (x - Double(i))
    }

    /// Incident solar radiation at the top of the atmosphere integrated over the hour before
    /// `date` (J/m2), (721, 1440) south first: 361 instants, trapezoidal rule.
    public static func toaRadiation(at date: Date) -> [Float] {
        let bins = 360
        let j2000 = date.timeIntervalSince1970 / secondsPerDay + 2440587.5 - 2451545.0
        let tsi = tsi(date)
        // Per instant: the orbital terms shared by every point.
        var sinDec = [Double](repeating: 0, count: bins + 1), cosDec = sinDec, factor = sinDec, phase = sinDec
        for b in 0 ... bins {
            let d = j2000 - 1.0 / 24 * Double(bins - b) / Double(bins)
            let theta = d / 365.25
            let rel = 1.7535 + 6.283076 * theta, rem = 6.240041 + 6.283020 * theta, rlls = 4.8951 + 6.283076 * theta
            let rllls = 4.8952 + 6.283320 * theta - 0.0075 * sin(rel) - 0.0326 * cos(rel) - 0.0003 * sin(2 * rel) + 0.0002 * cos(2 * rel)
            sinDec[b] = sin(0.409093) * sin(rllls)
            cosDec[b] = (1 - sinDec[b] * sinDec[b]).squareRoot()
            let eqt = 591.8 * sin(2 * rlls) - 459.4 * sin(rem) + 39.5 * sin(rem) * cos(2 * rlls) - 12.7 * sin(4 * rlls) - 4.8 * sin(2 * rem)
            let distance = 1.0001 - 0.0163 * sin(rel) + 0.0037 * cos(rel)
            factor[b] = tsi / (distance * distance) * (b == 0 || b == bins ? 0.5 : 1) * 3600 / Double(bins)
            let rotation = d - d.rounded(.down)
            phase[b] = 2 * .pi * (rotation + eqt / secondsPerDay)
        }
        var out = [Float](repeating: 0, count: 721 * 1440)
        // cos(hour angle) per (instant, longitude), shared by every latitude.
        var cosHour = [Double](repeating: 0, count: (bins + 1) * 1440)
        for b in 0 ... bins {
            for j in 0 ..< 1440 { cosHour[b * 1440 + j] = cos(phase[b] + Double(j) * 0.25 * .pi / 180) }
        }
        out.withUnsafeMutableBufferPointer { o in
            let dst = UnsafeSendable(o.baseAddress!)
            cosHour.withUnsafeBufferPointer { ch in
                let c = UnsafeSendable(ch.baseAddress!)
                parallel(721) { i in
                    let lat = (-90 + Double(i) * 0.25) * .pi / 180
                    let sl = sin(lat), cl = cos(lat)
                    var acc = [Double](repeating: 0, count: 1440)
                    for b in 0 ... bins {
                        let a = cl * cosDec[b], s0 = sl * sinDec[b], w = factor[b]
                        let row = c.value + b * 1440
                        for j in 0 ..< 1440 { acc[j] += w * max(a * row[j] + s0, 0) }
                    }
                    for j in 0 ..< 1440 { dst.value[i * 1440 + j] = Float(acc[j]) }
                }
            }
        }
        return out
    }
}
