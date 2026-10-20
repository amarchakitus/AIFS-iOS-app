import Foundation
import GraphCastANE

/// GraphCast operational's initial state from ECMWF open data, its messages through the
/// shared `OpenDataCache`: the IFS 0.25-degree analyses at the cycle and 6 h before (the HRES analyses Google fine-tuned it on), every field rolled
/// to start at 0 E and flipped to start at the south pole, as GraphCast's grid does. The
/// earlier time is stored normalised (fp16), the later in physical units (see GraphCastState).
public final class GraphCastOpenData: @unchecked Sendable {
    struct Source: Hashable, Sendable {
        let param: String
        let levtype: String
        let level: String?
    }

    let metadata: GraphCastRuntime.Metadata
    let mirrors: [OpenDataClient.Mirror]
    let session: URLSession

    public init(bundle: URL, mirrors: [OpenDataClient.Mirror] = OpenDataCache.mirrors, session: URLSession = .shared) throws {
        metadata = try GraphCastRuntime.Metadata.load(bundle)
        self.mirrors = mirrors
        self.session = session
    }

    /// IFS's message for a state variable (`variable` or `variable:level`).
    static func source(of key: String) -> Source? {
        let parts = key.split(separator: ":"), level = parts.count > 1 ? String(parts[1]) : nil
        let surface = ["2m_temperature": "2t", "mean_sea_level_pressure": "msl",
                       "10m_u_component_of_wind": "10u", "10m_v_component_of_wind": "10v"]
        let pressure = ["temperature": "t", "geopotential": "z", "u_component_of_wind": "u",
                        "v_component_of_wind": "v", "vertical_velocity": "w", "specific_humidity": "q"]
        if let p = surface[String(parts[0])], level == nil { return Source(param: p, levtype: "sfc", level: nil) }
        if let p = pressure[String(parts[0])], level != nil { return Source(param: p, levtype: "pl", level: level) }
        return nil
    }

    // MARK: - Availability

    private func isPublished(_ date: Date) async -> Bool {
        for mirror in mirrors {
            var request = URLRequest(url: OpenDataCache.fileURL(mirror, date: date, model: "ifs", stream: "oper", ext: "index"))
            request.httpMethod = "HEAD"
            if (try? await (session.data(for: request).1 as? HTTPURLResponse)?.statusCode) == 200 { return true }
        }
        return false
    }

    /// Cycles in the last `lookback` (newest first) whose state can be built: it and the
    /// cycle 6 h before are published.
    public func availableDates(now: Date = .now, lookback: Int = 20) async -> [Date] {
        let cycle = 6 * 3600.0
        let newest = (now.timeIntervalSince1970 / cycle).rounded(.down) * cycle
        let dates = (0 ... lookback).map { Date(timeIntervalSince1970: newest - Double($0) * cycle) }
        let published = await withTaskGroup(of: (Date, Bool).self) { group in
            for date in dates { group.addTask { (date, await self.isPublished(date)) } }
            var found: [Date: Bool] = [:]
            for await (date, ok) in group { found[date] = ok }
            return found
        }
        return dates.dropLast().filter { published[$0] == true && published[$0.addingTimeInterval(-cycle)] == true }
    }

    public func latestDate(now: Date = .now) async throws -> Date {
        guard let date = await availableDates(now: now, lookback: 8).first else {
            throw URLError(.resourceUnavailable, userInfo: [NSLocalizedDescriptionKey: "No complete open-data cycle in the last 48 hours"])
        }
        return date
    }

    // MARK: - Build

    /// The input times for the state at `date` (the later one).
    static func times(_ date: Date) -> [Date] { [date.addingTimeInterval(-6 * 3600), date] }

    /// Each state variable's message, in the state's order.
    func sources() throws -> [Source] {
        try metadata.stateVariables.map { key in
            guard let s = Self.source(of: key) else { throw AIFSError.invalidInitialCondition("no open-data source for \(key)") }
            return s
        }
    }

    func key(_ s: Source, _ time: Date) -> OpenDataCache.Key {
        OpenDataCache.Key(model: "ifs", stream: "oper", date: time, param: s.param, levtype: s.levtype, level: s.level)
    }

    /// Every open-data message the state at `date` is built from.
    public func keys(date: Date) throws -> [OpenDataCache.Key] {
        let sources = try sources()
        return Self.times(date).flatMap { time in sources.map { key($0, time) } }.sorted()
    }

    /// Downloads into `cache` the messages for `date` it doesn't have yet.
    public func download(date: Date, into cache: OpenDataCache, progress: (@Sendable (OpenDataClient.Progress) -> Void)? = nil) async throws {
        try await cache.fetch(keys(date: date), progress: progress)
    }

    /// Assembles the state for `date` (the later input time) from the messages in `cache`.
    public func build(date: Date, from cache: OpenDataCache) throws -> GraphCastState {
        let times = Self.times(date), sources = try sources()
        let state = GraphCastState(time: date, variables: metadata.stateVariables)
        let jobs = times.indices.flatMap { t in sources.indices.map { (t, $0) } }
        // Each message fills its own plane of the state: across the cores.
        let failure = FirstFailure()
        DispatchQueue.concurrentPerform(iterations: jobs.count) { j in
            let (t, v) = jobs[j]
            do {
                try store(GRIB2.decode(cache.data(key(sources[v], times[t]))), variable: v, slot: t, into: state)
            } catch {
                failure.set(error)
            }
        }
        if let error = failure.error { throw error }
        return state
    }

    /// One decoded message into the state: rolled to start at 0 E, flipped to start at
    /// -90; the earlier time normalised (fp16), the later kept as is.
    private func store(_ field: GRIB2Field, variable v: Int, slot t: Int, into state: GraphCastState) throws {
        let rows = metadata.grid.rows, cols = metadata.grid.cols
        guard field.ni == cols, field.nj == rows, abs(field.firstLatitude - 90) < 1e-6 else {
            throw GRIB2Error.unsupported("grid \(field.nj)x\(field.ni) from \(field.firstLatitude)N")
        }
        let shift = Int(((360 - field.firstLongitude) / (360.0 / Double(cols))).rounded()) % cols
        let key = state.variables[v], parts = key.split(separator: ":")
        let level = parts.count > 1 ? Int(parts[1]) : nil
        let plane = rows * cols
        field.values.withUnsafeBufferPointer { src in
            if t == 0 {
                let c = metadata.inputs.first { $0.variable == parts[0] && $0.level == level && $0.time == "0" }!
                let (m, s) = (Float(c.mean), Float(c.std))
                let dst = state.older + v * plane
                for r in 0 ..< rows {
                    let from = (rows - 1 - r) * cols
                    for j in 0 ..< cols { dst[r * cols + j] = Float16((src[from + (j + shift) % cols] - m) / s) }
                }
            } else {
                let dst = state.newer + v * plane
                for r in 0 ..< rows {
                    let from = (rows - 1 - r) * cols
                    for j in 0 ..< cols { dst[r * cols + j] = src[from + (j + shift) % cols] }
                }
            }
        }
    }

    public static func fileName(for date: Date) -> String { "graphcast_\(OpenDataClient.stamp(date).full).safetensors" }
}
