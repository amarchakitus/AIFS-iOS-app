import AuroraANE
import Foundation

/// Aurora 1.5's initial state from ECMWF open data, as `aurora-1p5/aurora_1p5/batch.py`
/// builds it, its messages through the shared `OpenDataCache`: the IFS 0.25-degree analyses
/// at the cycle and 6 h before, with two kinds of stand-in --
///
/// - lcc, mcc, hcc (not in IFS open data): AIFS's step 0, which is the IFS analysis passed
///   through AIFS's grid (its total cloud cover matches the IFS analysis's to 0.07 RMS);
/// - ci, sea-ice cover (not in open data at all): sea-ice thickness / 0.3 m, capped at 1.
///
/// Insolation is computed. Every field is rolled to start at 0 E, cropped to 720 rows (the
/// model drops the south pole's), normalised, and stored as fp16.
public final class AuroraOpenData: @unchecked Sendable {
    /// AWS first: ECMWF's own server rate-limits the many range requests a state needs.
    public static let mirrors = OpenDataCache.mirrors
    static let iceThicknessForFullCover: Float = 0.3  // m

    /// Where an input comes from: IFS (`ifs`) or AIFS (`aifs-single`), and which message.
    struct Source: Hashable, Sendable {
        let model: String
        let param: String
        let levtype: String
        let level: String?
    }

    let metadata: AuroraMetadata
    let mirrors: [OpenDataClient.Mirror]
    let session: URLSession

    public init(bundle: URL, mirrors: [OpenDataClient.Mirror] = AuroraOpenData.mirrors, session: URLSession = .shared) throws {
        metadata = try AuroraMetadata.load(bundle)
        self.mirrors = mirrors
        self.session = session
    }

    static func source(ofInput name: String) -> Source? {
        switch name {
        case "stl1": Source(model: "ifs", param: "sot", levtype: "sol", level: "1")
        case "swvl1": Source(model: "ifs", param: "vsw", levtype: "sol", level: "1")
        case "scaled_sd": Source(model: "ifs", param: "sd", levtype: "sfc", level: nil)
        case "ci": Source(model: "ifs", param: "sithick", levtype: "sfc", level: nil)
        case "lcc", "mcc", "hcc": Source(model: "aifs-single", param: name, levtype: "sfc", level: nil)
        case "insolation": nil
        default: Source(model: "ifs", param: name, levtype: "sfc", level: nil)
        }
    }

    var sources: [Source] {
        let surface = metadata.inputs.compactMap(Self.source(ofInput:))
        let atmosphere = metadata.atmosphere.flatMap { v in
            metadata.levels.map { Source(model: "ifs", param: v, levtype: "pl", level: String($0)) }
        }
        return surface + atmosphere
    }

    // MARK: - Availability

    private func isPublished(_ date: Date) async -> Bool {
        for mirror in mirrors {
            var ok = true
            for model in ["ifs", "aifs-single"] {
                var request = URLRequest(url: OpenDataCache.fileURL(mirror, date: date, model: model, stream: "oper", ext: "index"))
                request.httpMethod = "HEAD"
                if (try? await (session.data(for: request).1 as? HTTPURLResponse)?.statusCode) != 200 { ok = false; break }
            }
            if ok { return true }
        }
        return false
    }

    /// Cycles in the last `lookback` (newest first) whose state can be built: it and the
    /// cycle 6 h before are published, by both IFS and AIFS.
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

    func key(_ s: Source, _ time: Date) -> OpenDataCache.Key {
        OpenDataCache.Key(model: s.model, stream: "oper", date: time, param: s.param, levtype: s.levtype, level: s.level)
    }

    /// Every open-data message the state at `date` is built from.
    public func keys(date: Date) -> [OpenDataCache.Key] {
        Self.times(date).flatMap { time in sources.map { key($0, time) } }.sorted()
    }

    /// Downloads into `cache` the messages for `date` it doesn't have yet.
    public func download(date: Date, into cache: OpenDataCache, progress: (@Sendable (OpenDataClient.Progress) -> Void)? = nil) async throws {
        try await cache.fetch(keys(date: date), progress: progress)
    }

    /// Assembles the state for `date` (the later input time) from the messages in `cache`.
    public func build(date: Date, from cache: OpenDataCache) throws -> AuroraState {
        let times = Self.times(date)
        let rows = metadata.grid.rows, cols = metadata.grid.cols
        let state = AuroraState(time: date, rows: rows, cols: cols)
        let jobs = times.indices.flatMap { t in sources.map { (t, $0) } }
        // Each message fills its own plane of the state: across the cores.
        let failure = FirstFailure()
        DispatchQueue.concurrentPerform(iterations: jobs.count) { j in
            let (t, s) = jobs[j]
            do {
                try store(GRIB2.decode(cache.data(key(s, times[t]))), source: s, slot: t, into: state)
            } catch {
                failure.set(error)
            }
        }
        if let error = failure.error { throw error }

        // Insolation, computed for both times.
        let insolation = metadata.inputs.firstIndex(of: "insolation")!
        let plane = UnsafeMutablePointer<Float>.allocate(capacity: rows * cols)
        defer { plane.deallocate() }
        for (t, time) in times.enumerated() {
            AuroraInsolation.field(time, rows: rows, cols: cols, into: plane)
            let dst = state.surface(insolation, t)
            for i in 0 ..< rows * cols { dst[i] = Float16(metadata.normalise(plane[i], "insolation")) }
        }
        return state
    }

    /// One decoded message into the state: rolled to start at 0 E, cropped to the model's
    /// rows, turned into the model's variable, and normalised.
    private func store(_ field: GRIB2Field, source: Source, slot t: Int, into state: AuroraState) throws {
        let rows = metadata.grid.rows, cols = metadata.grid.cols
        guard field.ni == cols, field.nj >= rows, abs(field.firstLatitude - 90) < 1e-6 else {
            throw GRIB2Error.unsupported("grid \(field.nj)x\(field.ni) from \(field.firstLatitude)N")
        }
        let shift = Int(((360 - field.firstLongitude) / (360.0 / Double(cols))).rounded()) % cols
        let dst: UnsafeMutablePointer<Float16>
        let convert: (Float) -> Float
        if source.levtype == "pl" {
            let v = metadata.atmosphere.firstIndex(of: source.param)!, level = Int(source.level!)!
            dst = state.atmosphere(v, t, metadata.levels.firstIndex(of: level)!)
            let name = source.param
            convert = { [metadata] x in metadata.normalise(x, name, level: level) }
        } else {
            let name = metadata.inputs.first { Self.source(ofInput: $0) == source }!
            dst = state.surface(metadata.inputs.firstIndex(of: name)!, t)
            let meta = metadata
            switch name {
            case "ci": convert = { x in meta.normalise(min(max((x.isNaN ? 0 : x) / Self.iceThicknessForFullCover, 0), 1), name) }
            case "lcc", "mcc", "hcc": convert = { x in meta.normalise(min(max(x / 100, 0), 1), name) }  // AIFS gives percent
            default: convert = { x in meta.normalise(x.isNaN ? 0 : x, name) }
            }
        }
        field.values.withUnsafeBufferPointer { src in
            for r in 0 ..< rows {
                for c in 0 ..< cols { dst[r * cols + c] = Float16(convert(src[r * cols + (c + shift) % cols])) }
            }
        }
    }

    public static func fileName(for date: Date) -> String { "aurora_\(OpenDataClient.stamp(date).full).safetensors" }
}
