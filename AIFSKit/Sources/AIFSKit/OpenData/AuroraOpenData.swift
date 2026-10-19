import AuroraANE
import Foundation

/// Aurora 1.5's initial state from ECMWF open data, as `aurora-1p5/aurora_1p5/batch.py`
/// builds it: the IFS 0.25-degree analyses at the cycle and 6 h before, with two kinds of
/// stand-in --
///
/// - lcc, mcc, hcc (not in IFS open data): AIFS's step 0, which is the IFS analysis passed
///   through AIFS's grid (its total cloud cover matches the IFS analysis's to 0.07 RMS);
/// - ci, sea-ice cover (not in open data at all): sea-ice thickness / 0.3 m, capped at 1.
///
/// Insolation is computed. Every field is rolled to start at 0 E, cropped to 720 rows (the
/// model drops the south pole's), normalised, and stored as fp16.
public final class AuroraOpenData: @unchecked Sendable {
    /// AWS first: ECMWF's own server rate-limits the many range requests a state needs.
    public static let mirrors = [
        OpenDataClient.Mirror(name: "aws", baseURL: URL(string: "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com")!),
        OpenDataClient.Mirror(name: "ecmwf", baseURL: URL(string: "https://data.ecmwf.int/forecasts")!),
        OpenDataClient.Mirror(name: "azure", baseURL: URL(string: "https://ai4edataeuwest.blob.core.windows.net/ecmwf")!),
    ]
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
    let maxConcurrent: Int

    public init(bundle: URL, mirrors: [OpenDataClient.Mirror] = AuroraOpenData.mirrors, session: URLSession = .shared,
                maxConcurrent: Int = 6) throws {
        metadata = try AuroraMetadata.load(bundle)
        self.mirrors = mirrors
        self.session = session
        self.maxConcurrent = maxConcurrent
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

    // MARK: - Networking

    func fileURL(_ mirror: OpenDataClient.Mirror, date: Date, model: String, ext: String) -> URL {
        let s = OpenDataClient.stamp(date)
        return mirror.baseURL.appending(path: "\(s.day)/\(s.hour)z/\(model)/0p25/oper/\(s.full)-0h-oper-fc.\(ext)")
    }

    private func get(_ request: URLRequest) async throws -> Data {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "HTTP \(code) for \(request.url!.absoluteString)"])
        }
        return data
    }

    private func withMirrors<T>(_ what: String, _ make: (OpenDataClient.Mirror) async throws -> T) async throws -> T {
        var failures: [String] = []
        for mirror in mirrors {
            do { return try await make(mirror) } catch {
                failures.append("\(mirror.name): \(error.localizedDescription)")
            }
        }
        throw URLError(.resourceUnavailable, userInfo: [
            NSLocalizedDescriptionKey: "Every mirror failed for \(what):\n" + failures.joined(separator: "\n"),
        ])
    }

    private func index(date: Date, model: String) async throws -> [OpenDataClient.IndexEntry] {
        let data = try await withMirrors("\(model) index") { mirror in
            try await self.get(URLRequest(url: self.fileURL(mirror, date: date, model: model, ext: "index")))
        }
        let decoder = JSONDecoder()
        return try String(decoding: data, as: UTF8.self).split(separator: "\n").map {
            try decoder.decode(OpenDataClient.IndexEntry.self, from: Data($0.utf8))
        }
    }

    private func isPublished(_ date: Date) async -> Bool {
        for mirror in mirrors {
            var ok = true
            for model in ["ifs", "aifs-single"] {
                var request = URLRequest(url: fileURL(mirror, date: date, model: model, ext: "index"))
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

    /// Download and assemble the state for `date` (the later input time).
    public func build(date: Date, progress: (@Sendable (OpenDataClient.Progress) -> Void)? = nil) async throws -> AuroraState {
        let times = [date.addingTimeInterval(-6 * 3600), date]
        let wanted = sources
        let rows = metadata.grid.rows, cols = metadata.grid.cols

        // Each file's messages, merged into runs of adjacent byte ranges: a few requests per
        // file rather than one per field.
        struct Piece: Sendable { let source: Source; let offset: Int; let length: Int }
        struct Run: Sendable { let time: Int; let model: String; let start: Int; let end: Int; let pieces: [Piece] }
        var runs: [Run] = []
        for (t, time) in times.enumerated() {
            for model in ["ifs", "aifs-single"] {
                let entries = try await index(date: time, model: model)
                var pieces: [Piece] = []
                for s in wanted where s.model == model {
                    guard let e = entries.first(where: { $0.param == s.param && $0.levtype == s.levtype && $0.levelist == s.level }) else {
                        throw AIFSError.invalidInitialCondition(
                            "\(model) open data for \(OpenDataClient.stamp(time).full) lacks \(s.param)\(s.level.map { " at \($0)" } ?? "")")
                    }
                    pieces.append(Piece(source: s, offset: e._offset, length: e._length))
                }
                pieces.sort { $0.offset < $1.offset }
                var current: [Piece] = []
                for p in pieces {
                    if let last = current.last, last.offset + last.length != p.offset {
                        runs.append(Run(time: t, model: model, start: current[0].offset, end: last.offset + last.length, pieces: current))
                        current = []
                    }
                    current.append(p)
                }
                if let last = current.last {
                    runs.append(Run(time: t, model: model, start: current[0].offset, end: last.offset + last.length, pieces: current))
                }
            }
        }

        let state = AuroraState(time: date, rows: rows, cols: cols)
        let total = runs.reduce(0) { $0 + $1.pieces.count }
        var done = 0, bytes = 0
        progress?(.init(completed: 0, total: total, bytes: 0, message: "Downloading \(total) fields"))

        try await withThrowingTaskGroup(of: (Run, Data).self) { group in
            var pending = runs[...]
            func launch() {
                guard let run = pending.popFirst() else { return }
                group.addTask {
                    let data = try await self.withMirrors("\(run.model) fields") { mirror in
                        var request = URLRequest(url: self.fileURL(mirror, date: times[run.time], model: run.model, ext: "grib2"))
                        request.setValue("bytes=\(run.start)-\(run.end - 1)", forHTTPHeaderField: "Range")
                        let data = try await self.get(request)
                        guard data.count == run.end - run.start else {
                            throw URLError(.cannotParseResponse, userInfo: [NSLocalizedDescriptionKey: "range returned \(data.count) of \(run.end - run.start) bytes"])
                        }
                        return data
                    }
                    return (run, data)
                }
            }
            for _ in 0 ..< maxConcurrent { launch() }
            while let (run, data) = try await group.next() {
                for p in run.pieces {
                    let start = p.offset - run.start
                    let field = try GRIB2.decode(data.subdata(in: start ..< start + p.length))
                    try store(field, source: p.source, slot: run.time, into: state)
                    done += 1
                }
                bytes += data.count
                progress?(.init(completed: done, total: total, bytes: bytes, message: "\(run.model) \(OpenDataClient.stamp(times[run.time]).full)"))
                launch()
            }
        }

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
