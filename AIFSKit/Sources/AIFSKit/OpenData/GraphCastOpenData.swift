import Foundation
import GraphCastANE

/// GraphCast operational's initial state from ECMWF open data: the IFS 0.25-degree analyses
/// at the cycle and 6 h before (the HRES analyses Google fine-tuned it on), every field rolled
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
    let maxConcurrent: Int

    public init(bundle: URL, mirrors: [OpenDataClient.Mirror] = AuroraOpenData.mirrors, session: URLSession = .shared,
                maxConcurrent: Int = 6) throws {
        metadata = try GraphCastRuntime.Metadata.load(bundle)
        self.mirrors = mirrors
        self.session = session
        self.maxConcurrent = maxConcurrent
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

    // MARK: - Networking

    func fileURL(_ mirror: OpenDataClient.Mirror, date: Date, ext: String) -> URL {
        let s = OpenDataClient.stamp(date)
        return mirror.baseURL.appending(path: "\(s.day)/\(s.hour)z/ifs/0p25/oper/\(s.full)-0h-oper-fc.\(ext)")
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

    private func index(date: Date) async throws -> [OpenDataClient.IndexEntry] {
        let data = try await withMirrors("IFS index") { mirror in
            try await self.get(URLRequest(url: self.fileURL(mirror, date: date, ext: "index")))
        }
        let decoder = JSONDecoder()
        return try String(decoding: data, as: UTF8.self).split(separator: "\n").map {
            try decoder.decode(OpenDataClient.IndexEntry.self, from: Data($0.utf8))
        }
    }

    private func isPublished(_ date: Date) async -> Bool {
        for mirror in mirrors {
            var request = URLRequest(url: fileURL(mirror, date: date, ext: "index"))
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

    /// Download and assemble the state for `date` (the later input time).
    public func build(date: Date, progress: (@Sendable (OpenDataClient.Progress) -> Void)? = nil) async throws -> GraphCastState {
        let times = [date.addingTimeInterval(-6 * 3600), date]
        let keys = metadata.stateVariables
        let sources = try keys.map { key in
            guard let s = Self.source(of: key) else { throw AIFSError.invalidInitialCondition("no open-data source for \(key)") }
            return s
        }

        // Each file's messages, merged into runs of adjacent byte ranges.
        struct Piece: Sendable { let variable: Int; let offset: Int; let length: Int }
        struct Run: Sendable { let time: Int; let start: Int; let end: Int; let pieces: [Piece] }
        var runs: [Run] = []
        for (t, time) in times.enumerated() {
            let entries = try await index(date: time)
            var pieces: [Piece] = []
            for (v, s) in sources.enumerated() {
                guard let e = entries.first(where: { $0.param == s.param && $0.levtype == s.levtype && $0.levelist == s.level }) else {
                    throw AIFSError.invalidInitialCondition(
                        "IFS open data for \(OpenDataClient.stamp(time).full) lacks \(s.param)\(s.level.map { " at \($0)" } ?? "")")
                }
                pieces.append(Piece(variable: v, offset: e._offset, length: e._length))
            }
            pieces.sort { $0.offset < $1.offset }
            var current: [Piece] = []
            for p in pieces {
                if let last = current.last, last.offset + last.length != p.offset {
                    runs.append(Run(time: t, start: current[0].offset, end: last.offset + last.length, pieces: current))
                    current = []
                }
                current.append(p)
            }
            if let last = current.last {
                runs.append(Run(time: t, start: current[0].offset, end: last.offset + last.length, pieces: current))
            }
        }

        let state = GraphCastState(time: date, variables: keys)
        let total = runs.reduce(0) { $0 + $1.pieces.count }
        var done = 0, bytes = 0
        progress?(.init(completed: 0, total: total, bytes: 0, message: "Downloading \(total) fields"))

        try await withThrowingTaskGroup(of: (Run, Data).self) { group in
            var pending = runs[...]
            func launch() {
                guard let run = pending.popFirst() else { return }
                group.addTask {
                    let data = try await self.withMirrors("IFS fields") { mirror in
                        var request = URLRequest(url: self.fileURL(mirror, date: times[run.time], ext: "grib2"))
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
                    try store(field, variable: p.variable, slot: run.time, into: state)
                    done += 1
                }
                bytes += data.count
                progress?(.init(completed: done, total: total, bytes: bytes, message: "IFS \(OpenDataClient.stamp(times[run.time]).full)"))
                launch()
            }
        }
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
