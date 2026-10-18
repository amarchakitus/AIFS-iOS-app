import Foundation
import MLX

/// Builds initial conditions on device from ECMWF open data -- the Swift counterpart of
/// `aifs-mps/aifs_mps/opendata.py::build_input_state`, restricted to the fields the bundle's
/// model reads (`manifest.stateVars`), so nothing unused is downloaded.
///
/// Per input time: fetch the `.index` of the `oper` and `wave` step-0 files, then each
/// needed message by HTTP range, decode it (GRIB2/CCSDS), roll it to 0..360 E, regrid it
/// to N320 with the same operator the Python pipeline uses, and apply the same
/// transformations (mwd -> cos/sin, soil renames, gh -> z, land-only fields NaN at sea).
public final class OpenDataClient: @unchecked Sendable {
    public struct Mirror: Sendable, Hashable {
        public let name: String
        public let baseURL: URL
    }

    /// Tried in order per request. Azure was the Python default but answers 409 for
    /// recent runs as of 2026-10; ECMWF's own portal is fastest when not saturated.
    public static let defaultMirrors = [
        Mirror(name: "ecmwf", baseURL: URL(string: "https://data.ecmwf.int/forecasts")!),
        Mirror(name: "aws", baseURL: URL(string: "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com")!),
        Mirror(name: "azure", baseURL: URL(string: "https://ai4edataeuwest.blob.core.windows.net/ecmwf")!),
    ]

    public static let gravity = 9.80665

    public struct Progress: Sendable {
        public let completed: Int
        public let total: Int
        public let bytes: Int
        public let message: String
    }

    let manifest: Manifest
    let mirrors: [Mirror]
    let session: URLSession
    let maxConcurrent: Int
    private let regrid: SparseOperator
    private let sea: [Bool]

    /// `bundle` must contain `opendata.safetensors` (see `aifs-mps/aifs_mps/export.py`).
    public init(
        bundle: URL, manifest: Manifest, mirrors: [Mirror] = OpenDataClient.defaultMirrors,
        session: URLSession = .shared, maxConcurrent: Int = 6
    ) throws {
        self.manifest = manifest
        self.mirrors = mirrors
        self.session = session
        self.maxConcurrent = maxConcurrent
        // Read without MLX: the client also serves the cycle list, which has no business
        // starting a Metal device (and the simulator cannot).
        let (data, tensors, metadata) = try SafeTensors.open(bundle.appending(path: "opendata.safetensors"))
        guard let rows = metadata["rows"].flatMap(Int.init), let cols = metadata["cols"].flatMap(Int.init),
              let indptr = tensors["indptr"], let indices = tensors["indices"], let weights = tensors["weights"],
              let seaMask = tensors["sea"]
        else {
            throw AIFSError.invalidBundle("opendata.safetensors lacks the operator or the source grid shape")
        }
        regrid = SparseOperator(
            indptr: SafeTensors.array(data, indptr, as: Int32.self),
            indices: SafeTensors.array(data, indices, as: Int32.self),
            weights: SafeTensors.array(data, weights, as: Float.self),
            cols: rows * cols)
        sea = SafeTensors.array(data, seaMask, as: UInt8.self).map { $0 != 0 }
        guard regrid.rows == manifest.numDataNodes, sea.count == manifest.numDataNodes else {
            throw AIFSError.invalidBundle("open-data operator does not match the model grid")
        }
    }

    // MARK: - What to fetch

    /// One open-data message.
    struct Message: Hashable, Sendable {
        let stream: String  // oper | wave
        let param: String
        let levtype: String
        let level: String?
    }

    static let soil: [String: (param: String, level: String)] = [
        "stl1": ("sot", "1"), "stl2": ("sot", "2"), "swvl1": ("vsw", "1"), "swvl2": ("vsw", "2"),
    ]
    static let wave: Set<String> = [
        "wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "cdww", "mwp", "swh",
    ]
    /// Undefined over sea in the training data; NaN there so anemoi re-imputes them.
    static let landOnly: Set<String> = ["sd", "swvl1", "swvl2"]

    /// The message a state variable is derived from.
    static func source(of name: String) -> Message {
        if name == "cos_mwd" || name == "sin_mwd" {
            return Message(stream: "wave", param: "mwd", levtype: "sfc", level: nil)
        }
        if wave.contains(name) { return Message(stream: "wave", param: name, levtype: "sfc", level: nil) }
        if let s = soil[name] { return Message(stream: "oper", param: s.param, levtype: "sol", level: s.level) }
        if let sep = name.lastIndex(of: "_"), let level = Int(name[name.index(after: sep)...]) {
            let param = String(name[..<sep])
            return Message(stream: "oper", param: param == "z" ? "gh" : param, levtype: "pl", level: String(level))
        }
        return Message(stream: "oper", param: name, levtype: "sfc", level: nil)
    }

    /// Physical value of state variable `name` from its regridded source message.
    static func transform(_ name: String, _ v: Double) -> Double {
        switch name {
        case "cos_mwd": cos(v * .pi / 180)
        case "sin_mwd": sin(v * .pi / 180)
        case _ where name.hasPrefix("z_"): v * gravity
        default: v
        }
    }

    // MARK: - URLs

    static func stamp(_ date: Date) -> (day: String, hour: String, full: String) {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyyMMdd"
        let day = f.string(from: date)
        f.dateFormat = "HH"
        let hour = f.string(from: date)
        return (day, hour, day + hour + "0000")
    }

    func fileURL(_ mirror: Mirror, date: Date, stream: String, ext: String) -> URL {
        let s = Self.stamp(date)
        return mirror.baseURL.appending(
            path: "\(s.day)/\(s.hour)z/ifs/0p25/\(stream)/\(s.full)-0h-\(stream)-fc.\(ext)")
    }

    // MARK: - Networking

    private func get(_ request: URLRequest) async throws -> Data {
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200 ..< 300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "HTTP \(code) for \(request.url!.absoluteString)"])
        }
        return data
    }

    /// `make(mirror)` against each mirror in turn until one succeeds.
    private func withMirrors<T>(_ what: String, _ make: (Mirror) async throws -> T) async throws -> T {
        var failures: [String] = []
        for mirror in mirrors {
            do { return try await make(mirror) } catch {
                failures.append("\(mirror.name): \(error.localizedDescription)")
            }
        }
        throw URLError(
            .resourceUnavailable,
            userInfo: [NSLocalizedDescriptionKey: "Every mirror failed for \(what):\n" + failures.joined(separator: "\n")])
    }

    struct IndexEntry: Decodable {
        let param: String
        let levtype: String
        let levelist: String?
        let _offset: Int
        let _length: Int
    }

    private func index(date: Date, stream: String) async throws -> [IndexEntry] {
        let data = try await withMirrors("\(stream) index") { mirror in
            try await self.get(URLRequest(url: self.fileURL(mirror, date: date, stream: stream, ext: "index")))
        }
        let decoder = JSONDecoder()
        return try String(decoding: data, as: UTF8.self).split(separator: "\n").map {
            try decoder.decode(IndexEntry.self, from: Data($0.utf8))
        }
    }

    /// Most recent cycle whose step-0 files (both streams), and those of the cycle before
    /// it -- the model's first input time -- are published.
    public func latestDate(now: Date = .now, lookback: Int = 8) async throws -> Date {
        let cycle = 6 * 3600.0
        let newest = (now.timeIntervalSince1970 / cycle).rounded(.down) * cycle
        for k in 0 ..< lookback {
            let date = Date(timeIntervalSince1970: newest - Double(k) * cycle)
            if await isPublished(date), await isPublished(date.addingTimeInterval(-cycle)) {
                return date
            }
        }
        throw URLError(.resourceUnavailable, userInfo: [NSLocalizedDescriptionKey: "No complete open-data cycle in the last \(lookback * 6) hours"])
    }

    /// Every cycle in the last `lookback` (newest first) that can be built: it and the
    /// cycle before it are both published on some mirror. ECMWF's own server keeps a few
    /// days; the cloud mirrors keep longer. Probed in parallel, a few requests per cycle.
    public func availableDates(now: Date = .now, lookback: Int = 20) async -> [Date] {
        let cycle = 6 * 3600.0
        let newest = (now.timeIntervalSince1970 / cycle).rounded(.down) * cycle
        // One extra, older cycle: the oldest candidate's first input time.
        let dates = (0 ... lookback).map { Date(timeIntervalSince1970: newest - Double($0) * cycle) }
        let published = await withTaskGroup(of: (Date, Bool).self) { group in
            for date in dates { group.addTask { (date, await self.isPublished(date)) } }
            var found: [Date: Bool] = [:]
            for await (date, ok) in group { found[date] = ok }
            return found
        }
        return dates.dropLast().filter { published[$0] == true && published[$0.addingTimeInterval(-cycle)] == true }
    }

    private func isPublished(_ date: Date) async -> Bool {
        for mirror in mirrors {
            var ok = true
            for stream in ["oper", "wave"] {
                var request = URLRequest(url: fileURL(mirror, date: date, stream: stream, ext: "index"))
                request.httpMethod = "HEAD"
                let status = try? await (session.data(for: request).1 as? HTTPURLResponse)?.statusCode
                if status != 200 { ok = false; break }
            }
            if ok { return true }
        }
        return false
    }

    // MARK: - Build

    /// Download and assemble the initial condition for `date` (the latest input time).
    public func build(
        date: Date, progress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> InitialCondition {
        let dt = TimeInterval(manifest.timestepHours * 3600)
        let times = (0 ..< manifest.multiStep).map { date.addingTimeInterval(dt * Double($0 - manifest.multiStep + 1)) }
        let names = manifest.stateVars
        let messages = Array(Set(names.map(Self.source)))

        // Locate every message in each time's indexes.
        struct Job: Sendable { let time: Int; let message: Message; let offset: Int; let length: Int }
        var jobs: [Job] = []
        for (t, time) in times.enumerated() {
            for stream in ["oper", "wave"] {
                let entries = try await index(date: time, stream: stream)
                for m in messages where m.stream == stream {
                    guard let e = entries.first(where: {
                        $0.param == m.param && $0.levtype == m.levtype && $0.levelist == m.level
                    }) else {
                        throw AIFSError.invalidInitialCondition(
                            "open data for \(Self.stamp(time).full) lacks \(m.param)\(m.level.map { " at \($0)" } ?? "")")
                    }
                    jobs.append(Job(time: t, message: m, offset: e._offset, length: e._length))
                }
            }
        }

        let total = jobs.count, n = manifest.numDataNodes
        var regridded: [Message: [[Double]]] = [:]
        var done = 0, bytes = 0
        progress?(Progress(completed: 0, total: total, bytes: 0, message: "Downloading \(total) fields"))

        try await withThrowingTaskGroup(of: (Job, [Double], Int).self) { group in
            var pending = jobs[...]
            func launch() {
                guard let job = pending.popFirst() else { return }
                group.addTask {
                    let data = try await self.withMirrors(job.message.param) { mirror in
                        var request = URLRequest(url: self.fileURL(mirror, date: times[job.time], stream: job.message.stream, ext: "grib2"))
                        request.setValue("bytes=\(job.offset)-\(job.offset + job.length - 1)", forHTTPHeaderField: "Range")
                        let data = try await self.get(request)
                        guard data.count == job.length else {
                            throw URLError(.cannotParseResponse, userInfo: [NSLocalizedDescriptionKey: "range returned \(data.count) of \(job.length) bytes"])
                        }
                        return data
                    }
                    let field = try GRIB2.decode(data)
                    return (job, try self.toN320(field), data.count)
                }
            }
            for _ in 0 ..< maxConcurrent { launch() }
            while let (job, values, size) = try await group.next() {
                regridded[job.message, default: Array(repeating: [], count: times.count)][job.time] = values
                done += 1
                bytes += size
                progress?(Progress(completed: done, total: total, bytes: bytes, message: "\(job.message.param) \(Self.stamp(times[job.time]).full)"))
                launch()
            }
        }

        var fields: [String: MLXArray] = [:]
        for name in names {
            let source = regridded[Self.source(of: name)]!
            var flat = [Float](repeating: 0, count: times.count * n)
            for t in 0 ..< times.count {
                for i in 0 ..< n {
                    var v = Self.transform(name, source[t][i])
                    if Self.landOnly.contains(name), sea[i] { v = .nan }
                    flat[t * n + i] = Float(v)
                }
            }
            fields[name] = MLXArray(flat, [times.count, n])
        }
        return InitialCondition(date: date, fields: fields)
    }

    /// Roll an open-data field to start at 0 E (it starts at 180 E) and regrid to N320.
    func toN320(_ field: GRIB2Field) throws -> [Double] {
        let ni = field.ni, nj = field.nj
        guard ni * nj == regrid.cols, abs(field.firstLatitude - 90) < 1e-6 else {
            throw GRIB2Error.unsupported("grid \(nj)x\(ni) from \(field.firstLatitude)N")
        }
        let dlon = 360.0 / Double(ni)
        let shift = Int(((360 - field.firstLongitude) / dlon).rounded()) % ni
        var rolled = [Float](repeating: 0, count: ni * nj)
        field.values.withUnsafeBufferPointer { src in
            for j in 0 ..< nj {
                let row = j * ni
                for i in 0 ..< ni { rolled[row + i] = src[row + (i + shift) % ni] }
            }
        }
        return regrid.apply(rolled)
    }
}

/// A CSR matrix applied in double precision, as scipy does for the Python pipeline.
struct SparseOperator: Sendable {
    let indptr: [Int32]
    let indices: [Int32]
    let weights: [Float]
    var rows: Int { indptr.count - 1 }
    let cols: Int

    init(indptr: [Int32], indices: [Int32], weights: [Float], cols: Int) {
        self.indptr = indptr
        self.indices = indices
        self.weights = weights
        self.cols = cols
    }

    func apply(_ x: [Float]) -> [Double] {
        var out = [Double](repeating: 0, count: rows)
        indptr.withUnsafeBufferPointer { p in
            indices.withUnsafeBufferPointer { idx in
                weights.withUnsafeBufferPointer { w in
                    x.withUnsafeBufferPointer { x in
                        for r in 0 ..< p.count - 1 {
                            var acc = 0.0
                            for k in Int(p[r]) ..< Int(p[r + 1]) {
                                acc += Double(w[k]) * Double(x[Int(idx[k])])
                            }
                            out[r] = acc
                        }
                    }
                }
            }
        }
        return out
    }
}

extension InitialCondition {
    /// Write in the format `./aifs export ic` produces, so either source loads the same way.
    public func save(to url: URL) throws {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        let iso = String(f.string(from: date).dropLast())  // naive UTC, like Python's isoformat()
        let partial = url.deletingLastPathComponent().appending(path: url.lastPathComponent + ".partial.safetensors")
        try MLX.save(arrays: fields, metadata: ["date": iso, "model": "single"], url: partial)
        _ = try? FileManager.default.removeItem(at: url)
        try FileManager.default.moveItem(at: partial, to: url)
    }

    public static func fileName(for date: Date) -> String {
        "ic_\(OpenDataClient.stamp(date).day)T\(OpenDataClient.stamp(date).hour).safetensors"
    }
}
