import Foundation

/// ECMWF open-data messages as downloaded, shared by every model's initial conditions: a
/// field any model needs is fetched and stored once, whichever model asks for it first, and
/// each model decodes and processes it its own way when it builds its inputs.
///
/// A message is kept exactly as the server sent it -- one GRIB2 message, compressed --
/// keyed by everything that identifies it: the model that produced it (`ifs`,
/// `aifs-single`), the stream (`oper`, `wave`), the cycle, the parameter, level type and
/// level (step 0, 0.25 degrees). Nothing is applied before storing: not the roll to 0 E,
/// regridding, unit conversion, masking or normalisation, which differ from model to model.
/// Two models that read different messages for the same quantity keep both (AIFS reads
/// geopotential height `gh`, Aurora and GraphCast geopotential `z`).
///
///     <root>/<model>/<YYYYMMDDHH>/<stream>.<levtype>.<param>[.<level>].grib2
///
/// Every message is checked to be one whole GRIB2 message (its header, declared length and
/// end marker) when written and when read, and written under a temporary name first, so a
/// stopped download never leaves a half message that reads as a field.
public final class OpenDataCache: @unchecked Sendable {
    public struct Key: Hashable, Sendable, Comparable {
        /// The producing model's directory in open data: `ifs` or `aifs-single`.
        public let model: String
        public let stream: String
        /// The cycle (the analysis time; every message is step 0).
        public let date: Date
        public let param: String
        public let levtype: String
        public let level: String?

        public init(model: String, stream: String, date: Date, param: String, levtype: String, level: String?) {
            (self.model, self.stream, self.date, self.param, self.levtype, self.level) = (model, stream, date, param, levtype, level)
        }

        var fileName: String { [stream, levtype, param, level].compactMap { $0 }.joined(separator: ".") + ".grib2" }

        public static func < (a: Key, b: Key) -> Bool {
            (a.model, a.date, a.stream, a.levtype, a.param, a.level ?? "") < (b.model, b.date, b.stream, b.levtype, b.param, b.level ?? "")
        }
    }

    /// AWS first: ECMWF's own server rate-limits the many range requests a cycle needs.
    public static let mirrors = [
        OpenDataClient.Mirror(name: "aws", baseURL: URL(string: "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com")!),
        OpenDataClient.Mirror(name: "ecmwf", baseURL: URL(string: "https://data.ecmwf.int/forecasts")!),
        OpenDataClient.Mirror(name: "azure", baseURL: URL(string: "https://ai4edataeuwest.blob.core.windows.net/ecmwf")!),
    ]

    public let root: URL
    let mirrors: [OpenDataClient.Mirror]
    let session: URLSession
    let maxConcurrent: Int

    public init(root: URL, mirrors: [OpenDataClient.Mirror] = OpenDataCache.mirrors, session: URLSession = .shared,
                maxConcurrent: Int = 6) {
        self.root = root
        self.mirrors = mirrors
        self.session = session
        self.maxConcurrent = maxConcurrent
    }

    // MARK: - Where things are

    /// A step-0 file of `model`'s `stream` for the cycle `date` on `mirror`.
    public static func fileURL(_ mirror: OpenDataClient.Mirror, date: Date, model: String, stream: String, ext: String) -> URL {
        let s = OpenDataClient.stamp(date)
        return mirror.baseURL.appending(path: "\(s.day)/\(s.hour)z/\(model)/0p25/\(stream)/\(s.full)-0h-\(stream)-fc.\(ext)")
    }

    func cycleDirectory(model: String, date: Date) -> URL {
        let s = OpenDataClient.stamp(date)
        return root.appending(path: "\(model)/\(s.day)\(s.hour)", directoryHint: .isDirectory)
    }

    public func url(_ key: Key) -> URL { cycleDirectory(model: key.model, date: key.date).appending(path: key.fileName) }

    public func contains(_ key: Key) -> Bool { FileManager.default.fileExists(atPath: url(key).path) }

    /// Bytes on disk of the `keys` present.
    public func bytes(_ keys: [Key]) -> Int {
        keys.reduce(0) { $0 + ((try? url($1).resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
    }

    /// A cached message's bytes, checked to be one whole GRIB2 message.
    public func data(_ key: Key) throws -> Data {
        let data = try Data(contentsOf: url(key), options: .mappedIfSafe)
        guard Self.isWholeMessage(data) else {
            throw AIFSError.invalidInitialCondition("the cached \(key.param)\(key.level.map { " at \($0)" } ?? "") for \(OpenDataClient.stamp(key.date).full) is damaged")
        }
        return data
    }

    /// Cycles with anything cached, newest first.
    public func cycles() -> [Date] {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyyMMddHH"
        var dates = Set<Date>()
        for model in (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? [] {
            for name in (try? FileManager.default.contentsOfDirectory(atPath: root.appending(path: model).path)) ?? [] {
                if let d = f.date(from: name) { dates.insert(d) }
            }
        }
        return dates.sorted(by: >)
    }

    /// Deletes every message of the cycle `date` (any model, any stream) not in `keep`, and
    /// the cycle's directories once empty.
    public func retain(_ keep: Set<Key>, cycle date: Date) {
        let fm = FileManager.default
        for model in (try? fm.contentsOfDirectory(atPath: root.path)) ?? [] {
            let dir = cycleDirectory(model: model, date: date)
            let kept = Set(keep.filter { $0.model == model && $0.date == date }.map(\.fileName))
            for name in (try? fm.contentsOfDirectory(atPath: dir.path)) ?? [] where !kept.contains(name) {
                try? fm.removeItem(at: dir.appending(path: name))
            }
            if ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []).isEmpty { try? fm.removeItem(at: dir) }
        }
    }

    // MARK: - Downloading

    /// Downloads the `keys` not yet cached. Each file's wanted messages are located in its
    /// index and fetched in runs of adjacent byte ranges (a few requests per file, not one per
    /// message). `progress` counts every key, those already cached included.
    public func fetch(_ keys: [Key], progress: (@Sendable (OpenDataClient.Progress) -> Void)? = nil) async throws {
        let missing = Set(keys).filter { !contains($0) }.sorted()
        let total = Set(keys).count
        var done = total - missing.count, bytes = 0
        progress?(.init(completed: done, total: total, bytes: 0, message: missing.isEmpty ? "Already downloaded" : "Downloading \(missing.count) fields"))
        guard !missing.isEmpty else { return }

        struct File: Hashable { let model: String; let stream: String; let date: Date }
        struct Piece: Sendable { let key: Key; let offset: Int; let length: Int }
        struct Run: Sendable { let model: String; let stream: String; let date: Date; let start: Int; let end: Int; let pieces: [Piece] }
        var runs: [Run] = []
        for (file, wanted) in Dictionary(grouping: missing, by: { File(model: $0.model, stream: $0.stream, date: $0.date) }) {
            let entries = try await index(model: file.model, stream: file.stream, date: file.date)
            var pieces: [Piece] = []
            for k in wanted {
                guard let e = entries.first(where: { $0.param == k.param && $0.levtype == k.levtype && $0.levelist == k.level }) else {
                    throw AIFSError.invalidInitialCondition(
                        "\(k.model) open data for \(OpenDataClient.stamp(k.date).full) lacks \(k.param)\(k.level.map { " at \($0)" } ?? "")")
                }
                pieces.append(Piece(key: k, offset: e._offset, length: e._length))
            }
            pieces.sort { $0.offset < $1.offset }
            var current: [Piece] = []
            func close() {
                guard let first = current.first, let last = current.last else { return }
                runs.append(Run(model: file.model, stream: file.stream, date: file.date, start: first.offset,
                                end: last.offset + last.length, pieces: current))
                current = []
            }
            for p in pieces {
                if let last = current.last, last.offset + last.length != p.offset { close() }
                current.append(p)
            }
            close()
        }

        try await withThrowingTaskGroup(of: (Run, Int).self) { group in
            var pending = runs.sorted { ($0.date, $0.model, $0.start) < ($1.date, $1.model, $1.start) }[...]
            func launch() {
                guard let run = pending.popFirst() else { return }
                group.addTask {
                    let data = try await self.withMirrors("\(run.model) fields") { mirror in
                        var request = URLRequest(url: Self.fileURL(mirror, date: run.date, model: run.model, stream: run.stream, ext: "grib2"))
                        request.setValue("bytes=\(run.start)-\(run.end - 1)", forHTTPHeaderField: "Range")
                        let data = try await self.get(request)
                        guard data.count == run.end - run.start else {
                            throw URLError(.cannotParseResponse, userInfo: [NSLocalizedDescriptionKey: "range returned \(data.count) of \(run.end - run.start) bytes"])
                        }
                        return data
                    }
                    for p in run.pieces {
                        let start = p.offset - run.start
                        try self.store(data.subdata(in: start ..< start + p.length), p.key)
                    }
                    return (run, data.count)
                }
            }
            for _ in 0 ..< maxConcurrent { launch() }
            while let (run, size) = try await group.next() {
                done += run.pieces.count
                bytes += size
                progress?(.init(completed: done, total: total, bytes: bytes,
                                message: "\(run.model) \(OpenDataClient.stamp(run.date).full)"))
                launch()
            }
        }
    }

    /// Writes one message under a temporary name, then renames it into place.
    private func store(_ data: Data, _ key: Key) throws {
        guard Self.isWholeMessage(data) else {
            throw URLError(.cannotParseResponse, userInfo: [NSLocalizedDescriptionKey:
                "\(key.param)\(key.level.map { " at \($0)" } ?? "") is not one GRIB2 message"])
        }
        let dest = url(key)
        try FileManager.default.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = dest.deletingLastPathComponent().appending(path: ".\(key.fileName).\(UUID().uuidString)")
        try data.write(to: tmp)
        _ = try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: tmp, to: dest)
    }

    /// `GRIB`, edition 2, the length its header declares, and `7777` at the end.
    static func isWholeMessage(_ d: Data) -> Bool {
        guard d.count >= 16 + 4 else { return false }
        let b = [UInt8](d.prefix(16))
        guard b[0 ..< 4] == [0x47, 0x52, 0x49, 0x42], b[7] == 2 else { return false }
        let declared = b[8 ..< 16].reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
        return declared == UInt64(d.count) && [UInt8](d.suffix(4)) == [0x37, 0x37, 0x37, 0x37]
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

    private func index(model: String, stream: String, date: Date) async throws -> [OpenDataClient.IndexEntry] {
        let data = try await withMirrors("\(model) \(stream) index") { mirror in
            try await self.get(URLRequest(url: Self.fileURL(mirror, date: date, model: model, stream: stream, ext: "index")))
        }
        let decoder = JSONDecoder()
        return try String(decoding: data, as: UTF8.self).split(separator: "\n").map {
            try decoder.decode(OpenDataClient.IndexEntry.self, from: Data($0.utf8))
        }
    }
}
