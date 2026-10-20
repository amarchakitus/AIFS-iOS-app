import AIFSKit
import Foundation

/// An initial condition a model can run from: a cycle whose open-data messages the model
/// needs are all in the shared cache (`Documents/opendata`), or a state file of the model's
/// own in `Documents/ic` (written before the cache, or copied in from a Mac).
nonisolated struct ICFile: Identifiable, Hashable, Sendable {
    let date: Date
    let model: ForecastModel
    /// What it takes on disk: the model's messages (some shared with other models' initial
    /// conditions), or the file's size.
    let bytes: Int
    /// The model's own state file, when it isn't built from the cache.
    let file: URL?

    var id: String { "\(model.rawValue)_\(StoredForecast.stamp(date))" }

    /// A file in `Documents/ic`: Aurora states are `aurora_<date>`, GraphCast's
    /// `graphcast_<date>`, AIFS initial conditions `ic_<date>`.
    static func model(ofFile url: URL) -> ForecastModel? {
        let name = url.lastPathComponent
        guard url.pathExtension == "safetensors" else { return nil }
        if name.hasPrefix("aurora_") { return .aurora }
        if name.hasPrefix("graphcast_") { return .graphcast }
        return name.hasPrefix("ic_") ? .aifs : nil
    }
}

/// The initial conditions on disk: the shared open-data cache, and which of its cycles each
/// model can be built from.
///
/// A model's initial condition at a cycle reads messages of that cycle and the one 6 h
/// before, and shares most of them with the other models' (IFS fields every model reads)
/// and with its own at neighbouring cycles. So deleting one keeps every message another
/// complete initial condition still needs, and removes the rest of the two cycles' messages.
/// (Only installed models' initial conditions count: without its bundle a model's messages
/// can't be named, nor run.)
nonisolated enum InitialConditionStore {
    static let cache = OpenDataCache(root: AppPaths.openData)

    /// The messages `model`'s initial condition at `date` is built from, from its installed
    /// bundle's metadata; nil without the bundle.
    static func keys(_ model: ForecastModel, date: Date) -> [OpenDataCache.Key]? {
        guard AppPaths.isInstalled(model), let source = sources.source(model) else { return nil }
        return source(date)
    }

    /// Per model, its keys for a date: built once from its bundle (the bundle's metadata is
    /// small, but AIFS's open-data client would load its regridding operator to ask).
    private static let sources = Sources()

    private final class Sources: @unchecked Sendable {
        private var made: [ForecastModel: (Date, @Sendable (Date) -> [OpenDataCache.Key])] = [:]
        private let lock = NSLock()

        func source(_ model: ForecastModel) -> (@Sendable (Date) -> [OpenDataCache.Key])? {
            let bundle = AppPaths.bundle(for: model)
            // Rebuilt when the bundle changes (a model update).
            let stamp = (try? bundle.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            lock.lock()
            defer { lock.unlock() }
            if let (seen, f) = made[model], seen == stamp { return f }
            let f: (@Sendable (Date) -> [OpenDataCache.Key])?
            switch model {
            case .aifs:
                f = (try? Manifest.load(from: bundle.appending(path: "manifest.json"))).map { m in
                    { OpenDataClient.keys(date: $0, manifest: m) }
                }
            case .aurora:
                f = (try? AuroraOpenData(bundle: bundle)).map { c in { c.keys(date: $0) } }
            case .graphcast:
                f = (try? GraphCastOpenData(bundle: bundle)).map { c in { (try? c.keys(date: $0)) ?? [] } }
            }
            if let f { made[model] = (stamp, f) }
            return f
        }
    }

    /// Whether every message of `model`'s initial condition at `date` is cached.
    static func isComplete(_ model: ForecastModel, date: Date) -> Bool {
        guard let keys = keys(model, date: date), !keys.isEmpty else { return false }
        return keys.allSatisfy(cache.contains)
    }

    /// `model`'s initial conditions, newest first: the cache's complete cycles, and its own
    /// files for the cycles the cache doesn't have.
    static func list(_ model: ForecastModel) -> [ICFile] {
        var out: [ICFile] = []
        for date in cache.cycles() {
            guard let keys = keys(model, date: date), !keys.isEmpty, keys.allSatisfy(cache.contains) else { continue }
            out.append(ICFile(date: date, model: model, bytes: cache.bytes(keys), file: nil))
        }
        let cached = Set(out.map(\.date))
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: AppPaths.initialConditions, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        for url in urls where ICFile.model(ofFile: url) == model {
            guard let date = InitialCondition.peekDate(url), !cached.contains(date) else { continue }
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            out.append(ICFile(date: date, model: model, bytes: size, file: url))
        }
        return out.sorted { $0.date > $1.date }
    }

    /// Every message some complete initial condition (any model, any cycle) needs, but for
    /// `excluded`'s.
    static func liveKeys(excluding excluded: (ForecastModel, Date)? = nil) -> Set<OpenDataCache.Key> {
        var live = Set<OpenDataCache.Key>()
        for date in cache.cycles() {
            for model in ForecastModel.allCases {
                if let e = excluded, e.0 == model, e.1 == date { continue }
                guard let keys = keys(model, date: date), !keys.isEmpty, keys.allSatisfy(cache.contains) else { continue }
                live.formUnion(keys)
            }
        }
        return live
    }

    /// Deletes `ic`: its file, and its messages that no other complete initial condition
    /// needs (with whatever else of its two cycles nothing complete needs).
    static func delete(_ ic: ICFile) {
        if let file = ic.file { try? FileManager.default.removeItem(at: file) }
        guard let keys = keys(ic.model, date: ic.date) else { return }
        let live = liveKeys(excluding: (ic.model, ic.date))
        for date in Set(keys.map(\.date)) { cache.retain(live, cycle: date) }
    }

    /// After a stopped download of `model`'s initial condition at `date`: deletes what it
    /// fetched that no complete initial condition needs.
    static func discardIncomplete(_ model: ForecastModel, date: Date) {
        guard let keys = keys(model, date: date) else { return }
        let live = liveKeys()
        for date in Set(keys.map(\.date)) { cache.retain(live, cycle: date) }
    }

    /// Every model's initial conditions.
    static func all() -> [ICFile] { ForecastModel.allCases.flatMap(list) }

    /// When `ic` arrived: its file's time, or when the last of its messages was downloaded.
    static func added(_ ic: ICFile) -> Date {
        func modified(_ url: URL) -> Date {
            (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        }
        if let file = ic.file { return modified(file) }
        return keys(ic.model, date: ic.date)?.map { modified(cache.url($0)) }.max() ?? .distantPast
    }

    /// What initial conditions take on disk: the cache, and the models' own files.
    static func bytesOnDisk() -> Int {
        var total = 0
        for dir in [AppPaths.openData, AppPaths.initialConditions] {
            let files = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey])
            while let url = files?.nextObject() as? URL {
                total += (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            }
        }
        return total
    }

    /// Deletes the initial conditions that arrived since `since`, or all of them (nil), as
    /// clearing a browser's history does. A message an older initial condition still reads
    /// stays; so do forecasts.
    static func clear(since: Date?) {
        guard let since else {
            try? FileManager.default.removeItem(at: AppPaths.openData)
            try? FileManager.default.createDirectory(at: AppPaths.openData, withIntermediateDirectories: true)
            AppPaths.excludeFromBackup(AppPaths.openData)
            let files = (try? FileManager.default.contentsOfDirectory(at: AppPaths.initialConditions, includingPropertiesForKeys: nil)) ?? []
            for url in files where ICFile.model(ofFile: url) != nil { try? FileManager.default.removeItem(at: url) }
            return
        }
        let gone = all().filter { added($0) >= since }
        for ic in gone { if let file = ic.file { try? FileManager.default.removeItem(at: file) } }
        let goneIDs = Set(gone.filter { $0.file == nil }.map(\.id))
        // What every remaining complete initial condition reads.
        var live = Set<OpenDataCache.Key>()
        for ic in all() where ic.file == nil && !goneIDs.contains(ic.id) {
            live.formUnion(keys(ic.model, date: ic.date) ?? [])
        }
        // The cleared ones' cycles, and any other cycle a download touched since (what a
        // stopped one left): everything there that nothing remaining reads.
        var cycles = Set(gone.filter { $0.file == nil }.flatMap { (keys($0.model, date: $0.date) ?? []).map(\.date) })
        for date in cache.cycles() where cache.lastModified(cycle: date) >= since { cycles.insert(date) }
        for date in cycles { cache.retain(live, cycle: date) }
    }

    /// Cycles older than `age` that no complete initial condition reads: what a failed
    /// download left (kept that long so retrying resumes it).
    static func pruneStale(olderThan age: TimeInterval = 2 * 86400) {
        let live = liveKeys()
        let liveDates = Set(live.map(\.date))
        for date in cache.cycles() where !liveDates.contains(date) && date < Date.now.addingTimeInterval(-age) {
            cache.retain([], cycle: date)
        }
    }
}
