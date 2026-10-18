import AIFSKit
import Foundation

/// Where everything lives in the app's Documents folder, which Finder and the Files app
/// can see (UIFileSharingEnabled), so a model bundle can be dropped in from a Mac.
///
///     Documents/aifs-single-v2/        model bundle (`./aifs export bundle`)
///     Documents/ic/ic_<date>.safetensors     initial conditions (downloaded or exported)
///     Documents/forecasts/<init>/      one directory per forecast run
nonisolated enum AppPaths {
    static var documents: URL { URL.documentsDirectory }
    static var bundle: URL { documents.appending(path: "aifs-single-v2", directoryHint: .isDirectory) }
    static var initialConditions: URL { documents.appending(path: "ic", directoryHint: .isDirectory) }
    static var forecasts: URL { documents.appending(path: "forecasts", directoryHint: .isDirectory) }

    static func ensureDirectories() {
        for url in [initialConditions, forecasts] {
            try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        }
        // Out of iCloud backups: the model and initial conditions download again, and a
        // forecast reruns from them -- and between them they run to gigabytes.
        for url in [bundle, initialConditions, forecasts] { excludeFromBackup(url) }
        // Documents/coreml only ever holds models copied in for `--ane-probe`; reclaim it on
        // any other launch. (The forecast's own Core ML models live in the bundle.)
        if !ProcessInfo.processInfo.arguments.contains("--ane-probe") {
            try? FileManager.default.removeItem(at: documents.appending(path: "coreml"))
        }
    }

    static func excludeFromBackup(_ url: URL) {
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        var url = url
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
    }

    static var bundleIsPresent: Bool {
        FileManager.default.fileExists(atPath: bundle.appending(path: "manifest.json").path)
    }
}

/// An initial-condition file on disk.
nonisolated struct ICFile: Identifiable, Hashable, Sendable {
    let url: URL
    let date: Date
    let bytes: Int
    var id: URL { url }

    static func list() -> [ICFile] {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: AppPaths.initialConditions, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return urls.filter { $0.lastPathComponent.hasPrefix("ic_") && $0.pathExtension == "safetensors" }
            .compactMap { url in
                guard let date = InitialCondition.peekDate(url) else { return nil }
                let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                return ICFile(url: url, date: date, bytes: size)
            }
            .sorted { $0.date > $1.date }
    }
}

/// `forecast.json`: what a stored forecast is and how far it got.
nonisolated struct ForecastMeta: Codable, Hashable, Sendable {
    var initDate: Date
    var stepHours: Int
    var plannedSteps: Int
    var completedSteps: Int
    var fields: [String]
    var secondsPerStep: [Double]
    var peakMemoryBytes: Int
    var model: String
    var finished: Bool
}

/// A forecast directory: `forecast.json`, `static.safetensors` (land-sea mask), and one
/// `step_<hours>.safetensors` per completed step, written as the run progresses.
nonisolated struct StoredForecast: Identifiable, Hashable, Sendable {
    let url: URL
    var meta: ForecastMeta
    var id: URL { url }

    static let metaName = "forecast.json"

    static func directory(for initDate: Date) -> URL {
        AppPaths.forecasts.appending(path: "init_\(Self.stamp(initDate))", directoryHint: .isDirectory)
    }

    static func stamp(_ date: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "yyyyMMdd'T'HH"
        return f.string(from: date)
    }

    static func list() -> [StoredForecast] {
        let dirs = (try? FileManager.default.contentsOfDirectory(
            at: AppPaths.forecasts, includingPropertiesForKeys: nil)) ?? []
        return dirs.compactMap { load($0) }.sorted { $0.meta.initDate > $1.meta.initDate }
    }

    static func load(_ url: URL) -> StoredForecast? {
        guard let data = try? Data(contentsOf: url.appending(path: metaName)),
            let meta = try? Self.decoder.decode(ForecastMeta.self, from: data)
        else { return nil }
        return StoredForecast(url: url, meta: meta)
    }

    func save() throws {
        try Self.encoder.encode(meta).write(to: url.appending(path: Self.metaName), options: .atomic)
    }

    func stepURL(_ step: Int) -> URL {
        url.appending(path: String(format: "step_%03d.safetensors", step * meta.stepHours))
    }

    var staticURL: URL { url.appending(path: "static.safetensors") }

    func validDate(step: Int) -> Date {
        meta.initDate.addingTimeInterval(TimeInterval(step * meta.stepHours * 3600))
    }

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}

nonisolated extension Date {
    /// "Sat 19 Oct 00 UTC"
    var utcLabel: String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "UTC")
        f.dateFormat = "EEE d MMM HH 'UTC'"
        return f.string(from: self)
    }
}
