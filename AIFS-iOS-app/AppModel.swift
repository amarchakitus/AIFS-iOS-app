import AIFSKit
import Foundation
import Observation
import UIKit
import os

/// App state. Lives on the main actor; the heavy work (model load, open-data download,
/// forecast steps) runs on detached tasks and reports back.
@Observable
final class AppModel {
    enum ModelState: Equatable {
        case missing, loading, ready, failed(String)
    }

    struct Download: Equatable {
        var completed = 0
        var total = 0
        var megabytes = 0
        var message = "Finding the latest cycle…"
    }

    struct Run: Equatable {
        var initDate: Date
        var step = 0
        var totalSteps: Int
        /// Overall fraction of the run done, for the progress bar. Only ever increases.
        var progress = 0.0
        var phase = "Loading model"
        var lastStepSeconds: Double?
        var availableMemoryGB: Double?
        var peakMemoryGB: Double?
        /// Stop was pressed. A step can't be interrupted, so the run ends when this one does.
        var stopping = false
    }

    var modelState: ModelState = .missing
    var manifestDescription = ""
    var initialConditions: [ICFile] = []
    var forecasts: [StoredForecast] = []
    var download: Download?
    var run: Run?
    var errorMessage: String?
    var selectedForecastID: URL?
    /// Set when a run creates its forecast. The view decides whether to open it (it does on
    /// iPad, where the list stays visible beside it; on iPhone it would hide the progress).
    var startedForecastID: URL?
    /// Set when "Download latest" finishes; the view selects it as the run's input.
    var downloadedICID: URL?
    var thermalState = ProcessInfo.processInfo.thermalState
    /// Bumped whenever a running forecast writes a step, so the viewer can refresh.
    var forecastRevision = 0

    private let engine = Engine()
    private var runTask: Task<Void, Never>?
    private var downloadTask: Task<Void, Never>?

    init() {
        AppPaths.ensureDirectories()
        refresh()
        NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.thermalState = ProcessInfo.processInfo.thermalState }
        }
    }

    func refresh() {
        initialConditions = ICFile.list()
        forecasts = StoredForecast.list()
        if !AppPaths.bundleIsPresent {
            modelState = .missing
        } else if modelState == .missing {
            modelState = .ready
            // "AIFS Single v2 (aifs-single-mse-2.0)" -> "AIFS Single v2": the checkpoint name
            // stays in the bundle's manifest, it is just not shown.
            let description = (try? Manifest.load(from: AppPaths.bundle.appending(path: "manifest.json")))?
                .description ?? ""
            manifestDescription = description.components(separatedBy: " (").first ?? description
        }
    }

    var selectedForecast: StoredForecast? {
        forecasts.first { $0.id == selectedForecastID }
    }

    // MARK: - Initial conditions

    /// Cycles available to download (nil until looked up), and whether a lookup is running.
    var availableCycles: [Date]?
    var findingCycles = false

    func findAvailableCycles() {
        guard !findingCycles else { return }
        findingCycles = true
        let engine = engine
        Task {
            availableCycles = (try? await Task.detached { try await engine.availableCycles() }.value) ?? []
            findingCycles = false
        }
    }

    /// Download the latest cycle, or `date` when given (an earlier one).
    func downloadIC(date: Date? = nil) {
        guard downloadTask == nil else { return }
        download = Download()
        downloadedICID = nil  // so the same cycle twice still re-selects it
        errorMessage = nil
        let engine = engine
        let work = Task.detached(priority: .userInitiated) {
            try await engine.download(date: date) { progress in
                Task { @MainActor [weak self] in
                    self?.download = Download(
                        completed: progress.completed, total: progress.total,
                        megabytes: progress.bytes / 1_000_000, message: progress.message)
                }
            }
        }
        downloadTask = Task {
            do {
                // Detached work doesn't inherit cancellation; pass Cancel on to it.
                let url = try await withTaskCancellationHandler {
                    try await work.value
                } onCancel: {
                    work.cancel()
                }
                download = nil
                refresh()
                // Select what was just fetched (or found already on disk). Match by file name:
                // the listing's URL for the same file need not compare equal.
                downloadedICID = initialConditions.first {
                    $0.url.lastPathComponent == url.lastPathComponent
                }?.id
            } catch is CancellationError {
                errorMessage = nil
            } catch where (error as? URLError)?.code == .cancelled {
                errorMessage = nil
            } catch {
                errorMessage = "Download failed: \(error.localizedDescription)"
            }
            download = nil
            downloadTask = nil
            refresh()
        }
    }

    func cancelDownload() { downloadTask?.cancel() }

    func delete(_ ic: ICFile) {
        try? FileManager.default.removeItem(at: ic.url)
        refresh()
    }

    // MARK: - Forecasts

    func startForecast(from ic: ICFile, leadHours: Int) {
        guard runTask == nil, modelState == .ready || modelState == .loading else { return }
        errorMessage = nil
        let steps = leadHours / 6
        run = Run(initDate: ic.date, totalSteps: steps)
        UIApplication.shared.isIdleTimerDisabled = true  // the GPU stops when the screen locks
        let engine = engine
        let work = Task.detached(priority: .userInitiated) {
            try await engine.forecast(icURL: ic.url, steps: steps) { event in
                Task { @MainActor [weak self] in self?.handle(event) }
            }
        }
        runTask = Task {
            do {
                // A detached task does not inherit cancellation: pass Stop on explicitly,
                // or the engine never sees it.
                try await withTaskCancellationHandler {
                    try await work.value
                } onCancel: {
                    work.cancel()
                }
            } catch is CancellationError {
                errorMessage = nil
            } catch {
                errorMessage = "Forecast failed: \(error.localizedDescription)"
                print("[AIFS] forecast failed: \(error)")
            }
            run = nil
            runTask = nil
            UIApplication.shared.isIdleTimerDisabled = false
            refresh()
        }
    }

    func cancelForecast() {
        run?.stopping = true
        runTask?.cancel()
    }

    /// The stored forecast a run from `ic` would replace, if there is one.
    func existingForecast(for ic: ICFile) -> StoredForecast? {
        let dir = StoredForecast.directory(for: ic.date).lastPathComponent
        return forecasts.first { $0.id.lastPathComponent == dir }
    }

    /// `--autorun-steps N` on launch: run N steps from the newest initial condition. For
    /// measuring the device from a Mac (`xcrun devicectl device process launch --console`).
    func autorunIfRequested() {
        let args = ProcessInfo.processInfo.arguments
        if args.contains("--ane-probe") {
            // Every compiled model in Documents/coreml, each a (1, 1024, 1, S) MLP half
            // (2 matmuls of 1024x4096 per token).
            Task.detached(priority: .userInitiated) {
                let dir = AppPaths.documents.appending(path: "coreml")
                let models = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
                    .filter { $0.pathExtension == "mlmodelc" }.sorted { $0.path < $1.path }
                print("[AIFS] ane-probe: \(models.count) models, thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
                for url in models {
                    // name_<S>: S tokens per call. mlp_*: an MLP half; dec_*: the decoder's
                    // dense second half (projection, MLP, output head).
                    let name = url.deletingPathExtension().lastPathComponent
                    let s = Double(name.split(separator: "_").last.flatMap { Int($0) } ?? 0)
                    let mlpMACs: Double = 2 * 1024 * 4096
                    let decMACs: Double = mlpMACs + 1024 * 1024 + 1024 * 120
                    let macs = name.hasPrefix("dec") ? decMACs : mlpMACs
                    for line in await NeuralEngineProbe.report(modelURL: url, flopsPerCall: 2 * s * macs) {
                        print("[AIFS] ane " + line)
                    }
                }
                print("[AIFS] ane-probe done, thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
            }
            return
        }
        if args.contains("--bench") {
            Task.detached(priority: .userInitiated) {
                print("[AIFS] bench start, thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
                for line in AIFSBenchmark.report() { print("[AIFS] bench " + line) }
                print("[AIFS] bench done, thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
            }
            return
        }
        guard let i = args.firstIndex(of: "--autorun-steps"), i + 1 < args.count,
            let steps = Int(args[i + 1]), let ic = initialConditions.first
        else { return }
        print("[AIFS] autorun: \(steps) steps from \(ic.date.utcLabel), \(Double(os_proc_available_memory()) / 1e9) GB available")
        startForecast(from: ic, leadHours: steps * 6)
    }

    private func handle(_ event: Engine.Event) {
        guard var r = run else { return }
        switch event {
        case .started(let forecastURL):
            refresh()
            // Match by directory name: the URL the engine built and the one the directory
            // listing returns name the same folder but need not compare equal (trailing
            // slash, /private prefix).
            startedForecastID = forecasts.first {
                $0.url.lastPathComponent == forecastURL.lastPathComponent
            }?.id
        case .progress(let step, let phase, let fraction):
            // Updates arrive as separate main-actor tasks and can trail the step's completion;
            // one from a finished step would push the bar a whole step ahead. Ignore those,
            // and never move the bar backwards.
            guard step > r.step else { return }
            r.phase = phase.prefix(1).uppercased() + phase.dropFirst()  // "decoder" -> "Decoder"
            r.progress = max(r.progress, (Double(step - 1) + fraction) / Double(r.totalSteps))
        case .stepDone(let step, let seconds, let peak):
            print(String(
                format: "[AIFS] step %d/%d: %.1f s, peak %.2f GB, %.2f GB available, thermal %d",
                step, r.totalSteps, seconds, Double(peak) / 1e9,
                Double(os_proc_available_memory()) / 1e9, ProcessInfo.processInfo.thermalState.rawValue))
            r.step = step
            r.progress = max(r.progress, Double(step) / Double(r.totalSteps))
            r.lastStepSeconds = seconds
            r.peakMemoryGB = Double(peak) / 1e9
            refresh()
            forecastRevision += 1
        }
        r.availableMemoryGB = Double(os_proc_available_memory()) / 1e9
        run = r
    }

    func delete(_ forecast: StoredForecast) {
        if selectedForecastID == forecast.id { selectedForecastID = nil }
        try? FileManager.default.removeItem(at: forecast.url)
        refresh()
    }
}

/// The off-main-actor half: owns the loaded model and does the long-running work.
nonisolated final class Engine: @unchecked Sendable {
    enum Event: Sendable {
        case started(URL)
        /// Phase and fraction done of `step` (1-based), the one being computed.
        case progress(step: Int, phase: String, fraction: Double)
        case stepDone(step: Int, seconds: Double, peakBytes: Int)
    }

    private let lock = NSLock()
    private var model: AIFSModel?

    /// The model stays resident between runs (~540 MB); loading it takes a few seconds.
    func loadedModel() throws -> AIFSModel {
        lock.lock()
        defer { lock.unlock() }
        if let model { return model }
        // Cached buffers count against the app's memory limit like live ones; keep the
        // cache small so the step's peak, not leftovers, is what the limit sees.
        AIFSRuntime.setCacheLimit(256 << 20)
        // fp16, as the checkpoint autocasts; processor on the Neural Engine when the bundle
        // has its Core ML models (`--no-ane` at launch forces the GPU path, for comparison).
        let m = try AIFSModel(
            bundle: AppPaths.bundle, useNeuralEngine: !ProcessInfo.processInfo.arguments.contains("--no-ane"))
        print("[AIFS] model loaded, processor on \(m.usesNeuralEngine ? "Neural Engine" : "GPU")")
        model = m
        return m
    }

    private func openDataClient() throws -> OpenDataClient {
        let manifest = try Manifest.load(from: AppPaths.bundle.appending(path: "manifest.json"))
        return try OpenDataClient(bundle: AppPaths.bundle, manifest: manifest)
    }

    /// Cycles the mirrors can still serve, newest first.
    func availableCycles() async throws -> [Date] {
        let client = try openDataClient()
        return await client.availableDates()
    }

    /// Download and assemble the initial condition for `date`, or the latest cycle.
    func download(date: Date?, progress: @escaping @Sendable (OpenDataClient.Progress) -> Void) async throws -> URL {
        let client = try openDataClient()
        let latest: Date
        if let date { latest = date } else { latest = try await client.latestDate() }
        let date = latest
        let url = AppPaths.initialConditions.appending(path: InitialCondition.fileName(for: date))
        if FileManager.default.fileExists(atPath: url.path) { return url }
        let ic = try await client.build(date: date, progress: progress)
        try Task.checkCancellation()
        try ic.save(to: url)
        return url
    }

    func forecast(icURL: URL, steps: Int, report: @escaping @Sendable (Event) -> Void) async throws {
        report(.progress(step: 1, phase: "Loading model", fraction: 0))
        let model = try loadedModel()
        // Debug tuning from launch arguments (see AppModel.autorunIfRequested); none of
        // these changes the result, only speed and memory.
        let args = ProcessInfo.processInfo.arguments
        func arg(_ name: String) -> Int? {
            args.firstIndex(of: name).flatMap { $0 + 1 < args.count ? Int(args[$0 + 1]) : nil }
        }
        if let v = arg("--attn-block") { model.attentionBlock = v }
        if let v = arg("--enc-chunks") { model.encoderChunks = v }
        if let v = arg("--dec-chunks") { model.decoderChunks = v }
        // What the viewers draw, plus snowfall for the dashboard's conditions.
        let stored = Array(Set(DisplayField.storedOutputs + ["sf"])).sorted()

        // The initial condition's fields (~420 MB) are only needed to build the first input
        // window; scoping them here frees them before the rollout starts.
        let (startedForecast, initDate, statics): (Forecast, Date, [String: [Float]]) = try {
            let ic = try InitialCondition.load(from: icURL)
            // Land-sea mask (coastlines) and surface geopotential (globe relief).
            var statics: [String: [Float]] = [:]
            for name in ["lsm", "z"] { statics[name] = ic.latest(name) }
            return (try Forecast(model: model, initialCondition: ic), ic.date, statics)
        }()
        // Stopped while loading: leave any forecast already stored for this date alone.
        try Task.checkCancellation()
        let dir = StoredForecast.directory(for: initDate)
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var record = StoredForecast(
            url: dir,
            meta: ForecastMeta(
                initDate: initDate, stepHours: model.manifest.timestepHours, plannedSteps: steps,
                completedSteps: 0, fields: stored, secondsPerStep: [], peakMemoryBytes: 0,
                model: model.manifest.description, finished: false))
        if !statics.isEmpty {
            try FieldFile.write(statics, to: record.staticURL)
        }
        try record.save()
        report(.started(dir))

        var forecast: Forecast? = startedForecast
        defer {
            forecast = nil
            model.onProgress = nil
            AIFSRuntime.clearCache()
        }
        for s in 1 ... steps {
            try Task.checkCancellation()
            model.onProgress = { phase, fraction in report(.progress(step: s, phase: phase, fraction: fraction)) }
            AIFSRuntime.resetPeakMemory()
            let t = ContinuousClock.now
            let output = forecast!.advanceStep()
            let elapsed = ContinuousClock.now - t
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18

            try FieldFile.write(
                output.fields(stored), metadata: ["valid": output.date.ISO8601Format()], to: record.stepURL(s))
            record.meta.completedSteps = s
            record.meta.secondsPerStep.append(seconds)
            record.meta.peakMemoryBytes = max(record.meta.peakMemoryBytes, AIFSRuntime.peakMemory)
            record.meta.finished = s == steps
            try record.save()
            print("[AIFS] phases: " + model.lastTimings.map { String(format: "%@ %.2fs", $0.phase, $0.seconds) }
                .joined(separator: ", ") + " (block \(model.attentionBlock), enc \(model.encoderChunks), dec \(model.decoderChunks))")
            report(.stepDone(step: s, seconds: seconds, peakBytes: AIFSRuntime.peakMemory))
        }
    }
}
