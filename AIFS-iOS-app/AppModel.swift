import AIFSKit
import AuroraANE
import Foundation
import GraphCastANE
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
        /// Every finished step's time, for the estimate of time left.
        var stepSeconds: [Double] = []
        #if DEVELOPER
        var availableMemoryGB: Double?
        var peakMemoryGB: Double?
        #endif
        /// Stop was pressed. A step can't be interrupted, so the run ends when this one does.
        var stopping = false
        /// The model's programs being compiled for this device before the first step (first
        /// run, or after an update): the fraction done, and when compiling began.
        var compiling: Double?
        var compileStart = Date()

        /// Time to the end of compiling, from its pace so far.
        var compileSecondsLeft: Double? {
            guard let f = compiling, f > 0.02 else { return nil }
            return Date().timeIntervalSince(compileStart) / f * (1 - f)
        }

        /// Time to the end of the run: the typical recent step times the steps still to do.
        /// The median, because a step that waited out a pause in the background takes as
        /// long as the pause; recent steps, because a warm device slows down as it goes.
        /// Nil until the first step finishes.
        var secondsLeft: Double? {
            guard !stepSeconds.isEmpty else { return nil }
            let recent = stepSeconds.suffix(5).sorted()
            let median = recent.count.isMultiple(of: 2)
                ? (recent[recent.count / 2 - 1] + recent[recent.count / 2]) / 2
                : recent[recent.count / 2]
            return median * Double(totalSteps) * (1 - progress)
        }
    }

    /// The model the initial conditions, downloads and runs are for; remembered.
    var selectedModel: ForecastModel = ForecastModel(rawValue: UserDefaults.standard.string(forKey: "model.selected") ?? "") ?? .aifs {
        didSet {
            guard selectedModel != oldValue else { return }
            UserDefaults.standard.set(selectedModel.rawValue, forKey: "model.selected")
            availableCycles = nil
            refresh()
        }
    }
    /// The selected model's state.
    var modelState: ModelState = .missing
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
    var downloadedICID: String?
    var thermalState = ProcessInfo.processInfo.thermalState
    /// Hot enough that iOS is, or soon will be, slowing the device down.
    var isHot: Bool { thermalState == .serious || thermalState == .critical }
    /// Bumped whenever a running forecast writes a step, so the viewer can refresh.
    var forecastRevision = 0

    private let engine = Engine()
    /// The model bundle downloading: bytes so far and in all.
    var modelDownload: (done: Int64, total: Int64)?
    private var modelTask: Task<Void, Never>?
    private var runTask: Task<Void, Never>?
    private var downloadTask: Task<Void, Never>?

    init() {
        #if DEVELOPER
        AIFSRuntime.log = { print($0) }
        #endif
        AppPaths.ensureDirectories()
        refresh()
        Task.detached(priority: .utility) { InitialConditionStore.pruneStale() }
        NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.thermalState = ProcessInfo.processInfo.thermalState }
        }
    }

    func refresh() {
        initialConditions = InitialConditionStore.list(selectedModel)
        forecasts = StoredForecast.list()
        switch selectedModel {
        case .aifs:
            if !AppPaths.bundleIsPresent {
                modelState = .missing
            } else if modelState == .missing {
                modelState = .ready
            }
        case .aurora, .graphcast:
            modelState = AppPaths.isInstalled(selectedModel) ? .ready : .missing
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
        let engine = engine, model = selectedModel
        Task {
            let cycles = (try? await Task.detached { try await engine.availableCycles(model: model) }.value) ?? []
            if model == selectedModel { availableCycles = cycles }
            findingCycles = false
        }
    }

    /// Download the latest cycle, or `date` when given (an earlier one).
    func downloadIC(date: Date? = nil) {
        guard downloadTask == nil else { return }
        download = Download()
        downloadedICID = nil  // so the same cycle twice still re-selects it
        errorMessage = nil
        let engine = engine, model = selectedModel
        let work = Task.detached(priority: .userInitiated) {
            try await engine.download(date: date, model: model) { progress in
                // AppModel lives as long as the app, so holding it here is harmless.
                Task { @MainActor in
                    self.download = Download(
                        completed: progress.completed, total: progress.total,
                        megabytes: progress.bytes / 1_000_000, message: progress.message)
                }
            }
        }
        downloadTask = Task {
            do {
                // Detached work doesn't inherit cancellation; pass Cancel on to it.
                let date = try await withTaskCancellationHandler {
                    try await work.value
                } onCancel: {
                    work.cancel()
                }
                download = nil
                refresh()
                // Select what was just fetched (or found already on disk).
                downloadedICID = initialConditions.first { $0.date == date }?.id
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

    // MARK: - Model bundle

    /// Fetch the model bundle from Hugging Face (resumes where a stopped one left off).
    func downloadModel() {
        guard modelTask == nil else { return }
        errorMessage = nil
        modelDownload = (0, 0)
        UIApplication.shared.isIdleTimerDisabled = true  // 1-1.5 GB; don't let the screen lock stall it
        let model = selectedModel
        let work = Task.detached(priority: .userInitiated) {
            try await ModelDownloader.download(model) { done, total in
                Task { @MainActor in
                    if self.modelDownload != nil { self.modelDownload = (done, total) }
                }
            }
        }
        modelTask = Task {
            do {
                try await withTaskCancellationHandler { try await work.value } onCancel: { work.cancel() }
            } catch is CancellationError {
            } catch where (error as? URLError)?.code == .cancelled {
            } catch {
                errorMessage = "Model download failed: \(error.localizedDescription)"
            }
            modelDownload = nil
            modelTask = nil
            allowScreenLock()
            refresh()
        }
    }

    func cancelModelDownload() { modelTask?.cancel() }

    /// Whether models can be deleted now: not while one runs or anything downloads.
    var canDeleteModels: Bool { run == nil && download == nil && modelDownload == nil }

    /// Deletes `models`' bundles, and the initial conditions only they could use. Forecasts
    /// stay, and stay viewable: AIFS keeps the two files its maps and places need (its
    /// display operator and grid, 32 MB); Aurora's and GraphCast's need nothing.
    func deleteModels(_ models: Set<ForecastModel>) {
        guard canDeleteModels else { return }
        if models.contains(.aifs) { engine.unloadModel() }
        // Named while their bundles still say which messages they read.
        let initialConditions = models.flatMap { model in
            InitialConditionStore.list(model).map { ($0, InitialConditionStore.keys(model, date: $0.date) ?? []) }
        }
        for model in models {
            let bundle = AppPaths.bundle(for: model)
            let keep: Set<String> = model == .aifs ? ["display.safetensors", "grid.safetensors"] : []
            let files = (try? FileManager.default.contentsOfDirectory(at: bundle, includingPropertiesForKeys: nil)) ?? []
            for url in files where !keep.contains(url.lastPathComponent) { try? FileManager.default.removeItem(at: url) }
            if keep.isEmpty { try? FileManager.default.removeItem(at: bundle) }
            try? FileManager.default.removeItem(at: ModelDownloader.staging(for: model))
        }
        InitialConditionStore.forget(initialConditions)
        refresh()
    }

    func delete(_ ic: ICFile) {
        InitialConditionStore.delete(ic)
        refresh()
    }

    // MARK: - Forecasts

    func startForecast(from ic: ICFile, leadHours: Int) {
        guard runTask == nil, modelState == .ready || modelState == .loading else { return }
        errorMessage = nil
        // AIFS: a step peaks at ~3.1 GB of model memory plus the Core ML processor; the
        // weights themselves (~0.6 GB) are already counted once loaded. Aurora: ~2 GB at the
        // peak, its programs streamed onto the Neural Engine. GraphCast: ~3.1 GB, its
        // buffers and state. Refuse up front rather than have iOS end the app part-way
        // through a step.
        let model = ic.model
        let need = switch model {
        case .aurora: 2.5
        case .graphcast: 3.3
        case .aifs: engine.isModelLoaded ? 3.0 : 3.8
        }
        // On a Mac (the iPad app running there) there is no per-app memory cap and
        // os_proc_available_memory() means nothing; the machine's own memory is the limit.
        let free = ProcessInfo.processInfo.isiOSAppOnMac
            ? Double(ProcessInfo.processInfo.physicalMemory) / 1e9 / 2
            : Double(os_proc_available_memory()) / 1e9
        guard free >= need else {
            errorMessage = String(
                format: "Not enough memory for a forecast: %.1f GB is free for the app and a step needs about %.1f GB. "
                    + "Close other apps and try again, or use a device with more memory.", free, need)
            return
        }
        let steps = leadHours / 6
        run = Run(initDate: ic.date, totalSteps: steps)
        UIApplication.shared.isIdleTimerDisabled = true  // the GPU stops when the screen locks
        let engine = engine
        let work = Task.detached(priority: .userInitiated) {
            try await engine.forecast(ic: ic, steps: steps) { event in
                Task { @MainActor in self.handle(event) }
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
                #if DEVELOPER
                print("[AIFS] forecast failed: \(error)")
                #endif
            }
            run = nil
            runTask = nil
            allowScreenLock()
            refresh()
        }
    }

    func cancelForecast() {
        run?.stopping = true
        runTask?.cancel()
    }

    /// The stored forecast a run from `ic` would replace, if there is one.
    func existingForecast(for ic: ICFile) -> StoredForecast? {
        let dir = StoredForecast.directory(for: ic.date, model: ic.model).lastPathComponent
        return forecasts.first { $0.id.lastPathComponent == dir }
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
            r.compiling = nil
            r.phase = phase.prefix(1).uppercased() + phase.dropFirst()  // "decoder" -> "Decoder"
            r.progress = max(r.progress, (Double(step - 1) + fraction) / Double(r.totalSteps))
        case .compiling(let fraction, let started):
            r.compileStart = started
            r.compiling = fraction < 1 ? fraction : nil
        case .stepDone(let step, let seconds, let peak):
            #if DEVELOPER
            print(String(
                format: "[AIFS] step %d/%d: %.1f s, peak %.2f GB, %.2f GB available, thermal %d",
                step, r.totalSteps, seconds, Double(peak) / 1e9,
                Double(os_proc_available_memory()) / 1e9, ProcessInfo.processInfo.thermalState.rawValue))
            r.peakMemoryGB = Double(peak) / 1e9
            #endif
            r.step = step
            r.progress = max(r.progress, Double(step) / Double(r.totalSteps))
            r.lastStepSeconds = seconds
            r.stepSeconds.append(seconds)
            refresh()
            forecastRevision += 1
        }
        #if DEVELOPER
        // Meaningless for the iPad app on a Mac (see startForecast): don't show it there.
        r.availableMemoryGB = ProcessInfo.processInfo.isiOSAppOnMac ? nil : Double(os_proc_available_memory()) / 1e9
        #endif
        run = r
    }

    /// After a download or a run, the screen may lock again -- except in developer builds,
    /// which keep the device awake while the app is open (see DeveloperTools).
    private func allowScreenLock() {
        #if !DEVELOPER
        UIApplication.shared.isIdleTimerDisabled = false
        #endif
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
        /// The fraction of the model's programs compiled for the Neural Engine, since `started`.
        case compiling(fraction: Double, started: Date)
    }

    private let lock = NSLock()
    private var model: AIFSModel?

    var isModelLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return model != nil
    }

    /// Lets the loaded AIFS model go (its bundle is being deleted).
    func unloadModel() {
        lock.lock()
        model = nil
        lock.unlock()
    }

    /// The model stays resident between runs (~540 MB); loading it takes a few seconds, or
    /// minutes on the first run, when its Neural Engine models compile (see `compiling`).
    func loadedModel(compiling: (Double) throws -> Void) throws -> AIFSModel {
        lock.lock()
        defer { lock.unlock() }
        if let model { return model }
        // Cached buffers count against the app's memory limit like live ones; keep the
        // cache small so the step's peak, not leftovers, is what the limit sees.
        AIFSRuntime.setCacheLimit(256 << 20)
        // fp16, as the checkpoint autocasts; processor on the Neural Engine when the bundle
        // has its Core ML models (`--no-ane` at launch forces the GPU path, for comparison).
        let m = try AIFSModel(
            bundle: AppPaths.bundle, useNeuralEngine: !LaunchOptions.noNeuralEngine, compiling: compiling)
        #if DEVELOPER
        print("[AIFS] model loaded, processor on \(m.usesNeuralEngine ? "Neural Engine" : "GPU")")
        #endif
        model = m
        return m
    }

    private func openDataClient() throws -> OpenDataClient {
        let manifest = try Manifest.load(from: AppPaths.bundle.appending(path: "manifest.json"))
        return try OpenDataClient(bundle: AppPaths.bundle, manifest: manifest)
    }

    /// Cycles the mirrors can still serve for `model`, newest first.
    func availableCycles(model: ForecastModel) async throws -> [Date] {
        switch model {
        case .aifs:
            let client = try openDataClient()
            return await client.availableDates()
        case .aurora:
            let client = try AuroraOpenData(bundle: AppPaths.auroraBundle)
            return await client.availableDates()
        case .graphcast:
            let client = try GraphCastOpenData(bundle: AppPaths.graphcastBundle)
            return await client.availableDates()
        }
    }

    /// Downloads into the shared open-data cache whatever `model`'s initial condition for
    /// `date` (or the latest cycle) lacks -- often little, other models' having fetched most
    /// of it -- and returns its date. Stopped, it discards what it fetched that nothing
    /// complete needs.
    func download(date: Date?, model: ForecastModel, progress: @escaping @Sendable (OpenDataClient.Progress) -> Void) async throws -> Date {
        let date: Date = if let date { date } else {
            switch model {
            case .aifs: try await openDataClient().latestDate()
            case .aurora: try await AuroraOpenData(bundle: AppPaths.auroraBundle).latestDate()
            case .graphcast: try await GraphCastOpenData(bundle: AppPaths.graphcastBundle).latestDate()
            }
        }
        guard let keys = InitialConditionStore.keys(model, date: date) else { throw AIFSError.invalidBundle("\(model.title) is not installed") }
        #if DEVELOPER
        let cached = keys.filter(InitialConditionStore.cache.contains).count
        print("[AIFS] \(model.shortName) \(StoredForecast.stamp(date)): \(keys.count) messages, \(cached) already cached")
        #endif
        do {
            try await InitialConditionStore.cache.fetch(keys, progress: progress)
        } catch {
            if Task.isCancelled || error is CancellationError || (error as? URLError)?.code == .cancelled {
                InitialConditionStore.discardIncomplete(model, date: date)
            }
            throw error
        }
        return date
    }

    func forecast(ic source: ICFile, steps: Int, report: @escaping @Sendable (Event) -> Void) async throws {
        if source.model == .aurora { return try await auroraForecast(ic: source, steps: steps, report: report) }
        if source.model == .graphcast { return try await graphcastForecast(ic: source, steps: steps, report: report) }
        report(.progress(step: 1, phase: "Loading model", fraction: 0))
        let model = try loadedModel(compiling: Self.compileProgress(report))
        // Development tuning (Developer/LaunchOptions): speed and memory only.
        if let v = LaunchOptions.attentionBlock { model.attentionBlock = v }
        if let v = LaunchOptions.encoderChunks { model.encoderChunks = v }
        if let v = LaunchOptions.decoderChunks { model.decoderChunks = v }
        // What the viewers draw, plus snowfall for the dashboard's conditions.
        let stored = Array(Set(DisplayField.storedOutputs + ["sf"])).sorted()

        // The initial condition's fields (~420 MB) are only needed to build the first input
        // window; scoping them here frees them before the rollout starts.
        let (startedForecast, initDate, statics): (Forecast, Date, [String: [Float]]) = try {
            // From the shared open-data cache (decoded and regridded now), or the file.
            let ic = if let file = source.file {
                try InitialCondition.load(from: file)
            } else {
                try openDataClient().build(date: source.date, from: InitialConditionStore.cache)
            }
            // Land-sea mask (coastlines) and surface geopotential (globe relief).
            var statics: [String: [Float]] = [:]
            for name in ["lsm", "z"] { statics[name] = ic.latest(name) }
            return (try Forecast(model: model, initialCondition: ic), ic.date, statics)
        }()
        // Stopped while loading: leave any forecast already stored for this date alone.
        try Task.checkCancellation()
        let dir = StoredForecast.directory(for: initDate, model: .aifs)
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var record = StoredForecast(
            url: dir,
            meta: ForecastMeta(
                initDate: initDate, stepHours: model.manifest.timestepHours, plannedSteps: steps,
                completedSteps: 0, fields: stored, secondsPerStep: [], peakMemoryBytes: 0,
                model: model.manifest.description, finished: false, kind: .aifs))
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
            RunGate.shared.pass()
            try Task.checkCancellation()
            model.onProgress = { phase, fraction in
                // Between GPU dispatches: the one place a backgrounded run can safely wait.
                RunGate.shared.pass()
                report(.progress(step: s, phase: phase, fraction: fraction))
            }
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
            #if DEVELOPER
            print("[AIFS] phases: " + model.lastTimings.map { String(format: "%@ %.2fs", $0.phase, $0.seconds) }
                .joined(separator: ", ") + " (block \(model.attentionBlock), enc \(model.encoderChunks), dec \(model.decoderChunks))")
            #endif
            report(.stepDone(step: s, seconds: seconds, peakBytes: AIFSRuntime.peakMemory))
        }
    }

    /// Turns a model's load progress (0 before its first program, 1 after its last) into
    /// `.compiling` events, once loading turns out to be compiling: a program from the
    /// system's cache loads in milliseconds, one compiled for this device takes seconds. So
    /// a run whose programs are all cached never shows it.
    private static func compileProgress(_ report: @escaping @Sendable (Event) -> Void) -> (Double) throws -> Void {
        var start = Date(), last = start, compiling = false
        return { fraction in
            try Task.checkCancellation()
            let now = Date()
            // Not from its creation: AIFS reads its weights before it starts on its programs.
            if fraction == 0 { start = now; last = now }
            compiling = compiling || now.timeIntervalSince(last) > 1
            last = now
            #if DEVELOPER
            print(String(format: "[AIFS] compiled %.3f at %.1f s", fraction, now.timeIntervalSince(start)))
            #endif
            if compiling { report(.compiling(fraction: fraction, started: start)) }
        }
    }

    /// Aurora 1.5 from a state file: its programs streamed onto the Neural Engine, each step's
    /// fields written as AIFS's are, on Aurora's 0.25-degree grid (720 x 1440).
    private func auroraForecast(ic: ICFile, steps: Int, report: @escaping @Sendable (Event) -> Void) async throws {
        report(.progress(step: 1, phase: "Loading model", fraction: 0))
        let bundle = AppPaths.auroraBundle
        let state = if let file = ic.file {
            try AuroraState.load(file)
        } else {
            try AuroraOpenData(bundle: bundle).build(date: ic.date, from: InitialConditionStore.cache)
        }
        // Everything on the Neural Engine, its programs streamed (~1.6 GB at the peak).
        let forecaster = try AuroraForecaster(directory: bundle, state: state, options: .init(resident: false))
        let stored = Array(Set(DisplayField.storedOutputs + ["sf"])).sorted()
        // Compiled for the Neural Engine on the first run (minutes), from the cache after.
        try forecaster.runtime.prepare(progress: Self.compileProgress(report))
        try Task.checkCancellation()

        let dir = StoredForecast.directory(for: state.time, model: .aurora)
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var record = StoredForecast(
            url: dir,
            meta: ForecastMeta(
                initDate: state.time, stepHours: forecaster.metadata.step_hours, plannedSteps: steps,
                completedSteps: 0, fields: stored, secondsPerStep: [], peakMemoryBytes: 0,
                model: forecaster.metadata.model, finished: false, kind: .aurora))
        // Land-sea mask (coastlines) and surface geopotential (globe relief).
        let statics = try SafeTensorsFile(bundle.appending(path: "static.safetensors"))
        let n = forecaster.metadata.grid.rows * forecaster.metadata.grid.cols
        // Written without MLX: an Aurora run never starts the GPU.
        try SafeTensorsWriter.write(record.staticURL, fields: Dictionary(uniqueKeysWithValues: ["lsm", "z"].map { name in
            (name, Array(UnsafeBufferPointer(start: try! statics.pointer(name, as: Float.self), count: n)))
        }))
        try record.save()
        report(.started(dir))

        for s in 1 ... steps {
            RunGate.shared.pass()
            try Task.checkCancellation()
            forecaster.runtime.onProgress = { phase, fraction in
                // Between Neural Engine calls: where a backgrounded run can safely wait.
                RunGate.shared.pass()
                let overall = switch phase {
                case "encoder": 0.02
                case "processor": 0.08 + 0.85 * fraction
                default: 0.95
                }
                report(.progress(step: s, phase: phase, fraction: overall))
            }
            let t = ContinuousClock.now
            let (date, fields) = try forecaster.advance(wanted: stored)
            let elapsed = ContinuousClock.now - t
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            try SafeTensorsWriter.write(record.stepURL(s), metadata: ["valid": date.ISO8601Format()], fields: fields)
            let peak = ProcessMemory.footprint().peak
            record.meta.completedSteps = s
            record.meta.secondsPerStep.append(seconds)
            record.meta.peakMemoryBytes = max(record.meta.peakMemoryBytes, peak)
            record.meta.finished = s == steps
            try record.save()
            report(.stepDone(step: s, seconds: seconds, peakBytes: peak))
        }
        forecaster.runtime.onProgress = nil
    }

    /// GraphCast (operational) from a state file: its programs on the Neural Engine, the CPU
    /// moving data along the graphs between them; each step's fields written as Aurora's are,
    /// on the 0.25-degree grid with both poles (721 x 1440).
    private func graphcastForecast(ic: ICFile, steps: Int, report: @escaping @Sendable (Event) -> Void) async throws {
        report(.progress(step: 1, phase: "Loading model", fraction: 0))
        let bundle = AppPaths.graphcastBundle
        let state = if let file = ic.file {
            try GraphCastState.load(file)
        } else {
            try GraphCastOpenData(bundle: bundle).build(date: ic.date, from: InitialConditionStore.cache)
        }
        let forecaster = try GraphCastForecaster(directory: bundle, state: state)
        // Only what it predicts: the viewer offers every field the forecast lists.
        let stored = Array(Set(DisplayField.storedOutputs + ["sf"])).sorted().filter(GraphCastForecaster.predicts)
        // Compiled for the Neural Engine on the first run, from the cache after.
        try forecaster.runtime.prepare(progress: Self.compileProgress(report))
        try Task.checkCancellation()

        let dir = StoredForecast.directory(for: state.time, model: .graphcast)
        try? FileManager.default.removeItem(at: dir)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var record = StoredForecast(
            url: dir,
            meta: ForecastMeta(
                initDate: state.time, stepHours: forecaster.stepHours, plannedSteps: steps,
                completedSteps: 0, fields: stored, secondsPerStep: [], peakMemoryBytes: 0,
                model: forecaster.metadata.model, finished: false, kind: .graphcast))
        // Land-sea mask (coastlines) and surface geopotential (globe relief), north first.
        let statics = try SafeTensorsFile(bundle.appending(path: "static.safetensors"))
        try SafeTensorsWriter.write(record.staticURL, fields: [
            "lsm": GraphCastForecaster.flip(try statics.pointer("land_sea_mask", as: Float.self)),
            "z": GraphCastForecaster.flip(try statics.pointer("geopotential_at_surface", as: Float.self)),
        ])
        try record.save()
        report(.started(dir))

        for s in 1 ... steps {
            RunGate.shared.pass()
            try Task.checkCancellation()
            forecaster.runtime.onProgress = { phase, fraction in
                // Between Neural Engine calls: where a backgrounded run can safely wait.
                RunGate.shared.pass()
                // The phases' shares of a step's time.
                let overall = switch phase {
                case "encoder": 0.16 * fraction
                case "processor": 0.16 + 0.46 * fraction
                default: 0.62 + 0.38 * fraction
                }
                report(.progress(step: s, phase: phase, fraction: overall))
            }
            let t = ContinuousClock.now
            let (date, fields) = try forecaster.advance(wanted: stored)
            let elapsed = ContinuousClock.now - t
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            try SafeTensorsWriter.write(record.stepURL(s), metadata: ["valid": date.ISO8601Format()], fields: fields)
            let peak = ProcessMemory.footprint().peak
            record.meta.completedSteps = s
            record.meta.secondsPerStep.append(seconds)
            record.meta.peakMemoryBytes = max(record.meta.peakMemoryBytes, peak)
            record.meta.finished = s == steps
            try record.save()
            #if DEVELOPER
            print("[AIFS] graphcast " + forecaster.timings.sorted { $0.value > $1.value }
                .map { String(format: "%@ %.2f", $0.key, $0.value) }.joined(separator: ", ") + "; "
                + forecaster.runtime.timings.sorted { $0.value > $1.value }
                .map { String(format: "%@ %.2f", $0.key, $0.value) }.joined(separator: ", "))
            #endif
            report(.stepDone(step: s, seconds: seconds, peakBytes: peak))
        }
        forecaster.runtime.onProgress = nil
    }
}
