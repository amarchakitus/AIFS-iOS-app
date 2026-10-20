import Foundation

#if DEVELOPER
/// Launch arguments, for development only: none of them is reachable from the UI, and none
/// changes a forecast's result. Pass them from Xcode's scheme, `xcrun simctl launch`, or
/// `xcrun devicectl device process launch`, to a developer build (AIFS_DEVELOPER = YES).
nonisolated enum LaunchOptions {
    // MARK: Opening a screen directly (screenshots, the simulator)

    /// `--open-latest`: select the newest forecast.
    static var openLatest: Bool { flag("--open-latest") }
    /// `--viewer-mode map|globe|places`: the forecast viewer's starting tab.
    static var viewerMode: String? { value("--viewer-mode") }
    /// `--open-settings`: the globe settings sheet.
    static var openGlobeSettings: Bool { flag("--open-settings") }
    /// `--open-place`: the first saved place's detail.
    static var openPlace: Bool { flag("--open-place") }
    /// `--open-app-settings`: the app's Settings sheet; with `--open-attribution`, its
    /// licences page.
    static var openAppSettings: Bool { flag("--open-app-settings") }
    static var openAttribution: Bool { flag("--open-attribution") }
    /// With `--open-app-settings`, `--open-models`: its Models page.
    static var openModels: Bool { flag("--open-models") }
    /// With `--open-models`, `--edit-models`: in edit mode, the first model selected.
    static var editModels: Bool { flag("--edit-models") }
    /// `--confirm`: also ask the open page's confirmation (Settings' clear, or with
    /// `--edit-models` the Models page's delete), for screenshots.
    static var confirm: Bool { flag("--confirm") }
    /// `--delete-models aifs,aurora`: delete these models as Settings' Models page does.
    static var deleteModels: [String] { (value("--delete-models") ?? "").split(separator: ",").map(String.init) }
    /// `--open-earlier-cycles`: the earlier-cycles sheet.
    static var openEarlierCycles: Bool { flag("--open-earlier-cycles") }
    /// `--download-model`: start the model download if none is installed (first launch).
    static var downloadModel: Bool { flag("--download-model") }
    /// `--remove-documents a,b/c`: delete these items in Documents first (test leftovers).
    static var removeDocuments: [String] { (value("--remove-documents") ?? "").split(separator: ",").map(String.init) }
    /// `--select-model aifs|aurora`: the model to start with.
    static var selectModel: String? { value("--select-model") }
    /// `--delete-newest-ic`: delete the selected model's newest initial condition (tests the
    /// shared cache's bookkeeping).
    static var deleteNewestIC: Bool { flag("--delete-newest-ic") }
    /// `--clear-ics day|week|all`: what Settings' "Clear initial conditions" does.
    static var clearICs: String? { value("--clear-ics") }
    /// `--download-latest`: fetch the newest cycle for the selected model, then run
    /// `--autorun-steps` steps from it, if given.
    static var downloadLatest: Bool { flag("--download-latest") }

    // MARK: Measuring the model on a device

    /// `--autorun-steps N`: run N steps from the newest initial condition on launch.
    static var autorunSteps: Int? { int("--autorun-steps") }
    /// `--attn-block N`, `--enc-chunks N`, `--dec-chunks N`: speed/memory tuning.
    static var attentionBlock: Int? { int("--attn-block") }
    static var encoderChunks: Int? { int("--enc-chunks") }
    static var decoderChunks: Int? { int("--dec-chunks") }
    /// `--no-ane`: keep the processor on the GPU, for comparison with the Neural Engine.
    static var noNeuralEngine: Bool { flag("--no-ane") }
    /// `--bench`: time the banded attention kernel and exit to the log.
    static var bench: Bool { flag("--bench") }
    /// `--ane-probe`: time every Core ML model copied into Documents/coreml.
    static var neuralEngineProbe: Bool { flag("--ane-probe") }
    /// `--aurora-bench`: run Aurora 1.5's Neural Engine test case in Documents/aurora (from
    /// aurora-1p5's `ane_export` and `ane_run --write-test`) and log time, memory and
    /// accuracy; with `--aurora-stream`, loading each program only while it runs.
    static var auroraBench: Bool { flag("--aurora-bench") }
    static var auroraStream: Bool { flag("--aurora-stream") }
    /// `--aurora-gpu 1,2`: stages whose blocks the GPU runs too, beside the Neural Engine.
    static var auroraGPUStages: Set<Int> { Set((value("--aurora-gpu") ?? "").split(separator: ",").compactMap { Int($0) }) }
    /// `--aurora-dir NAME`: the bundle's folder in Documents (default `aurora`).
    static var auroraDirectory: String { value("--aurora-dir") ?? "aurora" }
    /// `--graphcast-bench`: run GraphCast's Neural Engine test case in Documents/graphcast (from
    /// graphcast-ane's `ane_export` and `bundle --write-test`) and log time, memory and
    /// accuracy; `--graphcast-dir NAME` for another folder.
    static var graphcastBench: Bool { flag("--graphcast-bench") }
    static var graphcastDirectory: String { value("--graphcast-dir") ?? "graphcast" }
    /// `--graphcast-probe N`: compile each program in Documents/probe-grid N times (see
    /// GraphCastProbe).
    static var graphcastProbe: Int? { value("--graphcast-probe").flatMap { Int($0) } }

    private static var args: [String] { ProcessInfo.processInfo.arguments }
    private static func flag(_ name: String) -> Bool { args.contains(name) }
    private static func value(_ name: String) -> String? {
        args.firstIndex(of: name).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }
    }
    private static func int(_ name: String) -> Int? { value(name).flatMap(Int.init) }
}
#else
/// Not a developer build: every option off, so the code behind them compiles away and no
/// launch argument does anything.
nonisolated enum LaunchOptions {
    static let openLatest = false
    static let viewerMode: String? = nil
    static let openGlobeSettings = false
    static let openPlace = false
    static let openAppSettings = false
    static let openAttribution = false
    static let openModels = false
    static let editModels = false
    static let confirm = false
    static let openEarlierCycles = false
    static let downloadModel = false
    static let attentionBlock: Int? = nil
    static let encoderChunks: Int? = nil
    static let decoderChunks: Int? = nil
    static let noNeuralEngine = false
    static let neuralEngineProbe = false
}
#endif
