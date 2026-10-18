import Foundation

/// Launch arguments, for development only: none of them is reachable from the UI, and none
/// changes a forecast's result. Pass them from Xcode's scheme, `xcrun simctl launch`, or
/// `xcrun devicectl device process launch`.
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
    /// `--open-earlier-cycles`: the earlier-cycles sheet.
    static var openEarlierCycles: Bool { flag("--open-earlier-cycles") }
    /// `--download-model`: start the model download if none is installed (first launch).
    static var downloadModel: Bool { flag("--download-model") }

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

    private static var args: [String] { ProcessInfo.processInfo.arguments }
    private static func flag(_ name: String) -> Bool { args.contains(name) }
    private static func value(_ name: String) -> String? {
        args.firstIndex(of: name).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }
    }
    private static func int(_ name: String) -> Int? { value(name).flatMap(Int.init) }
}
