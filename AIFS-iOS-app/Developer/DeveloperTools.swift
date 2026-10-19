#if DEVELOPER
import AIFSKit
import AuroraANE
import Foundation
import os
import UIKit

/// Development tooling run from launch arguments: the Neural Engine probe, the attention
/// benchmark, and unattended runs for measuring a device. Kept apart from the app proper,
/// and compiled only into developer builds (AIFS_DEVELOPER = YES).
extension AppModel {
    /// The measuring options, on launch (see LaunchOptions). Watch the log from a Mac with
    /// `xcrun devicectl device process launch --console`.
    func autorunIfRequested() {
        // Developer builds keep the device awake while the app is open: benchmarks and long
        // runs must not stall when the screen locks.
        UIApplication.shared.isIdleTimerDisabled = true
        for path in LaunchOptions.removeDocuments where !path.contains("..") {
            let url = AppPaths.documents.appending(path: path)
            let removed = (try? FileManager.default.removeItem(at: url)) != nil
            print("[AIFS] \(removed ? "removed" : "no") Documents/\(path)")
        }
        if !LaunchOptions.removeDocuments.isEmpty { refresh() }
        if LaunchOptions.neuralEngineProbe {
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
        if LaunchOptions.auroraBench {
            let stream = LaunchOptions.auroraStream, name = LaunchOptions.auroraDirectory
            let gpu = LaunchOptions.auroraGPUStages
            Task.detached(priority: .userInitiated) {
                let dir = AppPaths.documents.appending(path: LaunchOptions.auroraDirectory, directoryHint: .isDirectory)
                // Also to Documents/aurora/bench.log: a first run compiles every program for
                // the Neural Engine and can outlast a console session.
                let logURL = dir.appending(path: "bench.log")
                func log(_ line: String) {
                    let stamped = "\(Date().formatted(.iso8601)) \(line)"
                    print("[AIFS] aurora " + stamped)
                    if let h = try? FileHandle(forWritingTo: logURL) {
                        h.seekToEndOfFile()
                        h.write(Data((stamped + "\n").utf8))
                        try? h.close()
                    } else {
                        try? Data((stamped + "\n").utf8).write(to: logURL)
                    }
                }
                log("\(name): \(stream ? "streaming" : "resident") programs, GPU stages \(gpu.sorted()), thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
                do {
                    try AuroraSelfTest.run(directory: dir, options: .init(resident: !stream, gpuStages: gpu), steps: 2, log: log)
                } catch {
                    log("failed: \(error)")
                }
                log("done, thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
            }
            return
        }
        if LaunchOptions.bench {
            Task.detached(priority: .userInitiated) {
                print("[AIFS] bench start, thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
                for line in AIFSBenchmark.report() { print("[AIFS] bench " + line) }
                print("[AIFS] bench done, thermal \(ProcessInfo.processInfo.thermalState.rawValue)")
            }
            return
        }
        if let name = LaunchOptions.selectModel, let model = ForecastModel(rawValue: name) { selectedModel = model }
        if LaunchOptions.downloadLatest {
            print("[AIFS] autorun: downloading the latest \(selectedModel.title) cycle")
            downloadIC()
            Task {
                while download != nil { try? await Task.sleep(for: .seconds(1)) }
                guard let steps = LaunchOptions.autorunSteps, let ic = initialConditions.first else { return }
                print("[AIFS] autorun: \(steps) \(selectedModel.shortName) steps from \(ic.date.utcLabel)")
                startForecast(from: ic, leadHours: steps * 6)
            }
            return
        }
        guard let steps = LaunchOptions.autorunSteps, let ic = initialConditions.first else { return }
        print("[AIFS] autorun: \(steps) steps from \(ic.date.utcLabel), \(Double(os_proc_available_memory()) / 1e9) GB available")
        startForecast(from: ic, leadHours: steps * 6)
    }
}
#endif
