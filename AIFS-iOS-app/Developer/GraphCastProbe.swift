#if DEVELOPER
import AuroraANE
import CoreML
import Foundation

/// `--graphcast-probe N`: loads each program in Documents/probe-grid N times, each from a
/// fresh copy (so Core ML compiles it again), and logs the load time, the memory the load
/// took and the time per call. Some installs compile GraphCast's grid program to run 50x
/// slower; this measures how often, and whether a program's shape makes it so.
nonisolated enum GraphCastProbe {
    static func run(trials: Int, log: (String) -> Void) {
        let dir = AppPaths.documents.appending(path: "probe-grid", directoryHint: .isDirectory)
        let scratch = AppPaths.documents.appending(path: "probe-grid-run", directoryHint: .isDirectory)
        let models = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "mlmodelc" }.sorted { $0.path < $1.path }
        for trial in 1 ... trials {
            for url in models {
                let name = url.deletingPathExtension().lastPathComponent
                try? FileManager.default.removeItem(at: scratch)
                try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
                let copy = scratch.appending(path: "\(name)-\(UUID().uuidString.prefix(8)).mlmodelc")
                do {
                    try FileManager.default.copyItem(at: url, to: copy)
                    let before = ProcessMemory.footprint().current
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuAndNeuralEngine
                    var t = Date()
                    let model = try MLModel(contentsOf: copy, configuration: config)
                    let load = Date().timeIntervalSince(t)
                    let after = ProcessMemory.footprint().current
                    var feed: [String: MLFeatureValue] = [:]
                    for (input, d) in model.modelDescription.inputDescriptionsByName {
                        let a = try MLMultiArray(shape: d.multiArrayConstraint!.shape, dataType: .float16)
                        let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
                        for i in 0 ..< a.count { p[i] = Float16(Float(i % 97) / 97 - 0.5) }
                        feed[input] = MLFeatureValue(multiArray: a)
                    }
                    let provider = try MLDictionaryFeatureProvider(dictionary: feed)
                    _ = try model.prediction(from: provider)
                    var best = Double.infinity
                    for _ in 0 ..< 5 {
                        t = Date()
                        _ = try model.prediction(from: provider)
                        best = min(best, Date().timeIntervalSince(t))
                    }
                    log(String(format: "trial %d %@: load %.1f s, +%.0f MB, %.1f ms per call", trial, name, load,
                               Double(after - before) / 1e6, best * 1000))
                } catch {
                    log("trial \(trial) \(name): failed: \(error)")
                }
            }
        }
        try? FileManager.default.removeItem(at: scratch)
    }
}
#endif
