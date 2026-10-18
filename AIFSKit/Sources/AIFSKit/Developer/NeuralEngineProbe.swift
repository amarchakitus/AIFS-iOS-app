import CoreML
import Foundation

/// Times a compiled Core ML model with a `(1, 1024, 1, S)` fp16 input `x` on each compute
/// unit, and reports where Core ML places its operations -- to decide whether the Neural
/// Engine is worth using for a layer before building anything on it.
public enum NeuralEngineProbe {
    public static func report(modelURL: URL, flopsPerCall: Double) async -> [String] {
        var lines: [String] = []
        let name = modelURL.deletingPathExtension().lastPathComponent
        for (label, units) in [("ANE", MLComputeUnits.cpuAndNeuralEngine), ("GPU", .cpuAndGPU)] {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = units
                let t0 = ContinuousClock.now
                let model = try MLModel(contentsOf: modelURL, configuration: config)
                let load = ContinuousClock.now - t0
                var features: [String: MLFeatureValue] = [:]
                for (name, desc) in model.modelDescription.inputDescriptionsByName {
                    guard let shape = desc.multiArrayConstraint?.shape else { continue }
                    let x = try MLMultiArray(shape: shape, dataType: .float16)
                    let n = x.count
                    x.withUnsafeMutableBufferPointer(ofType: Float16.self) { p, _ in
                        for i in 0 ..< n { p[i] = Float16(Float(i % 97) / 97 - 0.5) }
                    }
                    features[name] = MLFeatureValue(multiArray: x)
                }
                let input = try MLDictionaryFeatureProvider(dictionary: features)
                _ = try await model.prediction(from: input)
                let repeats = 5
                let t = ContinuousClock.now
                for _ in 0 ..< repeats { _ = try await model.prediction(from: input) }
                let d = (ContinuousClock.now - t).components
                let s = (Double(d.seconds) + Double(d.attoseconds) / 1e18) / Double(repeats)
                lines.append(String(
                    format: "%@ on %@: %.1f ms/call, %.2f TFLOPS (load %@)", name, label, s * 1000,
                    flopsPerCall / s / 1e12, "\(load)"))
            } catch {
                lines.append("\(name) on \(label): \(error.localizedDescription)")
            }
        }
        // Where Core ML actually puts each op when allowed the Neural Engine.
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            let plan = try await MLComputePlan.load(contentsOf: modelURL, configuration: config)
            if case .program(let program) = plan.modelStructure, let main = program.functions["main"] {
                var counts: [String: Int] = [:]
                for op in main.block.operations {
                    let device: String
                    switch plan.deviceUsage(for: op)?.preferred {
                    case .neuralEngine: device = "ANE"
                    case .gpu: device = "GPU"
                    case .cpu: device = "CPU"
                    default: device = "none"
                    }
                    counts["\(op.operatorName)@\(device)", default: 0] += 1
                }
                lines.append("\(name) placement: " + counts.sorted { $0.key < $1.key }.map { "\($0.key)×\($0.value)" }.joined(separator: " "))
            }
        } catch {
            lines.append("\(name) compute plan: \(error.localizedDescription)")
        }
        return lines
    }
}
