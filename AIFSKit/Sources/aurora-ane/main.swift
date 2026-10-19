// Aurora 1.5 on this Mac's Neural Engine: runs the test case of an exported bundle and
// reports time, memory and accuracy.
//
//   aurora-ane --bundle ../../aurora-1p5/exports/ane [--stream] [--gpu 1,2] [--steps N]
//
// --stream loads each program just before use and releases it after, instead of keeping
// all of them loaded; --gpu runs the blocks on the GPU too, beside the Neural Engine.
import AuroraANE
import Foundation

let args = CommandLine.arguments
func value(_ flag: String) -> String? { args.firstIndex(of: flag).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }
guard let bundle = value("--bundle") else {
    print("usage: aurora-ane --bundle DIR [--stream] [--gpu 1,2] [--steps N]")
    exit(2)
}
// --gpu 1,2: stages whose blocks the GPU runs too.
let gpuStages = Set((value("--gpu") ?? "").split(separator: ",").compactMap { Int($0) })
let options = AuroraRuntime.Options(resident: !args.contains("--stream"), gpuStages: gpuStages)
if args.contains("--compile") {
    // Time preparing every program (clear ~/Library/Caches/aurora-ane first for a cold compile).
    do {
        let rt = try AuroraRuntime(directory: URL(filePath: bundle), options: options)
        let t = Date()
        try rt.prepare { _ in }
        print(String(format: "prepared %d programs in %.1f s", rt.programNames.count, Date().timeIntervalSince(t)))
    } catch {
        print("failed: \(error)")
        exit(1)
    }
    exit(0)
}
if args.contains("--forecaster") {
    // One step through AuroraForecaster from the test state: physical output against the
    // float32 reference, and the next input time against Python's (test_next.safetensors).
    do {
        let dir = URL(filePath: bundle)
        let state = try AuroraState.load(dir.appending(path: "test_input.safetensors"))
        let f = try AuroraForecaster(directory: dir, state: state, options: options)
        let t = Date()
        let wanted = ["10u", "10v", "2d", "2t", "msl", "sf", "t_850", "tcc", "tp", "u_250", "u_850", "v_250", "v_850", "z_500"]  // the app's
        let (date, fields) = try f.advance(wanted: wanted)
        print(String(format: "step to %@ in %.2f s: %@", date.ISO8601Format(), Date().timeIntervalSince(t),
                     f.timings.sorted { $0.key < $1.key }.map { String(format: "%@ %.2f", $0.key, $0.value) }.joined(separator: ", ")))
        if let out = value("--out") {
            try SafeTensorsWriter.write(URL(filePath: out), metadata: ["valid": date.ISO8601Format()], fields: fields)
            print("wrote \(out)")
        }
        let expected = try SafeTensorsFile(dir.appending(path: "test_expected.safetensors"))
        let es = try expected.pointer("surf", as: Float16.self), ea = try expected.pointer("atmos", as: Float16.self)
        let n = 720 * 1440, meta = f.metadata
        func rmse(_ a: [Float], _ ref: (Int) -> Float) -> Double {
            var s = 0.0
            for i in 0 ..< n { let d = Double(a[i] - ref(i)); s += d * d }
            return (s / Double(n)).squareRoot()
        }
        let i2t = meta.surface.firstIndex(of: "2t")!
        print(String(format: "2t vs float32: rms %.4f K", rmse(fields["2t"]!) { meta.physical(Float(es[i2t * n + $0]), "2t") }))
        print(String(format: "z500 vs float32: rms %.3f m2/s2", rmse(fields["z_500"]!) { meta.physical(Float(ea[(0 * 13 + 7) * n + $0]), "z", level: 500) }))
        let tp = fields["tp"]!
        print(String(format: "tp (6 h, est.): max %.1f mm, mean %.3f mm", (tp.max() ?? 0) * 1000, tp.reduce(0, +) / Float(n) * 1000))
        let nextFile = try SafeTensorsFile(dir.appending(path: "test_next.safetensors"))  // keeps the mapping alive
        let next = try nextFile.pointer("surf_next", as: Float16.self)
        for (v, name) in meta.inputs.enumerated() where ["2t", "tcc", "scaled_sd", "insolation", "ci"].contains(name) {
            let mine = state.surface(v, 1)
            var s = 0.0, r = 0.0
            for i in 0 ..< n { let d = Double(mine[i]) - Double(next[v * n + i]); s += d * d; r += Double(next[v * n + i]) * Double(next[v * n + i]) }
            print(String(format: "next input %@: rms difference %.2e (normalised; rms %.2f)", name, (s / Double(n)).squareRoot(), (r / Double(n)).squareRoot()))
        }
    } catch {
        print("failed: \(error)")
        exit(1)
    }
    exit(0)
}
do {
    try AuroraSelfTest.run(directory: URL(filePath: bundle), options: options, steps: Int(value("--steps") ?? "1") ?? 1) { print($0) }
} catch {
    print("failed: \(error)")
    exit(1)
}
