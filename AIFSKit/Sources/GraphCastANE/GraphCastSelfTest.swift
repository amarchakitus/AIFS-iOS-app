import AuroraANE
import Foundation
#if os(iOS)
import os
#endif

/// Runs the bundle's test case (`test_x.safetensors`, from `bundle.py --write-test`) and
/// compares the outputs with the port's float32 ones in `test_expected.safetensors`.
public enum GraphCastSelfTest {
    public static func run(directory: URL, steps: Int = 1, log: @escaping (String) -> Void) throws {
        func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1e9) }
        func mem() -> String {
            let f = ProcessMemory.footprint()
            var s = "footprint \(gb(f.current)), peak \(gb(f.peak))"
            #if os(iOS)
            s += ", available \(gb(os_proc_available_memory()))"
            #endif
            return s
        }
        log("start: \(mem())")
        let rt = try GraphCastRuntime(directory: directory)
        rt.log = log
        let input = try SafeTensorsFile(directory.appending(path: "test_x.safetensors"))
        let expected = try SafeTensorsFile(directory.appending(path: "test_expected.safetensors"))
        let n = rt.gridNodes, k = GraphCastRuntime.outputs
        let out = UnsafeMutablePointer<Float>.allocate(capacity: n * k)
        defer { out.deallocate() }
        let t0 = Date()
        var index = 0
        try rt.prepare { fraction in
            guard fraction > 0 else { return }
            log(String(format: "loaded %@ at %.1f s: %@", GraphCastRuntime.programNames[index], Date().timeIntervalSince(t0), mem()))
            index += 1
        }
        log(String(format: "programs loaded and checked in %.1f s (reloaded: %@): %@", Date().timeIntervalSince(t0),
                   rt.reloads.description, mem()))
        for s in 1 ... steps {
            let t = Date()
            let o = UnsafeSendable(out)
            try rt.step(features: try input.pointer("x", as: Float16.self)) { lo, count, rows in
                for i in 0 ..< count * k { o.value[lo * k + i] = Float(rows[i]) }
            }
            let parts = rt.timings.sorted { $0.value > $1.value }.map { String(format: "%@ %.2f", $0.key, $0.value) }
            log(String(format: "step %d: %.2f s (%@); %d gathers on the CPU; %@", s, Date().timeIntervalSince(t),
                       parts.joined(separator: ", "), rt.cpuGathers, mem()))
        }
        let ref = try expected.pointer("out", as: Float.self)
        var lines: [String] = [], worst = 0.0
        for (c, o) in rt.metadata.outputs.enumerated() {
            var num = 0.0, den = 0.0
            for i in 0 ..< n {
                let d = Double(out[i * k + c]) - Double(ref[i * k + c])
                num += d * d
                den += Double(ref[i * k + c]) * Double(ref[i * k + c])
            }
            let rel = (num / max(den, 1e-30)).squareRoot()
            worst = max(worst, rel)
            if o.level == nil || o.level == 500 || o.level == 850 {
                lines.append(String(format: "%@ %.1e", o.name, rel))
            }
        }
        log("rms error relative to float32 (normalised outputs): " + lines.joined(separator: ", "))
        log(String(format: "worst channel: %.1e", worst))
    }

    /// The forecaster on the bundle's test state (`bundle.py --write-test`): its features
    /// against `test_x`, then one step against Google's float32 prediction.
    public static func forecast(directory: URL, log: (String) -> Void) throws {
        let state = try GraphCastState.load(directory.appending(path: "test_state.safetensors"))
        let fc = try GraphCastForecaster(directory: directory, state: state)
        var t = Date()
        fc.buildFeatures(target: fc.nextDate)
        log(String(format: "features built in %.2f s", Date().timeIntervalSince(t)))
        let wantFile = try SafeTensorsFile(directory.appending(path: "test_x.safetensors"))  // keeps the mapping
        let want = try wantFile.pointer("x", as: Float16.self)
        let f = GraphCastRuntime.features, n = fc.runtime.gridNodes
        var worst = (0.0, "")
        for (c, input) in fc.metadata.inputs.enumerated() {
            var d = 0.0, r = 0.0
            for j in 0 ..< n {
                let a = Double(fc.features[j * f + c]), b = Double(want[j * f + c])
                d += (a - b) * (a - b)
                r += b * b
            }
            let rel = (d / max(r, 1e-30)).squareRoot()
            if rel > worst.0 { worst = (rel, input.name) }
        }
        log(String(format: "features vs test_x: worst channel %@ %.1e", worst.1, worst.0))

        t = Date()
        let names = ["2t", "msl", "10u", "tp", "z_500", "t_850", "q_700"]
        let (date, fields) = try fc.advance(wanted: names)
        log(String(format: "step to %@ in %.2f s (%@)", ISO8601DateFormatter().string(from: date), Date().timeIntervalSince(t),
                   fc.timings.sorted { $0.value > $1.value }.map { String(format: "%@ %.2f", $0.key, $0.value) }.joined(separator: ", ")))
        let ref = try SafeTensorsFile(directory.appending(path: "test_reference.safetensors"))
        var lines: [String] = []
        for name in names {
            guard let got = fields[name] else { lines.append("\(name) missing"); continue }
            let r = try ref.pointer(name, as: Float.self)
            let last = ref.tensors["\(name)@last"] != nil ? try ref.pointer("\(name)@last", as: Float.self) : nil
            var d = 0.0, change = 0.0
            for j in 0 ..< n {
                d += Double(got[j] - r[j]) * Double(got[j] - r[j])
                let c = Double(r[j]) - Double(last?[j] ?? 0)
                change += c * c
            }
            lines.append(String(format: "%@ %.1e", name, (d / change).squareRoot()))
        }
        log("vs Google float32 (rms difference / rms change over the step): " + lines.joined(separator: ", "))
    }
}
