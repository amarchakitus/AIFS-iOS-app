import Darwin
import Foundation
#if os(iOS)
import os
#endif

/// The process's memory as the system accounts it (the figure iOS's per-app limit applies
/// to): now, and the highest since the process started.
public enum ProcessMemory {
    public static func footprint() -> (current: Int, peak: Int) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return (0, 0) }
        return (Int(info.phys_footprint), Int(info.ledger_phys_footprint_peak))
    }
}

/// Runs the bundle's test case (`test_input.safetensors`, from `ane_run.py --write-test`)
/// and compares the prediction with the float32 reference in `test_expected.safetensors`.
public enum AuroraSelfTest {
    public static func run(directory: URL, options: AuroraRuntime.Options, steps: Int = 1,
                           log: (String) -> Void) throws {
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
        let rt = try AuroraRuntime(directory: directory, options: options)
        let input = try SafeTensorsFile(directory.appending(path: "test_input.safetensors"))
        let expected = try SafeTensorsFile(directory.appending(path: "test_expected.safetensors"))
        guard let iso = input.metadata["time"], let time = ISO8601DateFormatter().date(from: iso) else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [NSLocalizedDescriptionKey: "test input has no time"])
        }
        let H = AuroraRuntime.height, W = AuroraRuntime.width
        let surfOut = UnsafeMutablePointer<Float16>.allocate(capacity: 26 * H * W)
        let atmosOut = UnsafeMutablePointer<Float16>.allocate(capacity: 5 * 13 * H * W)
        defer { surfOut.deallocate(); atmosOut.deallocate() }
        log("buffers allocated: \(mem())")

        // Compile every program first (minutes on a first run), logging each, so that a
        // program the device is slow to compile shows by name.
        let compileStart = Date()
        var last = compileStart, index = 0
        let names = rt.programNames
        try rt.prepare { fraction in
            guard fraction > 0 else { return }
            let now = Date()
            if now.timeIntervalSince(last) > 5 {
                log(String(format: "compiled %@ in %.1f s (%.0f%% at %.0f s)", names[index], now.timeIntervalSince(last),
                           fraction * 100, now.timeIntervalSince(compileStart)))
            }
            last = now
            index += 1
        }
        log(String(format: "all programs compiled in %.1f s: %@", Date().timeIntervalSince(compileStart), mem()))
        if options.resident {
            let t = Date()
            try rt.preload()
            log(String(format: "preloaded every program in %.1f s: %@", Date().timeIntervalSince(t), mem()))
        }
        for s in 1 ... steps {
            let t = Date()
            try rt.step(surface: try input.pointer("surf", as: Float16.self), atmosphere: try input.pointer("atmos", as: Float16.self),
                        time: time, surfaceOut: surfOut, atmosphereOut: atmosOut)
            let total = Date().timeIntervalSince(t)
            let parts = rt.timings.sorted { $0.value > $1.value }.map { String(format: "%@ %.2f", $0.key, $0.value) }
            let split = rt.chunksRun.sorted { $0.key.rawValue < $1.key.rawValue }.map { "\($0.key.rawValue) \($0.value)" }
            log(String(format: "step %d: %.2f s (%@); block chunks %@; %d program loads; %@", s, total,
                       parts.joined(separator: ", "), split.joined(separator: ", "), rt.loads, mem()))
            if !rt.loadTimes.isEmpty {
                let slowest = rt.loadTimes.sorted { $0.value > $1.value }.prefix(6).map { String(format: "%@ %.2f", $0.key, $0.value) }
                log(String(format: "  loads: %.2f s in all; slowest %@", rt.loadTimes.values.reduce(0, +), slowest.joined(separator: ", ")))
            }
        }

        // Accuracy against the float32 reference, in normalised units.
        func rel(_ a: UnsafePointer<Float16>, _ b: UnsafePointer<Float16>, _ n: Int) -> Double {
            var num = 0.0, den = 0.0
            for i in 0 ..< n {
                let d = Double(a[i]) - Double(b[i])
                num += d * d
                den += Double(b[i]) * Double(b[i])
            }
            return (num / max(den, 1e-30)).squareRoot()
        }
        let es = try expected.pointer("surf", as: Float16.self), ea = try expected.pointer("atmos", as: Float16.self)
        var lines: [String] = []
        for (i, v) in rt.surfaceVariables.enumerated() where ["2t", "10u", "msl", "tcc", "scaled_tp_1h"].contains(v) {
            lines.append(String(format: "%@ %.1e", v, rel(surfOut + i * H * W, es + i * H * W, H * W)))
        }
        for (i, v) in ["z", "u", "v", "t", "q"].enumerated() {
            lines.append(String(format: "%@ %.1e", v, rel(atmosOut + i * 13 * H * W, ea + i * 13 * H * W, 13 * H * W)))
        }
        log("rms error relative to float32: " + lines.joined(separator: ", "))
    }
}
