import Foundation
import MLX

/// GPU microbenchmarks at the model's own shapes, to tell kernel inefficiency apart from
/// thermal limits on a device.
public enum AIFSBenchmark {
    static func seconds(_ f: () -> MLXArray, repeats: Int) -> Double {
        eval(f())  // warm up: compile kernels, allocate
        let t = ContinuousClock.now
        for _ in 0 ..< repeats { eval(f()) }
        let d = (ContinuousClock.now - t).components
        return (Double(d.seconds) + Double(d.attoseconds) / 1e18) / Double(repeats)
    }

    /// Achieved TFLOPS of `(rows, inner) @ (inner, cols)` with a transposed weight, as the
    /// model's linear layers run it.
    public static func matmul(rows: Int, inner: Int, cols: Int, dtype: DType, repeats: Int = 5) -> Double {
        let x = MLXRandom.normal([rows, inner]).asType(dtype)
        let w = MLXRandom.normal([cols, inner]).asType(dtype)
        eval(x, w)
        let s = seconds({ MLX.matmul(x, w.T) }, repeats: repeats)
        return 2 * Double(rows) * Double(inner) * Double(cols) / s / 1e12
    }

    /// Seconds for one processor layer's banded attention with query blocks of `block`.
    public static func bandedAttention(block: Int, tokens: Int = 40320, heads: Int = 16, dim: Int = 64,
                                       window: Int = 1120, repeats: Int = 3) -> Double {
        let q = MLXRandom.normal([1, heads, tokens, dim]).asType(.float16)
        let k = MLXRandom.normal([1, heads, tokens, dim]).asType(.float16)
        let v = MLXRandom.normal([1, heads, tokens, dim]).asType(.float16)
        var masks: [String: MLXArray] = [:]
        func mask(_ rows: Int, _ cols: Int, _ off: Int) -> MLXArray {
            let key = "\(rows)x\(cols)+\(off)"
            if let m = masks[key] { return m }
            var values = [Float](repeating: 0, count: rows * cols)
            for i in 0 ..< rows { for j in 0 ..< cols where abs(i + off - j) > window { values[i * cols + j] = -.infinity } }
            let m = MLXArray(values, [rows, cols]).asType(.float16)
            eval(m)
            masks[key] = m
            return m
        }
        eval(q, k, v)
        return seconds({
            var outs: [MLXArray] = []
            for s in stride(from: 0, to: tokens, by: block) {
                let e = min(s + block, tokens), lo = max(0, s - window), hi = min(tokens, e + window)
                outs.append(MLXFast.scaledDotProductAttention(
                    queries: q[0..., 0..., s ..< e], keys: k[0..., 0..., lo ..< hi],
                    values: v[0..., 0..., lo ..< hi], scale: 0.125, mask: mask(e - s, hi - lo, s - lo)))
            }
            return concatenated(outs, axis: 2)
        }, repeats: repeats)
    }

    /// Seconds for one elementwise/normalisation op at processor MLP shape (40320 x 4096).
    static func op(_ name: String, repeats: Int = 5) -> Double {
        let x = MLXRandom.normal([40320, 4096]).asType(.float16)
        let small = MLXRandom.normal([40320, 1024]).asType(.float16)
        let b = MLXRandom.normal([4096]).asType(.float16)
        let g = MLXRandom.normal([1024]).asType(.float16)
        let w = MLXRandom.normal([4096, 1024]).asType(.float16)
        eval(x, small, b, g, w)
        let gelu = compile(shapeless: true) { (x: MLXArray) in x * 0.5 * (1 + erf(x * Float(0.70710678))) }
        switch name {
        case "bias": return seconds({ x + b }, repeats: repeats)
        case "gelu": return seconds({ gelu(x) }, repeats: repeats)
        case "layernorm": return seconds({ MLXFast.layerNorm(small, weight: g, bias: g, eps: 1e-5) }, repeats: repeats)
        case "matmul+bias": return seconds({ MLX.matmul(small, w.T) + b }, repeats: repeats)
        case "addmm": return seconds({ addMM(b, small, w.T) }, repeats: repeats)
        default: return .nan
        }
    }

    /// A short report, one line per measurement.
    public static func report() -> [String] {
        var lines: [String] = []
        for rows in [4235, 8470, 16940, 33880, 67760] {
            let t = matmul(rows: rows, inner: 1024, cols: 4096, dtype: .float16)
            lines.append(String(format: "matmul %dx1024 @ 1024x4096 float16: %.2f TFLOPS", rows, t))
        }
        for name in ["bias", "gelu", "layernorm", "matmul+bias", "addmm"] {
            lines.append(String(format: "%@ at 40320 rows: %.1f ms", name, op(name) * 1000))
        }
        return lines
    }
}
