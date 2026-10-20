// Run AIFS Single v2 through AIFSKit on the Mac -- the same code the iPad runs -- and
// write each step's input and output so they can be checked against PyTorch:
//
//   aifs-mlx --bundle exports/aifs-single-v2-float16 --ic exports/ic/ic_20260924T00.safetensors \
//            --steps 3 --out /tmp/mlx-run [--dtype float32]
import AIFSKit
import AuroraANE
import Foundation
import GraphCastANE
import ImageIO
import MLX
import UniformTypeIdentifiers

AIFSRuntime.log = { print($0) }

func usage() -> Never {
    print("""
        usage: aifs-mlx --bundle DIR --ic FILE [--steps N] [--out DIR] [--dtype float16|float32]
                        [--render DIR]   (PNG of every display field, every step)
               aifs-mlx --bundle DIR --fetch-ic latest|YYYYMMDDTHH --out DIR
               aifs-mlx --bundle AURORA_DIR --aurora-ic YYYYMMDDTHH --out DIR
               aifs-mlx --bundle GRAPHCAST_DIR --graphcast-ic YYYYMMDDTHH --out DIR
               aifs-mlx --bundle GRAPHCAST_DIR --graphcast-forecast STATE --steps N
                        (scored against the IFS analyses at each valid time)
        Open-data messages go through a shared cache, --cache DIR (default: in the temporary
        folder), as the app's do: each is downloaded once, whichever command needs it first.
        """)
    exit(2)
}

var options: [String: String] = [:]
var argv = CommandLine.arguments.dropFirst()
while let flag = argv.popFirst() {
    guard flag.hasPrefix("--"), let value = argv.popFirst() else { usage() }
    options[String(flag.dropFirst(2))] = value
}
guard let bundle = options["bundle"] else { usage() }

/// The open-data messages every command shares.
let cache = OpenDataCache(root: options["cache"].map { URL(filePath: $0) }
    ?? FileManager.default.temporaryDirectory.appending(path: "aifs-opendata-cache"))

/// Downloads `keys` (only those not yet in the cache), reporting every 40th field.
func fetch(_ keys: [OpenDataCache.Key]) async throws {
    let before = keys.filter(cache.contains).count
    try await cache.fetch(keys) { p in
        if p.completed % 40 == 0 || p.completed == p.total { print("  \(p.completed)/\(p.total) fields, \(p.bytes / 1_000_000) MB") }
    }
    print("  \(keys.count) messages, \(before) already cached, \(keys.count - before) downloaded")
}

let cycleFormat: DateFormatter = {
    let f = DateFormatter()
    f.dateFormat = "yyyyMMdd'T'HH"
    f.timeZone = TimeZone(identifier: "UTC")
    return f
}()

if let which = options["graphcast-ic"] {
    // GraphCast's state from open data, into --out (never the bundle: that gets published).
    guard let date = cycleFormat.date(from: which), let out = options["out"] else { usage() }
    do {
        let t = Date()
        let client = try GraphCastOpenData(bundle: URL(filePath: bundle))
        try await fetch(client.keys(date: date))
        let state = try client.build(date: date, from: cache)
        let url = URL(filePath: out).appending(path: GraphCastOpenData.fileName(for: date))
        try FileManager.default.createDirectory(at: URL(filePath: out), withIntermediateDirectories: true)
        try state.save(to: url)
        print(String(format: "wrote %@ in %.1f s", url.path, Date().timeIntervalSince(t)))
        exit(0)
    } catch {
        print("failed: \(error)")
        exit(1)
    }
}

if let path = options["graphcast-forecast"] {
    // A rollout from a state file, each step scored against the IFS analysis at its valid
    // time (from open data), with persistence (the initial state) for comparison.
    let steps = Int(options["steps"] ?? "1") ?? 1
    do {
        let state = try GraphCastState.load(URL(filePath: path))
        let initial = state.time
        let fc = try GraphCastForecaster(directory: URL(filePath: bundle), state: state)
        let names = ["z_500", "t_850", "2t", "msl", "10u", "q_700"]
        // Persistence: the initial state's fields, north first.
        let start = try fc.persistence(names)
        let data = try GraphCastOpenData(bundle: URL(filePath: bundle))
        for s in 1 ... steps {
            let t = Date()
            let (valid, fields) = try fc.advance(wanted: names + ["tp"])
            let took = Date().timeIntervalSince(t)
            try await fetch(data.keys(date: valid))
            let truth = try data.build(date: valid, from: cache)
            let tf = try GraphCastForecaster.northFirst(truth, names)
            var line = String(format: "step %d (+%dh, %@) in %.1f s:", s, Int(valid.timeIntervalSince(initial) / 3600),
                              cycleFormat.string(from: valid), took)
            for name in names {
                line += String(format: " %@ %.4g (persistence %.4g)", name, rmse(fields[name]!, tf[name]!), rmse(start[name]!, tf[name]!))
            }
            let tp = fields["tp"]!
            line += String(format: "; tp mean %.3f mm, max %.1f mm", tp.reduce(0, +) / Float(tp.count) * 1000, (tp.max() ?? 0) * 1000)
            print(line)
        }
        exit(0)
    } catch {
        print("failed: \(error)")
        exit(1)
    }
}

/// Latitude-weighted RMSE of two 721 x 1440 fields, north first.
func rmse(_ a: [Float], _ b: [Float]) -> Double {
    var num = 0.0, den = 0.0
    for r in 0 ..< 721 {
        let w = cos((90 - Double(r) * 0.25) * .pi / 180)
        for c in 0 ..< 1440 {
            let d = Double(a[r * 1440 + c] - b[r * 1440 + c])
            num += w * d * d
        }
        den += w * 1440
    }
    return (num / den).squareRoot()
}

if let which = options["aurora-ic"] {
    // Aurora 1.5's state from open data (YYYYMMDDTHH), written beside the bundle's test
    // case and compared with it when that is for the same time (ane_run.py --write-test).
    let f = DateFormatter()
    f.dateFormat = "yyyyMMdd'T'HH"
    f.timeZone = TimeZone(identifier: "UTC")
    guard let date = f.date(from: which) else { usage() }
    do {
        let dir = URL(filePath: bundle)
        guard let out = options["out"] else { usage() }  // never the bundle: that gets published
        let client = try AuroraOpenData(bundle: dir)
        let t = ContinuousClock.now
        try await fetch(client.keys(date: date))
        let state = try client.build(date: date, from: cache)
        print("built in \(ContinuousClock.now - t)")
        try FileManager.default.createDirectory(at: URL(filePath: out), withIntermediateDirectories: true)
        let url = URL(filePath: out).appending(path: AuroraOpenData.fileName(for: date))
        try state.save(to: url)
        let reference = try AuroraState.load(dir.appending(path: "test_input.safetensors"))
        if reference.time == date {
            let meta = try AuroraMetadata.load(dir)
            let n = 720 * 1440
            func rms(_ a: UnsafePointer<Float16>, _ b: UnsafePointer<Float16>) -> (Double, Double) {
                var s = 0.0, m = 0.0
                for i in 0 ..< n { let d = abs(Double(a[i]) - Double(b[i])); s += d * d; m = max(m, d) }
                return ((s / Double(n)).squareRoot(), m)
            }
            for (v, name) in meta.inputs.enumerated() {
                let (r, m) = rms(state.surface(v, 1), reference.surface(v, 1))
                let (r0, _) = rms(state.surface(v, 0), reference.surface(v, 0))
                print(String(format: "  %-10@ rms %.1e (t-6h %.1e), max %.1e", name as NSString, r, r0, m))
            }
            for (v, name) in meta.atmosphere.enumerated() {
                var worst = 0.0
                for c in 0 ..< 13 { worst = max(worst, rms(state.atmosphere(v, 1, c), reference.atmosphere(v, 1, c)).0) }
                print(String(format: "  %-10@ worst level rms %.1e", name as NSString, worst))
            }
        }
    } catch {
        print("error: \(error)")
        exit(1)
    }
    exit(0)
}

if let which = options["fetch-ic"] {
    // Build initial conditions from ECMWF open data, exactly as the iPad does.
    guard let out = options["out"] else { usage() }
    do {
        let manifest = try Manifest.load(from: URL(filePath: bundle).appending(path: "manifest.json"))
        let client = try OpenDataClient(bundle: URL(filePath: bundle), manifest: manifest)
        let date: Date
        if which == "latest" {
            date = try await client.latestDate()
        } else {
            let f = DateFormatter()
            f.dateFormat = "yyyyMMdd'T'HH"
            f.timeZone = TimeZone(identifier: "UTC")
            guard let d = f.date(from: which) else { usage() }
            date = d
        }
        let t = ContinuousClock.now
        try await fetch(client.keys(date: date))
        let ic = try client.build(date: date, from: cache)
        let url = URL(filePath: out).appending(path: InitialCondition.fileName(for: date))
        try FileManager.default.createDirectory(at: URL(filePath: out), withIntermediateDirectories: true)
        try ic.save(to: url)
        print("IC \(date.formatted(.iso8601)) -> \(url.path) in \(ContinuousClock.now - t)")
    } catch {
        print("error: \(error.localizedDescription)")
        exit(1)
    }
    exit(0)
}

guard let icPath = options["ic"] else { usage() }
let steps = Int(options["steps"] ?? "1") ?? 1
let dtype: DType = options["dtype"] == "float32" ? .float32 : .float16
let out = options["out"].map { URL(filePath: $0) }

func elapsed(_ t: ContinuousClock.Instant) -> String {
    String(format: "%.1fs", Double((ContinuousClock.now - t).components.attoseconds) / 1e18
        + Double((ContinuousClock.now - t).components.seconds))
}

do {
    var t = ContinuousClock.now
    let model = try AIFSModel(bundle: URL(filePath: bundle), dtype: dtype, useNeuralEngine: options["ane"] != "off")
    print("loaded \(model.manifest.description) as \(dtype) in \(elapsed(t))" + (model.usesNeuralEngine ? ", processor on the Neural Engine" : ""))
    // Tuning knobs; none changes the result, only speed and memory.
    if let v = options["attn-block"].flatMap(Int.init) { model.attentionBlock = v }
    if let v = options["enc-chunks"].flatMap(Int.init) { model.encoderChunks = v }
    if let v = options["dec-chunks"].flatMap(Int.init) { model.decoderChunks = v }
    let ic = try InitialCondition.load(from: URL(filePath: icPath))
    let forecast = try Forecast(model: model, initialCondition: ic)
    let render = options["render"].map { URL(filePath: $0) }
    let renderer = try render.map { _ in try MapRenderer(bundle: URL(filePath: bundle)) }
    let coast = ic.latest("lsm").flatMap { lsm in renderer?.coastline(lsm: lsm) }
    if let render { try FileManager.default.createDirectory(at: render, withIntermediateDirectories: true) }
    if let out {
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    }
    for s in 0 ..< steps {
        if let out {
            try save(arrays: ["x": forecast.input], url: out.appending(path: "in_\(s).safetensors"))
        }
        Memory.peakMemory = 0
        t = ContinuousClock.now
        let (date, y) = forecast.advance()
        if let render, let renderer {
            let step = StepOutput(step: s + 1, date: date, y: y, names: model.manifest.outputVars)
            for field in DisplayField.all {
                let sources = field.sources.map { step.field($0)! }
                guard let map = renderer.render(field: field, sources: sources, coastline: coast) else { continue }
                let url = render.appending(path: String(format: "%@_%03d.png", field.id, (s + 1) * 6))
                let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
                CGImageDestinationAddImage(dest, map.image, nil)
                CGImageDestinationFinalize(dest)
            }
        }
        print("  " + model.lastTimings.map { String(format: "%@ %.2fs", $0.phase, $0.seconds) }.joined(separator: ", "))
        print("step \(s + 1) valid \(date.formatted(.iso8601)) in \(elapsed(t)), peak \(String(format: "%.2f", Double(Memory.peakMemory) / 1e9)) GB")
        if let out {
            try save(arrays: ["y": y], url: out.appending(path: "out_\(s).safetensors"))
        }
    }
} catch {
    print("error: \(error.localizedDescription)")
    exit(1)
}
