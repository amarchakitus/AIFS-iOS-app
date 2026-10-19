// Run AIFS Single v2 through AIFSKit on the Mac -- the same code the iPad runs -- and
// write each step's input and output so they can be checked against PyTorch:
//
//   aifs-mlx --bundle exports/aifs-single-v2-float16 --ic exports/ic/ic_20260924T00.safetensors \
//            --steps 3 --out /tmp/mlx-run [--dtype float32]
import AIFSKit
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

AIFSRuntime.log = { print($0) }

func usage() -> Never {
    print("""
        usage: aifs-mlx --bundle DIR --ic FILE [--steps N] [--out DIR] [--dtype float16|float32]
                        [--render DIR]   (PNG of every display field, every step)
               aifs-mlx --bundle DIR --fetch-ic latest|YYYYMMDDTHH --out DIR
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
        let ic = try await client.build(date: date) { p in
            if p.completed % 20 == 0 || p.completed == p.total {
                print("  \(p.completed)/\(p.total) fields, \(p.bytes / 1_000_000) MB")
            }
        }
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
