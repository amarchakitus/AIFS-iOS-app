import CoreGraphics
import Foundation

/// Weather maps from model output, shared by the iPad viewer and the Mac CLI (which
/// renders PNGs to check them).

/// A field the viewer can draw: which model outputs it needs, how to turn them into a
/// display value, and how to color it.
public struct DisplayField: Identifiable, Hashable, Sendable {
    public let id: String
    public let title: String
    public let units: String
    public let sources: [String]
    public let scale: ColorScale
    public let compute: @Sendable ([[Float]], Int) -> Float

    public static func == (a: Self, b: Self) -> Bool { a.id == b.id }
    public func hash(into h: inout Hasher) { h.combine(id) }

    private static let kelvinToC: @Sendable ([[Float]], Int) -> Float = { f, i in f[0][i] - 273.15 }
    private static let speed: @Sendable ([[Float]], Int) -> Float = { f, i in
        (f[0][i] * f[0][i] + f[1][i] * f[1][i]).squareRoot()
    }

    public static let all: [DisplayField] = [
        DisplayField(
            id: "t2m", title: "2 m temperature", units: "°C", sources: ["2t"],
            scale: .diverging(low: -40, mid: 0, high: 40), compute: kelvinToC),
        DisplayField(
            id: "msl", title: "Mean sea-level pressure", units: "hPa", sources: ["msl"],
            scale: .diverging(low: 960, mid: 1013, high: 1050), compute: { f, i in f[0][i] / 100 }),
        DisplayField(
            id: "wind10", title: "10 m wind speed", units: "m/s", sources: ["10u", "10v"],
            scale: .sequential(low: 0, high: 30, gamma: 1), compute: speed),
        DisplayField(
            id: "tp", title: "Precipitation (6 h)", units: "mm", sources: ["tp"],
            // Precipitation is heavily skewed; a square-root scale keeps drizzle visible
            // without saturating fronts.
            scale: .sequential(low: 0.1, high: 30, gamma: 0.5), compute: { f, i in max(0, f[0][i] * 1000) }),
        DisplayField(
            id: "tcc", title: "Total cloud cover", units: "%", sources: ["tcc"],
            scale: .gray(low: 0, high: 100), compute: { f, i in f[0][i] * 100 }),
        DisplayField(
            id: "z500", title: "500 hPa geopotential height", units: "dam", sources: ["z_500"],
            scale: .sequential(low: 480, high: 600, gamma: 1), compute: { f, i in f[0][i] / 9.80665 / 10 }),
        DisplayField(
            id: "t850", title: "850 hPa temperature", units: "°C", sources: ["t_850"],
            scale: .diverging(low: -30, mid: 0, high: 30), compute: kelvinToC),
        DisplayField(
            id: "wind250", title: "250 hPa wind speed", units: "m/s", sources: ["u_250", "v_250"],
            scale: .sequential(low: 0, high: 90, gamma: 1), compute: speed),
        DisplayField(
            id: "d2m", title: "2 m dew point", units: "°C", sources: ["2d"],
            scale: .diverging(low: -40, mid: 0, high: 30), compute: kelvinToC),
    ]

    /// Every model output any display field needs: what a run stores per step.
    public static var storedOutputs: [String] {
        Array(Set(all.flatMap(\.sources))).sorted()
    }
}

/// Color scales following the visualization rules: sequential = one hue light -> dark,
/// diverging = blue <-> red through a neutral gray midpoint, never a rainbow.
public enum ColorScale: Hashable, Sendable {
    case sequential(low: Float, high: Float, gamma: Float)
    case diverging(low: Float, mid: Float, high: Float)
    case gray(low: Float, high: Float)

    public var range: (Float, Float) {
        switch self {
        case .sequential(let l, let h, _), .gray(let l, let h): (l, h)
        case .diverging(let l, _, let h): (l, h)
        }
    }

    /// Position in [0, 1] along the color ramp.
    public func position(_ v: Float) -> Float {
        switch self {
        case .sequential(let l, let h, let g):
            return pow(min(max((v - l) / (h - l), 0), 1), g)
        case .gray(let l, let h):
            return min(max((v - l) / (h - l), 0), 1)
        case .diverging(let l, let m, let h):
            // Each arm scaled separately so the gray midpoint sits exactly at `mid`.
            if v < m { return 0.5 * min(max((v - l) / (m - l), 0), 1) }
            return 0.5 + 0.5 * min(max((v - m) / (h - m), 0), 1)
        }
    }

    /// The value at ramp position `p`, for legend ticks.
    public func value(at p: Float) -> Float {
        switch self {
        case .sequential(let l, let h, let g): l + (h - l) * pow(p, 1 / g)
        case .gray(let l, let h): l + (h - l) * p
        case .diverging(let l, let m, let h): p < 0.5 ? l + (m - l) * p * 2 : m + (h - m) * (p - 0.5) * 2
        }
    }

    /// Ramp stops (hex), interpolated in OKLab. Blue steps are the reference palette's
    /// sequential ramp; the red arm mirrors their lightness on the palette's red.
    public var stops: [UInt32] {
        switch self {
        case .sequential:
            [0xF0EFEC, 0xCDE2FB, 0x9EC5F4, 0x6DA7EC, 0x3987E5, 0x256ABF, 0x184F95, 0x0D366B]
        case .diverging:
            [0x0D366B, 0x256ABF, 0x5598E7, 0x9EC5F4, 0xF0EFEC, 0xF4B5AD, 0xEC7F78, 0xD03B3B, 0x8E1F1F]
        case .gray:
            [0x1A1A19, 0x4A4A47, 0x7D7C78, 0xB3B2AD, 0xF0EFEC]
        }
    }

    public var lut: ColorLUT { ColorLUT(stops: stops) }
}

/// 256 RGBA entries interpolated in OKLab between stops.
public struct ColorLUT: Sendable {
    let rgba: [UInt32]  // bytes R, G, B, A in memory (little-endian words)

    public init(stops: [UInt32], count: Int = 256) {
        let labs = stops.map(OKLab.init(hex:))
        rgba = (0 ..< count).map { k in
            let t = Double(k) / Double(count - 1) * Double(labs.count - 1)
            let i = min(Int(t), labs.count - 2)
            return labs[i].mix(labs[i + 1], t - Double(i)).rgba
        }
    }

    public func color(_ p: Float) -> UInt32 {
        rgba[Int((p * Float(rgba.count - 1)).rounded())]
    }

    /// sRGB components in [0, 1] at ramp position `p`.
    public func components(_ p: Float) -> (red: Double, green: Double, blue: Double) {
        let c = color(p)
        return (Double(c & 0xFF) / 255, Double(c >> 8 & 0xFF) / 255, Double(c >> 16 & 0xFF) / 255)
    }
}

struct OKLab {
    let l, a, b: Double

    init(l: Double, a: Double, b: Double) { (self.l, self.a, self.b) = (l, a, b) }

    init(hex: UInt32) {
        func lin(_ c: UInt32) -> Double {
            let v = Double(c) / 255
            return v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4)
        }
        let r = lin(hex >> 16 & 0xFF), g = lin(hex >> 8 & 0xFF), bl = lin(hex & 0xFF)
        let l_ = cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * bl)
        let m_ = cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * bl)
        let s_ = cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * bl)
        l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    }

    func mix(_ o: OKLab, _ t: Double) -> OKLab {
        OKLab(l: l + (o.l - l) * t, a: a + (o.a - a) * t, b: b + (o.b - b) * t)
    }

    var rgba: UInt32 {
        let l_ = l + 0.3963377774 * a + 0.2158037573 * b
        let m_ = l - 0.1055613458 * a - 0.0638541728 * b
        let s_ = l - 0.0894841775 * a - 1.2914855480 * b
        let (L, M, S) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_)
        func enc(_ v: Double) -> UInt32 {
            let c = v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1 / 2.4) - 0.055
            return UInt32((min(max(c, 0), 1) * 255).rounded())
        }
        let r = enc(4.0767416621 * L - 3.3077115913 * M + 0.2309699292 * S)
        let g = enc(-1.2684380046 * L + 2.6097574011 * M - 0.3413193965 * S)
        let bl = enc(-0.0041960863 * L - 0.7034186147 * M + 1.7076127010 * S)
        return r | g << 8 | bl << 16 | 0xFF << 24
    }
}

/// One rendered map: the image plus the regridded values behind it, for the readout.
public struct RenderedMap: @unchecked Sendable {
    public let image: CGImage
    public let values: [Float]  // (rows, cols), columns already shifted to start at 180W
    public let rows: Int
    public let cols: Int

    /// Value under a point in the image's unit square (0,0 = top left = 90N 180W).
    public func sample(u: Double, v: Double) -> (lat: Double, lon: Double, value: Float)? {
        guard (0 ... 1).contains(u), (0 ... 1).contains(v) else { return nil }
        let c = min(Int(u * Double(cols)), cols - 1), r = min(Int(v * Double(rows)), rows - 1)
        let lat = 90 - Double(r) * 180 / Double(rows - 1)
        let lon = -180 + Double(c) * 360 / Double(cols)
        return (lat, lon, values[r * cols + c])
    }
}

/// Regrids and colors fields. One instance per bundle; safe to use from any thread.
public final class MapRenderer: Sendable {
    public let grid: DisplayGrid

    public init(bundle: URL) throws {
        grid = try DisplayGrid(bundle: bundle)
    }

    /// Land-sea boundary pixels (columns shifted to start at 180W), from the N320 mask.
    public func coastline(lsm: [Float]) -> [Bool] {
        let land = shiftToDateline(grid.regrid(lsm)).map { $0 >= 0.5 }
        let rows = grid.rows, cols = grid.cols
        var edge = [Bool](repeating: false, count: land.count)
        for r in 0 ..< rows - 1 {
            for c in 0 ..< cols {
                let i = r * cols + c
                if land[i] != land[r * cols + (c + 1) % cols] || land[i] != land[i + cols] { edge[i] = true }
            }
        }
        return edge
    }

    /// Model grid columns run 0E -> 359.75E; maps read better centred on Greenwich.
    public func shiftToDateline(_ x: [Float]) -> [Float] {
        let rows = grid.rows, cols = grid.cols, half = cols / 2
        var out = [Float](repeating: 0, count: x.count)
        for r in 0 ..< rows {
            for c in 0 ..< cols { out[r * cols + c] = x[r * cols + (c + half) % cols] }
        }
        return out
    }

    public func render(field: DisplayField, sources: [[Float]], coastline: [Bool]?) -> RenderedMap? {
        let n = sources[0].count
        var display = [Float](repeating: 0, count: n)
        for i in 0 ..< n { display[i] = field.compute(sources, i) }
        let values = shiftToDateline(grid.regrid(display))
        let lut = field.scale.lut
        let coast: UInt32 = 0xFF19_1918  // near-black ink, RGBA little-endian
        let missing: UInt32 = 0xFFD8_D8D6
        var pixels = [UInt32](repeating: 0, count: values.count)
        for i in 0 ..< values.count {
            if let coastline, coastline[i] {
                pixels[i] = coast
            } else if values[i].isNaN {
                pixels[i] = missing
            } else {
                pixels[i] = lut.color(field.scale.position(values[i]))
            }
        }
        let cols = grid.cols, rows = grid.rows
        guard
            let provider = CGDataProvider(data: Data(bytes: pixels, count: pixels.count * 4) as CFData),
            let image = CGImage(
                width: cols, height: rows, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: cols * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                // Bytes are R, G, B, X in memory: default byte order, padding last.
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        else { return nil }
        return RenderedMap(image: image, values: values, rows: rows, cols: cols)
    }
}
