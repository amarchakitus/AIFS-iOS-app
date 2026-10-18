import AIFSKit
import Foundation
import Observation

/// A wind level the particles can follow: which stored outputs hold its components, and
/// the speed range used to match speeds across levels (as in AIFS-VIZ).
nonisolated enum WindLevel: String, Codable, CaseIterable, Identifiable, Sendable {
    case surface, hPa850, jet

    var id: String { rawValue }
    var title: String {
        switch self {
        case .surface: "10 m wind"
        case .hPa850: "850 hPa wind"
        case .jet: "250 hPa jet"
        }
    }
    var components: (u: String, v: String) {
        switch self {
        case .surface: ("10u", "10v")
        case .hPa850: ("u_850", "v_850")
        case .jet: ("u_250", "v_250")
        }
    }
    /// Top of the layer's usual speed range, m/s.
    var speedRange: Double {
        switch self {
        case .surface: 40
        case .hPa850: 50
        case .jet: 90
        }
    }
}

/// Colour ramps a layer can be drawn with instead of its own.
nonisolated enum RampChoice: String, Codable, CaseIterable, Identifiable, Sendable {
    case standard, blue, teal, purple, orange, gray, blueRed

    var id: String { rawValue }
    var title: String {
        switch self {
        case .standard: "Default"
        case .blue: "Blue"
        case .teal: "Teal"
        case .purple: "Purple"
        case .orange: "Orange"
        case .gray: "Gray"
        case .blueRed: "Blue–red"
        }
    }

    /// The named ramp a field's own scale already uses, so the picker can say which
    /// choice is the default instead of offering an anonymous "Default".
    static func matching(_ scale: ColorScale) -> RampChoice {
        switch scale {
        case .sequential: .blue
        case .diverging: .blueRed
        case .gray: .gray
        }
    }

    /// Stops light -> dark (or cool -> warm), interpolated in OKLab like the defaults.
    func stops(default scale: ColorScale) -> [UInt32] {
        switch self {
        case .standard: scale.stops
        case .blue: [0xF0EFEC, 0xCDE2FB, 0x9EC5F4, 0x6DA7EC, 0x3987E5, 0x256ABF, 0x184F95, 0x0D366B]
        case .teal: [0xEEF6F3, 0xC5E8DC, 0x8FD3BC, 0x4FB898, 0x1B9C77, 0x117D60, 0x0B5F49, 0x064233]
        case .purple: [0xF2F0FA, 0xDCD6F5, 0xBCB2EE, 0x9A8DE6, 0x7A6ADA, 0x6250D6, 0x4A3AA7, 0x2E2470]
        case .orange: [0xFBF1EA, 0xF8D9C4, 0xF4B996, 0xEF9565, 0xEB6834, 0xC94E1F, 0x9C3913, 0x6B250A]
        case .gray: [0x1A1A19, 0x4A4A47, 0x7D7C78, 0xB3B2AD, 0xF0EFEC]
        case .blueRed: [0x0D366B, 0x256ABF, 0x5598E7, 0x9EC5F4, 0xF0EFEC, 0xF4B5AD, 0xEC7F78, 0xD03B3B, 0x8E1F1F]
        }
    }
}

/// One globe: a field layer and, optionally, wind particles over it.
nonisolated struct GlobeSpec: Codable, Hashable, Identifiable, Sendable {
    var id = UUID()
    var layer: String  // DisplayField.id
    var overlay: WindLevel?
}

/// The basemap's switchable layers, in panel order.
nonisolated enum BasemapLayer: String, Codable, CaseIterable, Identifiable, Sendable {
    case night, relief, coast, graticule, atmosphere, stars

    var id: String { rawValue }
    var title: String {
        switch self {
        case .night: "Day / night"
        case .relief: "Relief"
        case .coast: "Coastlines"
        case .graticule: "Grid"
        case .atmosphere: "Atmosphere"
        case .stars: "Stars"
        }
    }
    var range: ClosedRange<Double> { self == .atmosphere || self == .stars ? 0 ... 1.5 : 0 ... 1 }
}

nonisolated enum GlobeLayout: String, Codable, CaseIterable, Identifiable, Sendable {
    case auto, row, column
    var id: String { rawValue }
    var title: String { rawValue.capitalized }
}

/// The globe panel, after AIFS-VIZ's `uisettings.SCHEMA`: same groups, same defaults and
/// ranges, so a setting that looks right on the desktop wallpaper looks right here.
nonisolated struct GlobeSettings: Codable, Equatable, Sendable {
    // Globes
    var globes = [GlobeSpec(layer: "t2m", overlay: .surface)]
    var layout = GlobeLayout.auto
    var showLabels = true

    // Motion
    var spin = true
    var spinSeconds = 240.0  // 20...3600, log
    var cycleSeconds = 180.0  // 20...7200, log: time to play the whole forecast
    var loopForecast = true
    var fpsCap = 60.0  // 10...120

    // Wind particles
    var particles = true
    var advect = 0.25  // 0.02...6, log: AIFS-VIZ uses 1; calmer on a small globe
    var speedNormalise = 1.0  // 0...1
    var particleDensity = 0.55  // 0.15...2.5: every particle is on the visible side
    var trailSpacing = 4.0  // 1...8: keeps streaks long at the calmer speed
    var trailLength = 16.0  // 4...32
    var particleLife = 12.0  // 0.5...120, log
    var trailWidth = 1.6  // 0.5...5 px
    var trailIntensity = 0.7  // 0.1...2
    var trailFade = 1.25  // 0.4...3
    var trailWhiteness = 0.8  // 0...1
    var vectorFieldOpacity = 0.32  // 0...1: a wind layer's own speed field beneath its particles

    // Basemap
    // Strengths are kept while a layer is off, so switching it back on restores it.
    var night = 0.85
    var relief = 0.75
    var coast = 0.85
    var graticule = 0.5
    var atmosphere = 1.0  // 0...1.5
    var stars = 1.0  // 0...1.5
    /// Layers switched off. Optional so settings saved before it existed still decode;
    /// nil means the default, grid lines off.
    var basemapOff: Set<BasemapLayer>?

    // Fields
    var opacity = 1.0  // 0.1...1
    var ramps: [String: RampChoice] = [:]

    // Readouts
    var localTime = false

    func isOn(_ layer: BasemapLayer) -> Bool { !(basemapOff ?? [.graticule]).contains(layer) }

    mutating func setOn(_ layer: BasemapLayer, _ on: Bool) {
        var off = basemapOff ?? [.graticule]
        if on { off.remove(layer) } else { off.insert(layer) }
        basemapOff = off
    }

    /// A layer's strength as stored, whether or not it is on.
    func strength(_ layer: BasemapLayer) -> Double {
        switch layer {
        case .night: night
        case .relief: relief
        case .coast: coast
        case .graticule: graticule
        case .atmosphere: atmosphere
        case .stars: stars
        }
    }

    mutating func setStrength(_ layer: BasemapLayer, _ v: Double) {
        switch layer {
        case .night: night = v
        case .relief: relief = v
        case .coast: coast = v
        case .graticule: graticule = v
        case .atmosphere: atmosphere = v
        case .stars: stars = v
        }
    }

    /// What the renderer draws: the strength, or nothing when the layer is off.
    func level(_ layer: BasemapLayer) -> Double { isOn(layer) ? strength(layer) : 0 }

    /// Advection rate for particles on `level`, degrees of arc per second per m/s (as in
    /// AIFS-VIZ): matched across levels by each level's own range, anchored on the jet so
    /// turning matching up speeds the slow layers up.
    func advection(for level: WindLevel) -> Double {
        advect * pow(90 / level.speedRange, speedNormalise)
    }

    func lut(for field: DisplayField) -> ColorLUT {
        ColorLUT(stops: (ramps[field.id] ?? .standard).stops(default: field.scale))
    }
}

/// The settings, persisted to UserDefaults on every change.
@Observable
final class GlobeSettingsStore {
    static let shared = GlobeSettingsStore()
    private static let key = "globeSettings.v5"

    var settings: GlobeSettings {
        didSet {
            if let data = try? JSONEncoder().encode(settings) {
                UserDefaults.standard.set(data, forKey: Self.key)
            }
        }
    }

    private init() {
        settings = UserDefaults.standard.data(forKey: Self.key)
            .flatMap { try? JSONDecoder().decode(GlobeSettings.self, from: $0) } ?? GlobeSettings()
    }

    func reset() { settings = GlobeSettings() }
}
