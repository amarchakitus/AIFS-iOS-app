import AIFSKit
import Foundation
import Observation

/// What a value measures, for picking its display unit. Fields without one (cloud cover
/// in %, geopotential height in dam) are shown as stored.
nonisolated enum Quantity: Sendable {
    case temperature  // base unit °C
    case speed  // m/s
    case precipitation  // mm
    case pressure  // hPa
}

nonisolated enum TemperatureUnit: String, Codable, CaseIterable, Identifiable, Sendable {
    case celsius, fahrenheit
    var id: String { rawValue }
    var symbol: String { self == .celsius ? "°C" : "°F" }
}

nonisolated enum SpeedUnit: String, Codable, CaseIterable, Identifiable, Sendable {
    case metresPerSecond, kilometresPerHour, milesPerHour, knots
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .metresPerSecond: "m/s"
        case .kilometresPerHour: "km/h"
        case .milesPerHour: "mph"
        case .knots: "kn"
        }
    }
    /// Display units per m/s.
    var perMetrePerSecond: Double {
        switch self {
        case .metresPerSecond: 1
        case .kilometresPerHour: 3.6
        case .milesPerHour: 3600 / 1609.344
        case .knots: 3600 / 1852
        }
    }
}

nonisolated enum PrecipitationUnit: String, Codable, CaseIterable, Identifiable, Sendable {
    case millimetres, inches
    var id: String { rawValue }
    var symbol: String { self == .millimetres ? "mm" : "in" }
}

nonisolated enum PressureUnit: String, Codable, CaseIterable, Identifiable, Sendable {
    case hectopascals, inchesOfMercury
    var id: String { rawValue }
    var symbol: String { self == .hectopascals ? "hPa" : "inHg" }
}

/// The app-wide choice of units. Conversions run from the base units the model output is
/// decoded into (°C, m/s, mm, hPa) and back; colour scales stay in base units, so a colour
/// always means the same physical value whatever the labels say.
nonisolated struct UnitPreferences: Codable, Equatable, Sendable {
    var temperature = TemperatureUnit.celsius
    var speed = SpeedUnit.kilometresPerHour
    var precipitation = PrecipitationUnit.millimetres
    var pressure = PressureUnit.hectopascals

    /// What the device's region uses: US customary in the US, miles per hour in the UK,
    /// metric elsewhere.
    static var regional: UnitPreferences {
        switch Locale.current.measurementSystem {
        case .us: UnitPreferences(temperature: .fahrenheit, speed: .milesPerHour, precipitation: .inches, pressure: .inchesOfMercury)
        case .uk: UnitPreferences(speed: .milesPerHour)
        default: UnitPreferences()
        }
    }

    /// Base units -> display units.
    func display(_ v: Double, _ q: Quantity) -> Double {
        switch q {
        case .temperature: temperature == .celsius ? v : v * 9 / 5 + 32
        case .speed: v * speed.perMetrePerSecond
        case .precipitation: precipitation == .millimetres ? v : v / 25.4
        case .pressure: pressure == .hectopascals ? v : v * 100 / 3386.389
        }
    }

    /// Display units -> base units.
    func base(_ v: Double, _ q: Quantity) -> Double {
        switch q {
        case .temperature: temperature == .celsius ? v : (v - 32) * 5 / 9
        case .speed: v / speed.perMetrePerSecond
        case .precipitation: precipitation == .millimetres ? v : v * 25.4
        case .pressure: pressure == .hectopascals ? v : v * 3386.389 / 100
        }
    }

    func symbol(_ q: Quantity) -> String {
        switch q {
        case .temperature: temperature.symbol
        case .speed: speed.symbol
        case .precipitation: precipitation.symbol
        case .pressure: pressure.symbol
        }
    }

    /// Decimals for a point readout: enough to tell neighbours apart, no more.
    func decimals(_ q: Quantity) -> Int {
        switch q {
        case .temperature, .speed: 1
        case .precipitation: precipitation == .millimetres ? 1 : 2
        case .pressure: pressure == .hectopascals ? 1 : 2
        }
    }
}

/// The units, persisted. Starts from the region's; changes apply everywhere at once.
@Observable
final class UnitStore {
    static let shared = UnitStore()
    private static let key = "units.v1"

    var preferences: UnitPreferences {
        didSet {
            if let data = try? JSONEncoder().encode(preferences) {
                UserDefaults.standard.set(data, forKey: Self.key)
            }
        }
    }

    private init() {
        preferences = UserDefaults.standard.data(forKey: Self.key)
            .flatMap { try? JSONDecoder().decode(UnitPreferences.self, from: $0) } ?? .regional
    }
}

extension DisplayField {
    /// What the field's values measure, from the base unit it is decoded into.
    nonisolated var quantity: Quantity? {
        switch units {
        case "°C": .temperature
        case "m/s": .speed
        case "mm": .precipitation
        case "hPa": .pressure
        default: nil
        }
    }

    /// Unit label in the user's units.
    var displayUnits: String {
        quantity.map(UnitStore.shared.preferences.symbol) ?? units
    }

    /// A field value (base units) as a readout in the user's units, e.g. "68.2 °F".
    func readout(_ v: Float) -> String {
        guard let q = quantity else { return String(format: "%.1f %@", v, units) }
        let p = UnitStore.shared.preferences
        return String(format: "%.\(p.decimals(q))f %@", p.display(Double(v), q), p.symbol(q))
    }
}

/// Legend ticks for a field in the user's units: round numbers *in those units*, each
/// placed where its base-unit value falls on the ramp, so labels never lie about colour.
nonisolated struct LegendTick: Identifiable, Sendable {
    let id: Int
    let position: Double  // 0...1 along the ramp
    let label: String

    static func ticks(for field: DisplayField, units p: UnitPreferences) -> [LegendTick] {
        let scale = field.scale
        let q = field.quantity
        let toDisplay = { (v: Double) in q.map { p.display(v, $0) } ?? v }
        let toBase = { (v: Double) in q.map { p.base(v, $0) } ?? v }
        let (lo, hi) = scale.range
        let dLo = toDisplay(Double(lo)), dHi = toDisplay(Double(hi))

        var values: [(value: Double, decimals: Int)]
        if case .sequential(_, _, let gamma) = scale, gamma != 1 {
            // A skewed ramp (precipitation): evenly spaced along the ramp, not the value,
            // or every label piles up at the wet end. One significant figure each.
            values = [0, 0.25, 0.5, 0.75, 1].map { t in
                roundedToOneFigure(toDisplay(Double(scale.value(at: Float(t)))))
            }
        } else {
            let step = niceStep((dHi - dLo) / 4.5)
            let decimals = max(0, Int(-floor(log10(step) + 1e-9)))
            var v = (dLo / step).rounded(.up) * step
            values = []
            while v <= dHi + step * 1e-6 {
                values.append((v == 0 ? 0 : v, decimals))  // no "-0"
                v += step
            }
            // A diverging temperature ramp is neutral at freezing: label it exactly, so
            // 32 °F sits on the grey, rather than a round number beside it.
            if case .diverging(_, let mid, _) = scale, q == .temperature {
                let m = toDisplay(Double(mid))
                values.removeAll { abs($0.value - m) < (dHi - dLo) * 0.09 }
                values.append((m, 0))
                values.sort { $0.value < $1.value }
            }
        }

        var ticks: [LegendTick] = []
        for (value, decimals) in values {
            let position = Double(scale.position(Float(toBase(value))))
            // Never two labels on top of each other.
            if let last = ticks.last, position - last.position < 0.12 { continue }
            ticks.append(LegendTick(id: ticks.count, position: position,
                                    label: String(format: "%.\(decimals)f", value)))
        }
        return ticks
    }

    private static func niceStep(_ raw: Double) -> Double {
        guard raw > 0 else { return 1 }
        let magnitude = pow(10, floor(log10(raw)))
        let f = raw / magnitude
        return ([1, 2, 2.5, 4, 5, 10].first { $0 >= f - 1e-9 } ?? 10) * magnitude
    }

    private static func roundedToOneFigure(_ v: Double) -> (value: Double, decimals: Int) {
        guard v != 0 else { return (0, 0) }
        let e = floor(log10(abs(v)))
        let magnitude = pow(10, e)
        return ((v / magnitude).rounded() * magnitude, max(0, Int(-e)))
    }
}
