import AIFSKit
import Foundation
import Testing
@testable import AIFS_iOS_app

/// Conversions and legend ticks. The colour scales stay in base units, so a wrong
/// conversion here is a legend that silently lies about what a colour means.
struct UnitsTests {
    private let metric = UnitPreferences(
        temperature: .celsius, speed: .metresPerSecond, precipitation: .millimetres, pressure: .hectopascals)
    private let us = UnitPreferences(
        temperature: .fahrenheit, speed: .milesPerHour, precipitation: .inches, pressure: .inchesOfMercury)

    @Test func knownConversions() {
        #expect(us.display(0, .temperature) == 32)
        #expect(us.display(100, .temperature) == 212)
        #expect(us.display(-40, .temperature) == -40)
        #expect(abs(us.display(10, .speed) - 22.369) < 0.001)
        #expect(abs(UnitPreferences(speed: .kilometresPerHour).display(10, .speed) - 36) < 1e-9)
        #expect(abs(UnitPreferences(speed: .knots).display(10, .speed) - 19.438) < 0.001)
        #expect(abs(us.display(25.4, .precipitation) - 1) < 1e-12)
        #expect(abs(us.display(1013.25, .pressure) - 29.921) < 0.001)
        #expect(metric.display(1013.25, .pressure) == 1013.25)
    }

    @Test(arguments: [Quantity.temperature, .speed, .precipitation, .pressure])
    func baseUndoesDisplay(_ q: Quantity) {
        for prefs in [metric, us, UnitPreferences(speed: .knots), UnitPreferences(speed: .kilometresPerHour)] {
            for v in [-35.0, 0, 0.3, 12.5, 980, 1040] {
                #expect(abs(prefs.base(prefs.display(v, q), q) - v) < 1e-9)
            }
        }
    }

    @Test func fieldQuantities() {
        let byID = Dictionary(uniqueKeysWithValues: DisplayField.all.map { ($0.id, $0.quantity) })
        #expect(byID["t2m"] == .temperature)
        #expect(byID["d2m"] == .temperature)
        #expect(byID["wind10"] == .speed)
        #expect(byID["tp"] == .precipitation)
        #expect(byID["msl"] == .pressure)
        #expect(byID["tcc"] == .some(nil))  // % has no alternative units
        #expect(byID["z500"] == .some(nil))
    }

    /// Every field, every unit system: ticks increase, never overlap, and each sits where
    /// its own value falls on the ramp.
    @Test func ticksAreHonestAndSpaced() {
        for field in DisplayField.all {
            for prefs in [metric, us, UnitPreferences(speed: .knots)] {
                let ticks = LegendTick.ticks(for: field, units: prefs)
                #expect(ticks.count >= 3, "\(field.id): \(ticks.map(\.label))")
                for (a, b) in zip(ticks, ticks.dropFirst()) {
                    #expect(b.position - a.position >= 0.12, "\(field.id) overlap: \(ticks.map(\.label))")
                }
                for tick in ticks {
                    guard let shown = Double(tick.label) else {
                        Issue.record("\(field.id): unparseable label \(tick.label)")
                        continue
                    }
                    let base = field.quantity.map { prefs.base(shown, $0) } ?? shown
                    let expected = Double(field.scale.position(Float(base)))
                    #expect(abs(expected - tick.position) < 1e-4, "\(field.id) \(tick.label)")
                }
            }
        }
    }

    @Test func fahrenheitLabelsFreezingOnTheNeutralGrey() throws {
        let t2m = try #require(DisplayField.all.first { $0.id == "t2m" })
        let ticks = LegendTick.ticks(for: t2m, units: us)
        let freezing = try #require(ticks.first { $0.label == "32" })
        #expect(abs(freezing.position - 0.5) < 1e-6)
        #expect(ticks.map(\.label) == ["-40", "0", "32", "80"])
    }

    @Test func celsiusTicksAreRoundNumbers() throws {
        let t2m = try #require(DisplayField.all.first { $0.id == "t2m" })
        #expect(LegendTick.ticks(for: t2m, units: metric).map(\.label) == ["-40", "-20", "0", "20", "40"])
    }

    @Test func skewedRainRampSpreadsItsTicks() throws {
        let tp = try #require(DisplayField.all.first { $0.id == "tp" })
        #expect(LegendTick.ticks(for: tp, units: metric).map(\.label) == ["0.1", "2", "8", "20", "30"])
        #expect(LegendTick.ticks(for: tp, units: us).map(\.label) == ["0.004", "0.08", "0.3", "0.7", "1"])
    }
}
