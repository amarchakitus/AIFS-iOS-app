import Foundation
import Testing
@testable import AIFS_iOS_app

/// Places' daily summaries: local days, and which day a 6-hour step belongs to.
struct PointForecastTests {
    private func date(_ iso: String) -> Date { ISO8601DateFormatter().date(from: iso)! }

    private func forecast(timeZone: String, steps: [(String, Double, Double)]) -> PointForecast {
        // (valid time, 2 m temperature °C, precipitation mm over the 6 h ending then)
        let location = SavedLocation(name: "Test", latitude: 0, longitude: 0, timeZoneID: timeZone)
        return PointForecast(location: location, steps: steps.map { iso, c, mm in
            PointStep(date: date(iso), t2m: c + 273.15, tp: mm / 1000)
        })
    }

    @Test func groupsByTheLocalDayInThePlacesTimeZone() {
        // 06Z and 12Z on the 18th are 01:00 and 07:00 in Chicago; 03Z on the 19th is
        // still the evening of the 18th there.
        let f = forecast(timeZone: "America/Chicago", steps: [
            ("2026-10-18T06:00:00Z", 10, 0), ("2026-10-18T12:00:00Z", 8, 0),
            ("2026-10-18T18:00:00Z", 16, 1), ("2026-10-19T03:00:00Z", 12, 2),
        ])
        #expect(f.days.count == 1)
        #expect(f.days.first?.low == 8)
        #expect(f.days.first?.high == 16)
        #expect(f.days.first?.precipitationMM == 3)
    }

    @Test func aMidnightStepClosesTheDayBefore() {
        // The 18-24Z window ends at midnight UTC: its rain, and its temperature, belong to
        // the 18th, not the 19th.
        let f = forecast(timeZone: "UTC", steps: [
            ("2026-10-18T18:00:00Z", 15, 0), ("2026-10-19T00:00:00Z", 11, 4),
            ("2026-10-19T06:00:00Z", 9, 1),
        ])
        #expect(f.days.count == 2)
        #expect(f.days[0].precipitationMM == 4)
        #expect(f.days[0].low == 11)
        #expect(f.days[1].precipitationMM == 1)
        #expect(f.days[1].high == 9)
    }

    @Test func currentIsTheNearestStep() {
        let f = forecast(timeZone: "UTC", steps: [
            ("2026-10-18T00:00:00Z", 1, 0), ("2026-10-18T06:00:00Z", 2, 0), ("2026-10-18T12:00:00Z", 3, 0),
        ])
        #expect(f.current(now: date("2026-10-18T07:30:00Z"))?.date == date("2026-10-18T06:00:00Z"))
        #expect(f.current(now: date("2026-10-18T10:00:00Z"))?.date == date("2026-10-18T12:00:00Z"))
        #expect(f.current(now: date("2027-01-01T00:00:00Z"))?.date == date("2026-10-18T12:00:00Z"))
    }
}
