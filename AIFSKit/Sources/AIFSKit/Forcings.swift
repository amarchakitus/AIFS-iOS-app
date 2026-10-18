import Foundation

/// The date/position forcings anemoi-inference computes with earthkit-data
/// (`earthkit/data/sources/forcings.py`, `earthkit/data/utils/meteo.py`), reproduced
/// formula for formula. Computed in Double and rounded to Float, as anemoi does when it
/// builds the fp32 input tensor.
public struct Forcings: Sendable {
    public let latitudes: [Double]
    public let longitudes: [Double]

    private static let utc: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        return c
    }()

    public init(latitudes: [Float], longitudes: [Float]) {
        self.latitudes = latitudes.map(Double.init)
        self.longitudes = longitudes.map(Double.init)
    }

    public static func isComputed(_ name: String) -> Bool { supported.contains(name) }

    static let supported: Set<String> = [
        "cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude",
        "cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time", "insolation",
    ]

    /// Fractional days since 1 January 00 UTC of `date`'s year.
    static func julianDay(_ date: Date) -> Double {
        let year = utc.component(.year, from: date)
        let start = utc.date(from: DateComponents(year: year, month: 1, day: 1))!
        return date.timeIntervalSince(start) / 86400
    }

    /// Hours since 00 UTC of `date`'s day.
    static func hoursSinceMidnight(_ date: Date) -> Double {
        date.timeIntervalSince(utc.startOfDay(for: date)) / 3600
    }

    private static func rad(_ deg: Double) -> Double { deg * .pi / 180 }

    /// One forcing field over the grid at `date`.
    public func field(_ name: String, at date: Date) -> [Float] {
        switch name {
        case "cos_latitude": return latitudes.map { Float(cos(Self.rad($0))) }
        case "sin_latitude": return latitudes.map { Float(sin(Self.rad($0))) }
        case "cos_longitude": return longitudes.map { Float(cos(Self.rad($0))) }
        case "sin_longitude": return longitudes.map { Float(sin(Self.rad($0))) }
        case "cos_julian_day", "sin_julian_day":
            let r = Self.julianDay(date) / 365.25 * .pi * 2
            let v = Float(name.hasPrefix("cos") ? cos(r) : sin(r))
            return [Float](repeating: v, count: latitudes.count)
        case "cos_local_time", "sin_local_time":
            let h = Self.hoursSinceMidnight(date)
            let useCos = name.hasPrefix("cos")
            return longitudes.map { lon in
                let localTime = (lon / 360 * 24 + h).truncatingRemainder(dividingBy: 24)
                // Python's % is floored; for negative longitudes this keeps it in [0, 24).
                let lt = localTime < 0 ? localTime + 24 : localTime
                let r = lt / 24 * .pi * 2
                return Float(useCos ? cos(r) : sin(r))
            }
        case "insolation":
            return cosSolarZenithAngle(at: date)
        default:
            preconditionFailure("\(name) is not a computed forcing")
        }
    }

    /// earthkit's `cos_solar_zenith_angle`, clipped at zero (night).
    func cosSolarZenithAngle(at date: Date) -> [Float] {
        let a = Self.julianDay(date) / 365.25 * .pi * 2
        let declination = 0.396372 - 22.91327 * cos(a) + 4.025430 * sin(a)
            - 0.387205 * cos(2 * a) + 0.051967 * sin(2 * a)
            - 0.154527 * cos(3 * a) + 0.084798 * sin(3 * a)
        let timeCorrection = 0.004297 + 0.107029 * cos(a) - 1.837877 * sin(a)
            - 0.837378 * cos(2 * a) - 2.340475 * sin(2 * a)
        let dec = Self.rad(declination)
        // earthkit uses the integer hour (date.hour), not fractional time.
        let hour = Double(Self.utc.component(.hour, from: date))
        return zip(latitudes, longitudes).map { lat, lon in
            let la = Self.rad(lat)
            let z = sin(dec) * sin(la)
                + cos(dec) * cos(la) * cos(Self.rad((hour - 12) * 15 + lon + timeCorrection))
            return Float(max(z, 0))
        }
    }
}
