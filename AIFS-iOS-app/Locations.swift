import AIFSKit
import CoreLocation
import Foundation
import MapKit
import Observation
import simd

// MARK: - Saved places

nonisolated struct SavedLocation: Codable, Identifiable, Hashable, Sendable {
    var id = UUID()
    var name: String
    var detail: String?
    var latitude: Double
    var longitude: Double
    var timeZoneID: String?
    var isCurrentLocation = false

    var timeZone: TimeZone { timeZoneID.flatMap(TimeZone.init(identifier:)) ?? .current }
}

/// The dashboard's places, kept in Documents/locations.json.
@MainActor
@Observable
final class LocationStore {
    static let shared = LocationStore()
    private(set) var locations: [SavedLocation] = []
    private let url = URL.documentsDirectory.appending(path: "locations.json")

    private init() {
        locations = (try? JSONDecoder().decode([SavedLocation].self, from: Data(contentsOf: url))) ?? []
    }

    func add(_ location: SavedLocation) {
        if location.isCurrentLocation { locations.removeAll { $0.isCurrentLocation } }
        locations.removeAll { abs($0.latitude - location.latitude) < 0.01 && abs($0.longitude - location.longitude) < 0.01 }
        if location.isCurrentLocation { locations.insert(location, at: 0) } else { locations.append(location) }
        save()
    }

    func remove(_ location: SavedLocation) {
        locations.removeAll { $0.id == location.id }
        save()
    }

    private func save() {
        try? JSONEncoder().encode(locations).write(to: url, options: .atomic)
    }
}

// MARK: - The forecast at a point

/// One 6-hourly output at a place. Raw model units: K, m, fraction, m/s, Pa.
nonisolated struct PointStep: Sendable {
    let date: Date
    var t2m: Double?
    var d2m: Double?
    var tp: Double?  // metres of water over the 6 h ending at `date`
    var sf: Double?  // snowfall, metres of water equivalent
    var tcc: Double?
    var u10: Double?
    var v10: Double?
    var msl: Double?
    var isDay = true

    var temperatureC: Double? { t2m.map { $0 - 273.15 } }
    var precipitationMM: Double { max(0, (tp ?? 0) * 1000) }
    var windSpeed: Double? { u10.flatMap { u in v10.map { v in (u * u + v * v).squareRoot() } } }
    /// Direction the wind blows *from*, degrees clockwise from north.
    var windFrom: Double? {
        guard let u = u10, let v = v10 else { return nil }
        let d = atan2(-u, -v) * 180 / .pi
        return d < 0 ? d + 360 : d
    }
    /// Relative humidity from temperature and dew point (Magnus formula), percent.
    var humidity: Double? {
        guard let t = temperatureC, let td = d2m.map({ $0 - 273.15 }) else { return nil }
        let rh = 100 * exp(17.625 * td / (243.04 + td)) / exp(17.625 * t / (243.04 + t))
        return min(max(rh, 0), 100)
    }
    var isSnow: Bool {
        if let sf, let tp, tp > 0 { return sf >= 0.5 * tp }
        return (temperatureC ?? 10) <= 0.5
    }

    var condition: WeatherCondition { WeatherCondition(self) }
}

/// A short description and an SF Symbol, from cloud, precipitation, snow, wind and daylight.
nonisolated struct WeatherCondition: Sendable, Hashable {
    let text: String
    let symbol: String

    init(text: String, symbol: String) {
        self.text = text
        self.symbol = symbol
    }

    init(_ s: PointStep) {
        let mm = s.precipitationMM
        if mm >= 0.3 {
            if s.isSnow {
                self.init(text: mm >= 3 ? "Heavy Snow" : "Snow", symbol: "cloud.snow.fill")
            } else if mm >= 6 {
                self.init(text: "Heavy Rain", symbol: "cloud.heavyrain.fill")
            } else if mm >= 1.5 {
                self.init(text: "Rain", symbol: "cloud.rain.fill")
            } else {
                self.init(text: "Drizzle", symbol: "cloud.drizzle.fill")
            }
            return
        }
        let cloud = s.tcc ?? 0
        if let w = s.windSpeed, w >= 14 {
            self.init(text: "Windy", symbol: "wind")
        } else if cloud < 0.15 {
            self.init(text: s.isDay ? "Sunny" : "Clear", symbol: s.isDay ? "sun.max.fill" : "moon.stars.fill")
        } else if cloud < 0.45 {
            self.init(text: "Mostly Clear", symbol: s.isDay ? "cloud.sun.fill" : "cloud.moon.fill")
        } else if cloud < 0.8 {
            self.init(text: "Partly Cloudy", symbol: s.isDay ? "cloud.sun.fill" : "cloud.moon.fill")
        } else {
            self.init(text: "Cloudy", symbol: "cloud.fill")
        }
    }
}

nonisolated struct PointDay: Identifiable, Sendable {
    let id: Date  // local midnight
    let low: Double  // degC
    let high: Double
    let precipitationMM: Double
    let condition: WeatherCondition
}

/// The forecast for one place, from a stored forecast run.
nonisolated struct PointForecast: Sendable {
    let location: SavedLocation
    let steps: [PointStep]
    /// The model grid point the values come from, and how far it is from the place.
    var gridLatitude = 0.0
    var gridLongitude = 0.0
    var gridDistanceKm = 0.0

    static let fields = ["2t", "2d", "tp", "sf", "tcc", "10u", "10v", "msl"]

    /// The step valid nearest to `now`, the dashboard's "current conditions".
    func current(now: Date = .now) -> PointStep? {
        steps.min { abs($0.date.timeIntervalSince(now)) < abs($1.date.timeIntervalSince(now)) }
    }

    /// Calendar days in the place's own time zone. Each step belongs to the day its 6-hour
    /// window ends in, so one valid at exactly local midnight closes the day before -- its
    /// temperature and its precipitation both.
    var days: [PointDay] {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = location.timeZone
        let grouped = Dictionary(grouping: steps) { cal.startOfDay(for: $0.date.addingTimeInterval(-1)) }
        return grouped.keys.sorted().compactMap { day in
            let s = grouped[day]!
            let temps = s.compactMap(\.temperatureC)
            guard let low = temps.min(), let high = temps.max() else { return nil }
            let rain = s.reduce(0) { $0 + $1.precipitationMM }
            // The day's weather: its wettest step if it rains, else its daytime cloud.
            let daytime = s.filter(\.isDay)
            let representative = s.max { $0.precipitationMM < $1.precipitationMM }.flatMap { $0.precipitationMM >= 0.3 ? $0 : nil }
                ?? (daytime.isEmpty ? s : daytime).sorted { ($0.tcc ?? 0) < ($1.tcc ?? 0) }[(daytime.isEmpty ? s : daytime).count / 2]
            var shown = representative
            shown.isDay = true
            return PointDay(id: day, low: low, high: high, precipitationMM: rain, condition: shown.condition)
        }
    }

    /// Read every step's values at the grid point nearest `location`: a few bytes per
    /// field per step, straight from the step files.
    static func load(forecast: StoredForecast, location: SavedLocation, grid: ModelGrid) throws -> PointForecast {
        let index = grid.nearestIndex(latitude: location.latitude, longitude: location.longitude)
        let here = SIMD3<Float>(
            Float(cos(location.latitude * .pi / 180) * cos(location.longitude * .pi / 180)),
            Float(sin(location.latitude * .pi / 180)),
            Float(cos(location.latitude * .pi / 180) * sin(location.longitude * .pi / 180)))
        var steps: [PointStep] = []
        for s in stride(from: 1, through: forecast.meta.completedSteps, by: 1) {
            let v = try FieldFile.values(forecast.stepURL(s), fields: fields, at: [index])
            func value(_ name: String) -> Double? { v[name].map { Double($0[0]) }.flatMap { $0.isNaN ? nil : $0 } }
            let date = forecast.validDate(step: s)
            let sun = sunDirection(at: date)
            let elevation = Double(simd_dot(SIMD3(sun.x, sun.y, sun.z), here))
            steps.append(PointStep(
                date: date, t2m: value("2t"), d2m: value("2d"), tp: value("tp"), sf: value("sf"),
                tcc: value("tcc"), u10: value("10u"), v10: value("10v"), msl: value("msl"),
                isDay: elevation > -0.01))
        }
        let gLat = Double(grid.latitudes[index]), gLon = Double(grid.longitudes[index])
        let there = SIMD3<Float>(
            Float(cos(gLat * .pi / 180) * cos(gLon * .pi / 180)),
            Float(sin(gLat * .pi / 180)),
            Float(cos(gLat * .pi / 180) * sin(gLon * .pi / 180)))
        let km = Double(acos(min(1, simd_dot(here, there)))) * 6371
        return PointForecast(location: location, steps: steps, gridLatitude: gLat,
                             gridLongitude: gLon > 180 ? gLon - 360 : gLon, gridDistanceKm: km)
    }
}

/// The model grid, loaded once (4 MB of coordinates).
@MainActor
final class ModelGridCache {
    static let shared = ModelGridCache()
    private var grid: ModelGrid?

    func grid() async -> ModelGrid? {
        if let grid { return grid }
        let g = await Task.detached { try? ModelGrid(bundle: AppPaths.bundle) }.value
        grid = g
        return g
    }
}

// MARK: - Units

/// Point values formatted in the user's units (see `UnitStore`).
enum WeatherUnits {
    private static var units: UnitPreferences { UnitStore.shared.preferences }

    static func temperature(_ c: Double?) -> String {
        guard let c else { return "--" }
        return "\(Int(units.display(c, .temperature).rounded()))°"
    }

    /// Temperature in display units, for scales.
    static func temperatureValue(_ c: Double) -> Double { units.display(c, .temperature) }

    static func wind(_ ms: Double?) -> String {
        guard let ms else { return "--" }
        return "\(Int(units.display(ms, .speed).rounded())) \(units.speed.symbol)"
    }

    static func precipitation(_ mm: Double) -> String {
        if units.precipitation == .inches {
            let inches = mm / 25.4
            return inches < 0.01 ? "0 in" : String(format: "%.2f in", inches)
        }
        return mm < 0.1 ? "0 mm" : String(format: mm < 10 ? "%.1f mm" : "%.0f mm", mm)
    }

    /// Precipitation in display units (mm or inches), for chart scales.
    static func precipitationValue(_ mm: Double) -> Double { units.display(mm, .precipitation) }

    static func precipitationAxis(_ v: Double) -> String {
        units.precipitation == .inches ? String(format: "%.2g in", v) : String(format: "%.0f mm", v)
    }

    static func pressure(_ pa: Double?) -> String {
        guard let pa else { return "--" }
        let v = units.display(pa / 100, .pressure)
        return units.pressure == .hectopascals ? String(format: "%.0f hPa", v) : String(format: "%.2f inHg", v)
    }

    static func compass(_ degrees: Double?) -> String {
        guard let degrees else { return "" }
        let points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        return points[Int((degrees / 22.5).rounded()) % 16]
    }
}

// MARK: - Finding places

/// Search-as-you-type over MapKit's places.
@MainActor
@Observable
final class PlaceSearch: NSObject, MKLocalSearchCompleterDelegate {
    var query = "" {
        didSet { completer.queryFragment = query }
    }
    private(set) var results: [MKLocalSearchCompletion] = []
    private let completer = MKLocalSearchCompleter()

    override init() {
        super.init()
        completer.resultTypes = .address
        completer.delegate = self
    }

    nonisolated func completerDidUpdateResults(_ completer: MKLocalSearchCompleter) {
        let results = completer.results
        MainActor.assumeIsolated { self.results = results }
    }

    nonisolated func completer(_ completer: MKLocalSearchCompleter, didFailWithError error: Error) {}

    /// Resolve a completion to a place with coordinates and a time zone.
    func resolve(_ completion: MKLocalSearchCompletion) async throws -> SavedLocation? {
        let response = try await MKLocalSearch(request: MKLocalSearch.Request(completion: completion)).start()
        guard let item = response.mapItems.first else { return nil }
        let c = item.location.coordinate
        return SavedLocation(
            name: item.addressRepresentations?.cityName ?? completion.title,
            detail: completion.subtitle.isEmpty ? nil : completion.subtitle,
            latitude: c.latitude, longitude: c.longitude, timeZoneID: item.timeZone?.identifier)
    }
}

/// One-shot "where am I", with the place named by reverse geocoding.
@MainActor
final class CurrentLocationFinder: NSObject, CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private var continuation: CheckedContinuation<CLLocation, Error>?

    func locate() async throws -> SavedLocation {
        let location: CLLocation = try await withCheckedThrowingContinuation { c in
            continuation = c
            manager.delegate = self
            manager.desiredAccuracy = kCLLocationAccuracyKilometer
            if manager.authorizationStatus == .notDetermined {
                manager.requestWhenInUseAuthorization()
            } else {
                manager.requestLocation()
            }
        }
        var name = "My Location"
        var tz: String?
        if let request = MKReverseGeocodingRequest(location: location),
            let item = try? await request.mapItems.first
        {
            name = item.addressRepresentations?.cityName ?? item.name ?? name
            tz = item.timeZone?.identifier
        }
        return SavedLocation(
            name: name, detail: "My Location", latitude: location.coordinate.latitude,
            longitude: location.coordinate.longitude, timeZoneID: tz, isCurrentLocation: true)
    }

    nonisolated func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        MainActor.assumeIsolated {
            switch manager.authorizationStatus {
            case .authorizedWhenInUse, .authorizedAlways: manager.requestLocation()
            case .denied, .restricted:
                continuation?.resume(throwing: CLError(.denied))
                continuation = nil
            default: break
            }
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        MainActor.assumeIsolated {
            guard let l = locations.last else { return }
            continuation?.resume(returning: l)
            continuation = nil
        }
    }

    nonisolated func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        MainActor.assumeIsolated {
            continuation?.resume(throwing: error)
            continuation = nil
        }
    }
}
