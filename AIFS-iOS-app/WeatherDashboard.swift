import AIFSKit
import Charts
import MapKit
import SwiftUI

/// Places and their forecasts from one finished run, laid out like the Weather app.
struct WeatherDashboard: View {
    let forecast: StoredForecast

    private var store = LocationStore.shared
    @State private var points: [UUID: PointForecast] = [:]
    @State private var adding = false
    @State private var shown: SavedLocation?

    init(forecast: StoredForecast) {
        self.forecast = forecast
    }

    var body: some View {
        ScrollView {
            LazyVStack(spacing: 10) {
                if store.locations.isEmpty {
                    ContentUnavailableView {
                        Label("No places yet", systemImage: "mappin.and.ellipse")
                    } description: {
                        Text("Add a place to see this forecast for it.")
                    } actions: {
                        Button("Add a place") { adding = true }.buttonStyle(.borderedProminent)
                    }
                    .padding(.top, 40)
                }
                ForEach(store.locations) { location in
                    Button {
                        shown = location
                    } label: {
                        LocationCard(location: location, point: points[location.id])
                    }
                    .buttonStyle(.plain)
                    .contextMenu {
                        Button("Remove", systemImage: "trash", role: .destructive) { store.remove(location) }
                    }
                }
            }
            .padding()
        }
        .background(Color(.systemGroupedBackground))
        .toolbar {
            ToolbarItem {
                Button {
                    adding = true
                } label: {
                    Image(systemName: "plus")
                }
                .accessibilityLabel("Add a place")
            }
        }
        .sheet(isPresented: $adding) {
            AddLocationSheet { store.add($0) }
        }
        .sheet(item: $shown) { location in
            LocationDetail(location: location, point: points[location.id], forecast: forecast)
                .presentationDragIndicator(.visible)
        }
        .task(id: "\(forecast.id.lastPathComponent)|\(store.locations.map(\.id))") { await load() }
    }

    private func load() async {
        guard let grid = await ModelGridCache.shared.grid(for: forecast.model) else { return }
        for location in store.locations where points[location.id] == nil {
            let forecast = forecast
            if let p = await Task.detached(priority: .userInitiated, operation: {
                try? PointForecast.load(forecast: forecast, location: location, grid: grid)
            }).value {
                points[location.id] = p
            }
        }
        if LaunchOptions.openPlace, shown == nil {  // development: open a place directly
            shown = store.locations.first
        }
    }
}

// MARK: - Card

/// Series colours: warm for air temperature, cool for moisture. Text never wears them.
private let temperatureColor = Color(red: 0.85, green: 0.38, blue: 0.23)
private let dewPointColor = Color(red: 0.17, green: 0.56, blue: 0.75)
private let snowColor = Color(red: 0.56, green: 0.49, blue: 0.77)

private struct LocationCard: View {
    let location: SavedLocation
    let point: PointForecast?

    var body: some View {
        let now = point?.current()
        HStack(spacing: 14) {
            ConditionIcon(symbol: now?.condition.symbol)
                .font(.title2)
                .frame(width: 36)
            VStack(alignment: .leading, spacing: 3) {
                // A line each: name, condition, local time. Sharing lines squeezed and cut
                // them off beside the range band.
                HStack(alignment: .firstTextBaseline, spacing: 5) {
                    Text(location.name).font(.headline).lineLimit(1)
                    if location.isCurrentLocation {
                        Image(systemName: "location.fill").font(.caption2).foregroundStyle(.secondary)
                    }
                }
                Text(now?.condition.text ?? "Loading…")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                if let now {
                    Text(localTime(now.date)).font(.caption).foregroundStyle(.tertiary).monospacedDigit()
                }
            }
            .layoutPriority(1)  // the name before the range band
            Spacer(minLength: 8)
            if let point, point.days.count > 1 {
                let days = point.days
                VStack(spacing: 3) {
                    RangeBand(days: days)
                        .frame(width: 72, height: 24)
                    // Says what the band is: the daily low-to-high over these days.
                    Text("\(days.count) days · \(WeatherUnits.temperature(days.map(\.low).min()))–\(WeatherUnits.temperature(days.map(\.high).max()))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                        .lineLimit(1)
                        .fixedSize()
                }
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Daily lows and highs over \(days.count) days")
            }
            Text(WeatherUnits.temperature(now?.temperatureC))
                .font(.system(.title, design: .rounded).weight(.semibold))
                .monospacedDigit()
                .frame(minWidth: 58, alignment: .trailing)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func localTime(_ date: Date) -> String {
        let f = DateFormatter()
        f.timeZone = location.timeZone
        f.dateFormat = "EEE HH:mm"
        return f.string(from: date)
    }
}

/// The run's daily lows and highs as one band: the trend at a glance, without the
/// day-night zigzag a 6-hourly line draws at this size.
private struct RangeBand: View {
    let days: [PointDay]

    var body: some View {
        let low = days.map(\.low).min() ?? 0, high = days.map(\.high).max() ?? 1
        Chart(days) { d in
            AreaMark(x: .value("Day", d.id), yStart: .value("Low", d.low), yEnd: .value("High", d.high))
                .interpolationMethod(.monotone)
                .foregroundStyle(temperatureColor.opacity(0.28))
            LineMark(x: .value("Day", d.id), y: .value("High", d.high))
                .interpolationMethod(.monotone)
                .lineStyle(StrokeStyle(lineWidth: 1.5, lineCap: .round))
                .foregroundStyle(temperatureColor)
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartYScale(domain: (low - 0.5) ... (high + 0.5))
    }
}

/// A condition symbol that reads on a light card or a dark one. Multicolour draws clouds
/// and the moon white, which vanish on white, so clouds are grey and each symbol's
/// accent (sun, moon, rain, snow) keeps its colour.
private struct ConditionIcon: View {
    let symbol: String?

    var body: some View {
        let name = symbol ?? "circle.dashed"
        let cloud = Color(.systemGray2)
        if name.hasPrefix("cloud.") && name != "cloud.fill" {
            Image(systemName: name)
                .symbolRenderingMode(.palette)
                .foregroundStyle(cloud, accent(name))
        } else if name == "cloud.fill" {
            Image(systemName: name).foregroundStyle(cloud)
        } else if name.contains("moon") {
            Image(systemName: name).symbolRenderingMode(.hierarchical).foregroundStyle(.indigo)
        } else {
            Image(systemName: name).symbolRenderingMode(.multicolor)
        }
    }

    private func accent(_ name: String) -> Color {
        if name.contains("sun") { return .yellow }
        if name.contains("moon") { return .indigo }
        if name.contains("snow") { return .cyan }
        return .blue  // rain, drizzle, heavy rain
    }
}

// MARK: - Detail

private struct LocationDetail: View {
    let location: SavedLocation
    let point: PointForecast?
    let forecast: StoredForecast
    @State private var selected: Date?

    var body: some View {
        let now = point?.current()
        ScrollView {
                VStack(spacing: 14) {
                    header(now)
                    if let point {
                        temperatureCard(point, now: now)
                        precipitationCard(point)
                        dailyCard(point)
                        conditionsCard(point, now: now)
                        Text(footnote(point))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.top, 4)
                    } else {
                        ProgressView().padding(40)
                    }
                }
                .padding()
                .padding(.top, 14)  // clear of the grabber
        }
        // A sheet keeps a toolbar's height clear at the top even with no bar in it.
        .ignoresSafeArea(.container, edges: .top)
        .background(Color(.systemGroupedBackground))
    }

    // No navigation bar: the sheet is swiped away, and the name heads the conditions
    // directly, its left edge on the icon's.
    private func header(_ now: PointStep?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(location.name)
                .font(.title.weight(.bold))
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            headerConditions(now)
        }
        .padding(.horizontal, 4)
    }

    private func headerConditions(_ now: PointStep?) -> some View {
        HStack(spacing: 14) {
            ConditionIcon(symbol: now?.condition.symbol)
                .font(.system(size: 46))
            VStack(alignment: .leading, spacing: 0) {
                Text(WeatherUnits.temperature(now?.temperatureC))
                    .font(.system(size: 54, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                Text(now?.condition.text ?? " ").font(.title3).foregroundStyle(.secondary)
            }
            Spacer()
            if let now, let point, let today = point.days.first(where: { sameLocalDay($0.id, now.date) }) {
                VStack(alignment: .trailing, spacing: 4) {
                    Text(formatter("EEE HH:mm").string(from: now.date)).font(.subheadline.weight(.medium))
                    Text("\(WeatherUnits.temperature(today.low)) to \(WeatherUnits.temperature(today.high))")
                        .font(.subheadline).foregroundStyle(.secondary).monospacedDigit()
                }
            }
        }
    }

    // Temperature and dew point share a unit, so they share the one axis.
    private func temperatureCard(_ point: PointForecast, now: PointStep?) -> some View {
        let rows = point.steps.flatMap { s -> [(date: Date, series: String, value: Double)] in
            [s.temperatureC.map { (s.date, "Temperature", WeatherUnits.temperatureValue($0)) },
             s.d2m.map { (s.date, "Dew point", WeatherUnits.temperatureValue($0 - 273.15)) }].compactMap { $0 }
        }
        let pick = selected.flatMap { d in point.steps.min { abs($0.date.timeIntervalSince(d)) < abs($1.date.timeIntervalSince(d)) } }
        return Card(title: "Temperature") {
            LegendRow(items: [("Air", temperatureColor), ("Dew point", dewPointColor)])
        } content: {
            Chart {
                ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                    LineMark(x: .value("Time", r.date), y: .value("Degrees", r.value))
                        .foregroundStyle(by: .value("Series", r.series == "Temperature" ? "Air" : r.series))
                        .interpolationMethod(.catmullRom)
                        .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round))
                }
                if let pick {
                    RuleMark(x: .value("Selected", pick.date))
                        .foregroundStyle(.secondary.opacity(0.6))
                        .lineStyle(StrokeStyle(lineWidth: 1))
                        .annotation(position: .top, spacing: 4,
                                    overflowResolution: .init(x: .fit(to: .chart), y: .disabled)) {
                            VStack(alignment: .leading, spacing: 1) {
                                Text(formatter("EEE HH:mm").string(from: pick.date)).foregroundStyle(.secondary)
                                Text("Air \(WeatherUnits.temperature(pick.temperatureC)) · dew \(WeatherUnits.temperature(pick.d2m.map { $0 - 273.15 }))")
                            }
                            .font(.caption.monospacedDigit())
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .background(Color(.tertiarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 8))
                        }
                } else if let now {
                    RuleMark(x: .value("Now", now.date))
                        .foregroundStyle(.secondary.opacity(0.35))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [3, 3]))
                }
            }
            .chartForegroundStyleScale(["Air": temperatureColor, "Dew point": dewPointColor])
            .chartLegend(.hidden)
            .chartXSelection(value: $selected)
            .chartYScale(domain: .automatic(includesZero: false))
            .chartXAxis { dayAxis(point) }
            .chartXScale(domain: span(point))
            .chartYAxis {
                AxisMarks(position: .leading, values: .automatic(desiredCount: 4)) { v in
                    AxisGridLine().foregroundStyle(.quaternary)
                    AxisValueLabel { if let d = v.as(Double.self) { Text("\(Int(d))°").font(.caption) } }
                }
            }
            .frame(height: 190)
        }
    }

    private func precipitationCard(_ point: PointForecast) -> some View {
        let total = point.steps.reduce(0) { $0 + $1.precipitationMM }
        let wet = point.steps.filter { $0.precipitationMM >= 0.05 }
        return Card(title: "Precipitation", trailing: wet.isEmpty ? nil : "\(WeatherUnits.precipitation(total)) over the run") {
            if wet.isEmpty {
                Text("None expected in this forecast.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                // Only wet steps: a zero-height bar still draws as a pill.
                Chart(wet, id: \.date) { s in
                    // Each total is the 6 h ending at its step; the bar spans that window,
                    // less a sliver so neighbouring bars stay separate. A RectangleMark with
                    // both ranges: a BarMark given an x range and a bare `y` draws a thin
                    // horizontal span at that height, not a column from zero.
                    let from: Date = s.date.addingTimeInterval(-6 * 3600 + 900)
                    let to: Date = s.date.addingTimeInterval(-900)
                    let amount: Double = WeatherUnits.precipitationValue(s.precipitationMM)
                    let kind: String = s.isSnow ? "Snow" : "Rain"
                    RectangleMark(
                        xStart: .value("From", from), xEnd: .value("To", to),
                        yStart: .value("None", 0.0), yEnd: .value("Amount", amount))
                        .foregroundStyle(by: .value("Type", kind))
                        .cornerRadius(1.5)
                }
                .chartForegroundStyleScale(["Rain": dewPointColor, "Snow": snowColor])
                .chartLegend(.hidden)
                .chartXAxis { dayAxis(point) }
            .chartXScale(domain: span(point))
                .chartYAxis {
                    AxisMarks(position: .leading, values: .automatic(desiredCount: 3)) { v in
                        AxisGridLine().foregroundStyle(.quaternary)
                        AxisValueLabel { if let d = v.as(Double.self) { Text(WeatherUnits.precipitationAxis(d)).font(.caption) } }
                    }
                }
                .frame(height: 110)
                if wet.contains(where: \.isSnow) {
                    LegendRow(items: [("Rain", dewPointColor), ("Snow", snowColor)])
                }
            }
        }
    }

    /// A gridline at every local midnight; labels only every few days on long runs,
    /// so a 15-day forecast gets about six instead of sixteen piled together. Spaced
    /// labels carry the date ("Wed 21"): a weekday alone is ambiguous across weeks.
    private func dayAxis(_ point: PointForecast) -> some AxisContent {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = location.timeZone
        var days: [Date] = []
        if let first = point.steps.first?.date, let last = point.steps.last?.date {
            var d = cal.startOfDay(for: first)
            if d < first { d = cal.date(byAdding: .day, value: 1, to: d)! }
            while d <= last {
                days.append(d)
                d = cal.date(byAdding: .day, value: 1, to: d)!
            }
        }
        let every = max(1, Int((Double(days.count) / 6).rounded(.up)))
        let labelled = Set(days.enumerated().filter { $0.offset % every == 0 }.map(\.element))
        let format = formatter(every == 1 ? "EEE" : "EEE d")
        return AxisMarks(values: days) { value in
            AxisGridLine().foregroundStyle(.quaternary)
            AxisValueLabel(anchor: .topLeading) {
                let date = value.as(Date.self)
                Text(date.map { labelled.contains($0) ? format.string(from: $0) : "" } ?? "").font(.caption)
            }
        }
    }

    private func dailyCard(_ point: PointForecast) -> some View {
        // From today on: the run's first day is often already over by the time it's read.
        let all = point.days
        let days = Array(all.drop { day in !isToday(day, point) && all.contains { isToday($0, point) } && day.id < (point.current()?.date ?? .distantPast) })
        return Card(title: "By day", trailing: nil) {
            Grid(horizontalSpacing: 12, verticalSpacing: 0) {
                ForEach(Array(days.enumerated()), id: \.element.id) { i, day in
                    if i > 0 { Divider().gridCellUnsizedAxes(.horizontal) }
                    GridRow {
                        Text(isToday(day, point) ? "Today" : formatter("EEEE").string(from: day.id))
                            .gridColumnAlignment(.leading)
                        ConditionIcon(symbol: day.condition.symbol)
                            .frame(width: 28)
                        Text(day.precipitationMM >= 1 ? WeatherUnits.precipitation(day.precipitationMM) : "")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .trailing)
                        Text(WeatherUnits.temperature(day.low))
                            .foregroundStyle(.secondary)
                            .frame(width: 40, alignment: .trailing)
                        Text(WeatherUnits.temperature(day.high))
                            .fontWeight(.semibold)
                            .frame(width: 40, alignment: .trailing)
                    }
                    .monospacedDigit()
                    .padding(.vertical, 9)
                }
            }
        }
    }

    private func conditionsCard(_ point: PointForecast, now: PointStep?) -> some View {
        let next24 = point.steps.filter { $0.date > (now?.date ?? .distantPast) }.prefix(4)
            .reduce(0) { $0 + $1.precipitationMM }
        let wind = now.map { s in
            [WeatherUnits.wind(s.windSpeed), s.windFrom.map { "from \(WeatherUnits.compass($0))" }]
                .compactMap { $0 }.joined(separator: " ")
        }
        let items: [(String, String, String)] = [
            ("wind", "Wind", wind ?? "--"),
            ("humidity", "Humidity", now?.humidity.map { "\(Int($0.rounded()))%" } ?? "--"),
            ("thermometer.medium", "Dew point", WeatherUnits.temperature(now?.d2m.map { $0 - 273.15 })),
            ("gauge.with.dots.needle.33percent", "Pressure", WeatherUnits.pressure(now?.msl)),
            ("cloud", "Cloud cover", now?.tcc.map { "\(Int(($0 * 100).rounded()))%" } ?? "--"),
            ("drop", "Next 24 h", WeatherUnits.precipitation(next24)),
        ]
        return Card(title: "Conditions", trailing: now.map { formatter("EEE HH:mm").string(from: $0.date) }) {
            LazyVGrid(columns: [GridItem(.flexible(), alignment: .leading), GridItem(.flexible(), alignment: .leading)],
                      alignment: .leading, spacing: 14) {
                ForEach(items, id: \.1) { icon, label, value in
                    VStack(alignment: .leading, spacing: 2) {
                        Label(label, systemImage: icon).font(.caption).foregroundStyle(.secondary)
                        Text(value).font(.body.weight(.medium)).monospacedDigit()
                    }
                }
            }
        }
    }

    private func footnote(_ point: PointForecast) -> String {
        let lat = point.gridLatitude, lon = point.gridLongitude
        let at = String(format: "%.2f°%@ %.2f°%@", abs(lat), lat >= 0 ? "N" : "S", abs(lon), lon >= 0 ? "E" : "W")
        let km = Int(point.gridDistanceKm.rounded())
        return "AIFS Single v2, init \(forecast.meta.initDate.utcLabel)\nModel grid point \(at), \(km) km away"
    }

    /// The whole run, so both charts share one time axis whatever they plot.
    private func span(_ point: PointForecast) -> ClosedRange<Date> {
        let first = point.steps.first?.date ?? .now
        return first ... max(point.steps.last?.date ?? first, first.addingTimeInterval(3600))
    }

    private func sameLocalDay(_ a: Date, _ b: Date) -> Bool {
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = location.timeZone
        return cal.isDate(a, inSameDayAs: b)
    }

    /// "Today" is the day of the step shown as now, not the run's first day, which can
    /// be yesterday by the time it is read.
    private func isToday(_ day: PointDay, _ point: PointForecast) -> Bool {
        guard let now = point.current() else { return false }
        return sameLocalDay(day.id, now.date)
    }

    private func formatter(_ format: String) -> DateFormatter {
        let f = DateFormatter()
        f.timeZone = location.timeZone
        f.dateFormat = format
        return f
    }
}

/// A grouped card with a title and something right-aligned beside it: a note or a legend.
private struct Card<Accessory: View, Content: View>: View {
    let title: String
    @ViewBuilder let accessory: Accessory
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                Text(title).font(.headline)
                Spacer()
                accessory
            }
            content
        }
        .padding(16)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}

extension Card where Accessory == Text? {
    init(title: String, trailing: String?, @ViewBuilder content: () -> Content) {
        self.title = title
        accessory = trailing.map { Text($0).font(.caption).foregroundStyle(.secondary).monospacedDigit() }
        self.content = content()
    }
}

/// Series names with their colours, for a card header.
private struct LegendRow: View {
    let items: [(String, Color)]

    var body: some View {
        HStack(spacing: 12) {
            ForEach(items, id: \.0) { name, color in
                HStack(spacing: 5) {
                    Capsule().fill(color).frame(width: 12, height: 3)
                    Text(name).font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }
}

// MARK: - Adding a place

private struct AddLocationSheet: View {
    let onAdd: (SavedLocation) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var search = PlaceSearch()
    @State private var locating = false
    @State private var finder = CurrentLocationFinder()
    @State private var error: String?

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button {
                        locating = true
                        Task {
                            do {
                                onAdd(try await finder.locate())
                                dismiss()
                            } catch {
                                self.error = "Couldn't get your location. Check Location access in Settings."
                            }
                            locating = false
                        }
                    } label: {
                        HStack {
                            Label("Use my current location", systemImage: "location.fill")
                            if locating { Spacer(); ProgressView() }
                        }
                    }
                    if let error { Text(error).font(.footnote).foregroundStyle(.secondary) }
                }
                Section {
                    ForEach(search.results, id: \.self) { result in
                        Button {
                            Task {
                                if let place = try? await search.resolve(result) {
                                    onAdd(place)
                                    dismiss()
                                }
                            }
                        } label: {
                            VStack(alignment: .leading) {
                                Text(result.title).foregroundStyle(.primary)
                                if !result.subtitle.isEmpty {
                                    Text(result.subtitle).font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
            .searchable(text: $search.query, placement: .navigationBarDrawer(displayMode: .always), prompt: "City or address")
            .navigationTitle("Add a place")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            }
        }
    }
}
