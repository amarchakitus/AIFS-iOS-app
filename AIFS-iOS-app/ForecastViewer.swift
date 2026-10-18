import AIFSKit
import SwiftUI

/// One stored forecast, three ways: a flat map (live while the run is generating), and,
/// once it has finished, globes and a per-place dashboard.
struct ForecastViewer: View {
    let forecast: StoredForecast
    /// Changes whenever a running forecast writes another step.
    let revision: Int
    /// The run writing this forecast is still going.
    let isGenerating: Bool

    enum Mode: String, CaseIterable, Identifiable {
        case map = "Map", globe = "Globe", places = "Places"
        var id: String { rawValue }
    }
    /// The last tab used, kept across forecasts and launches: leaving the globe for the
    /// list and opening a forecast again comes back to the globe.
    @AppStorage("viewer.mode") private var mode: Mode = .map

    var body: some View {
        VStack(spacing: 0) {
            // Generating shows the map without forgetting the tab you chose.
            Picker("View", selection: Binding(get: { isGenerating ? .map : mode }, set: { mode = $0 })) {
                ForEach(Mode.allCases) { m in
                    Text(m.rawValue).tag(m)
                        .selectionDisabled(isGenerating && m != .map)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.top, 2)
            if isGenerating && mode == .map {
                Text("Globe and Places open when the forecast finishes.")
                    .font(.caption).foregroundStyle(.secondary).padding(.top, 6)
            }
            switch isGenerating ? .map : mode {
            case .map: FlatMapViewer(forecast: forecast, revision: revision)
            case .globe: GlobeScreen(forecast: forecast)
            case .places: WeatherDashboard(forecast: forecast)
            }
        }
        // The page's own surface behind the selector too, so Places' grey doesn't start
        // in a hard edge just under it.
        .background((mode == .places && !isGenerating ? Color(.systemGroupedBackground) : Color(.systemBackground)).ignoresSafeArea())
        .navigationTitle("Init \(forecast.meta.initDate.utcLabel)")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            if let name = LaunchOptions.viewerMode,
               let m = Mode.allCases.first(where: { $0.rawValue.lowercased() == name }) {
                mode = m  // development: open on a given tab
            }
        }
    }
}

/// The flat map: pick a field, scrub or play through lead times, drag to read values.
/// Kept simple on purpose -- it is what you watch while a run is generating.
private struct FlatMapViewer: View {
    let forecast: StoredForecast
    let revision: Int

    @State private var field = DisplayField.all[0]
    @State private var step = 1
    @State private var frames = MapFrames()
    @State private var readout: (lat: Double, lon: Double, value: Float)?
    @State private var playing = false

    private var completed: Int { forecast.meta.completedSteps }

    /// Fields this forecast saved (older runs lack, e.g., 850 hPa wind).
    private var availableFields: [DisplayField] {
        DisplayField.all.filter { $0.sources.allSatisfy(forecast.meta.fields.contains) }
    }

    var body: some View {
        // Map, legend and time controls stay together at the top; in portrait the 2:1 map
        // is width-bound and the space below is simply left empty.
        VStack(spacing: 16) {
            header
            mapView
            Legend(field: field)
            timeControls
            Spacer(minLength: 0)
        }
        .padding()
        .toolbar {
            ToolbarItem {
                Picker("Field", selection: $field) {
                    ForEach(availableFields) { Text($0.title).tag($0) }
                }
                .pickerStyle(.menu)
            }
        }
        .onChange(of: FrameKey(forecast: forecast.id, field: field.id, step: step), initial: true) { _, key in
            frames.request(key, forecast: forecast, field: field, completed: completed)
        }
        .onChange(of: revision) {
            // A running forecast wrote another step: the requested one may exist now.
            frames.request(FrameKey(forecast: forecast.id, field: field.id, step: step),
                           forecast: forecast, field: field, completed: completed)
        }
        .task(id: playing) {
            while playing, !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(600))
                step = step >= completed ? 1 : step + 1
            }
        }
        .onChange(of: forecast.id) { step = 1 }
    }

    @ViewBuilder private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(field.title).font(.title2.weight(.semibold))
            Spacer()
            if let r = readout {
                Text(
                    field.readout(r.value) + String(format: "  at %.1f°%@ %.1f°%@",
                        abs(r.lat), r.lat >= 0 ? "N" : "S", abs(r.lon), r.lon >= 0 ? "E" : "W")
                )
                .monospacedDigit()
                .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var mapView: some View {
        // A clear 2:1 box sized by the available width, so the legend and controls stay put
        // while there is only a spinner to show.
        Color.clear.aspectRatio(2, contentMode: .fit).overlay {
            if let map = frames.shown {
                GeometryReader { geo in
                    Image(decorative: map.image, scale: 1)
                        .resizable()
                        .interpolation(.medium)
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { g in
                                    readout = map.sample(
                                        u: g.location.x / geo.size.width, v: g.location.y / geo.size.height)
                                }
                                .onEnded { _ in readout = nil })
                }
            } else if let renderError = frames.error {
                ContentUnavailableView(
                    "Cannot draw this step", systemImage: "exclamationmark.triangle",
                    description: Text(renderError))
            } else {
                ProgressView(completed == 0 ? "Generating first step…" : "")
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    @ViewBuilder private var timeControls: some View {
        HStack(spacing: 16) {
            Button {
                playing.toggle()
            } label: {
                Image(systemName: playing ? "pause.fill" : "play.fill").frame(width: 24)
            }
            .disabled(completed < 2)
            .accessibilityLabel(playing ? "Pause" : "Play")
            if completed > 1 {
                Slider(
                    value: Binding(get: { Double(step) }, set: { step = Int($0.rounded()) }),
                    in: 1 ... Double(completed), step: 1)
            } else {
                Spacer()
            }
            VStack(alignment: .trailing) {
                Text("+\(step * forecast.meta.stepHours) h").font(.headline).monospacedDigit()
                Text(forecast.validDate(step: step).utcLabel).font(.caption).foregroundStyle(.secondary)
            }
            .frame(minWidth: 130, alignment: .trailing)
        }
    }

}

/// One rendered frame of the flat map.
private struct FrameKey: Hashable {
    let forecast: URL
    let field: String
    let step: Int
}

/// The flat map's frames: rendered one at a time, always the newest request first, so a
/// fast scrub never queues a backlog and a slow frame finishing late can never replace a
/// newer one on screen. Recent frames are kept and the steps either side are rendered
/// ahead, so scrubbing back and forth or playing is mostly cache hits.
@MainActor @Observable
private final class MapFrames {
    private(set) var shown: RenderedMap?
    private(set) var error: String?

    private var wanted: FrameKey?
    private var context: (forecast: StoredForecast, field: DisplayField, completed: Int)?
    private var cache: [FrameKey: RenderedMap] = [:]
    private var order: [FrameKey] = []  // least recently used first
    private var working = false
    private static let capacity = 12  // ~8 MB a frame (image + values)

    func request(_ key: FrameKey, forecast: StoredForecast, field: DisplayField, completed: Int) {
        wanted = key
        context = (forecast, field, completed)
        guard completed >= 1 else {
            // The run has not written its first step yet: the spinner, not an error.
            shown = nil
            error = nil
            return
        }
        if let hit = cache[key] {
            touch(key)
            shown = hit
            error = nil
        }
        if !working { Task { await work() } }
    }

    /// Render whatever is most useful next until nothing is: the wanted frame, then its
    /// neighbours, nearest first.
    private func work() async {
        working = true
        defer { working = false }
        while let job = nextJob() {
            let (key, forecast, field) = job
            do {
                let map = try await Task.detached(priority: .userInitiated) {
                    try Self.render(forecast: forecast, field: field, step: key.step)
                }.value
                if let map { store(map, for: key) }
                if key == wanted {
                    shown = map
                    error = nil
                }
            } catch {
                if key == wanted { self.error = error.localizedDescription }
                return
            }
        }
    }

    private func nextJob() -> (FrameKey, StoredForecast, DisplayField)? {
        guard let wanted, let context, context.completed >= 1 else { return nil }
        let (forecast, field, completed) = context
        let candidates = [0, 1, -1, 2, -2, 3].map { wanted.step + $0 }.filter { (1 ... completed).contains($0) }
        for s in candidates {
            let key = FrameKey(forecast: wanted.forecast, field: wanted.field, step: s)
            if cache[key] == nil { return (key, forecast, field) }
        }
        return nil
    }

    private func store(_ map: RenderedMap, for key: FrameKey) {
        cache[key] = map
        touch(key)
        while order.count > Self.capacity { cache[order.removeFirst()] = nil }
    }

    private func touch(_ key: FrameKey) {
        order.removeAll { $0 == key }
        order.append(key)
    }

    nonisolated private static func render(forecast: StoredForecast, field: DisplayField, step: Int) throws -> RenderedMap? {
        let renderer = try RendererCache.shared.renderer()
        // Only the field's own inputs, by byte range: a step file holds every stored output.
        let fields = try FieldFile.read(forecast.stepURL(step), fields: field.sources)
        let sources = try field.sources.map { name in
            guard let f = fields[name] else { throw CocoaError(.fileReadCorruptFile) }
            return f
        }
        let coast = RendererCache.shared.coastline(for: forecast, renderer: renderer)
        return renderer.render(field: field, sources: sources, coastline: coast)
    }
}

/// One renderer (28 MB display operator) and one coastline per forecast, shared.
nonisolated final class RendererCache: @unchecked Sendable {
    static let shared = RendererCache()
    private let lock = NSLock()
    private var cached: MapRenderer?
    private var coastlines: [URL: [Bool]] = [:]

    func renderer() throws -> MapRenderer {
        lock.lock()
        defer { lock.unlock() }
        if let cached { return cached }
        let r = try MapRenderer(bundle: AppPaths.bundle)
        cached = r
        return r
    }

    func coastline(for forecast: StoredForecast, renderer: MapRenderer) -> [Bool]? {
        lock.lock()
        if let c = coastlines[forecast.id] { lock.unlock(); return c }
        lock.unlock()
        guard let lsm = try? FieldFile.read(forecast.staticURL)["lsm"] else { return nil }
        let c = renderer.coastline(lsm: lsm)
        lock.lock()
        coastlines[forecast.id] = c
        lock.unlock()
        return c
    }
}

/// Horizontal color bar with tick labels in text ink, in the user's units, units on the
/// right. Each tick sits where its value falls on the ramp, so it is aligned to the bar.
struct Legend: View {
    let field: DisplayField
    var lut: ColorLUT?
    private var units = UnitStore.shared

    init(field: DisplayField, lut: ColorLUT? = nil) {
        self.field = field
        self.lut = lut
    }

    var body: some View {
        let lut = lut ?? field.scale.lut
        let ticks = LegendTick.ticks(for: field, units: units.preferences)
        HStack(alignment: .bottom, spacing: 8) {
            VStack(spacing: 4) {
                LinearGradient(
                    stops: (0 ... 32).map { k in
                        let p = Float(k) / 32
                        return .init(color: lut.swiftUIColor(p), location: Double(p))
                    }, startPoint: .leading, endPoint: .trailing
                )
                .frame(height: 12)
                .clipShape(RoundedRectangle(cornerRadius: 3))
                TickRow(positions: ticks.map(\.position)) {
                    ForEach(ticks) { Text($0.label) }
                }
            }
            Text(field.displayUnits).foregroundStyle(.secondary)
        }
        .font(.caption)
        .monospacedDigit()
        .frame(maxWidth: 640)
    }
}

/// Places each label centred on its ramp position, kept inside the bar at the ends.
private struct TickRow: Layout {
    let positions: [Double]

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let height = subviews.map { $0.sizeThatFits(.unspecified).height }.max() ?? 0
        return CGSize(width: proposal.width ?? 240, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for (view, p) in zip(subviews, positions) {
            let size = view.sizeThatFits(.unspecified)
            let x = min(max(bounds.minX + p * bounds.width - size.width / 2, bounds.minX), bounds.maxX - size.width)
            view.place(at: CGPoint(x: x, y: bounds.minY), proposal: ProposedViewSize(size))
        }
    }
}
