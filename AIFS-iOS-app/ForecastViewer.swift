import AIFSKit
import SwiftUI

/// Map of one stored forecast: pick a field, scrub or play through lead times, drag on
/// the map to read the value under your finger.
struct ForecastViewer: View {
    let forecast: StoredForecast
    /// Changes whenever a running forecast writes another step.
    let revision: Int

    @State private var field = DisplayField.all[0]
    @State private var step = 1
    @State private var map: RenderedMap?
    @State private var readout: (lat: Double, lon: Double, value: Float)?
    @State private var playing = false
    @State private var renderError: String?

    private var completed: Int { forecast.meta.completedSteps }

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
        .navigationTitle("Init \(forecast.meta.initDate.utcLabel)")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem {
                Picker("Field", selection: $field) {
                    ForEach(DisplayField.all) { Text($0.title).tag($0) }
                }
                .pickerStyle(.menu)
            }
        }
        .task(id: RenderKey(forecast: forecast.id, field: field.id, step: step, revision: revision)) {
            await render()
        }
        .task(id: playing) {
            while playing, !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(600))
                step = step >= completed ? 1 : step + 1
            }
        }
        .onChange(of: forecast.id) { step = 1 }
    }

    private struct RenderKey: Hashable {
        let forecast: URL
        let field: String
        let step: Int
        let revision: Int
    }

    @ViewBuilder private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(field.title).font(.title2.weight(.semibold))
            Spacer()
            if let r = readout {
                Text(
                    String(format: "%.1f %@  at %.1f°%@ %.1f°%@", r.value, field.units,
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
            if let map {
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
            } else if let renderError {
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

    private func render() async {
        guard completed >= 1 else {
            // The run has not written its first step yet: show the spinner, not an error.
            map = nil
            renderError = nil
            return
        }
        let s = min(max(step, 1), completed)
        let forecast = forecast, field = field
        do {
            let rendered = try await Task.detached(priority: .userInitiated) {
                let renderer = try RendererCache.shared.renderer()
                let fields = try FieldFile.read(forecast.stepURL(s))
                let sources = try field.sources.map { name in
                    guard let f = fields[name] else { throw CocoaError(.fileReadCorruptFile) }
                    return f
                }
                let coast = RendererCache.shared.coastline(for: forecast, renderer: renderer)
                return renderer.render(field: field, sources: sources, coastline: coast)
            }.value
            map = rendered
            renderError = nil
        } catch {
            renderError = error.localizedDescription
        }
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

/// Horizontal color bar with end and midpoint labels in text ink, units on the right.
struct Legend: View {
    let field: DisplayField

    var body: some View {
        let lut = field.scale.lut
        let ticks: [Float] = [0, 0.25, 0.5, 0.75, 1]
        VStack(spacing: 4) {
            LinearGradient(
                stops: (0 ... 32).map { k in
                    let p = Float(k) / 32
                    return .init(color: lut.swiftUIColor(p), location: Double(p))
                }, startPoint: .leading, endPoint: .trailing
            )
            .frame(height: 12)
            .clipShape(RoundedRectangle(cornerRadius: 3))
            HStack {
                ForEach(ticks, id: \.self) { p in
                    Text(Self.format(field.scale.value(at: p)))
                    if p < 1 { Spacer() }
                }
                Text(field.units).foregroundStyle(.secondary).padding(.leading, 8)
            }
            .font(.caption)
            .monospacedDigit()
        }
        .frame(maxWidth: 640)
    }

    static func format(_ v: Float) -> String {
        abs(v) >= 100 || v == v.rounded() ? String(format: "%.0f", v) : String(format: "%.1f", v)
    }
}
