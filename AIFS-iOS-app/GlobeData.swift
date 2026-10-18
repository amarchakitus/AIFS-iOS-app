import AIFSKit
import Foundation
import Observation

/// Everything one globe draws for one forecast, on the 0.25-degree grid (columns from
/// 180W), filled in step by step in the background.
@MainActor
@Observable
final class GlobeLayerData {
    let field: DisplayField?
    let wind: WindLevel?
    let total: Int
    /// Field values per step, in display units (half floats: ~2 MB a step).
    private(set) var steps: [[Float16]] = []
    /// Wind per step, (u, v) interleaved, m/s.
    private(set) var winds: [[Float16]] = []
    private(set) var failed = false

    /// Steps loaded so far (field and wind arrive together, or wind is unavailable).
    var loaded: Int { max(steps.count, winds.count) }

    init(field: DisplayField?, wind: WindLevel?, total: Int) {
        self.field = field
        self.wind = wind
        self.total = total
    }

    fileprivate func append(field values: [Float16]?, wind uv: [Float16]?) {
        if let values { steps.append(values) }
        if let uv { winds.append(uv) }
    }

    fileprivate func fail() { failed = true }
}

/// Static basemap layers per forecast: land fraction (coastlines) and elevation (relief).
nonisolated struct GlobeBasemap: Sendable {
    let land: [Float]
    let elevation: [Float]?
}

@MainActor
final class GlobeDataCache {
    static let shared = GlobeDataCache()

    private var layers: [String: GlobeLayerData] = [:]
    private var order: [String] = []
    private var basemaps: [URL: GlobeBasemap] = [:]
    /// Bounds memory: each cached layer is ~2-6 MB per step.
    private let capacity = 6

    /// The data for `spec` on `forecast`, loading in the background if not cached.
    func layer(for spec: GlobeSpec, forecast: StoredForecast, particles: Bool) -> GlobeLayerData {
        let field = DisplayField.all.first { $0.id == spec.layer }
        let wind = particles ? spec.overlay : nil
        let key = "\(forecast.id.lastPathComponent)|\(spec.layer)|\(wind?.rawValue ?? "-")|\(forecast.meta.completedSteps)"
        if let cached = layers[key] {
            order.removeAll { $0 == key }
            order.append(key)
            return cached
        }
        let data = GlobeLayerData(field: field, wind: wind, total: forecast.meta.completedSteps)
        layers[key] = data
        order.append(key)
        while order.count > capacity { layers[order.removeFirst()] = nil }
        load(data, forecast: forecast)
        return data
    }

    func basemap(for forecast: StoredForecast) async -> GlobeBasemap? {
        if let b = basemaps[forecast.id] { return b }
        let b: GlobeBasemap? = await Task.detached(priority: .userInitiated) {
            guard let renderer = try? RendererCache.shared.renderer(),
                let statics = try? FieldFile.read(forecast.staticURL), let lsm = statics["lsm"]
            else { return nil }
            // Surface geopotential to metres; absent in forecasts made before it was saved.
            let elevation = statics["z"].map { z in
                renderer.shiftToDateline(renderer.grid.regrid(z.map { $0 / 9.80665 }))
            }
            return GlobeBasemap(land: renderer.landFraction(lsm: lsm), elevation: elevation)
        }.value
        if let b { basemaps[forecast.id] = b }
        return b
    }

    private func load(_ data: GlobeLayerData, forecast: StoredForecast) {
        let field = data.field, wind = data.wind, total = data.total
        Task { [weak data] in
            guard let renderer = try? RendererCache.shared.renderer() else { data?.fail(); return }
            for s in stride(from: 1, through: total, by: 1) {
                guard let data else { return }  // evicted
                let step: ([Float16]?, [Float16]?)? = await Task.detached(priority: .userInitiated) {
                    var names = field?.sources ?? []
                    if let wind { names += [wind.components.u, wind.components.v] }
                    guard let fields = try? FieldFile.read(forecast.stepURL(s), fields: names) else { return nil }
                    var values: [Float16]?
                    if let field {
                        let sources = field.sources.compactMap { fields[$0] }
                        guard sources.count == field.sources.count else { return nil }
                        values = renderer.displayValues(field: field, sources: sources).map(Float16.init)
                    }
                    var uv: [Float16]?
                    // No particles (rather than no globe) for forecasts saved without this level.
                    if let wind, let u = fields[wind.components.u], let v = fields[wind.components.v] {
                        let ug = renderer.shiftToDateline(renderer.grid.regrid(u))
                        let vg = renderer.shiftToDateline(renderer.grid.regrid(v))
                        var out = [Float16](repeating: 0, count: ug.count * 2)
                        for i in 0 ..< ug.count {
                            out[2 * i] = Float16(ug[i])
                            out[2 * i + 1] = Float16(vg[i])
                        }
                        uv = out
                    }
                    return (values, uv)
                }.value
                guard let step else { data.fail(); return }
                data.append(field: step.0, wind: step.1)
            }
        }
    }
}
