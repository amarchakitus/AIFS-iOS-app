import AIFSKit
import MetalKit
import SwiftUI
import simd

/// Camera and clock shared by every globe on screen, so they turn and play together.
@Observable
final class GlobeState {
    static let homeLatitude = 25.0
    static let homeZoom = 0.44

    /// Point at the centre of the disc, degrees.
    var latitude = homeLatitude
    var longitude = -40.0
    /// Disc radius as a fraction of the shorter side of each globe's cell.
    var zoom = homeZoom
    /// Forecast time in steps, continuous: 1 is the first output, 1.5 halfway to the second.
    var time = 1.0
    var playing = false
    /// Turning on its own. Per session: touching the globe stops it, `goHome` restarts it.
    var spinning = true
    /// An eased return to the home view, advanced by the frame clock.
    private var homing: (latitude: Double, zoom: Double, start: CFTimeInterval)?

    var isHome: Bool {
        spinning && abs(latitude - Self.homeLatitude) < 0.5 && abs(zoom - Self.homeZoom) < 0.005
    }

    /// The user took hold of the globe.
    func interact() {
        spinning = false
        homing = nil
    }

    /// Back to the zoomed-out view, turning again.
    func goHome() {
        homing = (latitude, zoom, CACurrentMediaTime())
        spinning = true
    }

    /// One frame of the return, if one is under way.
    func advanceHoming(now: CFTimeInterval) {
        guard let h = homing else { return }
        let p = min((now - h.start) / 0.7, 1)
        let e = p * p * (3 - 2 * p)
        latitude = h.latitude + (Self.homeLatitude - h.latitude) * e
        zoom = h.zoom + (Self.homeZoom - h.zoom) * e
        if p >= 1 { homing = nil }
    }

    /// Camera -> earth: columns east, north, out of the screen, at the centre point.
    var inverseRotation: simd_float3x3 {
        let (p, l) = (latitude * .pi / 180, longitude * .pi / 180)
        let out = SIMD3<Float>(Float(cos(p) * cos(l)), Float(sin(p)), Float(cos(p) * sin(l)))
        let north = SIMD3<Float>(Float(-sin(p) * cos(l)), Float(cos(p)), Float(-sin(p) * sin(l)))
        let east = SIMD3<Float>(Float(-sin(l)), 0, Float(cos(l)))
        return simd_float3x3(columns: (east, north, out))
    }

    /// Latitude/longitude under a point of a cell (points, y-down), or nil off the disc.
    func coordinate(at point: CGPoint, in size: CGSize) -> (lat: Double, lon: Double)? {
        let radius = zoom * min(size.width, size.height)
        let x = (point.x - size.width / 2) / radius, y = (size.height / 2 - point.y) / radius
        let r2 = x * x + y * y
        guard r2 <= 1 else { return nil }
        let e = inverseRotation * SIMD3<Float>(Float(x), Float(y), Float((1 - r2).squareRoot()))
        return (Double(asin(max(-1, min(1, e.y)))) * 180 / .pi, Double(atan2(e.z, e.x)) * 180 / .pi)
    }
}

/// One GlobeState per forecast, for as long as the app runs.
@MainActor
final class GlobeStateCache {
    static let shared = GlobeStateCache()
    private var states: [URL: GlobeState] = [:]

    func state(for forecast: URL) -> GlobeState {
        if let s = states[forecast] { return s }
        let s = GlobeState()
        s.spinning = GlobeSettingsStore.shared.settings.spin  // only when first opened
        states[forecast] = s
        return s
    }
}

/// The sub-solar point's direction in the shader's earth frame at `date` (NOAA's
/// low-precision solar position: well under a degree, plenty for a terminator).
nonisolated func sunDirection(at date: Date) -> SIMD4<Float> {
    var cal = Calendar(identifier: .gregorian)
    cal.timeZone = TimeZone(identifier: "UTC")!
    let day = Double(cal.ordinality(of: .day, in: .year, for: date) ?? 1)
    let c = cal.dateComponents([.hour, .minute], from: date)
    let hours = Double(c.hour ?? 0) + Double(c.minute ?? 0) / 60
    let g = 2 * Double.pi / 365 * (day - 1 + (hours - 12) / 24)
    let eqTime = 229.18 * (0.000075 + 0.001868 * cos(g) - 0.032077 * sin(g) - 0.014615 * cos(2 * g) - 0.040849 * sin(2 * g))
    let decl = 0.006918 - 0.399912 * cos(g) + 0.070257 * sin(g) - 0.006758 * cos(2 * g)
        + 0.000907 * sin(2 * g) - 0.002697 * cos(3 * g) + 0.00148 * sin(3 * g)
    let lon = -15 * (hours - 12 + eqTime / 60) * .pi / 180
    return SIMD4<Float>(Float(cos(decl) * cos(lon)), Float(sin(decl)), Float(cos(decl) * sin(lon)), 0)
}

// MARK: - Screen

/// The globes for a finished forecast, per the settings panel.
struct GlobeScreen: View {
    let forecast: StoredForecast

    /// Per forecast and kept while the app runs, so leaving and reopening a forecast
    /// returns to the same view of the globe and the same time.
    private var state: GlobeState { GlobeStateCache.shared.state(for: forecast.id) }
    @State private var showSettings = LaunchOptions.openGlobeSettings
    @State private var basemap: GlobeBasemap?
    private var store = GlobeSettingsStore.shared

    init(forecast: StoredForecast) {
        self.forecast = forecast
    }

    private var settings: GlobeSettings { store.settings }
    private var total: Int { forecast.meta.completedSteps }

    var body: some View {
        VStack(spacing: 12) {
            GeometryReader { geo in
                let (rows, cols) = arrangement(count: settings.globes.count, in: geo.size)
                let cell = CGSize(width: geo.size.width / CGFloat(cols), height: geo.size.height / CGFloat(rows))
                VStack(spacing: 0) {
                    ForEach(0 ..< rows, id: \.self) { r in
                        HStack(spacing: 0) {
                            ForEach(0 ..< cols, id: \.self) { c in
                                let i = r * cols + c
                                if i < settings.globes.count {
                                    GlobeCell(
                                        forecast: forecast, spec: settings.globes[i], state: state,
                                        basemap: basemap, drivesClock: i == 0
                                    )
                                    .frame(width: cell.width, height: cell.height)
                                } else {
                                    Color.clear.frame(width: cell.width, height: cell.height)
                                }
                            }
                        }
                    }
                }
            }
            .background(Color(red: 0.016, green: 0.024, blue: 0.043))
            .clipShape(RoundedRectangle(cornerRadius: 12))

            timeControls
        }
        .padding()
        .toolbar {
            if !state.isHome {
                ToolbarItem {
                    Button {
                        state.goHome()
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                    }
                    .accessibilityLabel("Reset view")
                }
            }
            ToolbarItem {
                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                }
                .accessibilityLabel("Globe settings")
            }
        }
        .sheet(isPresented: $showSettings) {
            GlobeSettingsView(store: store, fields: availableFields)
                .presentationDetents([.medium, .large])
                // Opaque: clear glass over a bright globe is unreadable.
                .presentationBackground(Color(.systemGroupedBackground))
                .presentationBackgroundInteraction(.enabled(upThrough: .medium))
        }
        .task(id: forecast.id) { basemap = await GlobeDataCache.shared.basemap(for: forecast) }
        .onChange(of: settings.spin) { _, on in state.spinning = on }
    }

    /// Fields this forecast actually saved (older ones lack, e.g., 850 hPa wind).
    private var availableFields: [DisplayField] {
        DisplayField.all.filter { $0.sources.allSatisfy(forecast.meta.fields.contains) }
    }

    /// Rows and columns: as configured, or (auto) whichever makes the globes biggest.
    private func arrangement(count: Int, in size: CGSize) -> (Int, Int) {
        let n = max(count, 1)
        switch settings.layout {
        case .row: return (1, n)
        case .column: return (n, 1)
        case .auto:
            let options = (1 ... n).map { cols in ((n + cols - 1) / cols, cols) }
            return options.max { a, b in
                min(size.width / CGFloat(a.1), size.height / CGFloat(a.0))
                    < min(size.width / CGFloat(b.1), size.height / CGFloat(b.0))
            } ?? (1, 1)
        }
    }

    private func validDate(_ time: Double) -> Date {
        forecast.meta.initDate.addingTimeInterval(time * Double(forecast.meta.stepHours) * 3600)
    }

    private var timeControls: some View {
        HStack(spacing: 16) {
            Button {
                state.playing.toggle()
            } label: {
                Image(systemName: state.playing ? "pause.fill" : "play.fill").frame(width: 24)
            }
            .disabled(total < 2)
            .accessibilityLabel(state.playing ? "Pause" : "Play")
            if total > 1 {
                Slider(value: Binding(get: { state.time }, set: { state.time = $0 }), in: 1 ... Double(total)) { editing in
                    if editing { state.playing = false }
                }
            } else {
                Spacer()
            }
            VStack(alignment: .trailing) {
                let hours = Int((state.time * Double(forecast.meta.stepHours)).rounded())
                Text("+\(hours) h").font(.headline).monospacedDigit()
                Text(timeLabel(validDate(state.time))).font(.caption).foregroundStyle(.secondary)
            }
            .frame(minWidth: 130, alignment: .trailing)
        }
    }

    private func timeLabel(_ date: Date) -> String {
        guard settings.localTime else { return date.utcLabel }
        let f = DateFormatter()
        f.dateFormat = "EEE d MMM HH:mm"
        return f.string(from: date)
    }
}

// MARK: - One globe

private struct GlobeCell: View {
    let forecast: StoredForecast
    let spec: GlobeSpec
    let state: GlobeState
    let basemap: GlobeBasemap?
    let drivesClock: Bool

    private var store = GlobeSettingsStore.shared
    @State private var readout: (lat: Double, lon: Double, value: Float)?
    @State private var dragStart: (lat: Double, lon: Double)?
    @State private var zoomStart: Double?

    init(forecast: StoredForecast, spec: GlobeSpec, state: GlobeState, basemap: GlobeBasemap?, drivesClock: Bool) {
        self.forecast = forecast
        self.spec = spec
        self.state = state
        self.basemap = basemap
        self.drivesClock = drivesClock
    }

    private var data: GlobeLayerData {
        GlobeDataCache.shared.layer(for: spec, forecast: forecast, particles: store.settings.particles)
    }

    var body: some View {
        let data = data
        VStack(spacing: 6) {
            GeometryReader { geo in
                GlobeMetalView(
                    data: data, basemap: basemap, settings: store.settings, state: state,
                    initDate: forecast.meta.initDate, stepHours: forecast.meta.stepHours, drivesClock: drivesClock
                )
                .gesture(
                    DragGesture(minimumDistance: 2)
                        .onChanged { g in
                            state.interact()
                            let start = dragStart ?? (state.latitude, state.longitude)
                            dragStart = start
                            let radius = state.zoom * min(geo.size.width, geo.size.height)
                            state.longitude = start.lon - g.translation.width / radius * 180 / .pi
                            state.latitude = max(-85, min(85, start.lat + g.translation.height / radius * 180 / .pi))
                        }
                        .onEnded { _ in dragStart = nil }
                )
                .simultaneousGesture(
                    MagnifyGesture()
                        .onChanged { g in
                            state.interact()
                            let start = zoomStart ?? state.zoom
                            zoomStart = start
                            state.zoom = max(0.25, min(4, start * g.magnification))
                        }
                        .onEnded { _ in zoomStart = nil }
                )
                .simultaneousGesture(
                    SpatialTapGesture().onEnded { tap in
                        state.interact()
                        readout = state.coordinate(at: tap.location, in: geo.size).flatMap { value(at: $0, data: data) }
                    }
                )
                .overlay(alignment: .topLeading) {
                    if let r = readout, let field = data.field {
                        Text(field.readout(r.value) + String(format: "  %.1f°%@ %.1f°%@",
                                    abs(r.lat), r.lat >= 0 ? "N" : "S", abs(r.lon), r.lon >= 0 ? "E" : "W"))
                            .font(.caption.monospacedDigit())
                            .padding(6)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 6))
                            .padding(8)
                    }
                }
                .overlay {
                    if data.loaded == 0 && !data.failed { ProgressView().tint(.white) }
                }
            }
            if store.settings.showLabels, let field = data.field {
                VStack(spacing: 2) {
                    Text(field.title + (data.wind.map { " · \($0.title) particles" } ?? ""))
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.white.opacity(0.85))
                    Legend(field: field, lut: store.settings.lut(for: field))
                        .environment(\.colorScheme, .dark)
                        .padding(.horizontal, 12)
                }
                .padding(.bottom, 8)
            }
        }
    }

    /// The field at a point, interpolated between the two steps around the current time.
    private func value(at c: (lat: Double, lon: Double), data: GlobeLayerData) -> (lat: Double, lon: Double, value: Float)? {
        let steps = data.steps
        guard !steps.isEmpty else { return nil }
        let col = (Int(((c.lon + 180) / 0.25).rounded()) % 1440 + 1440) % 1440
        let row = min(max(Int(((90 - c.lat) / 0.25).rounded()), 0), 720)
        let i = row * 1440 + col
        let t = min(max(state.time, 1), Double(steps.count)) - 1
        let a = Int(t), b = min(a + 1, steps.count - 1), f = Float(t - Double(a))
        let v = Float(steps[a][i]) * (1 - f) + Float(steps[b][i]) * f
        return v.isNaN ? nil : (c.lat, c.lon, v)
    }
}

// MARK: - Metal

/// Mirrors `GlobeUniforms` in Globe.metal (144 bytes; every member naturally aligned).
private struct GlobeUniforms {
    var invRotation: simd_float3x3
    var center: SIMD2<Float>
    var resolution: SIMD2<Float>
    var sunDir: SIMD4<Float>
    var radius: Float
    var blend: Float
    var hasField: Float
    var scaleKind: Float
    var low: Float
    var mid: Float
    var high: Float
    var gamma: Float
    var night: Float
    var relief: Float
    var coast: Float
    var graticule: Float
    var atmosphere: Float
    var stars: Float
    var opacity: Float
    var hasElevation: Float
}

/// Mirrors `ParticleUniforms`.
private struct ParticleUniforms {
    var invRotation: simd_float3x3
    var extent: SIMD2<Float>
    var blend: Float
    var dt: Float
    var advect: Float
    var lifeBase: Float
    var count: UInt32
    var stride: UInt32
    var trail: UInt32
    var head: UInt32
    var stamp: UInt32
    var frame: UInt32
}

/// Mirrors `TrailUniforms`.
private struct TrailUniforms {
    var rotation: simd_float3x3
    var center: SIMD2<Float>
    var resolution: SIMD2<Float>
    var radius: Float
    var width: Float
    var fade: Float
    var intensity: Float
    var whiteness: Float
    var slotSeconds: Float
    var blend: Float
    var speedRange: Float
    var count: UInt32
    var trail: UInt32
    var head: UInt32
    var stride: UInt32
}

private struct GlobeMetalView: UIViewRepresentable {
    let data: GlobeLayerData
    let basemap: GlobeBasemap?
    let settings: GlobeSettings
    let state: GlobeState
    let initDate: Date
    let stepHours: Int
    let drivesClock: Bool

    func makeCoordinator() -> GlobeRenderer { GlobeRenderer(state: state) }

    func makeUIView(context: Context) -> MTKView {
        let view = MTKView(frame: .zero, device: context.coordinator.device)
        view.colorPixelFormat = .bgra8Unorm
        view.sampleCount = GlobeRenderer.sampleCount
        view.delegate = context.coordinator
        return view
    }

    func updateUIView(_ view: MTKView, context: Context) {
        // Up to the display's maximum (120 on ProMotion; the app opts in to above 60 on
        // iPhone with CADisableMinimumFrameDurationOnPhone).
        let maximum = view.window?.windowScene?.screen.maximumFramesPerSecond ?? 120
        view.preferredFramesPerSecond = min(Int(settings.fpsCap), maximum)
        context.coordinator.update(
            data: data, basemap: basemap, settings: settings, initDate: initDate,
            stepHours: stepHours, drivesClock: drivesClock)
    }
}

@MainActor
final class GlobeRenderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    private let queue: MTLCommandQueue
    private let globePipeline: MTLRenderPipelineState
    private let trailPipeline: MTLRenderPipelineState
    private let particleKernel: MTLComputePipelineState
    private let state: GlobeState
    private let blank: MTLTexture

    private var data: GlobeLayerData?
    private var fieldTextures: [MTLTexture] = []
    private var windTextures: [MTLTexture] = []
    private var landTexture: MTLTexture?
    private var elevationTexture: MTLTexture?
    private var basemapLoaded = false
    private var lutTexture: MTLTexture?
    private var lutKey = ""
    private var settings = GlobeSettings()
    private var initDate = Date()
    private var stepHours = 6
    private var drivesClock = false

    // Particles
    private var stateBuffer: MTLBuffer?
    private var historyBuffer: MTLBuffer?
    private var particleCount = 0
    private var particleCapacity = 0
    private var trailSlots = 0
    private var head: UInt32 = 0
    private var simStep: UInt32 = 0
    private var sinceStamp = 0.0
    /// Trails remember a position every `trailSpacing` x this many seconds, whatever the
    /// frame rate (AIFS-VIZ's 30 Hz step, kept as the unit so its settings carry over).
    private static let slotUnit = 1.0 / 30
    static let sampleCount = 4

    private var lastFrame = CACurrentMediaTime()

    init(state: GlobeState) {
        self.state = state
        device = MTLCreateSystemDefaultDevice()!
        queue = device.makeCommandQueue()!
        let library = device.makeDefaultLibrary()!

        let g = MTLRenderPipelineDescriptor()
        g.vertexFunction = library.makeFunction(name: "globeVertex")
        g.fragmentFunction = library.makeFunction(name: "globeFragment")
        g.colorAttachments[0].pixelFormat = .bgra8Unorm
        g.rasterSampleCount = Self.sampleCount
        globePipeline = try! device.makeRenderPipelineState(descriptor: g)

        let t = MTLRenderPipelineDescriptor()
        t.vertexFunction = library.makeFunction(name: "trailVertex")
        t.fragmentFunction = library.makeFunction(name: "trailFragment")
        t.colorAttachments[0].pixelFormat = .bgra8Unorm
        t.rasterSampleCount = Self.sampleCount  // anti-aliased trail edges
        t.colorAttachments[0].isBlendingEnabled = true
        t.colorAttachments[0].sourceRGBBlendFactor = .one  // premultiplied
        t.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        t.colorAttachments[0].sourceAlphaBlendFactor = .one
        t.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        trailPipeline = try! device.makeRenderPipelineState(descriptor: t)

        particleKernel = try! device.makeComputePipelineState(function: library.makeFunction(name: "particleUpdate")!)
        blank = device.makeTexture(descriptor: .texture2DDescriptor(
            pixelFormat: .r16Float, width: 1, height: 1, mipmapped: false))!
        super.init()
    }

    private func texture(_ bytes: UnsafeRawBufferPointer, format: MTLPixelFormat, width: Int, height: Int, bytesPerPixel: Int) -> MTLTexture {
        let tex = device.makeTexture(descriptor: .texture2DDescriptor(
            pixelFormat: format, width: width, height: height, mipmapped: false))!
        tex.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                    withBytes: bytes.baseAddress!, bytesPerRow: width * bytesPerPixel)
        return tex
    }

    func update(data: GlobeLayerData, basemap: GlobeBasemap?, settings: GlobeSettings, initDate: Date,
                stepHours: Int, drivesClock: Bool) {
        self.settings = settings
        self.initDate = initDate
        self.stepHours = stepHours
        self.drivesClock = drivesClock
        if data !== self.data {
            self.data = data
            fieldTextures = []
            windTextures = []
        }
        if let field = data.field {
            let key = "\(field.id)|\((settings.ramps[field.id] ?? .standard).rawValue)"
            if key != lutKey {
                lutKey = key
                lutTexture = settings.lut(for: field).rgba.withUnsafeBytes {
                    texture($0, format: .rgba8Unorm, width: $0.count / 4, height: 1, bytesPerPixel: 4)
                }
            }
        }
        if !basemapLoaded, let basemap {
            basemapLoaded = true
            landTexture = basemap.land.withUnsafeBytes {
                texture($0, format: .r32Float, width: 1440, height: 721, bytesPerPixel: 4)
            }
            elevationTexture = basemap.elevation?.withUnsafeBytes {
                texture($0, format: .r32Float, width: 1440, height: 721, bytesPerPixel: 4)
            }
        }
    }

    /// Pull any newly loaded steps into textures (cheap when nothing changed).
    private func syncTextures() {
        guard let data else { return }
        for s in fieldTextures.count ..< data.steps.count {
            fieldTextures.append(data.steps[s].withUnsafeBytes {
                texture($0, format: .r16Float, width: 1440, height: 721, bytesPerPixel: 2)
            })
        }
        for s in windTextures.count ..< data.winds.count {
            windTextures.append(data.winds[s].withUnsafeBytes {
                texture($0, format: .rg16Float, width: 1440, height: 721, bytesPerPixel: 4)
            })
        }
    }

    /// Size the particle system to what is on screen. Allocated once for the most this
    /// cell can show (fully zoomed in), so a pinch changes only how many are live and
    /// never restarts the ones already flying.
    private func ensureParticles(radiusPx: Double, cell: CGSize) {
        func budget(_ areaPx: Double) -> Int {
            min(max(Int(settings.particleDensity * areaPx / 150) / 64 * 64, 512), 65536)
        }
        let cellArea = Double(cell.width * cell.height)
        let count = budget(min(.pi * radiusPx * radiusPx, cellArea))
        let slots = Int(settings.trailLength)
        if stateBuffer == nil || slots != trailSlots || count > particleCapacity {
            let capacity = max(budget(cellArea), count)
            // Zeroed: lifetime 0 marks "never lived", spawned on the first step.
            stateBuffer = device.makeBuffer(length: capacity * 16, options: .storageModeShared)
            historyBuffer = device.makeBuffer(length: capacity * slots * 8, options: .storageModePrivate)
            particleCapacity = capacity
            trailSlots = slots
            particleCount = 0
            head = 0
        }
        if count > particleCount, let state = stateBuffer {
            // Newly live: whatever they held is stale, so mark them never lived.
            memset(state.contents() + particleCount * 16, 0, (count - particleCount) * 16)
        }
        particleCount = count
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        // No GPU work unless in the foreground: iOS refuses it from the background.
        guard UIApplication.shared.applicationState == .active else { return }
        syncTextures()
        let now = CACurrentMediaTime()
        let dt = min(now - lastFrame, 0.1)
        lastFrame = now
        let count = fieldTextures.count > 0 ? fieldTextures.count : windTextures.count
        if drivesClock {
            state.advanceHoming(now: now)
            if state.spinning { state.longitude += 360 / settings.spinSeconds * dt }
            if state.playing, let total = data?.total, total > 1 {
                state.time += Double(total - 1) / settings.cycleSeconds * dt
                if state.time > Double(total) {
                    if settings.loopForecast { state.time = 1 } else { state.time = Double(total); state.playing = false }
                }
            }
        }

        guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
            let buffer = queue.makeCommandBuffer()
        else { return }
        let size = view.drawableSize
        let radius = state.zoom * min(size.width, size.height)
        let t = min(max(state.time, 1), Double(max(count, 1))) - 1
        let a = Int(t), b = min(a + 1, max(count - 1, 0))
        let blend = Float(t - Double(a))
        let invRotation = state.inverseRotation

        // --- particles: one step per frame, by the frame's own duration -----------------
        let windReady = settings.particles && data?.wind != nil && windTextures.count > max(a, b)
        let slotSeconds = Self.slotUnit * max(settings.trailSpacing, 1)
        if windReady, let level = data?.wind, dt > 0, let encoder = buffer.makeComputeCommandEncoder() {
            ensureParticles(radiusPx: radius, cell: size)
            simStep &+= 1
            sinceStamp += dt
            let stamp = sinceStamp >= slotSeconds
            if stamp {
                sinceStamp = sinceStamp.truncatingRemainder(dividingBy: slotSeconds)
                head = (head + 1) % UInt32(trailSlots)
            }
            do {
                var pu = ParticleUniforms(
                    invRotation: invRotation,
                    extent: SIMD2(Float(size.width / 2 / radius), Float(size.height / 2 / radius)),
                    blend: blend, dt: Float(dt),
                    // Slower in degrees as the globe grows, so streaks keep their on-screen
                    // length and pace when zoomed in instead of carpeting the view.
                    advect: Float(settings.advection(for: level) / max(1, state.zoom / GlobeState.homeZoom)),
                    lifeBase: Float(settings.particleLife), count: UInt32(particleCount),
                    stride: UInt32(particleCapacity), trail: UInt32(trailSlots), head: head,
                    stamp: stamp ? 1 : 0, frame: simStep)
                encoder.setComputePipelineState(particleKernel)
                encoder.setBuffer(stateBuffer, offset: 0, index: 0)
                encoder.setBuffer(historyBuffer, offset: 0, index: 1)
                encoder.setBytes(&pu, length: MemoryLayout<ParticleUniforms>.stride, index: 2)
                encoder.setTexture(windTextures[a], index: 0)
                encoder.setTexture(windTextures[b], index: 1)
                let w = particleKernel.threadExecutionWidth
                encoder.dispatchThreadgroups(
                    MTLSize(width: (particleCount + w - 1) / w, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
                encoder.endEncoding()
            }
        }

        guard let encoder = buffer.makeRenderCommandEncoder(descriptor: pass) else { return }

        // --- globe ---------------------------------------------------------------------
        let field = data?.field
        let scale = field?.scale ?? .sequential(low: 0, high: 1, gamma: 1)
        let (kind, low, mid, high, gamma): (Float, Float, Float, Float, Float) = {
            switch scale {
            case .sequential(let l, let h, let g): (0, l, 0, h, g)
            case .diverging(let l, let m, let h): (1, l, m, h, 1)
            case .gray(let l, let h): (2, l, 0, h, 1)
            }
        }()
        // A wind layer drawn under its own particles is only a wash, so the trails read.
        let isOwnWind = field.map { f in data?.wind.map { f.sources == [$0.components.u, $0.components.v] } ?? false } ?? false
        let opacity = settings.opacity * (isOwnWind && windReady ? settings.vectorFieldOpacity : 1)
        var u = GlobeUniforms(
            invRotation: invRotation,
            center: SIMD2(Float(size.width / 2), Float(size.height / 2)),
            resolution: SIMD2(Float(size.width), Float(size.height)),
            // The terminator follows the frame's (interpolated) valid time, not the clock.
            sunDir: sunDirection(at: initDate.addingTimeInterval(state.time * Double(stepHours) * 3600)),
            radius: Float(radius), blend: blend, hasField: fieldTextures.isEmpty ? 0 : 1,
            scaleKind: kind, low: low, mid: mid, high: high, gamma: gamma,
            night: Float(settings.level(.night)), relief: Float(settings.level(.relief)),
            coast: Float(settings.level(.coast)), graticule: Float(settings.level(.graticule)),
            atmosphere: Float(settings.level(.atmosphere)), stars: Float(settings.level(.stars)),
            opacity: Float(opacity),
            hasElevation: elevationTexture == nil ? 0 : 1)
        encoder.setRenderPipelineState(globePipeline)
        encoder.setVertexBytes(&u, length: MemoryLayout<GlobeUniforms>.stride, index: 0)
        encoder.setFragmentBytes(&u, length: MemoryLayout<GlobeUniforms>.stride, index: 0)
        encoder.setFragmentTexture(fieldTextures.isEmpty ? blank : fieldTextures[min(a, fieldTextures.count - 1)], index: 0)
        encoder.setFragmentTexture(fieldTextures.isEmpty ? blank : fieldTextures[min(b, fieldTextures.count - 1)], index: 1)
        encoder.setFragmentTexture(landTexture ?? blank, index: 2)
        encoder.setFragmentTexture(lutTexture ?? blank, index: 3)
        encoder.setFragmentTexture(elevationTexture ?? blank, index: 4)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

        // --- trails --------------------------------------------------------------------
        if windReady, particleCount > 0, trailSlots > 1 {
            var tu = TrailUniforms(
                rotation: invRotation.transpose,
                center: u.center, resolution: u.resolution, radius: Float(radius),
                // Device pixels at 2x, as on the Retina displays AIFS-VIZ is tuned for.
                width: Float(settings.trailWidth * view.contentScaleFactor / 2), fade: Float(settings.trailFade),
                intensity: Float(settings.trailIntensity), whiteness: Float(settings.trailWhiteness),
                slotSeconds: Float(slotSeconds), blend: blend,
                speedRange: Float(data?.wind?.speedRange ?? 40),
                count: UInt32(particleCount), trail: UInt32(trailSlots), head: head,
                stride: UInt32(particleCapacity))
            encoder.setRenderPipelineState(trailPipeline)
            encoder.setVertexBuffer(stateBuffer, offset: 0, index: 0)
            encoder.setVertexBuffer(historyBuffer, offset: 0, index: 1)
            encoder.setVertexBytes(&tu, length: MemoryLayout<TrailUniforms>.stride, index: 2)
            encoder.setFragmentBytes(&tu, length: MemoryLayout<TrailUniforms>.stride, index: 0)
            encoder.setVertexTexture(windTextures[a], index: 0)
            encoder.setVertexTexture(windTextures[b], index: 1)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6,
                                   instanceCount: particleCount * trailSlots)
        }
        encoder.endEncoding()
        buffer.present(drawable)
        buffer.commit()
    }
}
