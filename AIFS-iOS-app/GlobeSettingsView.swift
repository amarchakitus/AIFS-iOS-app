import AIFSKit
import SwiftUI

/// The globe control panel, grouped as in AIFS-VIZ: globes, motion, wind particles,
/// basemap, fields, readouts. Every change applies on the next frame.
struct GlobeSettingsView: View {
    @Bindable var store: GlobeSettingsStore
    let fields: [DisplayField]
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                globesSection
                Section("Motion") {
                    Toggle("Rotate", isOn: $store.settings.spin)
                    LogSlider("Rotation period", value: $store.settings.spinSeconds, in: 20 ... 3600, unit: "s")
                    LogSlider("Forecast playback", value: $store.settings.cycleSeconds, in: 20 ... 7200, unit: "s")
                    Toggle("Loop playback", isOn: $store.settings.loopForecast)
                    ValueSlider("Frame rate cap", value: $store.settings.fpsCap, in: 10 ... 120, step: 1, format: "%.0f fps")
                }
                particlesSection
                basemapSection
                Section("Fields") {
                    ValueSlider("Field opacity", value: $store.settings.opacity, in: 0.1 ... 1)
                    ForEach(fields) { field in
                        RampMenu(field: field, ramps: $store.settings.ramps)
                    }
                }
                Section("Readouts") {
                    Picker("Times shown in", selection: $store.settings.localTime) {
                        Text("UTC").tag(false)
                        Text("Local").tag(true)
                    }
                    Toggle("Captions under each globe", isOn: $store.settings.showLabels)
                }
                Section {
                    Button("Reset all", role: .destructive) { store.reset() }
                }
            }
            .navigationTitle("Globe settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }

    @ViewBuilder private var globesSection: some View {
        Section {
            ForEach($store.settings.globes) { $globe in
                VStack(alignment: .leading, spacing: 8) {
                    Picker("Layer", selection: $globe.layer) {
                        ForEach(fields) { Text($0.title).tag($0.id) }
                    }
                    Picker("Particles", selection: $globe.overlay) {
                        Text("None").tag(WindLevel?.none)
                        ForEach(WindLevel.allCases) { Text($0.title).tag(WindLevel?.some($0)) }
                    }
                }
            }
            .onDelete { store.settings.globes.remove(atOffsets: $0) }
            if store.settings.globes.count < 3 {
                Button {
                    store.settings.globes.append(GlobeSpec(layer: fields.first?.id ?? "t2m", overlay: nil))
                } label: {
                    Label("Add a globe", systemImage: "plus")
                }
            }
            Picker("Arrangement", selection: $store.settings.layout) {
                ForEach(GlobeLayout.allCases) { Text($0.title).tag($0) }
            }
        } header: {
            Text("Globes")
        } footer: {
            Text("Up to three, turning and playing together. Swipe a globe to remove it.")
        }
    }

    /// One row of chips switches layers on and off; only the layers that are on get a
    /// strength slider, so turning things off also tidies the section.
    @ViewBuilder private var basemapSection: some View {
        Section {
            FlowLayout(spacing: 8) {
                ForEach(BasemapLayer.allCases) { layer in
                    Toggle(layer.title, isOn: Binding(
                        get: { store.settings.isOn(layer) },
                        set: { store.settings.setOn(layer, $0) }
                    ))
                    .toggleStyle(.button)
                    .buttonBorderShape(.capsule)
                    .controlSize(.small)
                }
            }
            .padding(.vertical, 4)
            ForEach(BasemapLayer.allCases.filter(store.settings.isOn)) { layer in
                ValueSlider(layer.title, value: Binding(
                    get: { store.settings.strength(layer) },
                    set: { store.settings.setStrength(layer, $0) }
                ), in: layer.range)
            }
        } header: {
            Text("Basemap")
        }
    }

    @ViewBuilder private var particlesSection: some View {
        Section {
            Toggle("Draw wind particles", isOn: $store.settings.particles)
            Group {
                LogSlider("Speed", value: $store.settings.advect, in: 0.02 ... 6, unit: "×")
                ValueSlider("Match speeds across levels", value: $store.settings.speedNormalise, in: 0 ... 1)
                ValueSlider("Density", value: $store.settings.particleDensity, in: 0.15 ... 2.5, format: "%.2f×")
                ValueSlider("Trail length", value: $store.settings.trailLength, in: 4 ... 32, step: 1, format: "%.0f steps")
                ValueSlider("Trail spacing", value: $store.settings.trailSpacing, in: 1 ... 8, step: 1, format: "%.0f×")
                LogSlider("Lifetime", value: $store.settings.particleLife, in: 0.5 ... 120, unit: "s")
                ValueSlider("Trail width", value: $store.settings.trailWidth, in: 0.5 ... 5, format: "%.1f px")
                ValueSlider("Brightness", value: $store.settings.trailIntensity, in: 0.1 ... 2)
                ValueSlider("Tail fade", value: $store.settings.trailFade, in: 0.4 ... 3)
                ValueSlider("Whiteness", value: $store.settings.trailWhiteness, in: 0 ... 1)
                ValueSlider("Speed wash beneath", value: $store.settings.vectorFieldOpacity, in: 0 ... 1)
            }
            .disabled(!store.settings.particles)
        } header: {
            Text("Wind particles")
        } footer: {
            Text("Relative speeds within a level are exact; speed only sets how fast the whole field is drawn. "
                + "Trail spacing lengthens streaks without speeding the air up.")
        }
    }
}

/// A field's colour ramp. Collapsed, an untouched field reads just "Default"; in the menu
/// the default ramp carries a grey "Default" subtitle. Picking it clears the override, so
/// the field follows its own scale again. A Menu, not a Picker: menu pickers drop an
/// option's second line, where a menu item's label keeps it as the subtitle.
private struct RampMenu: View {
    let field: DisplayField
    @Binding var ramps: [String: RampChoice]

    var body: some View {
        let standard = RampChoice.matching(field.scale)
        let chosen = ramps[field.id].flatMap { $0 == .standard ? nil : $0 }
        LabeledContent(field.title) {
            Menu {
                // Toggles, not buttons: the menu then reserves a checkmark column for every
                // item, so the ticked one lines up with the rest instead of being indented.
                ForEach(RampChoice.allCases.filter { $0 != .standard }) { choice in
                    Toggle(isOn: Binding(
                        get: { choice == (chosen ?? standard) },
                        set: { _ in ramps[field.id] = choice == standard ? nil : choice }
                    )) {
                        Text(choice.title)
                        if choice == standard { Text("Default") }
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Text(chosen?.title ?? "Default")
                    Image(systemName: "chevron.up.chevron.down").font(.footnote.weight(.medium))
                }
            }
            .tint(Color(.secondaryLabel))  // grey like the Picker rows, not the accent
        }
    }
}

/// Lays its children out in rows, wrapping when a row is full.
private struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = rows(width: proposal.width ?? .infinity, subviews: subviews)
        let height = rows.map(\.height).reduce(0, +) + spacing * CGFloat(max(rows.count - 1, 0))
        return CGSize(width: proposal.width ?? rows.map(\.width).max() ?? 0, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var y = bounds.minY
        for row in rows(width: bounds.width, subviews: subviews) {
            var x = bounds.minX
            for i in row.indices {
                let size = subviews[i].sizeThatFits(.unspecified)
                subviews[i].place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
            y += row.height + spacing
        }
    }

    private func rows(width: CGFloat, subviews: Subviews) -> [(indices: [Int], width: CGFloat, height: CGFloat)] {
        var rows: [(indices: [Int], width: CGFloat, height: CGFloat)] = []
        var current: [Int] = [], x: CGFloat = 0, height: CGFloat = 0
        for (i, view) in subviews.enumerated() {
            let size = view.sizeThatFits(.unspecified)
            if !current.isEmpty, x + spacing + size.width > width {
                rows.append((current, x, height))
                current = []; x = 0; height = 0
            }
            x += (current.isEmpty ? 0 : spacing) + size.width
            height = max(height, size.height)
            current.append(i)
        }
        if !current.isEmpty { rows.append((current, x, height)) }
        return rows
    }
}

/// A labelled slider with its value shown.
private struct ValueSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    var step: Double?
    var format = "%.2f"

    init(_ title: String, value: Binding<Double>, in range: ClosedRange<Double>, step: Double? = nil, format: String = "%.2f") {
        self.title = title
        _value = value
        self.range = range
        self.step = step
        self.format = format
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(String(format: format, value)).foregroundStyle(.secondary).monospacedDigit()
            }
            if let step {
                Slider(value: $value, in: range, step: step)
            } else {
                Slider(value: $value, in: range)
            }
        }
    }
}

/// A slider over the logarithm of its value: these are ratios, and a linear slider over
/// 0.02...6 spends nine tenths of its travel above 1 (AIFS-VIZ's reasoning, kept).
private struct LogSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let unit: String

    init(_ title: String, value: Binding<Double>, in range: ClosedRange<Double>, unit: String) {
        self.title = title
        _value = value
        self.range = range
        self.unit = unit
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                Spacer()
                Text(value < 10 ? String(format: "%.2f%@", value, unit) : String(format: "%.0f%@", value, unit))
                    .foregroundStyle(.secondary).monospacedDigit()
            }
            Slider(
                value: Binding(get: { log(value) }, set: { value = exp($0) }),
                in: log(range.lowerBound) ... log(range.upperBound))
        }
    }
}
