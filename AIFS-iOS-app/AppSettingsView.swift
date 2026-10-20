import SwiftUI

/// App-wide settings: units, storage, licences and attribution, and the privacy policy.
struct AppSettingsView: View {
    @Bindable var units: UnitStore
    let app: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var showAttribution = LaunchOptions.openAttribution
    @State private var initialConditionBytes: Int?
    @State private var confirmClear = false
    @State private var clearing = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Picker("Temperature", selection: $units.preferences.temperature) {
                        ForEach(TemperatureUnit.allCases) { Text($0.symbol).tag($0) }
                    }
                    Picker("Wind speed", selection: $units.preferences.speed) {
                        ForEach(SpeedUnit.allCases) { Text($0.symbol).tag($0) }
                    }
                    Picker("Precipitation", selection: $units.preferences.precipitation) {
                        ForEach(PrecipitationUnit.allCases) { Text($0.symbol).tag($0) }
                    }
                    Picker("Pressure", selection: $units.preferences.pressure) {
                        ForEach(PressureUnit.allCases) { Text($0.symbol).tag($0) }
                    }
                    if units.preferences != .regional {
                        Button("Use my region's units") { units.preferences = .regional }
                    }
                } header: {
                    Text("Units")
                }
                Section {
                    LabeledContent("Initial conditions") {
                        if let bytes = initialConditionBytes {
                            Text(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file))
                        } else {
                            ProgressView().controlSize(.small)
                        }
                    }
                    Button("Clear initial conditions…", role: .destructive) { confirmClear = true }
                        .disabled(clearing || app.download != nil || app.run != nil || initialConditionBytes == 0)
                } header: {
                    Text("Storage")
                }
                .confirmationDialog("Clear initial conditions", isPresented: $confirmClear, titleVisibility: .visible) {
                    Button("Past day", role: .destructive) { clear(since: .now.addingTimeInterval(-86400)) }
                    Button("Past week", role: .destructive) { clear(since: .now.addingTimeInterval(-7 * 86400)) }
                    Button("All time", role: .destructive) { clear(since: nil) }
                } message: {
                    Text("Forecasts are kept.")
                }
                Section {
                    Button {
                        showAttribution = true
                    } label: {
                        HStack {
                            Text("Licences & attribution").foregroundStyle(.primary)
                            Spacer()
                            Image(systemName: "chevron.right")
                                .font(.footnote.weight(.semibold))
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .tint(.primary)  // a row that opens a page, not a blue action
                    NavigationLink("Privacy policy") { PrivacyPolicyView() }
                }
            }
            .navigationDestination(isPresented: $showAttribution) { AttributionView() }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .task { await measure() }
        }
    }

    private func measure() async {
        initialConditionBytes = await Task.detached { InitialConditionStore.bytesOnDisk() }.value
    }

    /// Deletes the initial conditions downloaded since `since` (all of them for nil).
    private func clear(since: Date?) {
        clearing = true
        Task {
            await Task.detached { InitialConditionStore.clear(since: since) }.value
            app.refresh()
            await measure()
            clearing = false
        }
    }
}
