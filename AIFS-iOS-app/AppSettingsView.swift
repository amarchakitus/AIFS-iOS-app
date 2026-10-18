import SwiftUI

/// App-wide settings: units, and the licences and attribution page.
struct AppSettingsView: View {
    @Bindable var units: UnitStore
    @Environment(\.dismiss) private var dismiss
    /// Debug: `--open-attribution` opens the licences page (with `--open-app-settings`).
    @State private var showAttribution = ProcessInfo.processInfo.arguments.contains("--open-attribution")

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
                } footer: {
                    Text("Applies to maps, globes and places. Colours stay tied to the physical value, "
                        + "so a colour means the same weather in any units; only the labels change.")
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
                }
            }
            .navigationDestination(isPresented: $showAttribution) { AttributionView() }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}
