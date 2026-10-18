import SwiftUI

/// App-wide settings: for now, the units every screen shows.
struct AppSettingsView: View {
    @Bindable var units: UnitStore
    @Environment(\.dismiss) private var dismiss

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
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
    }
}
