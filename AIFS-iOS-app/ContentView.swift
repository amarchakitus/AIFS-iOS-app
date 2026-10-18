import SwiftUI

struct ContentView: View {
    @State private var app = AppModel()
    @State private var selectedIC: URL?
    @State private var leadHours = 48
    @State private var typingLeadTime = false
    @State private var leadTimeText = ""
    /// Debug: `--open-earlier-cycles` opens the earlier-cycles sheet.
    @State private var showEarlierCycles = ProcessInfo.processInfo.arguments.contains("--open-earlier-cycles")
    /// A run that would replace an existing forecast, waiting for confirmation.
    @State private var confirmReplace: (ic: ICFile, existing: StoredForecast)?
    /// Debug: `--open-app-settings` opens the settings sheet.
    @State private var showSettings = ProcessInfo.processInfo.arguments.contains("--open-app-settings")
    // Model steps are 6 h. The slider covers whole days up to 15; a typed value can be any
    // number of steps, with no upper limit.
    @Environment(\.horizontalSizeClass) private var sizeClass

    var body: some View {
        NavigationSplitView {
            List(selection: $app.selectedForecastID) {
                modelSection
                icSection
                runSection
                forecastSection
            }
            .navigationTitle("AIFS")
            .refreshable { app.refresh() }
            .toolbar {
                ToolbarItem {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                    .accessibilityLabel("Settings")
                }
            }
            .sheet(isPresented: $showSettings) {
                AppSettingsView(units: UnitStore.shared)
            }
            .sheet(isPresented: $showEarlierCycles) {
                EarlierCyclesSheet(app: app) { date in
                    showEarlierCycles = false
                    app.downloadIC(date: date)
                }
                .presentationDetents([.medium, .large])
            }
        } detail: {
            if let forecast = app.selectedForecast {
                ForecastViewer(
                    forecast: forecast, revision: app.forecastRevision,
                    isGenerating: app.run != nil && app.startedForecastID == forecast.id)
            } else {
                ContentUnavailableView(
                    "No forecast selected", systemImage: "globe.europe.africa",
                    description: Text("Download initial conditions, run a forecast, then pick it from the list."))
            }
        }
        .alert("Something went wrong", isPresented: .constant(app.errorMessage != nil)) {
            Button("OK") { app.errorMessage = nil }
        } message: {
            Text(app.errorMessage ?? "")
        }
        .onAppear {
            if selectedIC == nil { selectedIC = app.initialConditions.first?.url }
            app.autorunIfRequested()
            // Debug: `--open-latest` opens the newest forecast (screenshots, simulator).
            if ProcessInfo.processInfo.arguments.contains("--open-latest") {
                app.selectedForecastID = app.forecasts.first?.id
            }
        }
        .onChange(of: app.downloadedICID) { _, id in
            if let id { selectedIC = id }
        }
        .onChange(of: app.startedForecastID) { _, id in
            // Side by side on iPad, so show the run as it fills in; on iPhone stay on the
            // list, where the progress is.
            if sizeClass == .regular, let id { app.selectedForecastID = id }
        }
        .onChange(of: app.initialConditions) {
            if selectedIC == nil || !app.initialConditions.contains(where: { $0.url == selectedIC }) {
                selectedIC = app.initialConditions.first?.url
            }
        }
    }

    // MARK: - Sections

    @ViewBuilder private var modelSection: some View {
        Section("Model") {
            switch app.modelState {
            case .missing:
                Label("Model bundle not installed", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                Text(
                    "Run `./aifs export bundle` on a Mac, then copy the **aifs-single-v2** folder into this app's Documents (Finder → iPad → Files → AIFS). Pull to refresh."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
            case .loading:
                Label("Loading…", systemImage: "hourglass")
            case .ready:
                Label(app.manifestDescription, systemImage: "cpu")
            case .failed(let why):
                Label(why, systemImage: "xmark.octagon").foregroundStyle(.red)
            }
        }
    }

    // Downloaded initial conditions only; fetching more lives in the header's download
    // menu, so actions and data don't share one list.
    @ViewBuilder private var icSection: some View {
        Section {
            if let d = app.download {
                VStack(alignment: .leading, spacing: 6) {
                    ProgressView(value: Double(d.completed), total: Double(max(d.total, 1)))
                    Text(d.total > 0 ? "\(d.completed)/\(d.total) fields · \(d.megabytes) MB" : d.message)
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Cancel", role: .cancel) { app.cancelDownload() }
                        .font(.caption)
                        .buttonStyle(.borderless)
                }
            } else if app.initialConditions.isEmpty {
                // First run: the one thing to do, in plain sight.
                Button {
                    app.downloadIC()
                } label: {
                    Label("Download the latest cycle", systemImage: "icloud.and.arrow.down")
                }
                .disabled(app.modelState == .missing)
            }
            ForEach(app.initialConditions) { ic in
                HStack {
                    Image(systemName: selectedIC == ic.url ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(selectedIC == ic.url ? Color.accentColor : .secondary)
                    VStack(alignment: .leading) {
                        Text(ic.date.utcLabel)
                        Text(ByteCountFormatter.string(fromByteCount: Int64(ic.bytes), countStyle: .file))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .contentShape(Rectangle())
                .onTapGesture { selectedIC = ic.url }
                .swipeActions { Button("Delete", role: .destructive) { app.delete(ic) } }
            }
        } header: {
            HStack {
                Text("Initial conditions")
                Spacer()
                Menu {
                    Button("Latest cycle", systemImage: "icloud.and.arrow.down") { app.downloadIC() }
                    Button("Earlier cycle…", systemImage: "clock.arrow.circlepath") { showEarlierCycles = true }
                } label: {
                    Image(systemName: "icloud.and.arrow.down")
                        .font(.body.weight(.medium))
                }
                .disabled(app.modelState == .missing || app.download != nil)
                .accessibilityLabel("Download initial conditions")
            }
        }
    }

    @ViewBuilder private var runSection: some View {
        Section("Run") {
            if let r = app.run {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Step \(min(r.step + 1, r.totalSteps)) of \(r.totalSteps)")
                        Spacer()
                        NetworkGlyph(phase: r.phase)
                    }
                    ProgressView(value: r.progress)
                    HStack(spacing: 12) {
                        if let s = r.lastStepSeconds { Text(String(format: "%.0f s/step", s)) }
                        if let p = r.peakMemoryGB { Text(String(format: "peak %.1f GB", p)) }
                        if let a = r.availableMemoryGB { Text(String(format: "%.1f GB free", a)) }
                        thermalLabel
                    }
                    .font(.caption).foregroundStyle(.secondary)
                    if r.stopping {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.mini)
                            Text("Stopping after this step…")
                        }
                        .font(.caption).foregroundStyle(.secondary)
                    } else {
                        // Borderless, so only the button takes the tap, not the whole row.
                        Button("Stop", role: .destructive) { app.cancelForecast() }
                            .font(.caption)
                            .buttonStyle(.borderless)
                    }
                }
            } else {
                leadTimeControl
                Button {
                    if let ic = app.initialConditions.first(where: { $0.url == selectedIC }) {
                        if let existing = app.existingForecast(for: ic) {
                            confirmReplace = (ic, existing)
                        } else {
                            app.startForecast(from: ic, leadHours: leadHours)
                        }
                    }
                } label: {
                    Label("Run forecast", systemImage: "play.fill")
                }
                .disabled(app.modelState != .ready || selectedIC == nil)
                .confirmationDialog(
                    "Replace the existing forecast?",
                    isPresented: Binding(get: { confirmReplace != nil }, set: { if !$0 { confirmReplace = nil } }),
                    titleVisibility: .visible
                ) {
                    Button("Replace", role: .destructive) {
                        if let c = confirmReplace { app.startForecast(from: c.ic, leadHours: leadHours) }
                        confirmReplace = nil
                    }
                    Button("Cancel", role: .cancel) { confirmReplace = nil }
                } message: {
                    if let c = confirmReplace {
                        let hours = c.existing.meta.completedSteps * c.existing.meta.stepHours
                        Text("There is already a forecast from \(c.ic.date.utcLabel) (+\(hours) h\(c.existing.meta.finished ? "" : ", stopped early")). "
                            + "Running again deletes it and starts over.")
                    }
                }
            }
        }
    }

    @ViewBuilder private var leadTimeControl: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Lead time")
                Spacer()
                Button {
                    leadTimeText = "\(leadHours)"
                    typingLeadTime = true
                } label: {
                    Text(leadTimeLabel)
                }
                .buttonStyle(.borderless)
                .monospacedDigit()
                .accessibilityHint("Enter the lead time in hours")
            }
            Slider(
                value: Binding(
                    get: { min(max(Double(leadHours) / 24, 1), 15) },
                    set: { leadHours = Int($0.rounded()) * 24 }),
                in: 1 ... 15, step: 1
            ) {
                Text("Lead time in days")
            } minimumValueLabel: {
                Text("1 d").font(.caption).foregroundStyle(.secondary)
            } maximumValueLabel: {
                Text("15 d").font(.caption).foregroundStyle(.secondary)
            }
        }
        .alert("Lead time in hours", isPresented: $typingLeadTime) {
            TextField("Hours", text: $leadTimeText).keyboardType(.numberPad)
            Button("Set") {
                if let hours = Int(leadTimeText.trimmingCharacters(in: .whitespaces)) {
                    // Nearest whole model step; at least one.
                    let steps = Int((Double(hours) / 6).rounded())
                    leadHours = max(steps, 1) * 6
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Rounded to a multiple of 6.")
        }
    }

    /// In days: "1 day", "2 days", or "1.25 days" for a typed value that is not whole
    /// (lead times are multiples of 6 h, so at most two decimals).
    private var leadTimeLabel: String {
        let days = Double(leadHours) / 24
        var number = String(format: "%.2f", days)
        while number.hasSuffix("0") { number.removeLast() }
        if number.hasSuffix(".") { number.removeLast() }
        return "\(number) day\(days == 1 ? "" : "s")"
    }

    @ViewBuilder private var thermalLabel: some View {
        switch app.thermalState {
        case .serious: Label("hot", systemImage: "thermometer.high").foregroundStyle(.orange)
        case .critical: Label("throttling", systemImage: "thermometer.high").foregroundStyle(.red)
        default: EmptyView()
        }
    }

    @ViewBuilder private var forecastSection: some View {
        Section("Forecasts") {
            if app.forecasts.isEmpty {
                Text("None yet").foregroundStyle(.secondary)
            }
            ForEach(app.forecasts) { f in
                let generating = app.run != nil && app.startedForecastID == f.id
                HStack {
                    VStack(alignment: .leading) {
                        Text("Init \(f.meta.initDate.utcLabel)")
                        Text(
                            (generating ? "Generating · " : "")
                                + "+\(f.meta.completedSteps * f.meta.stepHours) h"
                                + (f.meta.finished ? "" : " of \(f.meta.plannedSteps * f.meta.stepHours) h")
                        )
                        .font(.caption).foregroundStyle(.secondary)
                    }
                    if generating {
                        Spacer()
                        ProgressView().controlSize(.small)
                    }
                }
                .tag(f.id)
                .swipeActions {
                    // Deleting the directory a run is writing into would fail the run.
                    if !generating { Button("Delete", role: .destructive) { app.delete(f) } }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}

/// The cycles ECMWF's mirrors can still serve, to download one from the last few days.
private struct EarlierCyclesSheet: View {
    let app: AppModel
    let onPick: (Date) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if let cycles = app.availableCycles, !app.findingCycles {
                    if cycles.isEmpty {
                        ContentUnavailableView(
                            "Nothing found", systemImage: "icloud.slash",
                            description: Text("No mirror answered for the last five days. Check the connection and try again."))
                    }
                    Section {
                        ForEach(cycles, id: \.self) { date in
                            let have = app.initialConditions.contains { $0.date == date }
                            Button {
                                onPick(date)
                            } label: {
                                // Ordinary text ink, not the button tint.
                                HStack {
                                    Text(date.utcLabel).foregroundStyle(have ? .secondary : .primary)
                                    Spacer()
                                    if have {
                                        Text("Downloaded").font(.caption).foregroundStyle(.secondary)
                                    } else {
                                        Text(age(date)).font(.caption).foregroundStyle(.secondary)
                                    }
                                }
                            }
                            .disabled(have)
                            .tint(.primary)
                        }
                    } footer: {
                        if !cycles.isEmpty {
                            Text("Each is about 420 MB. ECMWF keeps the last few days of cycles; older ones come from the cloud mirrors.")
                        }
                    }
                } else {
                    ProgressView().frame(maxWidth: .infinity)
                }
            }
            .navigationTitle("Earlier cycles")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            }
        }
        .presentationBackground(Color(.systemGroupedBackground))
        .onAppear { app.findAvailableCycles() }
    }

    private func age(_ date: Date) -> String {
        let hours = Int(Date.now.timeIntervalSince(date) / 3600)
        return hours < 48 ? "\(hours) h ago" : "\(hours / 24) days ago"
    }
}
