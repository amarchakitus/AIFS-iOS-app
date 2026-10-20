import SwiftUI

/// The downloaded models and what each takes, for deleting one (swipe) or several (Edit).
struct ModelStorageView: View {
    let app: AppModel
    /// Installed models' bundle sizes; nil until measured.
    @Binding var sizes: [ForecastModel: Int]?
    @State private var selection = Set<ForecastModel>()
    /// Ours, not the navigation stack's: the list and the toolbar's buttons must share it (an
    /// `EditButton` in the toolbar toggles the stack's, which this list doesn't read).
    @State private var editMode: EditMode = LaunchOptions.editModels ? .active : .inactive
    @State private var pending: Set<ForecastModel> = []
    @State private var confirming = false  // an alert: a centred question with a clear Cancel

    private var installed: [ForecastModel] {
        ForecastModel.allCases.filter { (sizes?[$0] ?? 0) > 0 }
    }

    var body: some View {
        // Selectable only while editing: outside it a tap would just leave a row highlighted.
        List(selection: editMode.isEditing ? $selection : nil) {
            ForEach(installed, id: \.self) { model in
                LabeledContent(model.title, value: ByteCountFormatter.string(fromByteCount: Int64(sizes?[model] ?? 0), countStyle: .file))
                    .swipeActions {
                        if app.canDeleteModels {
                            Button("Delete", role: .destructive) { ask(about: [model]) }
                        }
                    }
            }
        }
        .overlay {
            if sizes == nil {
                ProgressView()
            } else if installed.isEmpty {
                ContentUnavailableView("No models downloaded", systemImage: "cpu")
            }
        }
        .environment(\.editMode, $editMode)
        .navigationTitle("Models")
        .task {
            if sizes == nil { sizes = await Self.measure() }
            if LaunchOptions.editModels, let first = installed.first {
                selection = [first]
                if LaunchOptions.confirm { ask(about: selection) }
            }
        }
        .toolbar {
            if !installed.isEmpty {
                ToolbarItem(placement: .primaryAction) {
                    Button(editMode.isEditing ? "Done" : "Edit") { setEditing(!editMode.isEditing) }
                        .fontWeight(editMode.isEditing ? .semibold : .regular)
                        .disabled(!app.canDeleteModels)
                }
            }
            if editMode.isEditing {
                ToolbarItem(placement: .bottomBar) {
                    Button(selection.isEmpty ? "Delete" : "Delete (\(selection.count))", role: .destructive) { ask(about: selection) }
                        .tint(.red)
                        .disabled(selection.isEmpty || !app.canDeleteModels)
                }
            }
        }
        .alert(title, isPresented: $confirming) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                app.deleteModels(pending)
                setEditing(false)
                Task { sizes = await Self.measure() }
            }
        } message: {
            Text(pending.count == 1 ? "Its initial conditions go too. Forecasts are kept."
                : "Their initial conditions go too. Forecasts are kept.")
        }
    }

    private var title: String {
        pending.count == 1 ? "Delete \(pending.first!.title)?" : "Delete \(pending.count) models?"
    }

    private func setEditing(_ editing: Bool) {
        withAnimation {
            editMode = editing ? .active : .inactive
            if !editing { selection = [] }
        }
    }

    private func ask(about models: Set<ForecastModel>) {
        pending = models
        confirming = true
    }

    /// Each installed model's bundle size (uninstalled ones absent), off the main thread.
    static func measure() async -> [ForecastModel: Int] {
        await Task.detached {
            var out: [ForecastModel: Int] = [:]
            for model in ForecastModel.allCases {
                let bytes = InitialConditionStore.bundleBytes(model)
                if bytes > 0 { out[model] = bytes }
            }
            return out
        }.value
    }
}
