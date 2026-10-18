import SwiftUI

@main struct MyApp: App {
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        // No GPU work while not in the foreground: a running forecast waits at its next
        // safe point and resumes on return (see RunGate).
        .onChange(of: scenePhase) { _, phase in RunGate.shared.setOpen(phase == .active) }
    }
}
