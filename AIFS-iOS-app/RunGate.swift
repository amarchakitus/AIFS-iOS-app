import Foundation

/// Holds a running forecast while the app is not in the foreground. iOS refuses GPU work
/// from a backgrounded app, and MLX treats the refused command buffer as fatal, so the
/// compute thread waits here -- between GPU dispatches, where nothing is in flight --
/// until the app is active again. The run then simply carries on.
nonisolated final class RunGate: @unchecked Sendable {
    static let shared = RunGate()
    private let condition = NSCondition()
    private var open = true

    func setOpen(_ isOpen: Bool) {
        condition.lock()
        open = isOpen
        condition.broadcast()
        condition.unlock()
    }

    /// Returns at once in the foreground; otherwise when the app is back, or the run
    /// is stopped (so Stop still works while paused).
    func pass() {
        condition.lock()
        defer { condition.unlock() }
        while !open && !Task.isCancelled {
            condition.wait(until: Date().addingTimeInterval(0.25))
        }
    }
}
