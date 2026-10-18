import Foundation
import MLX

/// Initial conditions: the two input times of every field the model reads from the state,
/// as written by `./aifs export ic` (`ic_<date>.safetensors`).
public struct InitialCondition: @unchecked Sendable {
    /// Valid time of the *latest* input slot; the first is `timestepHours` earlier.
    public let date: Date
    /// name -> `(2, N)` float32.
    public let fields: [String: MLXArray]

    public init(date: Date, fields: [String: MLXArray]) {
        self.date = date
        self.fields = fields
    }

    public static func load(from url: URL) throws -> InitialCondition {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        guard let iso = metadata["date"], let date = parseDate(iso) else {
            throw AIFSError.invalidInitialCondition("\(url.lastPathComponent) has no date")
        }
        return InitialCondition(date: date, fields: arrays)
    }

    /// Python's `datetime.isoformat()` of a naive UTC datetime, e.g. `2026-09-24T00:00:00`.
    static func parseDate(_ text: String) -> Date? {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f.date(from: text.hasSuffix("Z") || text.contains("+") ? text : text + "Z")
    }
}

/// A rollout in progress: the `(2, N, V)` input window and the date of its latest slot.
///
/// Mirrors anemoi-inference's `Runner.forecast`: after each step the window rolls back by
/// one, the prognostic outputs fill the newest slot, and the dynamic forcings are
/// recomputed for that slot's valid time. Constant fields carry over unchanged.
public final class Forecast: @unchecked Sendable {
    public let model: AIFSModel
    public let initDate: Date
    public private(set) var step = 0
    public private(set) var input: MLXArray

    private let forcings: Forcings
    private let computedColumns: [(column: Int, name: String)]
    private enum Source { case output(Int), forcing(String), carried }
    /// Where each input column of the next window's newest slot comes from.
    private let nextInputSource: [Source]

    public init(model: AIFSModel, initialCondition ic: InitialCondition) throws {
        self.model = model
        let m = model.manifest
        initDate = ic.date
        let forcings = Forcings(latitudes: model.latitudes, longitudes: model.longitudes)
        self.forcings = forcings

        let shape = [m.multiStep, m.numDataNodes]
        var computed: [(Int, String)] = []
        for (i, name) in m.inputVars.enumerated() where m.computedForcings.contains(name) {
            guard Forcings.isComputed(name) else {
                throw AIFSError.invalidBundle("no on-device formula for forcing \(name)")
            }
            computed.append((i, name))
        }
        computedColumns = computed

        let dt = TimeInterval(m.timestepHours * 3600)
        var columns: [MLXArray] = []
        for name in m.inputVars {
            if m.computedForcings.contains(name) {
                let slots = (0 ..< m.multiStep).map { t in
                    forcings.field(name, at: ic.date.addingTimeInterval(dt * Double(t - m.multiStep + 1)))
                }
                columns.append(MLXArray(slots.flatMap { $0 }, shape))
            } else {
                guard let f = ic.fields[name] else {
                    throw AIFSError.invalidInitialCondition("missing field \(name)")
                }
                guard f.shape == shape else {
                    throw AIFSError.invalidInitialCondition("\(name) has shape \(f.shape), want \(shape)")
                }
                columns.append(f.asType(.float32))
            }
        }
        input = stacked(columns, axis: 2)
        eval(input)

        var source = [Source](repeating: .carried, count: m.inputVars.count)
        for (i, o) in zip(m.prognosticInput, m.prognosticOutput) { source[i] = .output(o) }
        for c in computed { source[c.0] = .forcing(c.1) }
        nextInputSource = source
    }

    /// The newest slot of the next window: prognostic outputs, forcings recomputed for
    /// `date`, and -- for constant fields -- whatever the window already held. (anemoi
    /// rolls the window, which leaves the old first slot here; constants are equal in both.)
    private func nextSlot(output y: MLXArray, date: Date) -> MLXArray {
        let n = model.manifest.numDataNodes
        let oldest = input[0]
        let columns = nextInputSource.enumerated().map { i, source in
            switch source {
            case .output(let o): y[0..., o]
            case .forcing(let name): MLXArray(forcings.field(name, at: date), [n])
            case .carried: oldest[0..., i]
            }
        }
        return stacked(columns, axis: 1)
    }

    /// Valid time of the next step's output.
    public var nextValidDate: Date {
        initDate.addingTimeInterval(TimeInterval((step + 1) * model.manifest.timestepHours * 3600))
    }

    /// Run one step; returns the physical `(N, 120)` output valid at the returned date.
    public func advance() -> (date: Date, output: MLXArray) {
        let date = nextValidDate
        let y = model.predict(input)
        step += 1
        input = stacked([input[1], nextSlot(output: y, date: date)], axis: 0)
        eval(input)
        Memory.clearCache()
        return (date, y)
    }
}
