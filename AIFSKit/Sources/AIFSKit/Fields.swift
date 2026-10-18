import Foundation
import MLX

/// Plain-array access to model outputs and stored fields, so apps need not import MLX.

/// One forecast step's output, valid at `date`.
public struct StepOutput: @unchecked Sendable {
    public let step: Int
    public let date: Date
    let y: MLXArray
    let names: [String]

    public init(step: Int, date: Date, y: MLXArray, names: [String]) {
        (self.step, self.date, self.y, self.names) = (step, date, y, names)
    }

    /// `name` over the N320 grid, or nil if the model does not output it.
    public func field(_ name: String) -> [Float]? {
        guard let i = names.firstIndex(of: name) else { return nil }
        return y[0..., i].asArray(Float.self)
    }

    public func fields(_ wanted: [String]) -> [String: [Float]] {
        var out: [String: [Float]] = [:]
        for name in wanted { out[name] = field(name) }
        return out
    }
}

extension Forecast {
    /// `advance()`, wrapped for callers that do not use MLX.
    public func advanceStep() -> StepOutput {
        let (date, y) = advance()
        return StepOutput(step: step, date: date, y: y, names: model.manifest.outputVars)
    }
}

extension InitialCondition {
    /// The latest input slot of `name`, e.g. the land-sea mask for drawing coastlines.
    public func latest(_ name: String) -> [Float]? {
        fields[name].map { $0[-1].asArray(Float.self) }
    }

    /// Valid time of the latest slot, read from a file without loading the fields.
    public static func peekDate(_ url: URL) -> Date? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let head = try? handle.read(upToCount: 8), head.count == 8 else { return nil }
        let n = head.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard n < 10_000_000, let json = try? handle.read(upToCount: Int(n)),
            let header = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
            let meta = header["__metadata__"] as? [String: String], let iso = meta["date"]
        else { return nil }
        return parseDate(iso)
    }
}

/// Named float32 fields in a safetensors file.
public enum FieldFile {
    public static func write(_ fields: [String: [Float]], metadata: [String: String] = [:], to url: URL) throws {
        let arrays = fields.mapValues { MLXArray($0) }
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
    }

    public static func read(_ url: URL) throws -> [String: [Float]] {
        try loadArrays(url: url).mapValues { $0.asType(.float32).asArray(Float.self) }
    }
}

/// Interpolation from the model grid to the 0.25-degree lat/lon grid for drawing maps:
/// the operator the Python Zarr writer uses, from the bundle's `display.safetensors`.
/// Output is row-major `(rows, cols)`, rows 90N -> 90S, columns 0E -> 359.75E.
public final class DisplayGrid: Sendable {
    public let rows: Int
    public let cols: Int
    private let op: SparseOperator

    public init(bundle: URL) throws {
        let (arrays, metadata) = try loadArraysAndMetadata(url: bundle.appending(path: "display.safetensors"))
        guard let rows = metadata["rows"].flatMap(Int.init), let cols = metadata["cols"].flatMap(Int.init)
        else { throw AIFSError.invalidBundle("display.safetensors lacks the grid shape") }
        self.rows = rows
        self.cols = cols
        let indices = arrays["indices"]!.asArray(Int32.self)
        op = SparseOperator(
            indptr: arrays["indptr"]!.asArray(Int32.self), indices: indices,
            weights: arrays["weights"]!.asArray(Float.self), cols: Int(indices.max() ?? 0) + 1)
        guard op.rows == rows * cols else { throw AIFSError.invalidBundle("display operator shape") }
    }

    public func regrid(_ field: [Float]) -> [Float] {
        op.apply(field).map(Float.init)
    }
}

/// Process-wide MLX memory knobs and readings.
public enum AIFSRuntime {
    /// Bytes MLX may keep cached for reuse. On iPad, keep this small: cached buffers count
    /// against the app's memory limit just like live ones.
    public static func setCacheLimit(_ bytes: Int) { Memory.cacheLimit = bytes }
    public static var activeMemory: Int { Memory.activeMemory }
    public static var peakMemory: Int { Memory.peakMemory }
    public static func resetPeakMemory() { Memory.peakMemory = 0 }
    public static func clearCache() { Memory.clearCache() }
}
