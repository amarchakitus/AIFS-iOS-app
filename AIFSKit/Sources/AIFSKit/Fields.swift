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

/// A minimal safetensors reader in plain Swift, for files the viewers need (display and
/// model grids, step files): no MLX, so drawing a map never starts the MLX runtime -- and
/// the viewers run where MLX cannot (the iOS simulator).
enum SafeTensors {
    struct Tensor {
        let dtype: String
        let shape: [Int]
        let range: Range<Int>  // byte range within the file
    }

    static func open(_ url: URL) throws -> (data: Data, tensors: [String: Tensor], metadata: [String: String]) {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 8 else { throw CocoaError(.fileReadCorruptFile) }
        let n = Int(data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        guard n < 10_000_000, 8 + n <= data.count,
            var header = try JSONSerialization.jsonObject(with: data[8 ..< 8 + n]) as? [String: Any]
        else { throw CocoaError(.fileReadCorruptFile) }
        let metadata = header.removeValue(forKey: "__metadata__") as? [String: String] ?? [:]
        var tensors: [String: Tensor] = [:]
        for (name, value) in header {
            guard let info = value as? [String: Any], let dtype = info["dtype"] as? String,
                let shape = info["shape"] as? [Int], let o = info["data_offsets"] as? [Int], o.count == 2
            else { continue }
            tensors[name] = Tensor(dtype: dtype, shape: shape, range: (8 + n + o[0]) ..< (8 + n + o[1]))
        }
        return (data, tensors, metadata)
    }

    /// Copy a tensor's elements out (memcpy, so file alignment does not matter).
    static func array<T>(_ data: Data, _ tensor: Tensor, as: T.Type) -> [T] where T: Numeric {
        let count = tensor.range.count / MemoryLayout<T>.size
        var out = [T](repeating: 0, count: count)
        out.withUnsafeMutableBytes { dst in
            data.withUnsafeBytes { src in
                dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[tensor.range]))
            }
        }
        return out
    }
}

/// Named float32 fields in a safetensors file.
public enum FieldFile {
    public static func write(_ fields: [String: [Float]], metadata: [String: String] = [:], to url: URL) throws {
        let arrays = fields.mapValues { MLXArray($0) }
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
    }

    public static func read(_ url: URL) throws -> [String: [Float]] {
        let (data, tensors, _) = try SafeTensors.open(url)
        return tensors.filter { $0.value.dtype == "F32" }.mapValues { SafeTensors.array(data, $0, as: Float.self) }
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
        let (data, tensors, metadata) = try SafeTensors.open(bundle.appending(path: "display.safetensors"))
        guard let rows = metadata["rows"].flatMap(Int.init), let cols = metadata["cols"].flatMap(Int.init),
            let indptr = tensors["indptr"], let indices = tensors["indices"], let weights = tensors["weights"]
        else { throw AIFSError.invalidBundle("display.safetensors lacks the grid shape") }
        self.rows = rows
        self.cols = cols
        let idx = SafeTensors.array(data, indices, as: Int32.self)
        op = SparseOperator(
            indptr: SafeTensors.array(data, indptr, as: Int32.self), indices: idx,
            weights: SafeTensors.array(data, weights, as: Float.self), cols: Int(idx.max() ?? 0) + 1)
        guard op.rows == rows * cols else { throw AIFSError.invalidBundle("display operator shape") }
    }

    public func regrid(_ field: [Float]) -> [Float] {
        op.apply(field).map(Float.init)
    }
}

/// Process-wide MLX memory knobs and readings, and where diagnostics go.
public enum AIFSRuntime {
    /// Receives the library's diagnostics (a Neural Engine fallback taken, say). Silent
    /// unless set: the CLI and developer builds of the app print them. Set once at launch.
    nonisolated(unsafe) public static var log: @Sendable (String) -> Void = { _ in }
    /// Bytes MLX may keep cached for reuse. On iPad, keep this small: cached buffers count
    /// against the app's memory limit just like live ones.
    public static func setCacheLimit(_ bytes: Int) { Memory.cacheLimit = bytes }
    public static var activeMemory: Int { Memory.activeMemory }
    public static var peakMemory: Int { Memory.peakMemory }
    public static func resetPeakMemory() { Memory.peakMemory = 0 }
    public static func clearCache() { Memory.clearCache() }
}

/// The model's own grid points, for looking up the forecast at a place.
public final class ModelGrid: Sendable {
    public let latitudes: [Float]
    public let longitudes: [Float]

    public init(bundle: URL) throws {
        let (data, tensors, _) = try SafeTensors.open(bundle.appending(path: "grid.safetensors"))
        guard let lat = tensors["latitudes"], let lon = tensors["longitudes"], lat.dtype == "F32", lon.dtype == "F32"
        else { throw AIFSError.invalidBundle("grid.safetensors lacks float32 coordinates") }
        latitudes = SafeTensors.array(data, lat, as: Float.self)
        longitudes = SafeTensors.array(data, lon, as: Float.self)
    }

    /// Index of the grid point nearest `(latitude, longitude)` (degrees; any longitude
    /// convention), by great-circle distance. N320 spacing is ~31 km, so this is within
    /// ~16 km of any point.
    public func nearestIndex(latitude: Double, longitude: Double) -> Int {
        func unit(_ lat: Double, _ lon: Double) -> (Double, Double, Double) {
            let (p, l) = (lat * .pi / 180, lon * .pi / 180)
            return (cos(p) * cos(l), cos(p) * sin(l), sin(p))
        }
        let target = unit(latitude, longitude)
        var best = 0, bestDot = -2.0
        for i in 0 ..< latitudes.count {
            // Cheap latitude prefilter: points over ~1 degree away cannot be the nearest.
            if abs(Double(latitudes[i]) - latitude) > 1 { continue }
            let u = unit(Double(latitudes[i]), Double(longitudes[i]))
            let dot = u.0 * target.0 + u.1 * target.1 + u.2 * target.2
            if dot > bestDot { (best, bestDot) = (i, dot) }
        }
        return best
    }
}

extension FieldFile {
    /// Only the named `fields` of a step file (each read by its byte range, so the globe
    /// pays ~2 MB per field instead of the whole ~24 MB file). Missing fields are omitted.
    public static func read(_ url: URL, fields: [String]) throws -> [String: [Float]] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let (header, dataStart) = try readHeader(handle)
        var out: [String: [Float]] = [:]
        for field in fields {
            guard let info = header[field] as? [String: Any], info["dtype"] as? String == "F32",
                let offsets = info["data_offsets"] as? [Int]
            else { continue }
            try handle.seek(toOffset: dataStart + UInt64(offsets[0]))
            guard let bytes = try handle.read(upToCount: offsets[1] - offsets[0]),
                bytes.count == offsets[1] - offsets[0]
            else { throw CocoaError(.fileReadCorruptFile) }
            out[field] = bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        return out
    }

    static func readHeader(_ handle: FileHandle) throws -> (header: [String: Any], dataStart: UInt64) {
        guard let head = try handle.read(upToCount: 8), head.count == 8 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let n = Int(head.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        guard n < 10_000_000, let json = try handle.read(upToCount: n),
            let header = try JSONSerialization.jsonObject(with: json) as? [String: Any]
        else { throw CocoaError(.fileReadCorruptFile) }
        return (header, UInt64(8 + n))
    }

    /// Values of `fields` at grid `indices`, read directly from the file without loading
    /// whole fields (a step file is ~24 MB; a place needs a few bytes of it). Fields the
    /// file lacks are omitted. Result: field -> one value per index.
    public static func values(_ url: URL, fields: [String], at indices: [Int]) throws -> [String: [Float]] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let (header, dataStart) = try readHeader(handle)
        var out: [String: [Float]] = [:]
        for field in fields {
            guard let info = header[field] as? [String: Any], info["dtype"] as? String == "F32",
                let offsets = info["data_offsets"] as? [Int]
            else { continue }
            out[field] = try indices.map { i in
                try handle.seek(toOffset: dataStart + UInt64(offsets[0] + i * 4))
                guard let bytes = try handle.read(upToCount: 4), bytes.count == 4 else {
                    throw CocoaError(.fileReadCorruptFile)
                }
                return bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
            }
        }
        return out
    }
}
