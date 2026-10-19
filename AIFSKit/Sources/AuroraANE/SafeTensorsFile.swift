import Foundation

/// A safetensors file mapped into memory: tensors are pointers into the mapping, so large
/// read-only tables cost address space, not memory the system must keep resident.
public final class SafeTensorsFile: @unchecked Sendable {
    public struct Tensor: Sendable {
        public let dtype: String
        public let shape: [Int]
        public let offset: Int  // bytes from the start of the file
        public let bytes: Int
    }

    public let tensors: [String: Tensor]
    public let metadata: [String: String]
    private let base: UnsafeMutableRawPointer
    private let length: Int

    public init(_ url: URL) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw CocoaError(.fileReadNoSuchFile, userInfo: [NSFilePathErrorKey: url.path]) }
        defer { close(fd) }
        var st = stat()
        guard fstat(fd, &st) == 0, st.st_size >= 8 else { throw CocoaError(.fileReadCorruptFile) }
        length = Int(st.st_size)
        guard let p = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0), p != MAP_FAILED else {
            throw CocoaError(.fileReadUnknown)
        }
        base = p
        let n = Int(p.loadUnaligned(as: UInt64.self))
        guard n < 100_000_000, 8 + n <= length,
            var header = try JSONSerialization.jsonObject(
                with: Data(bytesNoCopy: p + 8, count: n, deallocator: .none)) as? [String: Any]
        else { throw CocoaError(.fileReadCorruptFile) }
        metadata = header.removeValue(forKey: "__metadata__") as? [String: String] ?? [:]
        var tensors: [String: Tensor] = [:]
        for (name, value) in header {
            guard let info = value as? [String: Any], let dtype = info["dtype"] as? String,
                let shape = info["shape"] as? [Int], let o = info["data_offsets"] as? [Int], o.count == 2
            else { continue }
            tensors[name] = Tensor(dtype: dtype, shape: shape, offset: 8 + n + o[0], bytes: o[1] - o[0])
        }
        self.tensors = tensors
    }

    deinit { munmap(base, length) }

    public func tensor(_ name: String) throws -> Tensor {
        guard let t = tensors[name] else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [NSLocalizedDescriptionKey: "no tensor \(name)"])
        }
        return t
    }

    /// The tensor's elements as `T` (whose size must match its dtype). The pointer lives as
    /// long as this file object. Mapped read-only: never write through it.
    public func pointer<T>(_ name: String, as: T.Type) throws -> UnsafeMutablePointer<T> {
        let t = try tensor(name)
        precondition(t.bytes % MemoryLayout<T>.size == 0)
        return (base + t.offset).bindMemory(to: T.self, capacity: t.bytes / MemoryLayout<T>.size)
    }
}
