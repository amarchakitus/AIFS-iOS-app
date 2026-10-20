import AuroraANE
import CoreVideo
import Foundation
import Metal

/// The edge gathers on the GPU: an edge's row is its sender's projection plus its receiver's
/// (`a[ia[i] - aBase] + b[ib[i]]`, 512 fp16 each). These are random 1 KB reads, which the
/// CPU's few cores can't issue fast enough to keep up with the Neural Engine (on an iPhone 18
/// Pro Max, 2.1 s a step against its 1.7 s of edge MLPs); the GPU can.
///
/// Every buffer is shared, not copied: the Neural Engine's IOSurfaces and the bundle's
/// mmap'd graph and tables, wrapped as no-copy Metal buffers. iOS refuses GPU work from an app
/// in the background, so a failed command buffer reports false and the caller gathers on the
/// CPU instead.
final class GraphCastGPU: @unchecked Sendable {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let gatherAdd: MTLComputePipelineState
    let graph: MTLBuffer
    let tables: MTLBuffer
    private var surfaces: [ObjectIdentifier: MTLBuffer] = [:]
    private let lock = NSLock()

    static let source = """
        #include <metal_stdlib>
        using namespace metal;
        // out[i] = a[ia[i] - aBase] + b[ib[i]], rows of 128 half4 (512 halves).
        kernel void gather_add(device half4 *out [[buffer(0)]], device const half4 *a [[buffer(1)]],
                               device const half4 *b [[buffer(2)]], device const int *ia [[buffer(3)]],
                               device const int *ib [[buffer(4)]], constant int &aBase [[buffer(5)]],
                               constant uint &rows [[buffer(6)]], uint2 g [[thread_position_in_grid]]) {
            if (g.y >= rows) return;
            out[g.y * 128 + g.x] = a[(ia[g.y] - aBase) * 128 + g.x] + b[ib[g.y] * 128 + g.x];
        }
        """

    init?(graph graphFile: SafeTensorsFile, tables tablesFile: SafeTensorsFile) {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let library = try? device.makeLibrary(source: Self.source, options: nil),
              let function = library.makeFunction(name: "gather_add"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let graph = Self.wrap(device, graphFile.mapping), let tables = Self.wrap(device, tablesFile.mapping)
        else { return nil }
        self.device = device
        self.queue = queue
        gatherAdd = pipeline
        self.graph = graph
        self.tables = tables
    }

    /// A no-copy buffer over page-aligned memory (an mmap or an IOSurface), its length
    /// rounded up to whole pages, which the mapping covers.
    static func wrap(_ device: MTLDevice, _ m: (base: UnsafeRawPointer, length: Int)) -> MTLBuffer? {
        let page = Int(getpagesize())
        let length = (m.length + page - 1) / page * page
        return device.makeBuffer(bytesNoCopy: UnsafeMutableRawPointer(mutating: m.base), length: length,
                                 options: .storageModeShared, deallocator: nil)
    }

    /// The Surface's memory as a Metal buffer (made once; the Surface outlives it).
    func buffer(_ s: Surface) -> MTLBuffer? {
        lock.lock()
        defer { lock.unlock() }
        if let b = surfaces[ObjectIdentifier(s)] { return b }
        CVPixelBufferLockBaseAddress(s.buffer, .readOnly)
        let base = CVPixelBufferGetBaseAddress(s.buffer)!
        let size = IOSurfaceGetAllocSize(CVPixelBufferGetIOSurface(s.buffer)!.takeUnretainedValue())
        CVPixelBufferUnlockBaseAddress(s.buffer, .readOnly)
        guard let b = Self.wrap(device, (UnsafeRawPointer(base), size)) else { return nil }
        surfaces[ObjectIdentifier(s)] = b
        return b
    }

    struct Rows {
        let buffer: MTLBuffer
        let offset: Int
    }

    /// `rows` rows of out = a[ia - aBase] + b[ib], run to completion. False if the GPU
    /// would not run it (the app in the background).
    func gatherAdd(out: Rows, a: Rows, aBase: Int, b: Rows, ia: Rows, ib: Rows, rows: Int) -> Bool {
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(gatherAdd)
        for (i, r) in [out, a, b, ia, ib].enumerated() { enc.setBuffer(r.buffer, offset: r.offset, index: i) }
        var base = Int32(aBase), count = UInt32(rows)
        enc.setBytes(&base, length: 4, index: 5)
        enc.setBytes(&count, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: 128, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 4, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return cb.status == .completed
    }
}
