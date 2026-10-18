import CAEC
import Foundation

/// Just enough GRIB2 to read ECMWF open data: a regular lat/lon grid (template 3.0) with
/// CCSDS packing (5.42, what open data uses) or simple packing (5.0), and an optional
/// bitmap. Anything else is rejected rather than misread.
public struct GRIB2Field: Sendable {
    public let ni: Int
    public let nj: Int
    /// First grid point, degrees.
    public let firstLatitude: Double
    public let firstLongitude: Double
    /// Row-major `(nj, ni)` in scanning order; NaN where the bitmap marks a point missing.
    public let values: [Float]
}

public enum GRIB2Error: LocalizedError {
    case malformed(String)
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .malformed(let why): "Malformed GRIB2 message: \(why)"
        case .unsupported(let why): "Unsupported GRIB2 feature: \(why)"
        }
    }
}

public enum GRIB2 {
    /// Decode the single message in `data`.
    public static func decode(_ data: Data) throws -> GRIB2Field {
        try data.withUnsafeBytes { raw in
            try decode(Reader(bytes: raw.bindMemory(to: UInt8.self)))
        }
    }

    struct Reader {
        let bytes: UnsafeBufferPointer<UInt8>

        func u8(_ i: Int) -> Int { Int(bytes[i]) }
        func u16(_ i: Int) -> Int { u8(i) << 8 | u8(i + 1) }
        func u32(_ i: Int) -> Int { u16(i) << 16 | u16(i + 2) }
        func u64(_ i: Int) -> Int { u32(i) << 32 | u32(i + 4) }
        /// GRIB2 signed integers are sign-and-magnitude, not two's complement.
        func s16(_ i: Int) -> Int {
            let v = u16(i)
            return v & 0x8000 != 0 ? -(v & 0x7FFF) : v
        }
        func s32(_ i: Int) -> Int {
            let v = u32(i)
            return v & 0x8000_0000 != 0 ? -(v & 0x7FFF_FFFF) : v
        }
        func f32(_ i: Int) -> Float { Float(bitPattern: UInt32(u32(i))) }
    }

    static func decode(_ r: Reader) throws -> GRIB2Field {
        let n = r.bytes.count
        guard n >= 16, r.u32(0) == 0x4752_4942 /* "GRIB" */ else {
            throw GRIB2Error.malformed("no GRIB header")
        }
        guard r.u8(7) == 2 else { throw GRIB2Error.unsupported("edition \(r.u8(7))") }
        let total = r.u64(8)
        guard total <= n else { throw GRIB2Error.malformed("truncated: \(n) of \(total) bytes") }

        var ni = 0, nj = 0, points = 0, lat1 = 0.0, lon1 = 0.0
        var template5 = -1, encodedValues = 0
        var reference: Float = 0, binaryScale = 0, decimalScale = 0, bits = 0
        var ccsdsFlags = 0, blockSize = 0, rsi = 0
        var bitmap: Range<Int>?
        var payload: Range<Int>?

        var offset = 16
        while offset + 4 <= total {
            if r.u32(offset) == 0x3737_3737 /* "7777" */ { break }
            let length = r.u32(offset)
            guard length >= 5, offset + length <= total else {
                throw GRIB2Error.malformed("section at \(offset) overruns the message")
            }
            let s = offset
            switch r.u8(s + 4) {
            case 3:
                points = r.u32(s + 6)
                let template = r.u16(s + 12)
                guard template == 0 else { throw GRIB2Error.unsupported("grid template 3.\(template)") }
                ni = r.u32(s + 30)
                nj = r.u32(s + 34)
                // Microdegrees unless basic angle/subdivisions say otherwise (0/missing).
                lat1 = Double(r.s32(s + 46)) * 1e-6
                lon1 = Double(r.s32(s + 50)) * 1e-6
                let scanning = r.u8(s + 71)
                // 0 = west to east, north to south, rows consecutive: all open data uses.
                guard scanning == 0 else { throw GRIB2Error.unsupported("scanning mode \(scanning)") }
            case 5:
                encodedValues = r.u32(s + 5)
                template5 = r.u16(s + 9)
                reference = r.f32(s + 11)
                binaryScale = r.s16(s + 15)
                decimalScale = r.s16(s + 17)
                bits = r.u8(s + 19)
                switch template5 {
                case 0:
                    break
                case 42:
                    ccsdsFlags = r.u8(s + 21)
                    blockSize = r.u8(s + 22)
                    rsi = r.u16(s + 23)
                default:
                    throw GRIB2Error.unsupported("data representation template 5.\(template5)")
                }
            case 6:
                switch r.u8(s + 5) {
                case 0: bitmap = (s + 6) ..< (s + length)
                case 255: bitmap = nil
                case let indicator: throw GRIB2Error.unsupported("bitmap indicator \(indicator)")
                }
            case 7:
                payload = (s + 5) ..< (s + length)
            default:
                break
            }
            offset += length
        }
        guard ni > 0, nj > 0, ni * nj == points else { throw GRIB2Error.malformed("grid section") }
        guard template5 >= 0, let payload else { throw GRIB2Error.malformed("no data section") }

        let packed = try unpack(
            r, payload, count: encodedValues, bits: bits, template: template5,
            ccsdsFlags: ccsdsFlags, blockSize: blockSize, rsi: rsi)

        // Y = (R + X * 2^E) / 10^D, evaluated in double as eccodes does.
        let bscale = pow(2.0, Double(binaryScale))
        let dscale = pow(10.0, Double(-decimalScale))
        let ref = Double(reference)
        func value(_ x: UInt32) -> Float { Float((ref + Double(x) * bscale) * dscale) }

        var values = [Float](repeating: .nan, count: points)
        if let bitmap {
            var k = 0
            for i in 0 ..< points {
                let byte = r.bytes[bitmap.lowerBound + i >> 3]
                if byte & (0x80 >> UInt8(i & 7)) != 0 {
                    guard k < packed.count else { throw GRIB2Error.malformed("bitmap/data count") }
                    values[i] = value(packed[k])
                    k += 1
                }
            }
        } else {
            guard packed.count == points else { throw GRIB2Error.malformed("data count") }
            for i in 0 ..< points { values[i] = value(packed[i]) }
        }
        return GRIB2Field(ni: ni, nj: nj, firstLatitude: lat1, firstLongitude: lon1, values: values)
    }

    /// The packed integers X of section 7.
    static func unpack(
        _ r: Reader, _ payload: Range<Int>, count: Int, bits: Int, template: Int,
        ccsdsFlags: Int, blockSize: Int, rsi: Int
    ) throws -> [UInt32] {
        if bits == 0 { return [UInt32](repeating: 0, count: count) }  // constant field
        guard bits <= 32 else { throw GRIB2Error.unsupported("\(bits) bits per value") }

        if template == 0 {
            var out = [UInt32](repeating: 0, count: count)
            var bit = payload.lowerBound * 8
            for i in 0 ..< count {
                var x: UInt32 = 0
                for _ in 0 ..< bits {
                    let b = r.bytes[bit >> 3] >> UInt8(7 - bit & 7) & 1
                    x = x << 1 | UInt32(b)
                    bit += 1
                }
                out[i] = x
            }
            return out
        }

        // CCSDS: as eccodes does (ECC-1602), ask libaec for native-endian 1/2/4-byte
        // samples rather than the 3-byte big-endian layout the flags may request.
        var flags = UInt32(ccsdsFlags)
        flags &= ~UInt32(AEC_DATA_3BYTE)
        flags &= ~UInt32(AEC_DATA_MSB)  // Apple platforms are little-endian
        let sampleBytes = bits <= 8 ? 1 : bits <= 16 ? 2 : 4
        var output = [UInt8](repeating: 0, count: count * sampleBytes)
        let status: Int32 = output.withUnsafeMutableBufferPointer { out in
            var strm = aec_stream()
            strm.next_in = r.bytes.baseAddress! + payload.lowerBound
            strm.avail_in = payload.count
            strm.next_out = out.baseAddress!
            strm.avail_out = out.count
            strm.bits_per_sample = UInt32(bits)
            strm.block_size = UInt32(blockSize)
            strm.rsi = UInt32(rsi)
            strm.flags = flags
            return aec_buffer_decode(&strm)
        }
        guard status == AEC_OK else { throw GRIB2Error.malformed("CCSDS decode failed (\(status))") }

        return output.withUnsafeBytes { raw in
            switch sampleBytes {
            case 1: raw.bindMemory(to: UInt8.self).map(UInt32.init)
            case 2: raw.bindMemory(to: UInt16.self).map(UInt32.init)
            default: Array(raw.bindMemory(to: UInt32.self))
            }
        }
    }
}
