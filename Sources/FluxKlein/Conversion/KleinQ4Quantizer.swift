import Foundation

/// Affine 4-bit, group-size-64 quantization of bf16 matrices, computed on the CPU with the exact arithmetic of MLX 0.32's Metal
/// `affine_quantize` kernel, which `mflux-save` runs (through `nn.quantize`) on the GPU. The result is byte-identical to that kernel's,
/// so no GPU kernel is needed. MLX's own CPU quantizer (`fast::Quantize::eval_cpu`) is not used because it differs from the Metal
/// kernel in two places and so gives different bytes for some groups:
///
/// - the kernel starts the group maximum at 0 (not at minus infinity), so a group of only negative values gets `w_max = 0`;
/// - the kernel rounds with Metal's `round` (half away from zero), the CPU path with `rint` (half to even).
///
/// Per group of 64 values, in float32 (the Metal library is built with `-fno-fast-math`, so every operation is IEEE-rounded):
/// `scale = max((w_max - w_min) / 15, 1e-7)`, negated unless `|w_min| > |w_max|`; `edge` is `w_min` or `w_max` accordingly;
/// `q0 = round(edge / scale)`; unless `q0 == 0`, `scale = edge / q0` and `bias = edge` (else `bias = 0`). Each value is stored as
/// `min(round((w - bias) / scale), 15)`, eight per little-endian UInt32, the first in the low bits. Scales and biases are stored as bf16
/// (round to nearest even); the values are quantized with the float32 scale and bias, not the rounded ones.
public enum KleinQ4Quantizer {
    public static let bits = 4
    public static let groupSize = 64
    /// Quantized values per UInt32 word.
    public static let valuesPerWord = 32 / bits

    /// Packed words, scales and biases for one matrix `[rows, columns]`.
    public struct Output: Equatable {
        public var weight: [UInt32]
        public var scales: [UInt16]
        public var biases: [UInt16]
    }

    /// Quantizes a row-major bf16 matrix given as raw bf16 bit patterns. `columns` must be a multiple of 64.
    public static func quantize(bf16 values: [UInt16], rows: Int, columns: Int) -> Output {
        precondition(values.count == rows * columns && columns % groupSize == 0, "matrix must be [rows, columns] with columns a multiple of 64")
        let groups = rows * columns / groupSize
        var out = Output(weight: [UInt32](repeating: 0, count: rows * columns / valuesPerWord),
                         scales: [UInt16](repeating: 0, count: groups), biases: [UInt16](repeating: 0, count: groups))
        values.withUnsafeBufferPointer { src in
            out.weight.withUnsafeMutableBufferPointer { w in
                out.scales.withUnsafeMutableBufferPointer { s in
                    out.biases.withUnsafeMutableBufferPointer { b in
                        quantizeGroups(source: src.baseAddress!, groupCount: groups, weight: w.baseAddress!, scales: s.baseAddress!, biases: b.baseAddress!)
                    }
                }
            }
        }
        return out
    }

    /// Quantizes `groupCount` consecutive groups of 64 bf16 values (a whole number of rows when called per row block). Writes
    /// `groupCount * 8` words and `groupCount` scales and biases.
    @inline(__always)
    public static func quantizeGroups(
        source: UnsafePointer<UInt16>, groupCount: Int,
        weight: UnsafeMutablePointer<UInt32>, scales: UnsafeMutablePointer<UInt16>, biases: UnsafeMutablePointer<UInt16>
    ) {
        let nBins: Float = 15
        let eps: Float = 1e-7
        let bf16Max = bf16ToFloat(0x7F7F)  // Limits<bfloat16_t>::max, the kernel's start value for the minimum
        var values = [Float](repeating: 0, count: groupSize)
        values.withUnsafeMutableBufferPointer { v in
            for g in 0..<groupCount {
                let src = source + g * groupSize
                var wMin = bf16Max
                var wMax: Float = 0
                for i in 0..<groupSize {
                    let x = bf16ToFloat(src[i])
                    v[i] = x
                    wMin = Swift.min(wMin, x)
                    wMax = Swift.max(wMax, x)
                }
                var scale = Swift.max((wMax - wMin) / nBins, eps)
                let side = abs(wMin) > abs(wMax)
                scale = side ? scale : -scale
                let edge = side ? wMin : wMax
                let q0 = (edge / scale).rounded(.toNearestOrAwayFromZero)
                let atZero = q0 == 0
                scale = atZero ? scale : edge / q0
                let bias: Float = atZero ? 0 : edge
                scales[g] = floatToBF16(scale)
                biases[g] = floatToBF16(bias)
                let words = weight + g * (groupSize / valuesPerWord)
                for w in 0..<(groupSize / valuesPerWord) {
                    var word: UInt32 = 0
                    for k in 0..<valuesPerWord {
                        let q = Swift.min(((v[w * valuesPerWord + k] - bias) / scale).rounded(.toNearestOrAwayFromZero), nBins)
                        // Values are non-negative by construction (at most -0.0); clamp anyway rather than trap on a conversion.
                        word |= UInt32(Swift.max(q, 0)) << UInt32(bits * k)
                    }
                    words[w] = word
                }
            }
        }
    }

    @inline(__always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }

    /// float32 to bf16, round to nearest even (the conversion of Metal's `bfloat` and of MLX's CPU `bfloat16_t`); NaN stays NaN.
    @inline(__always)
    public static func floatToBF16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        if value.isNaN { return UInt16(truncatingIfNeeded: (bits >> 16) | 0x40) }
        let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }
}
