import Foundation
import MLX

enum KleinTensorValues {
    // mlx-swift's byte accessors build an unsafe buffer pointer that is invalid for zero-element arrays, so the conversion is
    // guarded: a zero-element array has no values, after (not instead of) the dtype/shape checks.
    static func floats(_ array: MLXArray) -> [Float] { array.size == 0 ? [] : array.asType(.float32).asArray(Float.self) }

    static func bytes(_ array: MLXArray) -> Data { array.size == 0 ? Data() : array.asData(access: .copy).data }
}
