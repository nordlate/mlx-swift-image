import MLX

/// The pieces of the Qwen3 q4 text encoder that Z-Image Turbo's and Klein's encoders compute identically (same MLX calls, same dtypes).
/// Everything that differs between them (norm formulation and eps, dtype flow, mask, rotation order, attention precision) stays in
/// each model's own encoder.
package enum QwenOps {
    /// `quantized_matmul(x, W, scales, biases, transpose=True)` for the q4 matrix stored under `name`.
    package static func linear(_ x: MLXArray, _ name: String, tensor: (String) -> MLXArray) -> MLXArray {
        quantizedMM(
            x, tensor(name + ".weight"), scales: tensor(name + ".scales"), biases: tensor(name + ".biases"),
            transpose: true, groupSize: TextEncoderLayout.groupSize, bits: TextEncoderLayout.bits)
    }

    /// The dequantized rows of the q4 `embed_tokens` table for `ids` (bfloat16, as stored).
    package static func embed(_ ids: MLXArray, tensor: (String) -> MLXArray) -> MLXArray {
        let name = "embed_tokens"
        return dequantized(
            tensor(name + ".weight")[ids], scales: tensor(name + ".scales")[ids], biases: tensor(name + ".biases")[ids],
            groupSize: TextEncoderLayout.groupSize, bits: TextEncoderLayout.bits)
    }

    package static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }
}
