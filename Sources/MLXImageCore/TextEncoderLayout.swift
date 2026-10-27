import Foundation

/// The architecture of the Qwen3 q4 text encoders as mflux defines them, and the exact tensor names/dtypes/shapes a q4 snapshot must
/// contain. Z-Image (Turbo and Base) and FLUX.2 Klein 4B use the Qwen3 4B decoder (`qwen3_4B`); FLUX.2 Klein 9B uses the Qwen3 8B decoder
/// (`qwen3_8B`), which differs only in its hidden and MLP widths. The numerics that differ between the models (norm eps, dtype flow) live
/// with each model's own encoder. Constants come from mflux's `TextEncoder` defaults and `text_encoder_overrides` and were checked against
/// the snapshot headers; the loaders refuse anything that differs.
public struct TextEncoderLayout: Equatable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int

    /// Qwen3 4B (Z-Image, FLUX.2 Klein 4B): hidden 2560, MLP 9728.
    public static let qwen3_4B = TextEncoderLayout(hiddenSize: 2560, intermediateSize: 9728)
    /// Qwen3 8B (FLUX.2 Klein 9B): hidden 4096, MLP 12288.
    public static let qwen3_8B = TextEncoderLayout(hiddenSize: 4096, intermediateSize: 12288)

    // Shared by both sizes.
    public static let vocabSize = 151_936
    public static let layerCount = 36
    public static let attentionHeads = 32
    public static let keyValueHeads = 8
    public static let headDim = 128
    public static let ropeTheta: Float = 1_000_000
    public static let bits = 4
    public static let groupSize = 64

    /// A 4-bit affine-quantized matrix `[out, in]`: packed U32 `[out, in*bits/32]`, BF16 scales/biases `[out, in/group]`.
    package struct Quantized {
        package let name: String
        package let out: Int
        package let input: Int
    }

    package func layerMatrices(_ layer: Int) -> [Quantized] {
        let p = "layers.\(layer)."
        let qDim = Self.attentionHeads * Self.headDim
        let kvDim = Self.keyValueHeads * Self.headDim
        return [
            Quantized(name: p + "self_attn.q_proj", out: qDim, input: hiddenSize),
            Quantized(name: p + "self_attn.k_proj", out: kvDim, input: hiddenSize),
            Quantized(name: p + "self_attn.v_proj", out: kvDim, input: hiddenSize),
            Quantized(name: p + "self_attn.o_proj", out: hiddenSize, input: qDim),
            Quantized(name: p + "mlp.gate_proj", out: intermediateSize, input: hiddenSize),
            Quantized(name: p + "mlp.up_proj", out: intermediateSize, input: hiddenSize),
            Quantized(name: p + "mlp.down_proj", out: hiddenSize, input: intermediateSize),
        ]
    }

    /// Every q4 matrix of the architecture, embedding table first, then the seven matrices of each layer in order.
    package func quantizedMatrices() -> [Quantized] {
        [Quantized(name: "embed_tokens", out: Self.vocabSize, input: hiddenSize)] + (0..<Self.layerCount).flatMap(layerMatrices)
    }

    /// Every tensor the snapshot must hold, with its stored dtype and shape.
    public func expectedTensors() -> [String: (dtype: String, shape: [Int])] {
        var result: [String: (dtype: String, shape: [Int])] = [:]
        let (bits, groupSize, headDim) = (Self.bits, Self.groupSize, Self.headDim)
        func quantized(_ m: Quantized) {
            result[m.name + ".weight"] = ("U32", [m.out, m.input * bits / 32])
            result[m.name + ".scales"] = ("BF16", [m.out, m.input / groupSize])
            result[m.name + ".biases"] = ("BF16", [m.out, m.input / groupSize])
        }
        quantized(Quantized(name: "embed_tokens", out: Self.vocabSize, input: hiddenSize))
        result["norm.weight"] = ("BF16", [hiddenSize])
        result["rotary_emb.inv_freq"] = ("F32", [headDim / 2])
        for layer in 0..<Self.layerCount {
            let p = "layers.\(layer)."
            result[p + "input_layernorm.weight"] = ("BF16", [hiddenSize])
            result[p + "post_attention_layernorm.weight"] = ("BF16", [hiddenSize])
            result[p + "self_attn.q_norm.weight"] = ("BF16", [headDim])
            result[p + "self_attn.k_norm.weight"] = ("BF16", [headDim])
            layerMatrices(layer).forEach(quantized)
        }
        return result
    }
}
