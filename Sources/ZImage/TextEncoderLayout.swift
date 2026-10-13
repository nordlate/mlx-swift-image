import Foundation

/// The architecture of Z-Image Turbo's text encoder (a Qwen3 4B decoder) as mflux 0.18 defines it, and the exact
/// tensor names/dtypes/shapes the existing q4 snapshot must contain. Constants come from mflux's `TextEncoder`
/// defaults and were checked against the snapshot headers; the loader refuses anything that differs.
public enum TextEncoderLayout {
    public static let vocabSize = 151_936
    public static let hiddenSize = 2560
    public static let layerCount = 36
    public static let attentionHeads = 32
    public static let keyValueHeads = 8
    public static let intermediateSize = 9728
    public static let headDim = 128
    public static let ropeTheta: Float = 1_000_000
    public static let layerNormEps: Float = 1e-6
    /// mflux builds q_norm/k_norm with `nn.RMSNorm(head_dim)`, i.e. MLX's default eps, not the layer eps.
    public static let qkNormEps: Float = 1e-5
    public static let bits = 4
    public static let groupSize = 64

    /// A 4-bit affine-quantized matrix `[out, in]`: packed U32 `[out, in*bits/32]`, BF16 scales/biases `[out, in/group]`.
    struct Quantized {
        let name: String
        let out: Int
        let input: Int
    }

    static func layerMatrices(_ layer: Int) -> [Quantized] {
        let p = "layers.\(layer)."
        let qDim = attentionHeads * headDim
        let kvDim = keyValueHeads * headDim
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

    /// Every tensor the snapshot must hold, with its stored dtype and shape.
    public static func expectedTensors() -> [String: (dtype: String, shape: [Int])] {
        var result: [String: (dtype: String, shape: [Int])] = [:]
        func quantized(_ m: Quantized) {
            result[m.name + ".weight"] = ("U32", [m.out, m.input * bits / 32])
            result[m.name + ".scales"] = ("BF16", [m.out, m.input / groupSize])
            result[m.name + ".biases"] = ("BF16", [m.out, m.input / groupSize])
        }
        quantized(Quantized(name: "embed_tokens", out: vocabSize, input: hiddenSize))
        result["norm.weight"] = ("BF16", [hiddenSize])
        result["rotary_emb.inv_freq"] = ("F32", [headDim / 2])
        for layer in 0..<layerCount {
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
