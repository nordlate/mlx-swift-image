import Foundation

/// Architecture of FLUX.2 Klein 4B's transformer: mflux `Flux2Transformer` built with `ModelConfig.flux2_klein_4b().transformer_overrides`
/// (5 double-stream blocks, 20 single-stream blocks, 24 heads of 128, `joint_attention_dim` 7680, `guidance_embeds=False`). The expected
/// stored tensors are derived from these constants, not read from the snapshot: the headers are validated against them.
///
/// Every `nn.Linear` is stored quantized (4-bit, group size 64, affine): `<name>.weight` (U32, `[out, in * bits / 32]`) and
/// `<name>.scales` / `<name>.biases` (BF16, `[out, in / group]`). The 60 attention RMS-norm weights are BF16 `[128]` (the header shows
/// no float32 tensor anywhere in the transformer, unlike the text encoder's `inv_freq`). 109 matrices x 3 + 60 norms = 387 tensors.
public enum KleinTransformerLayout {
    public static let doubleBlocks = 5
    public static let singleBlocks = 20
    public static let heads = 24
    public static let headDim = 128
    public static let dim = heads * headDim  // 3072
    public static let inChannels = 128
    public static let jointAttentionDim = 7680
    public static let timestepChannels = 256
    public static let mlpHidden = dim * 3  // int(dim * mlp_ratio 3.0)
    public static let bits = 4
    public static let groupSize = 64
    public static let ropeTheta: Float = 2000
    public static let axesDims = [32, 32, 32, 32]
    /// `nn.RMSNorm(dim_head, eps=1e-5)` of the attention q/k norms.
    public static let attentionNormEps: Float = 1e-5
    /// `nn.LayerNorm(dim, eps=1e-6, affine=False)` of every block norm and of `AdaLayerNormContinuous`.
    public static let layerNormEps: Float = 1e-6
    public static let tensorCount = 387

    public struct Matrix: Equatable, Sendable {
        public let name: String
        public let out: Int
        public let input: Int
    }

    /// The 109 quantized matrices: (module path, out features, in features).
    public static func matrices() -> [Matrix] {
        var result: [Matrix] = [
            Matrix(name: "time_guidance_embed.linear_1", out: dim, input: timestepChannels),
            Matrix(name: "time_guidance_embed.linear_2", out: dim, input: dim),
            Matrix(name: "double_stream_modulation_img.linear", out: dim * 6, input: dim),
            Matrix(name: "double_stream_modulation_txt.linear", out: dim * 6, input: dim),
            Matrix(name: "single_stream_modulation.linear", out: dim * 3, input: dim),
            Matrix(name: "x_embedder", out: dim, input: inChannels),
            Matrix(name: "context_embedder", out: dim, input: jointAttentionDim),
            Matrix(name: "norm_out.linear", out: dim * 2, input: dim),
            Matrix(name: "proj_out", out: inChannels, input: dim),
        ]
        for i in 0..<doubleBlocks {
            let p = "transformer_blocks.\(i)."
            for name in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out"] {
                result.append(Matrix(name: p + name, out: dim, input: dim))
            }
            for ff in ["ff", "ff_context"] {
                result.append(Matrix(name: p + ff + ".linear_in", out: mlpHidden * 2, input: dim))
                result.append(Matrix(name: p + ff + ".linear_out", out: dim, input: mlpHidden))
            }
        }
        for i in 0..<singleBlocks {
            let p = "single_transformer_blocks.\(i).attn."
            result.append(Matrix(name: p + "to_qkv_mlp_proj", out: dim * 3 + mlpHidden * 2, input: dim))
            result.append(Matrix(name: p + "to_out", out: dim, input: dim + mlpHidden))
        }
        return result
    }

    /// The 60 un-quantized attention norm weights.
    public static func normNames() -> [String] {
        var result: [String] = []
        for i in 0..<doubleBlocks {
            for name in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] { result.append("transformer_blocks.\(i).attn.\(name).weight") }
        }
        for i in 0..<singleBlocks {
            for name in ["norm_q", "norm_k"] { result.append("single_transformer_blocks.\(i).attn.\(name).weight") }
        }
        return result
    }

    /// Every stored tensor: safetensors dtype string and shape.
    public static func expectedTensors() -> [String: (dtype: String, shape: [Int])] {
        var result: [String: (dtype: String, shape: [Int])] = [:]
        for m in matrices() {
            result[m.name + ".weight"] = ("U32", [m.out, m.input * bits / 32])
            result[m.name + ".scales"] = ("BF16", [m.out, m.input / groupSize])
            result[m.name + ".biases"] = ("BF16", [m.out, m.input / groupSize])
        }
        for name in normNames() { result[name] = ("BF16", [headDim]) }
        return result
    }
}
