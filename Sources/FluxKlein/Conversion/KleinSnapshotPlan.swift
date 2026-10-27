import Foundation
import MLXImageCore

/// How one stored tensor of the converted snapshot is made from the official bf16 weights.
enum KleinTensorRecipe: Equatable {
    /// The bf16 source tensor, unchanged.
    case copy(source: String)
    /// A PyTorch convolution kernel `[out, in, kh, kw]` stored as MLX's `[out, kh, kw, in]` (`WeightTransforms.transpose_conv2d_weight`).
    case transposedConv(source: String)
    /// One of the three outputs of quantizing the bf16 matrix `source` (`KleinQ4Quantizer`).
    case quantized(source: String, part: QuantizedPart)
    /// Fixed bytes (`rotary_emb.inv_freq`, which has no source tensor).
    case constant([UInt8])

    enum QuantizedPart: Equatable { case weight, scales, biases }

    var source: String? {
        switch self {
        case .copy(let s), .transposedConv(let s), .quantized(let s, _): s
        case .constant: nil
        }
    }
}

/// One stored tensor, in the order `mflux-save` writes it.
struct KleinPlannedTensor: Equatable {
    let name: String
    let dtype: String
    let shape: [Int]
    let recipe: KleinTensorRecipe

    var byteCount: Int { shape.reduce(1, *) * (dtype == "BF16" ? 2 : 4) }
}

/// One folder of the converted snapshot (`transformer`, `text_encoder` or `vae`): its tensors in `mflux-save`'s parameter order
/// (`tree_flatten(model.parameters())` of the quantized mflux module), the source tensors each comes from, and the shard split.
struct KleinComponentPlan {
    let folder: String
    let tensors: [KleinPlannedTensor]
    /// Source tensors of the official release that `mflux-save` does not store (`lm_head.weight`, `bn.num_batches_tracked`).
    let droppedSources: [String: (dtype: String, shape: [Int])]
    /// Expected bf16 shape of every source tensor that is used.
    let sources: [String: [Int]]

    /// Tensor indices per shard: `ModelSaver._split_weights`, greedy in parameter order, a new shard whenever the next tensor would
    /// take the current one over 2 GiB.
    var shards: [[Int]] { Self.split(byteCounts: tensors.map(\.byteCount)) }

    static let maxShardBytes = 2 << 30

    static func split(byteCounts: [Int], limit: Int = maxShardBytes) -> [[Int]] {
        var shards: [[Int]] = []
        var shard: [Int] = []
        var size = 0
        for (index, bytes) in byteCounts.enumerated() {
            // As in mflux: the check runs before the first tensor too, so a single tensor over the limit would leave an empty first shard.
            if size + bytes > limit {
                shards.append(shard)
                shard = []
                size = 0
            }
            shard.append(index)
            size += bytes
        }
        if !shard.isEmpty { shards.append(shard) }
        return shards
    }
}

/// The converted snapshot for a variant: the three weight folders in the order `mflux-save` writes them.
struct KleinSnapshotPlan {
    let variant: KleinVariant
    let components: [KleinComponentPlan]

    init(variant: KleinVariant) {
        self.variant = variant
        components = [Self.vae(), Self.transformer(variant.transformer), Self.textEncoder(variant.textEncoder)]
    }

    // MARK: Builders

    private struct Builder {
        var tensors: [KleinPlannedTensor] = []
        var sources: [String: [Int]] = [:]

        mutating func quantized(_ name: String, from source: String, out: Int, input: Int, bias: Bool = false) {
            let (bits, group) = (KleinQ4Quantizer.bits, KleinQ4Quantizer.groupSize)
            tensors.append(KleinPlannedTensor(name: name + ".weight", dtype: "U32", shape: [out, input * bits / 32],
                                              recipe: .quantized(source: source + ".weight", part: .weight)))
            tensors.append(KleinPlannedTensor(name: name + ".scales", dtype: "BF16", shape: [out, input / group],
                                              recipe: .quantized(source: source + ".weight", part: .scales)))
            tensors.append(KleinPlannedTensor(name: name + ".biases", dtype: "BF16", shape: [out, input / group],
                                              recipe: .quantized(source: source + ".weight", part: .biases)))
            sources[source + ".weight"] = [out, input]
            if bias { copy(name + ".bias", from: source + ".bias", shape: [out]) }
        }

        mutating func copy(_ name: String, from source: String, shape: [Int]) {
            tensors.append(KleinPlannedTensor(name: name, dtype: "BF16", shape: shape, recipe: .copy(source: source)))
            sources[source] = shape
        }

        mutating func conv(_ name: String, from source: String, input: Int, output: Int, kernel: Int = 3) {
            tensors.append(KleinPlannedTensor(name: name + ".weight", dtype: "BF16", shape: [output, kernel, kernel, input],
                                              recipe: .transposedConv(source: source + ".weight")))
            sources[source + ".weight"] = [output, input, kernel, kernel]
            copy(name + ".bias", from: source + ".bias", shape: [output])
        }

        /// `nn.GroupNorm` registers its bias before its weight.
        mutating func groupNorm(_ name: String, channels: Int) {
            copy(name + ".bias", from: name + ".bias", shape: [channels])
            copy(name + ".weight", from: name + ".weight", shape: [channels])
        }
    }

    /// mflux `Flux2Transformer`: the embedders and modulations, the double-stream blocks, the single-stream blocks, then the output layers.
    static func transformer(_ layout: KleinTransformerLayout) -> KleinComponentPlan {
        var b = Builder()
        let (dim, mlp, head) = (layout.dim, layout.mlpHidden, KleinTransformerLayout.headDim)
        func linear(_ name: String, _ out: Int, _ input: Int, source: String? = nil) {
            b.quantized(name, from: source ?? name, out: out, input: input)
        }
        func norm(_ name: String) { b.copy(name + ".weight", from: name + ".weight", shape: [head]) }
        linear("time_guidance_embed.linear_1", dim, KleinTransformerLayout.timestepChannels, source: "time_guidance_embed.timestep_embedder.linear_1")
        linear("time_guidance_embed.linear_2", dim, dim, source: "time_guidance_embed.timestep_embedder.linear_2")
        linear("double_stream_modulation_img.linear", dim * 6, dim)
        linear("double_stream_modulation_txt.linear", dim * 6, dim)
        linear("single_stream_modulation.linear", dim * 3, dim)
        linear("x_embedder", dim, KleinTransformerLayout.inChannels)
        linear("context_embedder", dim, layout.jointAttentionDim)
        for i in 0..<layout.doubleBlocks {
            let p = "transformer_blocks.\(i)."
            linear(p + "attn.to_q", dim, dim)
            linear(p + "attn.to_k", dim, dim)
            linear(p + "attn.to_v", dim, dim)
            norm(p + "attn.norm_q")
            norm(p + "attn.norm_k")
            linear(p + "attn.to_out", dim, dim, source: p + "attn.to_out.0")
            norm(p + "attn.norm_added_q")
            norm(p + "attn.norm_added_k")
            linear(p + "attn.add_q_proj", dim, dim)
            linear(p + "attn.add_k_proj", dim, dim)
            linear(p + "attn.add_v_proj", dim, dim)
            linear(p + "attn.to_add_out", dim, dim)
            for ff in ["ff", "ff_context"] {
                linear(p + ff + ".linear_in", mlp * 2, dim)
                linear(p + ff + ".linear_out", dim, mlp)
            }
        }
        for i in 0..<layout.singleBlocks {
            let p = "single_transformer_blocks.\(i).attn."
            linear(p + "to_qkv_mlp_proj", dim * 3 + mlp * 2, dim)
            norm(p + "norm_q")
            norm(p + "norm_k")
            linear(p + "to_out", dim, dim + mlp)
        }
        linear("norm_out.linear", dim * 2, dim)
        linear("proj_out", KleinTransformerLayout.inChannels, dim)
        return KleinComponentPlan(folder: "transformer", tensors: b.tensors, droppedSources: [:], sources: b.sources)
    }

    /// mflux's Qwen3 text encoder: the embedding, each layer, the final norm and the rotary `inv_freq` (computed, not loaded).
    static func textEncoder(_ layout: TextEncoderLayout) -> KleinComponentPlan {
        var b = Builder()
        let hidden = layout.hiddenSize, head = TextEncoderLayout.headDim
        b.quantized("embed_tokens", from: "model.embed_tokens", out: TextEncoderLayout.vocabSize, input: hidden)
        for layer in 0..<TextEncoderLayout.layerCount {
            let p = "layers.\(layer)."
            let m = layout.layerMatrices(layer)  // q, k, v, o, gate, up, down
            func linear(_ q: TextEncoderLayout.Quantized) { b.quantized(q.name, from: "model." + q.name, out: q.out, input: q.input) }
            b.copy(p + "input_layernorm.weight", from: "model." + p + "input_layernorm.weight", shape: [hidden])
            m[0...3].forEach(linear)
            b.copy(p + "self_attn.q_norm.weight", from: "model." + p + "self_attn.q_norm.weight", shape: [head])
            b.copy(p + "self_attn.k_norm.weight", from: "model." + p + "self_attn.k_norm.weight", shape: [head])
            b.copy(p + "post_attention_layernorm.weight", from: "model." + p + "post_attention_layernorm.weight", shape: [hidden])
            m[4...6].forEach(linear)
        }
        b.copy("norm.weight", from: "model.norm.weight", shape: [hidden])
        b.tensors.append(KleinPlannedTensor(name: "rotary_emb.inv_freq", dtype: "F32", shape: [head / 2],
                                            recipe: .constant(KleinConversionConstants.qwen3InvFreq)))
        return KleinComponentPlan(folder: "text_encoder", tensors: b.tensors,
                                  droppedSources: ["lm_head.weight": ("BF16", [TextEncoderLayout.vocabSize, hidden])], sources: b.sources)
    }

    /// mflux's Flux2 VAE (the same for every Klein size): encoder, decoder, the quant convolutions and the latent batch-norm statistics.
    static func vae() -> KleinComponentPlan {
        var b = Builder()
        func resnet(_ n: String, _ i: Int, _ o: Int) {
            b.groupNorm(n + ".norm1", channels: i)
            b.conv(n + ".conv1", from: n + ".conv1", input: i, output: o)
            b.groupNorm(n + ".norm2", channels: o)
            b.conv(n + ".conv2", from: n + ".conv2", input: o, output: o)
            if i != o { b.conv(n + ".conv_shortcut", from: n + ".conv_shortcut", input: i, output: o, kernel: 1) }
        }
        func midBlock(_ n: String) {
            for j in 0..<2 { resnet(n + ".resnets.\(j)", 512, 512) }
            let a = n + ".attentions.0"
            b.groupNorm(a + ".group_norm", channels: 512)
            for proj in ["to_q", "to_k", "to_v"] { b.quantized(a + "." + proj, from: a + "." + proj, out: 512, input: 512, bias: true) }
            b.quantized(a + ".to_out", from: a + ".to_out.0", out: 512, input: 512, bias: true)
        }
        b.conv("encoder.conv_in", from: "encoder.conv_in", input: 3, output: 128)
        let enc = [128, 256, 512, 512]
        for i in 0..<4 {
            for j in 0..<2 { resnet("encoder.down_blocks.\(i).resnets.\(j)", j == 0 && i > 0 ? enc[i - 1] : enc[i], enc[i]) }
            if i < 3 { b.conv("encoder.down_blocks.\(i).downsamplers.0.conv", from: "encoder.down_blocks.\(i).downsamplers.0.conv", input: enc[i], output: enc[i]) }
        }
        midBlock("encoder.mid_block")
        b.groupNorm("encoder.conv_norm_out", channels: 512)
        b.conv("encoder.conv_out", from: "encoder.conv_out", input: 512, output: 64)
        b.conv("decoder.conv_in", from: "decoder.conv_in", input: 32, output: 512)
        midBlock("decoder.mid_block")
        let dec = [512, 512, 256, 128]
        for i in 0..<4 {
            for j in 0..<3 { resnet("decoder.up_blocks.\(i).resnets.\(j)", j == 0 && i > 0 ? dec[i - 1] : dec[i], dec[i]) }
            if i < 3 { b.conv("decoder.up_blocks.\(i).upsamplers.0.conv", from: "decoder.up_blocks.\(i).upsamplers.0.conv", input: dec[i], output: dec[i]) }
        }
        b.groupNorm("decoder.conv_norm_out", channels: 128)
        b.conv("decoder.conv_out", from: "decoder.conv_out", input: 128, output: 3)
        b.conv("quant_conv", from: "quant_conv", input: 64, output: 64, kernel: 1)
        b.conv("post_quant_conv", from: "post_quant_conv", input: 32, output: 32, kernel: 1)
        b.copy("bn.running_mean", from: "bn.running_mean", shape: [128])
        b.copy("bn.running_var", from: "bn.running_var", shape: [128])
        return KleinComponentPlan(folder: "vae", tensors: b.tensors, droppedSources: ["bn.num_batches_tracked": ("I64", [])], sources: b.sources)
    }
}
