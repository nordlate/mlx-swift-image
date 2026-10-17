import Foundation
import MLX
import MLXFast
import MLXNN

/// The intermediates both sides record from an eager forward pass, by name. The reference records them from the production modules
/// (wrapping the real `Flux2Transformer`'s submodules); the native side records the same points. Block inputs of the selected blocks
/// feed the isolated one-block checks; every `.in_*` / `.in` key is an input, every other an output.
public enum KleinTransformerTrace {
    public static let doubleBlocks = [0, 2, 4]  // first, middle, last
    public static let singleBlocks = [0, 10, 19]

    /// Embeddings, timestep embedding, modulation parameters and rotary tables.
    public static let embeddingKeys: [String] =
        ["timestep_scaled", "temb_raw", "temb", "x_embed", "ctx_embed"]
        + (0..<6).map { "mod_img.\($0)" } + (0..<6).map { "mod_txt.\($0)" } + (0..<3).map { "mod_single.\($0)" }
        + ["rope_img_cos", "rope_img_sin", "rope_txt_cos", "rope_txt_sin"]

    public static func doubleKeys(_ i: Int) -> [String] {
        ["double.\(i).in_hidden", "double.\(i).in_encoder", "double.\(i).out_hidden", "double.\(i).out_encoder"]
    }
    public static func singleKeys(_ i: Int) -> [String] { ["single.\(i).in", "single.\(i).out"] }

    public static let blockKeys: [String] = doubleBlocks.flatMap(doubleKeys) + singleBlocks.flatMap(singleKeys)
    public static let finalKeys = ["norm_out_in", "norm_out_out"]
    /// Everything an eager pass records (the final output is saved separately as `noise_eager`).
    public static let allKeys: [String] = embeddingKeys + blockKeys + finalKeys
}

/// Port of mflux's `Flux2Transformer` (`flux2/model/flux2_transformer/`), op for op, for FLUX.2 Klein 4B q4 as the production path
/// runs it: `guidance=None`, `kv_cache=None`, no reference-image blending, a single batch row. The dtype flow is the reference's: the
/// timestep is scaled in the latents' dtype, q/k norms run in float32 (joint casts back; single always casts to bfloat16), and RoPE
/// mixes in float32 and casts back. Float32 img2img residuals are preserved. No classifier-free-guidance branch, KV cache, edit conditioning or
/// training support exists here.
public struct KleinTransformer {
    let weights: KleinTransformerWeights

    public init(weights: KleinTransformerWeights) {
        self.weights = weights
        _ = Self.attentionScale  // evaluated now, never inside a compile trace
    }

    /// `1 / mx.sqrt(query.shape[-1])` computed by MLX on the default device, as the reference does, then passed to SDPA as a float.
    static let attentionScale: Float = {
        let scale = MLXArray(Float(1)) / MLX.sqrt(MLXArray(Int32(KleinTransformerLayout.headDim)))
        return scale.item(Float.self)
    }()

    // MARK: Building blocks

    /// `nn.QuantizedLinear.__call__` (no bias stored): `quantized_matmul(x, weight, scales, biases, transpose=True)`.
    public func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        quantizedMM(
            x, weights.tensor(name + ".weight"), scales: weights.tensor(name + ".scales"), biases: weights.tensor(name + ".biases"),
            transpose: true, groupSize: KleinTransformerLayout.groupSize, bits: KleinTransformerLayout.bits)
    }

    /// Joint AttentionUtils restores the projection dtype; parallel single attention explicitly selects ModelConfig.precision (BF16).
    func qkNorm(_ x: MLXArray, _ name: String, outputDType: DType? = nil) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weights.tensor(name + ".weight"), eps: KleinTransformerLayout.attentionNormEps).asType(outputDType ?? x.dtype)
    }

    /// `nn.LayerNorm(dim, eps=1e-6, affine=False)`.
    static func layerNorm(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, eps: KleinTransformerLayout.layerNormEps)
    }

    /// `Flux2SwiGLU`: `silu(x1) * x2` over the two halves of the last axis.
    static func swiGLU(_ x: MLXArray) -> MLXArray {
        let halves = x.split(parts: 2, axis: -1)
        return silu(halves[0]) * halves[1]
    }

    // MARK: Timestep embedding

    /// `Flux2TimestepGuidanceEmbeddings._timestep_embedding` (`flip_sin_to_cos`): `[cos, sin]` of `t * f`, `f = exp(-ln(10000) * i / half)`.
    static func timestepFrequencies(_ timesteps: MLXArray, dim: Int = KleinTransformerLayout.timestepChannels) -> MLXArray {
        let half = dim / 2
        let freqs = MLX.exp(Float(-log(10000.0)) * MLXArray(0 ..< Int32(half)).asType(.float32) / Float(half))
        let args = timesteps.asType(.float32)[0..., .newAxis] * freqs[.newAxis]
        let emb = concatenated([MLX.sin(args), MLX.cos(args)], axis: -1)
        return concatenated([emb[0..., half...], emb[0..., ..<half]], axis: -1)
    }

    /// The scaled timestep as `Flux2Transformer.__call__` prepares it: a scalar becomes `[batch]` in the latents' dtype, and the
    /// `1000` scale applies when the largest value is at most 1.
    static func scaledTimestep(_ timestep: MLXArray, dtype: DType, batch: Int) -> MLXArray {
        var t = timestep
        if t.ndim == 0 { t = MLXArray.full([batch], values: t, dtype: dtype) }
        t = t.asType(dtype)
        let scale = MLX.where(MLX.max(t) .<= MLXArray(Float(1)), MLXArray(Float(1000)), MLXArray(Float(1))).asType(dtype)
        return t * scale
    }

    public func timestepEmbedding(_ scaled: MLXArray) -> MLXArray {
        linear(silu(linear(Self.timestepFrequencies(scaled), "time_guidance_embed.linear_1")), "time_guidance_embed.linear_2")
    }

    // MARK: Rotary

    /// `Flux2PosEmbed.__call__`: per-axis (theta 2000, 4 axes of 32) cos/sin of `id * omega`, concatenated: `[N, 64]` float32 each.
    public static func rotaryTables(ids: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        let pos = ids.asType(.float32)
        var cosOut: [MLXArray] = [], sinOut: [MLXArray] = []
        for (axis, dim) in KleinTransformerLayout.axesDims.enumerated() {
            let scale = MLXArray((0..<(dim / 2)).map { Float($0 * 2) }) / Float(dim)
            let omega = Float(1) / MLX.pow(MLXArray(KleinTransformerLayout.ropeTheta), scale)
            let out = expandedDimensions(pos[0..., axis], axis: -1) * expandedDimensions(omega, axis: 0)
            cosOut.append(MLX.cos(out))
            sinOut.append(MLX.sin(out))
        }
        return (concatenated(cosOut, axis: -1), concatenated(sinOut, axis: -1))
    }

    /// `AttentionUtils.apply_rope_bshd`: interleaved (even, odd) pairs rotated by cos/sin in float32, cast back to the input dtype.
    static func applyRope(_ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
        let cosB = cos.reshaped(1, 1, cos.dim(0), cos.dim(1)), sinB = sin.reshaped(1, 1, sin.dim(0), sin.dim(1))
        func mix(_ x: MLXArray) -> MLXArray {
            let pairs = x.asType(.float32).reshaped(Array(x.shape.dropLast()) + [-1, 2])
            let real = pairs[.ellipsis, 0], imag = pairs[.ellipsis, 1]
            let out0 = real * cosB + (-imag) * sinB
            let out1 = imag * cosB + real * sinB
            return stacked([out0, out1], axis: -1).reshaped(x.shape)
        }
        return (mix(q).asType(q.dtype), mix(k).asType(k.dtype))
    }

    // MARK: Attention

    /// `AttentionUtils.process_qkv`: projections reshaped to `[B, H, S, D]`, q and k normalized in float32.
    func processQKV(_ x: MLXArray, q: String, k: String, v: String, normQ: String, normK: String) -> (MLXArray, MLXArray, MLXArray) {
        let (b, s, h, d) = (x.dim(0), x.dim(1), KleinTransformerLayout.heads, KleinTransformerLayout.headDim)
        func heads(_ name: String) -> MLXArray { linear(x, name).reshaped(b, s, h, d).transposed(0, 2, 1, 3) }
        return (qkNorm(heads(q), normQ), qkNorm(heads(k), normK), heads(v))
    }

    /// `AttentionUtils.compute_attention` (no mask): SDPA, `[B, H, S, D]` to `[B, S, H * D]`.
    static func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: attentionScale, mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
        return out.transposed(0, 2, 1, 3).reshaped(q.dim(0), -1, KleinTransformerLayout.heads * KleinTransformerLayout.headDim)
    }

    /// `Flux2Attention.__call__` with `added_kv_proj_dim` set: text tokens first, then image tokens. Returns (image, text) outputs.
    func jointAttention(_ i: Int, hidden: MLXArray, encoder: MLXArray, cos: MLXArray, sin: MLXArray) -> (hidden: MLXArray, encoder: MLXArray) {
        let p = "transformer_blocks.\(i).attn."
        var (query, key, value) = processQKV(hidden, q: p + "to_q", k: p + "to_k", v: p + "to_v", normQ: p + "norm_q", normK: p + "norm_k")
        let (encQuery, encKey, encValue) = processQKV(
            encoder, q: p + "add_q_proj", k: p + "add_k_proj", v: p + "add_v_proj", normQ: p + "norm_added_q", normK: p + "norm_added_k")
        query = concatenated([encQuery, query], axis: 2)
        key = concatenated([encKey, key], axis: 2)
        value = concatenated([encValue, value], axis: 2)
        (query, key) = Self.applyRope(query, key, cos: cos, sin: sin)
        let merged = Self.attention(query, key, value)
        let encLength = encoder.dim(1)
        let textOut = merged[0..., ..<encLength]
        let imageOut = merged[0..., encLength...]
        return (linear(imageOut, p + "to_out"), linear(textOut, p + "to_add_out"))
    }

    // MARK: Blocks

    func feedForward(_ x: MLXArray, _ name: String) -> MLXArray {
        linear(Self.swiGLU(linear(x, name + ".linear_in")), name + ".linear_out")
    }

    /// `Flux2Modulation.__call__`: `[B, 1, 3 * sets * dim]` split into `sets` (shift, scale, gate) triples.
    public func modulation(_ temb: MLXArray, name: String, sets: Int) -> [[MLXArray]] {
        var mod = linear(silu(temb), name + ".linear")
        if mod.ndim == 2 { mod = expandedDimensions(mod, axis: 1) }
        let parts = mod.split(parts: 3 * sets, axis: -1)
        return (0..<sets).map { Array(parts[(3 * $0)..<(3 * ($0 + 1))]) }
    }

    /// `Flux2TransformerBlock.__call__`. Returns (encoder, hidden).
    public func doubleBlock(
        _ i: Int, hidden: MLXArray, encoder: MLXArray, modImg: [[MLXArray]], modTxt: [[MLXArray]], cos: MLXArray, sin: MLXArray
    ) -> (encoder: MLXArray, hidden: MLXArray) {
        let p = "transformer_blocks.\(i)."
        let (shiftMsa, scaleMsa, gateMsa) = (modImg[0][0], modImg[0][1], modImg[0][2])
        let (shiftMlp, scaleMlp, gateMlp) = (modImg[1][0], modImg[1][1], modImg[1][2])
        let (cShiftMsa, cScaleMsa, cGateMsa) = (modTxt[0][0], modTxt[0][1], modTxt[0][2])
        let (cShiftMlp, cScaleMlp, cGateMlp) = (modTxt[1][0], modTxt[1][1], modTxt[1][2])

        var normHidden = (1 + scaleMsa) * Self.layerNorm(hidden) + shiftMsa
        var normEncoder = (1 + cScaleMsa) * Self.layerNorm(encoder) + cShiftMsa
        let (attnOut, encAttnOut) = jointAttention(i, hidden: normHidden, encoder: normEncoder, cos: cos, sin: sin)
        var h = hidden + gateMsa * attnOut
        var e = encoder + cGateMsa * encAttnOut

        normHidden = (1 + scaleMlp) * Self.layerNorm(h) + shiftMlp
        h = h + gateMlp * feedForward(normHidden, p + "ff")
        normEncoder = (1 + cScaleMlp) * Self.layerNorm(e) + cShiftMlp
        e = e + cGateMlp * feedForward(normEncoder, p + "ff_context")
        return (e, h)
    }

    /// `Flux2SingleTransformerBlock.__call__` with `Flux2ParallelSelfAttention`: one fused qkv + SwiGLU-MLP projection.
    public func singleBlock(_ i: Int, hidden: MLXArray, mod: [MLXArray], cos: MLXArray, sin: MLXArray) -> MLXArray {
        let p = "single_transformer_blocks.\(i).attn."
        let (shift, scale, gate) = (mod[0], mod[1], mod[2])
        let normHidden = (1 + scale) * Self.layerNorm(hidden) + shift
        let proj = linear(normHidden, p + "to_qkv_mlp_proj")
        let inner = KleinTransformerLayout.dim
        let top = proj.split(indices: [inner * 3], axis: -1)
        let qkv = top[0].split(parts: 3, axis: -1)
        let (batch, seq) = (qkv[0].dim(0), qkv[0].dim(1))
        let (h, d) = (KleinTransformerLayout.heads, KleinTransformerLayout.headDim)
        func heads(_ x: MLXArray) -> MLXArray { x.reshaped(batch, seq, h, d).transposed(0, 2, 1, 3) }
        // Flux2ParallelSelfAttention casts normalized Q/K to BF16 before RoPE, even with float32 projections/residuals.
        var query = qkNorm(heads(qkv[0]), p + "norm_q", outputDType: .bfloat16)
        var key = qkNorm(heads(qkv[1]), p + "norm_k", outputDType: .bfloat16)
        let value = heads(qkv[2])
        (query, key) = Self.applyRope(query, key, cos: cos, sin: sin)
        let attn = Self.attention(query, key, value)
        let out = linear(concatenated([attn, Self.swiGLU(top[1])], axis: -1), p + "to_out")
        return hidden + gate * out
    }

    /// `AdaLayerNormContinuous.__call__`: `norm(x) * (1 + scale) + shift` from `linear(silu(temb))`.
    public func normOut(_ x: MLXArray, temb: MLXArray) -> MLXArray {
        let t = linear(silu(temb).asType(.bfloat16), "norm_out.linear")
        let dim = KleinTransformerLayout.dim
        let scale = t[0..., 0 ..< dim], shift = t[0..., dim ..< (2 * dim)]
        return Self.layerNorm(x) * expandedDimensions(1 + scale, axis: 1) + expandedDimensions(shift, axis: 1)
    }

    // MARK: Forward

    /// `Flux2Transformer.__call__` for `guidance=None`, `kv_cache=None`. `hiddenStates` is the packed latents `[B, N, 128]`,
    /// `encoderHiddenStates` the prompt embeddings `[B, S, 7680]`, `timestep` the scheduler's float32 timestep (scalar or `[B]`),
    /// `imgIDs` `[B, N, 4]` / `[N, 4]` and `txtIDs` `[B, S, 4]` / `[S, 4]` (int32). Returns the predicted noise `[B, N, 128]`.
    /// `trace` receives the named intermediates of `KleinTransformerTrace`; it never changes the computation.
    public func forward(
        hiddenStates: MLXArray, encoderHiddenStates: MLXArray, timestep: MLXArray, imgIDs: MLXArray, txtIDs: MLXArray,
        trace: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let dtype = hiddenStates.dtype
        let scaled = Self.scaledTimestep(timestep, dtype: dtype, batch: hiddenStates.dim(0))
        trace?("timestep_scaled", scaled)
        let tembRaw = timestepEmbedding(scaled)
        trace?("temb_raw", tembRaw)
        let temb = tembRaw.asType(.bfloat16)  // ModelConfig.precision
        trace?("temb", temb)

        var hidden = linear(hiddenStates, "x_embedder")
        var encoder = linear(encoderHiddenStates, "context_embedder")
        trace?("x_embed", hidden)
        trace?("ctx_embed", encoder)

        let img = imgIDs.ndim == 3 ? imgIDs[0] : imgIDs
        let txt = txtIDs.ndim == 3 ? txtIDs[0] : txtIDs
        let imageRope = Self.rotaryTables(ids: img), textRope = Self.rotaryTables(ids: txt)
        let cos = concatenated([textRope.cos, imageRope.cos], axis: 0), sin = concatenated([textRope.sin, imageRope.sin], axis: 0)
        if let trace {
            trace("rope_img_cos", imageRope.cos); trace("rope_img_sin", imageRope.sin)
            trace("rope_txt_cos", textRope.cos); trace("rope_txt_sin", textRope.sin)
        }

        let modImg = modulation(temb, name: "double_stream_modulation_img", sets: 2)
        let modTxt = modulation(temb, name: "double_stream_modulation_txt", sets: 2)
        if let trace {
            for (k, a) in (modImg[0] + modImg[1]).enumerated() { trace("mod_img.\(k)", a) }
            for (k, a) in (modTxt[0] + modTxt[1]).enumerated() { trace("mod_txt.\(k)", a) }
        }

        for i in 0..<KleinTransformerLayout.doubleBlocks {
            let traced = trace != nil && KleinTransformerTrace.doubleBlocks.contains(i)
            if traced { trace?("double.\(i).in_hidden", hidden); trace?("double.\(i).in_encoder", encoder) }
            (encoder, hidden) = doubleBlock(i, hidden: hidden, encoder: encoder, modImg: modImg, modTxt: modTxt, cos: cos, sin: sin)
            if traced { trace?("double.\(i).out_hidden", hidden); trace?("double.\(i).out_encoder", encoder) }
        }

        let textLength = encoder.dim(1)
        hidden = concatenated([encoder, hidden], axis: 1)
        let modSingle = modulation(temb, name: "single_stream_modulation", sets: 1)[0]
        if let trace { for (k, a) in modSingle.enumerated() { trace("mod_single.\(k)", a) } }
        for i in 0..<KleinTransformerLayout.singleBlocks {
            let traced = trace != nil && KleinTransformerTrace.singleBlocks.contains(i)
            if traced { trace?("single.\(i).in", hidden) }
            hidden = singleBlock(i, hidden: hidden, mod: modSingle, cos: cos, sin: sin)
            if traced { trace?("single.\(i).out", hidden) }
        }

        hidden = hidden[0..., textLength...]
        trace?("norm_out_in", hidden)
        hidden = normOut(hidden, temb: temb)
        trace?("norm_out_out", hidden)
        return linear(hidden, "proj_out")
    }

    /// `Flux2Klein._predict`'s compiled `predict` for the production call: `mx.compile` of the transformer with `guidance=None` (no
    /// negative branch, since the guidance is 1.0). Arguments: latents, latent IDs, prompt embeddings, text IDs, timestep.
    /// The weights are materialized first so every compiled input signature shares one evaluated copy.
    public func compiledPredict() -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray {
        weights.materialize()
        let compiled = compile { (inputs: [MLXArray]) -> [MLXArray] in
            [self.forward(hiddenStates: inputs[0], encoderHiddenStates: inputs[2], timestep: inputs[4], imgIDs: inputs[1], txtIDs: inputs[3])]
        }
        return { compiled([$0, $1, $2, $3, $4])[0] }
    }
}
