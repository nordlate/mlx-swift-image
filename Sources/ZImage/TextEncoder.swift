import Foundation
import MLX
import MLXFast
import MLXNN

/// Port of mflux's Z-Image `TextEncoder` (Qwen3 4B, q4) and `PromptEncoder.encode_prompt`.
///
/// Op-for-op the same graph as the Python reference, including its dtype flow: embeddings are dequantized to
/// bfloat16 and cast to float32, every layer runs in float32 against bf16 norm weights and q4 matrices, and the
/// result is the output of layer 34 (`all_hidden_states[-2]`) cast to bfloat16. Layer 35 and the final norm do not
/// influence that output, so they are not evaluated. `rotary_emb.inv_freq` is taken from the snapshot (float32), as
/// mflux's weight application overwrites its computed value with the stored one.
public struct TextEncoder {
    let weights: TextEncoderWeights
    let invFreq: MLXArray

    public init(weights: TextEncoderWeights) {
        self.weights = weights
        self.invFreq = weights.tensor("rotary_emb.inv_freq")
    }

    /// Number of decoder layers whose output feeds the result (`all_hidden_states[-2]` is layer index 34).
    static let evaluatedLayers = TextEncoderLayout.layerCount - 1

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        quantizedMM(
            x, weights.tensor(name + ".weight"), scales: weights.tensor(name + ".scales"),
            biases: weights.tensor(name + ".biases"), transpose: true,
            groupSize: TextEncoderLayout.groupSize, bits: TextEncoderLayout.bits)
    }

    private func embed(_ ids: MLXArray) -> MLXArray {
        let name = "embed_tokens"
        return dequantized(
            weights.tensor(name + ".weight")[ids], scales: weights.tensor(name + ".scales")[ids],
            biases: weights.tensor(name + ".biases")[ids],
            groupSize: TextEncoderLayout.groupSize, bits: TextEncoderLayout.bits)
    }

    private func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    /// Additive float32 mask `[B, 1, S, S]`: causal, plus -inf on padded key positions.
    static func additiveMask(attentionMask: MLXArray, seqLen: Int) -> MLXArray {
        let idx = MLXArray(0 ..< Int32(seqLen))
        let causal = MLX.where(
            idx[0..., .newAxis] .>= idx[.newAxis, 0...],
            MLXArray.zeros([seqLen, seqLen], dtype: .float32),
            MLXArray.full([seqLen, seqLen], values: MLXArray(-Float.infinity), dtype: .float32))
        let batch = attentionMask.dim(0)
        let padding = MLX.where(
            attentionMask[0..., .newAxis, .newAxis, 0...] .== 1,
            MLXArray.zeros([batch, 1, 1, seqLen], dtype: .float32),
            MLXArray.full([batch, 1, 1, seqLen], values: MLXArray(-Float.infinity), dtype: .float32))
        return causal[.newAxis, .newAxis, 0..., 0...] + padding
    }

    private func attention(
        _ h: MLXArray, layer: Int, mask: MLXArray, cos: MLXArray, sin: MLXArray, trace: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let p = "layers.\(layer).self_attn."
        let (batch, seq) = (h.dim(0), h.dim(1))
        let heads = TextEncoderLayout.attentionHeads, kvHeads = TextEncoderLayout.keyValueHeads, dim = TextEncoderLayout.headDim
        var q = linear(h, p + "q_proj").reshaped(batch, seq, heads, dim)
        var k = linear(h, p + "k_proj").reshaped(batch, seq, kvHeads, dim)
        var v = linear(h, p + "v_proj").reshaped(batch, seq, kvHeads, dim)
        if layer == 0 { trace?("l0.q_proj", q); trace?("l0.k_proj", k); trace?("l0.v_proj", v) }
        q = MLXFast.rmsNorm(q, weight: weights.tensor(p + "q_norm.weight"), eps: TextEncoderLayout.qkNormEps)
        k = MLXFast.rmsNorm(k, weight: weights.tensor(p + "k_norm.weight"), eps: TextEncoderLayout.qkNormEps)
        let c = expandedDimensions(cos, axis: 2), s = expandedDimensions(sin, axis: 2)
        if layer == 0 { trace?("l0.q_norm", q) }
        q = q * c + rotateHalf(q) * s
        k = k * c + rotateHalf(k) * s
        if layer == 0 { trace?("l0.q_rope", q) }
        let groups = heads / kvHeads
        k = repeated(k, count: groups, axis: 2)
        v = repeated(v, count: groups, axis: 2)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: Float(pow(Double(dim), -0.5)), mask: .array(mask))
        let merged = out.transposed(0, 2, 1, 3).reshaped(batch, seq, -1)
        if layer == 0 { trace?("l0.sdpa", merged) }
        return linear(merged, p + "o_proj")
    }

    /// `hidden_states[-2]` for `inputIDs`/`attentionMask` of shape `[B, S]`, as bfloat16 `[B, S, 2560]`.
    /// `trace`, when given, receives the float32 hidden state after the embedding (`embed`) and after each layer (`layerN`).
    public func hiddenStates(inputIDs: MLXArray, attentionMask: MLXArray, trace: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let seq = inputIDs.dim(1)
        var h = embed(inputIDs).asType(.float32)
        trace?("embed", h)
        let freqs = outer(MLXArray(0 ..< Int32(seq)).asType(.float32), invFreq)
        let emb = concatenated([freqs, freqs], axis: -1)
        let cos = MLX.cos(emb)[.newAxis, 0..., 0...].asType(h.dtype)
        let sin = MLX.sin(emb)[.newAxis, 0..., 0...].asType(h.dtype)
        let mask = Self.additiveMask(attentionMask: attentionMask, seqLen: seq)
        for layer in 0..<Self.evaluatedLayers {
            let p = "layers.\(layer)."
            let ln1 = MLXFast.rmsNorm(h, weight: weights.tensor(p + "input_layernorm.weight"), eps: TextEncoderLayout.layerNormEps)
            if layer == 0 { trace?("l0.ln1", ln1) }
            let attn = attention(ln1, layer: layer, mask: mask, cos: cos, sin: sin, trace: trace)
            if layer == 0 { trace?("l0.attn", attn) }
            h = h + attn
            if layer == 0 { trace?("l0.resid1", h) }
            let normed = MLXFast.rmsNorm(h, weight: weights.tensor(p + "post_attention_layernorm.weight"), eps: TextEncoderLayout.layerNormEps)
            if layer == 0 { trace?("l0.ln2", normed) }
            let gate = silu(linear(normed, p + "mlp.gate_proj"))
            let mlp = linear(gate * linear(normed, p + "mlp.up_proj"), p + "mlp.down_proj")
            if layer == 0 { trace?("l0.mlp", mlp) }
            h = h + mlp
            trace?("layer\(layer)", h)
        }
        return h.asType(.bfloat16)
    }

    /// Text embedding of one tokenized prompt: valid rows only, `[numValid, 2560]` bfloat16.
    public func encode(_ tokens: TokenizedPrompt) throws -> MLXArray {
        guard !tokens.inputIDs.isEmpty else { throw WeightLoadError.invalid("empty prompt has no embedding (the reference tokenizer returns a [1, 0] batch)") }
        let ids = MLXArray(tokens.inputIDs).reshaped(1, -1)
        let mask = MLXArray(tokens.attentionMask).reshaped(1, -1)
        let out = hiddenStates(inputIDs: ids, attentionMask: mask)[0, ..<tokens.validCount, 0...]
        eval(out)
        return out
    }
}
