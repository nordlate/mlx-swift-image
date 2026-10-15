import Foundation
import MLX
import MLXFast
import MLXNN
import ZImageNative

/// Port of mflux's `Qwen3TextEncoder` (`flux2/model/flux2_text_encoder/qwen3_text_encoder.py`) as Klein 4B runs it, op for op.
///
/// This is deliberately not the Z-Image encoder (`TextEncoder`): the dtype flow, norms, mask, rotation order and attention
/// precision all differ.
///  - Everything stays in the embedding dtype (bfloat16: the dequantized embedding is not cast to float32). The mask, cos and sin
///    are created in that dtype.
///  - `Qwen3VLRMSNorm` is explicit: cast to float32, `mean(square)`, `rsqrt(var + eps)`, multiply by the float32 weight, cast back.
///    The q/k norms use the same class with eps 1e-6 (not Z-Image's MLX-default 1e-5).
///  - q and k are normalized, then transposed to `[B, H, S, D]`, then rotated (cos/sin `[B, 1, S, D]`); Z-Image rotates first.
///  - Attention runs in float32: q, k, v are cast up, the additive bfloat16 mask is applied, the result is cast back to bfloat16.
///    GQA expansion is `expand_dims` + `broadcast_to` + `reshape`.
///  - The rotary `inv_freq` is the stored float32 tensor: mflux's saved-format loader applies every stored tensor, and
///    `rotary_emb.inv_freq` is one of the 905, so it overwrites the value the module computed.
///  - `hidden_states_list[0]` is the embedding output and `[i]` the output of layer `i - 1`. Layers past the highest requested
///    index and the final norm cannot influence the requested states, so they are not evaluated.
public struct KleinTextEncoder {
    public static let selectedLayers: [Int] = [9, 18, 27]
    static let rmsNormEps: Float = 1e-6
    static let attentionScaling: Float = 1.0

    let weights: KleinTextWeights
    let invFreq: MLXArray

    public init(weights: KleinTextWeights) {
        self.weights = weights
        self.invFreq = weights.tensor("rotary_emb.inv_freq")
    }

    // MARK: Ops

    /// `arange(n)` as int32. mlx-swift's Swift-collection initializers hand an unsafe buffer pointer to the core that is invalid for
    /// zero elements, so a zero-length range is built by shape instead (the same empty int32 array `mx.arange(0)` gives).
    static func arange32(_ n: Int) -> MLXArray {
        n == 0 ? MLXArray.zeros([0], dtype: .int32) : MLXArray(0 ..< Int32(n))
    }

    /// A token batch of `shape` and `dtype` from Swift values; zero elements are built by shape, never from an empty Swift array.
    public static func tokenArray(_ values: [Int64], shape: [Int], dtype: DType) -> MLXArray {
        values.isEmpty ? MLXArray.zeros(shape, dtype: dtype) : MLXArray(values).asType(dtype).reshaped(shape)
    }

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        quantizedMM(
            x, weights.tensor(name + ".weight"), scales: weights.tensor(name + ".scales"), biases: weights.tensor(name + ".biases"),
            transpose: true, groupSize: TextEncoderLayout.groupSize, bits: TextEncoderLayout.bits)
    }

    private func embed(_ ids: MLXArray) -> MLXArray {
        let name = "embed_tokens"
        return dequantized(
            weights.tensor(name + ".weight")[ids], scales: weights.tensor(name + ".scales")[ids],
            biases: weights.tensor(name + ".biases")[ids], groupSize: TextEncoderLayout.groupSize, bits: TextEncoderLayout.bits)
    }

    /// `Qwen3VLRMSNorm.__call__`.
    static func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float = rmsNormEps) -> MLXArray {
        let f = x.asType(.float32)
        let variance = mean(square(f), axis: -1, keepDims: true)
        let normalized = f * rsqrt(variance + eps)
        return (weight.asType(.float32) * normalized).asType(x.dtype)
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    /// `Qwen3TextEncoder.__call__`'s additive mask `[B, 1, S, S]` in `dtype`: causal (-inf above the diagonal) plus -inf on padded keys.
    static func additiveMask(attentionMask: MLXArray, seqLen: Int, dtype: DType) -> MLXArray {
        let batch = attentionMask.dim(0)
        let zerosMask = MLXArray.zeros(attentionMask.shape, dtype: dtype)
        let negInfMask = MLXArray.full(attentionMask.shape, values: MLXArray(-Float.infinity), dtype: dtype)
        let padding = MLX.where(attentionMask .== 1, zerosMask, negInfMask)
        let paddingMask = expandedDimensions(expandedDimensions(padding, axis: 1), axis: 1)
        let causal: MLXArray
        if seqLen == 1 {
            causal = MLXArray.zeros([batch, 1, 1, 1], dtype: dtype)
        } else {
            let idx = arange32(seqLen)
            let above = expandedDimensions(idx, axis: 0) .> expandedDimensions(idx, axis: 1)  // [r, c] = c > r
            let tri = MLX.where(
                above, MLXArray.full([seqLen, seqLen], values: MLXArray(-Float.infinity), dtype: dtype),
                MLXArray.zeros([seqLen, seqLen], dtype: dtype))
            causal = broadcast(expandedDimensions(expandedDimensions(tri, axis: 0), axis: 0), to: [batch, 1, seqLen, seqLen])
        }
        return causal + paddingMask
    }

    /// `Qwen3TextRotaryEmbedding.__call__` for positions `0..<seqLen`: cos/sin `[B, S, 128]` cast to `dtype`.
    func positionEmbeddings(batch: Int, seqLen: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
        let positionIDs = broadcast(expandedDimensions(Self.arange32(seqLen), axis: 0), to: [batch, seqLen])
        let pos = expandedDimensions(positionIDs.asType(.float32), axis: -1)
        let inv = expandedDimensions(expandedDimensions(invFreq, axis: 0), axis: 0)
        let freqs = pos * inv
        let emb = concatenated([freqs, freqs], axis: -1)
        return ((MLX.cos(emb) * Self.attentionScaling).asType(dtype), (MLX.sin(emb) * Self.attentionScaling).asType(dtype))
    }

    private func attention(_ h: MLXArray, layer: Int, mask: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let p = "layers.\(layer).self_attn."
        let (batch, seq) = (h.dim(0), h.dim(1))
        let heads = TextEncoderLayout.attentionHeads, kvHeads = TextEncoderLayout.keyValueHeads, dim = TextEncoderLayout.headDim
        var q = linear(h, p + "q_proj").reshaped(batch, seq, heads, dim)
        var k = linear(h, p + "k_proj").reshaped(batch, seq, kvHeads, dim)
        var v = linear(h, p + "v_proj").reshaped(batch, seq, kvHeads, dim)
        q = Self.rmsNorm(q, weight: weights.tensor(p + "q_norm.weight"))
        k = Self.rmsNorm(k, weight: weights.tensor(p + "k_norm.weight"))
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        let c = expandedDimensions(cos, axis: 1), s = expandedDimensions(sin, axis: 1)
        q = q * c + Self.rotateHalf(q) * s
        k = k * c + Self.rotateHalf(k) * s
        let groups = heads / kvHeads
        func repeatKV(_ x: MLXArray) -> MLXArray {
            let (b, n, len, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            return broadcast(expandedDimensions(x, axis: 2), to: [b, n, groups, len, d]).reshaped(b, n * groups, len, d)
        }
        let keys = repeatKV(k), values = repeatKV(v)
        let scale = Float(1.0 / Double(dim).squareRoot())
        let out = MLXFast.scaledDotProductAttention(
            queries: q.asType(.float32), keys: keys.asType(.float32), values: values.asType(.float32), scale: scale, mask: .array(mask))
        let merged = out.asType(q.dtype).transposed(0, 2, 1, 3).reshaped(batch, seq, heads * dim)
        return linear(merged, p + "o_proj")
    }

    // MARK: Forward

    /// The states of `hidden_states_list` at the requested indices, each `[B, S, 2560]` bfloat16 (index 0 is the embedding output).
    /// Nothing past the highest requested index is evaluated; the returned arrays are lazy until the caller evaluates them.
    public func hiddenStates(inputIDs: MLXArray, attentionMask: MLXArray, indices: Set<Int>) -> [Int: MLXArray] {
        precondition(!indices.isEmpty && indices.allSatisfy { (0...TextEncoderLayout.layerCount).contains($0) })
        let (batch, seq) = (inputIDs.dim(0), inputIDs.dim(1))
        var h = embed(inputIDs)
        let mask = Self.additiveMask(attentionMask: attentionMask, seqLen: seq, dtype: h.dtype)
        let (cos, sin) = positionEmbeddings(batch: batch, seqLen: seq, dtype: h.dtype)
        var captured: [Int: MLXArray] = [:]
        if indices.contains(0) { captured[0] = h }
        for layer in 0..<(indices.max()!) {
            let p = "layers.\(layer)."
            let ln1 = Self.rmsNorm(h, weight: weights.tensor(p + "input_layernorm.weight"))
            h = h + attention(ln1, layer: layer, mask: mask, cos: cos, sin: sin)
            let ln2 = Self.rmsNorm(h, weight: weights.tensor(p + "post_attention_layernorm.weight"))
            let gate = silu(linear(ln2, p + "mlp.gate_proj"))
            h = h + linear(gate * linear(ln2, p + "mlp.up_proj"), p + "mlp.down_proj")
            if indices.contains(layer + 1) { captured[layer + 1] = h }
        }
        return captured
    }

    /// `Qwen3TextEncoder.get_prompt_embeds`'s assembly: stack the selected states on axis 1 -> `[B, L, S, H]`, transpose to
    /// `[B, S, L, H]` and reshape to `[B, S, L * H]` (layer-major within each token).
    public static func assemble(_ states: [MLXArray]) -> MLXArray {
        let stack = stacked(states, axis: 1)
        let (batch, layers, seq, hidden) = (stack.dim(0), stack.dim(1), stack.dim(2), stack.dim(3))
        return stack.transposed(0, 2, 1, 3).reshaped(batch, seq, layers * hidden)
    }
}
