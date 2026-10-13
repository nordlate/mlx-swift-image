import Foundation
import MLX
import MLXFast
import MLXNN

/// Port of the input side of mflux's Z-Image `ZImageTransformer` and of its `ZImageTransformerBlock`: patch, caption and
/// timestep embeddings, rotary tables, and one modulated block. Op-for-op the same graph as the Python reference, so
/// the dtype flow is too (for example, the timestep embedding is float32 and promotes the block's activations to float32).
///
/// Not ported yet (later slices): noise/context refiner passes, the 30-layer loop, the final layer and unpatchify.
public struct ZImageTransformer {
    let weights: TransformerWeights
    let rope: ZImageRope

    public init(weights: TransformerWeights) {
        self.weights = weights
        self.rope = ZImageRope(theta: TransformerLayout.ropeTheta, axesDims: TransformerLayout.axesDims, axesLens: TransformerLayout.axesLens)
    }

    /// `nn.QuantizedLinear.__call__`: `quantized_matmul(x, weight, scales, biases, transpose=True)` plus `bias` when stored.
    func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let y = quantizedMM(
            x, weights.tensor(name + ".weight"), scales: weights.tensor(name + ".scales"), biases: weights.tensor(name + ".biases"),
            transpose: true, groupSize: TransformerLayout.groupSize, bits: TransformerLayout.bits)
        guard let bias = weights.tensors[name + ".bias"] else { return y }
        return y + bias
    }

    func rmsNorm(_ x: MLXArray, _ name: String, eps: Float = TransformerLayout.normEps) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weights.tensor(name + ".weight"), eps: eps)
    }

    // MARK: Timestep embedding

    /// mflux `TimestepEmbedder._timestep_embedding`: `[cos(t * f), sin(t * f)]` with `f = exp(-ln(max_period) * i / half)`.
    static func frequencyEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Double) -> MLXArray {
        precondition(dim % 2 == 0, "mflux zero-pads odd widths; the Z-Image embedder is 256 wide")
        let half = dim / 2
        let freqs = MLX.exp(Float(-log(maxPeriod)) * MLXArray(0 ..< Int32(half)).asType(.float32) / Float(half))
        let args = t[0..., .newAxis].asType(.float32) * freqs[.newAxis]
        return concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
    }

    /// `t_embedder(timestep.astype(float32) * t_scale)` for a `[1]` (or `[N]`) timestep, where mflux's sampler passes
    /// `1 - sigma`. Result `[N, 256]`, float32.
    public func timestepEmbedding(_ timestep: MLXArray) -> MLXArray {
        let scaled = timestep.asType(.float32) * TransformerLayout.timestepScale
        let freq = Self.frequencyEmbedding(scaled, dim: TransformerLayout.timestepFrequencySize, maxPeriod: TransformerLayout.timestepMaxPeriod)
        return linear(silu(linear(freq, "t_embedder.linear1")), "t_embedder.linear2")
    }

    // MARK: Patch and caption embedding

    /// `_create_coord_grid`: `[prod(size), 3]` int32 coordinates, each axis starting at `start`.
    static func coordinateGrid(size: [Int], start: [Int]) -> MLXArray {
        let axes = zip(start, size).map { MLXArray(Int32($0) ..< Int32($0 + $1)) }
        return stacked(meshGrid(axes, indexing: .ij), axis: -1).reshaped(-1, 3)
    }

    /// Rows `[n - 1]` repeated `count` times and appended, as mflux pads token sequences to a multiple of 32.
    static func padByRepeatingLastRow(_ x: MLXArray, count: Int) -> MLXArray {
        guard count > 0 else { return x }
        return concatenated([x, repeated(x[(x.dim(0) - 1)..., 0...], count: count, axis: 0)], axis: 0)
    }

    /// `_patchify`: image `[C, F, H, W]` into padded tokens `[N, pF*pH*pW*C]`, plus the caption features padded to a
    /// multiple of 32 by repeating the last row, position ids and pad masks (true = padding).
    public struct Patchified {
        public let imageTokens: MLXArray
        public let captionFeatures: MLXArray
        public let imageSize: [Int]
        public let imagePositionIDs: MLXArray
        public let captionPositionIDs: MLXArray
        public let imagePadMask: MLXArray
        public let captionPadMask: MLXArray
    }

    public static func patchify(image: MLXArray, captionFeatures: MLXArray) -> Patchified {
        let (pH, pW) = (TransformerLayout.patchSize, TransformerLayout.patchSize)
        let pF = TransformerLayout.framePatchSize
        let capOriginal = captionFeatures.dim(0)
        let capPadding = (-capOriginal).positiveModulo(32)
        let capPositionIDs = coordinateGrid(size: [capOriginal + capPadding, 1, 1], start: [1, 0, 0])
        let capPadMask = concatenated([MLXArray.zeros([capOriginal], dtype: .bool), MLXArray.ones([capPadding], dtype: .bool)])
        let capPadded = padByRepeatingLastRow(captionFeatures, count: capPadding)

        let (c, f, h, w) = (image.dim(0), image.dim(1), image.dim(2), image.dim(3))
        let (fTokens, hTokens, wTokens) = (f / pF, h / pH, w / pW)
        var tokens = image.reshaped(c, fTokens, pF, hTokens, pH, wTokens, pW)
            .transposed(1, 3, 5, 2, 4, 6, 0)
            .reshaped(fTokens * hTokens * wTokens, pF * pH * pW * c)
        let imageOriginal = tokens.dim(0)
        let imagePadding = (-imageOriginal).positiveModulo(32)
        let imagePositionIDs = coordinateGrid(size: [fTokens, hTokens, wTokens], start: [capOriginal + capPadding + 1, 0, 0])
        let positionIDs = imagePadding > 0
            ? concatenated([imagePositionIDs, MLXArray.zeros([imagePadding, 3], dtype: .int32)], axis: 0) : imagePositionIDs
        tokens = padByRepeatingLastRow(tokens, count: imagePadding)
        let imagePadMask = concatenated([MLXArray.zeros([imageOriginal], dtype: .bool), MLXArray.ones([imagePadding], dtype: .bool)])
        return Patchified(
            imageTokens: tokens, captionFeatures: capPadded, imageSize: [f, h, w], imagePositionIDs: positionIDs,
            captionPositionIDs: capPositionIDs, imagePadMask: imagePadMask, captionPadMask: capPadMask)
    }

    /// The transformer's embedded inputs: everything the noise refiner, the context refiner and the main blocks consume.
    public struct EmbeddedInputs {
        /// `[1, Nx, 3840]` after the x embedder and pad-token substitution (the noise refiner's input).
        public let imageEmbedding: MLXArray
        /// `[1, Nc, 3840]` after the caption RMSNorm + linear and pad-token substitution (the context refiner's input).
        public let captionEmbedding: MLXArray
        /// `[Nx, 64, 2]` and `[Nc, 64, 2]` cos/sin tables for the two sequences.
        public let imageFreqsCis: MLXArray
        public let captionFreqsCis: MLXArray
        /// `[1, 256]` float32 timestep embedding.
        public let timestepEmbedding: MLXArray
    }

    /// Embeds `latents` `[16, 1, H/8, W/8]`, a `[1]` timestep (`1 - sigma`) and caption features `[Nc, 2560]`.
    public func embed(latents: MLXArray, timestep: MLXArray, captionFeatures: MLXArray) -> EmbeddedInputs {
        let tEmb = timestepEmbedding(timestep)
        let p = Self.patchify(image: latents, captionFeatures: captionFeatures)
        var x = linear(p.imageTokens, "all_x_embedder.\(TransformerLayout.embedderKey)")
        x = MLX.where(p.imagePadMask[0..., .newAxis], weights.tensor("x_pad_token"), x)
        let xFreqs = rope(ids: p.imagePositionIDs)
        var cap = linear(rmsNorm(p.captionFeatures, "cap_embedder.0"), "cap_embedder.1")
        cap = MLX.where(p.captionPadMask[0..., .newAxis], weights.tensor("cap_pad_token"), cap)
        let capFreqs = rope(ids: p.captionPositionIDs)
        return EmbeddedInputs(
            imageEmbedding: expandedDimensions(x, axis: 0), captionEmbedding: expandedDimensions(cap, axis: 0),
            imageFreqsCis: xFreqs, captionFreqsCis: capFreqs, timestepEmbedding: tEmb)
    }

    // MARK: Block

    /// `ZImageAttention.__call__`. `freqsCis` is `[N, 64, 2]`; `attentionMask` is the `[1, N]` boolean valid mask.
    func attention(_ h: MLXArray, prefix: String, attentionMask: MLXArray, freqsCis: MLXArray, trace: ((String, MLXArray) -> Void)?) -> MLXArray {
        let p = prefix + "attention."
        let (batch, seq) = (h.dim(0), h.dim(1))
        let (heads, headDim) = (TransformerLayout.heads, TransformerLayout.headDim)
        var q = linear(h, p + "to_q").reshaped(batch, seq, heads, headDim)
        var k = linear(h, p + "to_k").reshaped(batch, seq, heads, headDim)
        let v = linear(h, p + "to_v").reshaped(batch, seq, heads, headDim)
        q = rmsNorm(q, p + "norm_q")
        k = rmsNorm(k, p + "norm_k")
        trace?("attn.q_norm", q)
        q = Self.applyRotary(q, freqsCis)
        k = Self.applyRotary(k, freqsCis)
        trace?("attn.q_rope", q)
        let mask = MLX.where(
            attentionMask[0..., .newAxis, .newAxis, 0...], MLXArray(Float(0)), MLXArray(-Float.infinity))
        let out = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
            scale: Float(pow(Double(headDim), -0.5)), mask: .array(mask))
        let merged = out.transposed(0, 2, 1, 3).reshaped(batch, seq, TransformerLayout.dim)
        trace?("attn.sdpa", merged)
        return linear(merged, p + "to_out.0")
    }

    /// `ZImageAttention._apply_rotary_emb`: rotates interleaved (even, odd) pairs by the cos/sin table, in the promoted dtype.
    static func applyRotary(_ x: MLXArray, _ freqsCis: MLXArray) -> MLXArray {
        let (batch, seq, heads, headDim) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let pairs = x.reshaped(batch, seq, heads, headDim / 2, 2)
        let table = expandedDimensions(expandedDimensions(freqsCis, axis: 0), axis: 2)
        let (xReal, xImag) = (pairs[.ellipsis, 0], pairs[.ellipsis, 1])
        let (cosine, sine) = (table[.ellipsis, 0], table[.ellipsis, 1])
        let outReal = xReal * cosine - xImag * sine
        let outImag = xReal * sine + xImag * cosine
        return stacked([outReal, outImag], axis: -1).reshaped(batch, seq, heads, headDim)
    }

    /// `FeedForward.__call__`: `w2(silu(w1(x)) * w3(x))`.
    func feedForward(_ x: MLXArray, prefix: String) -> MLXArray {
        let p = prefix + "feed_forward."
        return linear(silu(linear(x, p + "w1")) * linear(x, p + "w3"), p + "w2")
    }

    /// `ZImageTransformerBlock.__call__` for the block at `prefix` (`layers.0.` is the first main block).
    /// `x` is `[1, N, 3840]`, `attentionMask` `[1, N]` boolean, `freqsCis` `[N, 64, 2]`, `timestepEmbedding` `[1, 256]`.
    /// `trace` receives named intermediates (`modulation`, `attn.*`, `attn_out`, `after_attn`, `ffn_out`).
    public func block(
        prefix: String = "layers.0.", x: MLXArray, attentionMask: MLXArray, freqsCis: MLXArray, timestepEmbedding: MLXArray,
        trace: ((String, MLXArray) -> Void)? = nil
    ) -> MLXArray {
        let modulation = expandedDimensions(linear(timestepEmbedding, prefix + "adaLN_modulation.0"), axis: 1)
        trace?("modulation", modulation)
        let parts = modulation.split(parts: 4, axis: 2)
        let scaleMsa = 1.0 + parts[0], gateMsa = MLX.tanh(parts[1])
        let scaleMlp = 1.0 + parts[2], gateMlp = MLX.tanh(parts[3])

        let attnIn = rmsNorm(x, prefix + "attention_norm1") * scaleMsa
        trace?("attn_in", attnIn)
        let attnOut = attention(attnIn, prefix: prefix, attentionMask: attentionMask, freqsCis: freqsCis, trace: trace)
        trace?("attn_out", attnOut)
        var h = x + gateMsa * rmsNorm(attnOut, prefix + "attention_norm2")
        trace?("after_attn", h)

        let ffnOut = feedForward(rmsNorm(h, prefix + "ffn_norm1") * scaleMlp, prefix: prefix)
        trace?("ffn_out", ffnOut)
        h = h + gateMlp * rmsNorm(ffnOut, prefix + "ffn_norm2")
        return h
    }
}

/// mflux `RopeEmbedder`: per-axis cos/sin tables, gathered by integer position ids.
public struct ZImageRope {
    let tables: [MLXArray]

    public init(theta: Float, axesDims: [Int], axesLens: [Int]) {
        tables = zip(axesDims, axesLens).map { d, e in
            let exponent = MLXArray(stride(from: 0, to: d, by: 2).map { Float($0) }) / Float(d)
            let freqs = outer(MLXArray(0 ..< Int32(e)).asType(.float32), Float(1) / MLX.pow(theta, exponent))
            return stacked([MLX.cos(freqs), MLX.sin(freqs)], axis: -1)
        }
        eval(tables)
    }

    /// `ids` `[N, 3]` int32 to `[N, sum(axesDims) / 2, 2]`.
    public func callAsFunction(ids: MLXArray) -> MLXArray {
        concatenated(tables.enumerated().map { $1[ids[0..., $0].asType(.int32)] }, axis: 1)
    }
}

extension Int {
    /// Python's `%` for a positive modulus (the result is never negative).
    fileprivate func positiveModulo(_ m: Int) -> Int { ((self % m) + m) % m }
}
