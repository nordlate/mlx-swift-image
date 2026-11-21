import Foundation
import MLX
import MLXFast

/// Port of MFLUX 0.20's `SeedVR2Transformer` (the 3B NaDiT: 32 blocks, the first 10 with separate video and text weights, windowed
/// attention with the text appended to every window, 3D RoPE on video and text), op for op and in the reference's dtypes, for one batch
/// row and one frame. Linear layers run quantized (`SeedVR2Weights`); `vid_in.proj` stays fp16 as in MFLUX.
public struct SeedVR2Transformer {
    let weights: [String: MLXArray]

    public init(weights: SeedVR2Weights) {
        self.weights = weights.transformer
    }

    /// `1 / sqrt(head_dim)` as Python computes `head_dim ** -0.5`.
    static let attentionScale = Float(Foundation.pow(Double(SeedVR2Layout.headDim), -0.5))

    /// `SeedVR2Transformer.__call__(vid, txt, timestep)`: `vid` `[1, 33, 1, H, W]` (noise, condition latent, mask), `txt` the positive
    /// embedding `[1, 58, 5120]`, `timestep` a scalar. Returns the prediction `[1, 16, 1, H, W]`.
    public func callAsFunction(vid: MLXArray, txt: MLXArray, timestep: MLXArray) -> MLXArray {
        var txt = SeedVR2Ops.quantizedLinear(txt, weights, "txt_in")
        let textLength = txt.dim(1)
        var (vid, grid) = patchIn(vid)
        let emb = timeEmbedding(timestep).reshaped(-1, SeedVR2Layout.vidDim, 2, 3)
        let windows = [false, true].map { SeedVR2Windows(t: grid[0], h: grid[1], w: grid[2], shift: $0) }
        for i in 0 ..< SeedVR2Layout.layers {
            (vid, txt) = block(i, vid: vid, txt: txt, emb: emb, windows: windows[i % 2], textLength: textLength)
        }
        vid = SeedVR2Ops.rmsNorm(vid, weight: weights.tensor("vid_out_norm.weight"))
        let shiftA = emb[0..., 0..., 0, 0].expandedDimensions(axis: 1)
        let scaleA = emb[0..., 0..., 0, 1].expandedDimensions(axis: 1)
        vid = vid * (scaleA + weights.tensor("vid_out_ada.out_scale")) + (shiftA + weights.tensor("vid_out_ada.out_shift"))
        return patchOut(vid, grid: grid)
    }

    // MARK: Embeddings

    /// `PatchIn`: `[B, C, T, H, W]` to `(t h w)` tokens of `t*h*w*C` values, projected by the fp16 `vid_in.proj`.
    func patchIn(_ vid: MLXArray) -> (MLXArray, [Int]) {
        let (pt, ph, pw) = (SeedVR2Layout.patch.t, SeedVR2Layout.patch.h, SeedVR2Layout.patch.w)
        let (b, c, t, h, w) = (vid.dim(0), vid.dim(1), vid.dim(2), vid.dim(3), vid.dim(4))
        let (tp, hp, wp) = (t / pt, h / ph, w / pw)
        var x = vid.reshaped(b, c, tp, pt, hp, ph, wp, pw)
        x = x.transposed(0, 2, 4, 6, 3, 5, 7, 1)
        x = x.reshaped(b, tp, hp, wp, pt * ph * pw * c)
        x = SeedVR2Ops.linear(x, weights, "vid_in.proj")
        return (x.reshaped(b, -1, x.dim(-1)), [tp, hp, wp])
    }

    /// `PatchOut`: project to `t*h*w*16` per token and fold back to `[B, 16, T, H, W]`.
    func patchOut(_ vid: MLXArray, grid: [Int]) -> MLXArray {
        let (pt, ph, pw) = (SeedVR2Layout.patch.t, SeedVR2Layout.patch.h, SeedVR2Layout.patch.w)
        var x = SeedVR2Ops.quantizedLinear(vid, weights, "vid_out.proj")
        let b = x.dim(0), c = x.dim(-1) / (pt * ph * pw)
        x = x.reshaped(b, grid[0], grid[1], grid[2], pt, ph, pw, c)
        x = x.transposed(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshaped(b, c, grid[0] * pt, grid[1] * ph, grid[2] * pw)
    }

    /// `TimeEmbedding`: `[sin, cos]` of `t * exp(-ln(10000) * i / 128)`, then three linear layers with SiLU between them.
    func timeEmbedding(_ timestep: MLXArray) -> MLXArray {
        let t = timestep.ndim == 0 ? timestep.expandedDimensions(axis: 0) : timestep
        let half = SeedVR2Layout.timestepChannels / 2
        let exponent = MLXArray((0 ..< half).map { Float($0) }) * Float(-Foundation.log(10000.0) / Double(half))
        let args = t.expandedDimensions(axis: 1).asType(.float32) * exp(exponent)
        var emb = concatenated([sin(args), cos(args)], axis: -1)
        emb = SeedVR2Ops.quantizedLinear(emb, weights, "emb_in.proj_in")
        emb = SeedVR2Ops.silu(emb)
        emb = SeedVR2Ops.quantizedLinear(emb, weights, "emb_in.proj_hid")
        emb = SeedVR2Ops.silu(emb)
        return SeedVR2Ops.quantizedLinear(emb, weights, "emb_in.proj_out")
    }

    // MARK: Blocks

    /// `AdaModulation._apply_modulation`: `layer` 0 (attention) or 1 (MLP); `in` is `x * (scale) + shift`, `out` is `x * gate`.
    func modulate(_ x: MLXArray, emb: MLXArray, params: String, layer: Int, input: Bool) -> MLXArray {
        let mod = emb[0..., 0..., layer]
        let name = layer == 0 ? "attn" : "mlp"
        if input {
            let shift = mod[.ellipsis, 0].expandedDimensions(axis: 1) + weights.tensor("\(params).\(name)_shift")
            let scale = mod[.ellipsis, 1].expandedDimensions(axis: 1) + weights.tensor("\(params).\(name)_scale")
            return x * scale + shift
        }
        let gate = mod[.ellipsis, 2].expandedDimensions(axis: 1) + weights.tensor("\(params).\(name)_gate")
        return x * gate
    }

    /// `TransformerBlock.__call__`. The last block (31) leaves the text unchanged; its text MLP is never evaluated (MLX is lazy and the
    /// reference discards it too).
    func block(_ i: Int, vid: MLXArray, txt: MLXArray, emb: MLXArray, windows: SeedVR2Windows, textLength: Int) -> (MLXArray, MLXArray) {
        let shared = SeedVR2Layout.isShared(i), last = i == SeedVR2Layout.layers - 1
        let b = "blocks.\(i)"
        let vidParams = shared ? "\(b).ada.all" : "\(b).ada.vid", txtParams = shared ? "\(b).ada.all" : "\(b).ada.txt"
        func modVid(_ x: MLXArray, _ layer: Int, _ input: Bool) -> MLXArray { modulate(x, emb: emb, params: vidParams, layer: layer, input: input) }
        func modTxt(_ x: MLXArray, _ layer: Int, _ input: Bool) -> MLXArray {
            last ? x : modulate(x, emb: emb, params: txtParams, layer: layer, input: input)
        }

        var vidAttn = modVid(SeedVR2Ops.plainRMSNorm(vid), 0, true)
        var txtAttn = modTxt(SeedVR2Ops.plainRMSNorm(txt), 0, true)
        (vidAttn, txtAttn) = attention(i, vid: vidAttn, txt: txtAttn, windows: windows, textLength: textLength)
        vidAttn = modVid(vidAttn, 0, false)
        txtAttn = modTxt(txtAttn, 0, false)
        var vid = vid + vidAttn
        var txt = txt
        if !last { txt = txt + txtAttn }

        var vidMLP = modVid(SeedVR2Ops.plainRMSNorm(vid), 1, true)
        var txtMLP = modTxt(last ? txt : SeedVR2Ops.plainRMSNorm(txt), 1, true)
        let vidGroup = shared ? "\(b).mlp.all" : "\(b).mlp.vid", txtGroup = shared ? "\(b).mlp.all" : "\(b).mlp.txt"
        vidMLP = chunkedSwiGLU(vidMLP, vidGroup)
        if !last { txtMLP = swiGLU(txtMLP, txtGroup) }
        vidMLP = modVid(vidMLP, 1, false)
        txtMLP = modTxt(txtMLP, 1, false)
        vid = vid + vidMLP
        if !last { txt = txt + txtMLP }
        return (vid, txt)
    }

    /// `SwiGLUMLP`: `proj_out(silu(proj_in_gate(x)) * proj_in(x))`, no biases.
    func swiGLU(_ x: MLXArray, _ name: String) -> MLXArray {
        let gate = SeedVR2Ops.silu(SeedVR2Ops.quantizedLinear(x, weights, name + ".proj_in_gate"))
        let y = SeedVR2Ops.quantizedLinear(x, weights, name + ".proj_in")
        return SeedVR2Ops.quantizedLinear(gate * y, weights, name + ".proj_out")
    }

    /// Above this many video tokens (a 2048 x 2048 output has 16384) the video MLP runs in evaluated chunks of this many tokens, so
    /// its hidden activations (6912 wide, float32) exist for one chunk at a time. The MLP acts on each token alone.
    static let mlpChunkTokens = 16384

    /// `swiGLU` on `[1, L, D]` video tokens, in chunks of `mlpChunkTokens` when `L` is larger.
    func chunkedSwiGLU(_ x: MLXArray, _ name: String) -> MLXArray {
        let length = x.dim(1)
        guard length > Self.mlpChunkTokens else { return swiGLU(x, name) }
        var chunks: [MLXArray] = []
        for start in stride(from: 0, to: length, by: Self.mlpChunkTokens) {
            let chunk = swiGLU(x[0..., start ..< min(start + Self.mlpChunkTokens, length)], name)
            eval(chunk)
            chunks.append(chunk)
        }
        return concatenated(chunks, axis: 1)
    }

    /// `MMAttention.__call__` for one batch row: video q/k/v gathered window by window, q/k RMS-normed, RoPE applied, the text tokens
    /// appended to every window, one scaled-dot-product attention per window, the video outputs scattered back and the text outputs
    /// averaged over the windows. The text q/k/v and their RoPE are the same for every window, so they are computed once (the
    /// reference repeats them per window; the values are identical).
    func attention(_ i: Int, vid: MLXArray, txt: MLXArray, windows: SeedVR2Windows, textLength: Int) -> (MLXArray, MLXArray) {
        let shared = SeedVR2Layout.isShared(i)
        let a = "blocks.\(i).attn"
        let (vg, tg) = shared ? ("all", "all") : ("vid", "txt")
        let (heads, headDim) = (SeedVR2Layout.heads, SeedVR2Layout.headDim)
        let (batch, length) = (vid.dim(0), vid.dim(1))
        let (txtBatch, txtLength) = (txt.dim(0), txt.dim(1))

        // The reference projects and then gathers the rows into window order; gathering the (3x narrower) input rows first gives the
        // same rows without a second copy of the projected q/k/v.
        let forward = MLXArray(windows.forward)
        let qkvVid = SeedVR2Ops.quantizedLinear(vid.reshaped(-1, vid.dim(-1)).take(forward, axis: 0), weights, "\(a).proj_qkv.\(vg)")
            .reshaped(-1, 3, heads, headDim)
        let qkvTxt = SeedVR2Ops.quantizedLinear(txt.reshaped(-1, txt.dim(-1)), weights, "\(a).proj_qkv.\(tg)").reshaped(-1, 3, heads, headDim)

        var qVid = SeedVR2Ops.rmsNorm(qkvVid[0..., 0], weight: weights.tensor("\(a).norm_q.\(vg).weight"))
        var kVid = SeedVR2Ops.rmsNorm(qkvVid[0..., 1], weight: weights.tensor("\(a).norm_k.\(vg).weight"))
        let vVid = qkvVid[0..., 2]
        var qTxt = SeedVR2Ops.rmsNorm(qkvTxt[0..., 0], weight: weights.tensor("\(a).norm_q.\(tg).weight"))
        var kTxt = SeedVR2Ops.rmsNorm(qkvTxt[0..., 1], weight: weights.tensor("\(a).norm_k.\(tg).weight"))
        let vTxt = qkvTxt[0..., 2]

        let tables = SeedVR2RoPE.tables(freqs: weights.tensor("\(a).rope.rope.freqs"), windows: windows.windows, textLength: textLength)
        let videoFreqs = tables.video.expandedDimensions(axis: 1), textFreqs = tables.text.expandedDimensions(axis: 1)
        qVid = SeedVR2RoPE.apply(videoFreqs, qVid)
        kVid = SeedVR2RoPE.apply(videoFreqs, kVid)
        qTxt = SeedVR2RoPE.apply(textFreqs, qTxt)
        kTxt = SeedVR2RoPE.apply(textFreqs, kTxt)

        var videoOutputs: [MLXArray] = []
        var textOutputs: [MLXArray] = []
        var start = 0
        for window in windows.windows {
            let end = start + window.count
            func joined(_ video: MLXArray, _ text: MLXArray) -> MLXArray {
                concatenated([video[start ..< end], text], axis: 0).expandedDimensions(axis: 0).transposed(0, 2, 1, 3)
            }
            let o = MLXFast.scaledDotProductAttention(
                queries: joined(qVid, qTxt), keys: joined(kVid, kTxt), values: joined(vVid, vTxt), scale: Self.attentionScale, mask: nil)
            let out = o.transposed(0, 2, 1, 3).squeezed(axis: 0).reshaped(-1, heads * headDim)
            videoOutputs.append(out[0 ..< window.count])
            textOutputs.append(out[window.count...])
            start = end
        }
        let videoOut = concatenated(videoOutputs, axis: 0)
        let textOut = stacked(textOutputs).mean(axis: 0)
        let vidResult = SeedVR2Ops.quantizedLinear(videoOut.take(MLXArray(windows.reverse), axis: 0), weights, "\(a).proj_out.\(vg)")
        let txtResult = SeedVR2Ops.quantizedLinear(textOut, weights, "\(a).proj_out.\(tg)")
        return (vidResult.reshaped(batch, length, -1), txtResult.reshaped(txtBatch, txtLength, -1))
    }
}
