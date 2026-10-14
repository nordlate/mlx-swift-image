import Foundation
import MLX
import MLXFast
import MLXNN

/// Port of mflux's Z-Image `VAE.decode` (the `Decoder` only): `latents / scaling + shift`, the 16 -> 512 input convolution,
/// the mid block (resnet, single-head attention over all pixels, resnet), four up blocks of three resnets (the first three
/// followed by a nearest-neighbour 2x upsample and a convolution), the output norm, SiLU and the 3-channel output convolution.
///
/// The Python modules transpose NCHW to NHWC and back inside every layer; here the activations stay NHWC between layers
/// and are transposed once at each end. That only changes the layout, not a single value: every op below is the same one
/// in the same order and the same dtype as mflux's (bfloat16 throughout, except that the output norm and the attention's
/// group norm run in float32 and are cast back, as mflux does).
public struct ZImageVAE {
    let weights: VAEWeights

    public init(weights: VAEWeights) { self.weights = weights }

    // MARK: Layers (NHWC)

    /// `nn.Conv2d(kernel_size=k, stride=1, padding=k/2)`: `conv2d(x, weight) + bias`.
    func conv(_ x: MLXArray, _ name: String, padding: Int = 1) -> MLXArray {
        conv2d(x, weights.tensor(name + ".weight"), padding: IntOrPair(padding)) + weights.tensor(name + ".bias")
    }

    /// `nn.GroupNorm(32, c, eps=1e-6, pytorch_compatible=True)`: normalizes over each group of channels, then scales and shifts.
    func groupNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        let batch = x.dim(0)
        let dims = x.dim(-1)
        let rest = Array(x.shape.dropFirst().dropLast())
        let groupSize = dims / VAELayout.normGroups
        var y = x.reshaped(batch, -1, VAELayout.normGroups, groupSize)
        y = y.transposed(0, 2, 1, 3).reshaped(batch, VAELayout.normGroups, -1)
        y = MLXFast.layerNorm(y, weight: nil, bias: nil, eps: VAELayout.normEps)
        y = y.reshaped(batch, VAELayout.normGroups, -1, groupSize)
        y = y.transposed(0, 2, 1, 3).reshaped([batch] + rest + [dims])
        return weights.tensor(name + ".weight") * y + weights.tensor(name + ".bias")
    }

    /// `nn.QuantizedLinear.__call__` (4 bits, group size 64, affine) with the stored bias.
    func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let y = quantizedMM(
            x, weights.tensor(name + ".weight"), scales: weights.tensor(name + ".scales"), biases: weights.tensor(name + ".biases"),
            transpose: true, groupSize: VAELayout.groupSize, bits: VAELayout.bits)
        return y + weights.tensor(name + ".bias")
    }

    /// `ResnetBlock2D`: `norm, silu, conv, norm, silu, conv`, plus the (optionally 1x1-convolved) input.
    func resnet(_ x: MLXArray, _ name: String) -> MLXArray {
        var h = conv(silu(groupNorm(x, name + ".norm1")), name + ".conv1")
        h = conv(silu(groupNorm(h, name + ".norm2")), name + ".conv2")
        let shortcut = weights.tensors[name + ".conv_shortcut.weight"] != nil ? conv(x, name + ".conv_shortcut", padding: 0) : x
        return shortcut + h
    }

    /// The scale mflux hands the attention: `1.0 / sqrt(array(C, dtype=queries.dtype))`, computed in bfloat16 and then used as a float.
    static func attentionScale(channels: Int, dtype: DType) -> Float {
        let one = MLXArray(Float(1)).asType(dtype)
        let scale = one / MLX.sqrt(MLXArray(Float(channels)).asType(dtype))
        return scale.asType(.float32).item(Float.self)
    }

    /// `Attention`: group norm in float32 (cast back to bfloat16), q/k/v over the flattened pixels as one head, SDPA, output
    /// projection and residual.
    func attention(_ x: MLXArray, _ prefix: String) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let hidden = groupNorm(x.asType(.float32), prefix + ".group_norm").asType(.bfloat16)
        func heads(_ name: String) -> MLXArray { linear(hidden, prefix + "." + name).reshaped(b, h * w, 1, c).transposed(0, 2, 1, 3) }
        let (q, k, v) = (heads("to_q"), heads("to_k"), heads("to_v"))
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: Self.attentionScale(channels: c, dtype: q.dtype), mask: .none)
        let merged = out.transposed(0, 2, 1, 3).reshaped(b, h, w, c)
        return x + linear(merged, prefix + ".to_out.0")
    }

    /// `UpSampler._upsample_nearest` then the convolution.
    func upsample(_ x: MLXArray, _ name: String) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let expanded = broadcast(x[0..., 0..., .newAxis, 0..., .newAxis, 0...], to: [b, h, 2, w, 2, c]).reshaped(b, h * 2, w * 2, c)
        return conv(expanded, name + ".conv")
    }

    // MARK: Decoder

    /// `Decoder.__call__` on NHWC input `[B, h, w, 16]`; returns NHWC `[B, 8h, 8w, 3]`.
    func decoder(_ x: MLXArray) -> MLXArray {
        var h = conv(x, "decoder.conv_in.conv")
        h = resnet(h, "decoder.mid_block.resnets.0")
        h = attention(h, "decoder.mid_block.attentions.0")
        h = resnet(h, "decoder.mid_block.resnets.1")
        for (i, block) in VAELayout.decoderBlocks.enumerated() {
            for j in 0 ..< VAELayout.decoderResnetsPerBlock { h = resnet(h, "decoder.up_blocks.\(i).resnets.\(j)") }
            if block.upsample { h = upsample(h, "decoder.up_blocks.\(i).upsamplers.0") }
        }
        h = groupNorm(h.asType(.float32), "decoder.conv_norm_out.norm").asType(.bfloat16)
        return conv(silu(h), "decoder.conv_out.conv")
    }

    /// `VAE.decode` for latents `[1, 16, h, w]` (mflux's `unpack_latents` output): returns `[1, 3, 8h, 8w]`.
    public func decode(_ latents: MLXArray) -> MLXArray {
        let scaled = latents / VAELayout.scalingFactor + VAELayout.shiftFactor
        let decoded = decoder(scaled.transposed(0, 2, 3, 1))
        return decoded.transposed(0, 3, 1, 2)
    }

    /// mflux's `ZImageLatentCreator.unpack_latents`: `[16, 1, h, w]` to `[1, 16, h, w]`.
    public static func unpack(_ latents: MLXArray) -> MLXArray {
        squeezed(expandedDimensions(latents, axis: 0), axis: 2)
    }

    /// `VAEUtil.decode` for the sampler's final latents `[16, 1, h, w]` (no tiling: mflux's `tiling_config` is `None`).
    public func decode(samplerLatents latents: MLXArray) -> MLXArray {
        decode(Self.unpack(latents))
    }
}
