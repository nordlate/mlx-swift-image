import Foundation
import MLX
import MLXFast

/// Port of MFLUX 0.20's `SeedVR2VAE` (a causal 3D VAE: `Encoder3D` and `Decoder3D` with 128/256/512/512 channels), op for op, for a
/// single frame. Group norms run in float32 and cast to bfloat16 (`ModelConfig.precision`), convolutions use the stored fp16 weights
/// (MLX promotes the mixed inputs), and the mid-block attention projections are quantized as in MFLUX. Tiling is `SeedVR2VAETiler`.
public struct SeedVR2VAE {
    let weights: [String: MLXArray]

    public init(weights: SeedVR2Weights) {
        self.weights = weights.vae
    }

    /// `SeedVR2VAE.encode`: `[1, 3, H, W]` (or 5D) to the scaled latent mean `[1, 16, 1, H/8, W/8]`.
    public func encode(_ x: MLXArray) -> MLXArray {
        let x5 = x.ndim == 4 ? x.expandedDimensions(axis: 2) : x
        let h = encoder(x5)
        let mean = h.split(parts: 2, axis: 1)[0]
        return mean * SeedVR2Layout.vaeScalingFactor
    }

    /// `SeedVR2VAE.decode`: `[1, 16, 1, h, w]` (or 4D) to `[1, 3, 1, 8h, 8w]`.
    public func decode(_ z: MLXArray) -> MLXArray {
        let z5 = z.ndim == 4 ? z.expandedDimensions(axis: 2) : z
        return decoder(z5 / SeedVR2Layout.vaeScalingFactor)
    }

    // MARK: Layers

    /// `CausalConv3d`: on a `[B, C, T, H, W]` input, repeat the first frame in front (`kt - 1` frames, or `2 * pt` with
    /// `use_padding_causal`) instead of temporal padding, then `conv_general` channels-last with the spatial padding, plus the bias.
    func conv(_ x: MLXArray, _ name: String, stride: [Int] = [1, 1, 1], padding: [Int], paddingCausal: Bool = false) -> MLXArray {
        let weight = weights.tensor(name + ".weight")
        let kt = weight.dim(1)
        var x = x
        var temporalPadding = padding[0]
        if kt > 1 {
            let causalPad = paddingCausal ? 2 * padding[0] : kt - 1
            if causalPad > 0 {
                let first = x[0..., 0..., 0 ..< 1]
                x = concatenated([repeated(first, count: causalPad, axis: 2), x], axis: 2)
            }
            temporalPadding = 0
        }
        x = x.transposed(0, 2, 3, 4, 1)
        var out = convGeneral(x, weight, strides: .array(stride), padding: .array([temporalPadding, padding[1], padding[2]]))
        out = out + weights.tensor(name + ".bias")
        return out.transposed(0, 4, 1, 2, 3)
    }

    /// Channels-first group norm as the reference runs it: to channels-last, float32 norm, cast to bfloat16, back.
    func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        let y = SeedVR2Ops.groupNorm(
            x.transposed(0, 2, 3, 4, 1).asType(.float32), weight: weights.tensor(name + ".weight"), bias: weights.tensor(name + ".bias"))
        return y.asType(.bfloat16).transposed(0, 4, 1, 2, 3)
    }

    /// `ResnetBlock3D` (encoder and decoder are the same).
    func resnet(_ x: MLXArray, _ name: String) -> MLXArray {
        var h = SeedVR2Ops.silu(norm(x, name + ".norm1"))
        h = conv(h, name + ".conv1", padding: [1, 1, 1])
        h = SeedVR2Ops.silu(norm(h, name + ".norm2"))
        h = conv(h, name + ".conv2", padding: [1, 1, 1])
        let residual = weights[name + ".conv_shortcut.weight"] != nil ? conv(x, name + ".conv_shortcut", padding: [0, 0, 0]) : x
        return h + residual
    }

    /// `Attention3D`: per frame, group norm (float32, cast to bfloat16), quantized q/k/v, single-head attention over all positions.
    func attention(_ x: MLXArray, _ name: String) -> MLXArray {
        let (b, c, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        var y = x.transposed(0, 2, 1, 3, 4).reshaped(b * t, c, h * w).transposed(0, 2, 1)
        y = SeedVR2Ops.groupNorm(
            y.asType(.float32), weight: weights.tensor(name + ".group_norm.weight"), bias: weights.tensor(name + ".group_norm.bias")
        ).asType(.bfloat16)
        let q = SeedVR2Ops.quantizedLinear(y, weights, name + ".to_q").expandedDimensions(axis: 1)
        let k = SeedVR2Ops.quantizedLinear(y, weights, name + ".to_k").expandedDimensions(axis: 1)
        let v = SeedVR2Ops.quantizedLinear(y, weights, name + ".to_v").expandedDimensions(axis: 1)
        let scale = Float(Foundation.pow(Double(c), -0.5))
        y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil).squeezed(axis: 1)
        y = SeedVR2Ops.quantizedLinear(y, weights, name + ".to_out.0")
        y = y.transposed(0, 2, 1).reshaped(b, t, c, h, w).transposed(0, 2, 1, 3, 4)
        return y + x
    }

    /// `MidBlock3D`: resnet, attention, resnet.
    func midBlock(_ x: MLXArray, _ name: String) -> MLXArray {
        var y = resnet(x, name + ".resnets.0")
        y = attention(y, name + ".attentions.0")
        return resnet(y, name + ".resnets.1")
    }

    /// `Downsample3D`: pad right and bottom by one, then a stride-2 conv (and stride 2 in time on the temporal blocks).
    func downsample(_ x: MLXArray, _ name: String, temporal: Bool) -> MLXArray {
        let padded = MLX.padded(x, widths: [0, 0, 0, [0, 1], [0, 1]] as [IntOrPair])
        return conv(padded, name + ".conv", stride: [temporal ? 2 : 1, 2, 2], padding: [temporal ? 1 : 0, 0, 0])
    }

    /// `Upsample3D`: a 1x1x1 conv to `C * 2 * 2 (* 2)` channels, a pixel shuffle, the first frame only for a single input frame, and a
    /// causal 3x3x3 conv with `use_padding_causal`.
    func upsample(_ x: MLXArray, _ name: String, temporal: Bool) -> MLXArray {
        let (b, c, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let (sf, tf) = (2, temporal ? 2 : 1)
        var y = conv(x, name + ".upscale_conv", padding: [0, 0, 0])
        y = y.reshaped(b, sf, sf, tf, c, t, h, w)
        y = y.transposed(0, 4, 5, 3, 6, 1, 7, 2)
        y = y.reshaped(b, c, t * tf, h * sf, w * sf)
        if t == 1 && tf > 1 { y = y[0..., 0..., 0 ..< 1] }
        return conv(y, name + ".conv", padding: [1, 1, 1], paddingCausal: true)
    }

    // MARK: Encoder and decoder

    func encoder(_ x: MLXArray) -> MLXArray {
        let blocks = SeedVR2Layout.vaeBlockChannels.count
        var y = conv(x, "encoder.conv_in", padding: [1, 1, 1])
        for i in 0 ..< blocks {
            for r in 0 ..< SeedVR2Layout.encoderLayersPerBlock { y = resnet(y, "encoder.down_blocks.\(i).resnets.\(r)") }
            if i < blocks - 1 {
                y = downsample(y, "encoder.down_blocks.\(i).downsamplers.0", temporal: SeedVR2Layout.encoderTemporalDown(i))
            }
        }
        y = midBlock(y, "encoder.mid_block")
        y = SeedVR2Ops.silu(norm(y, "encoder.conv_norm_out"))
        return conv(y, "encoder.conv_out", padding: [1, 1, 1])
    }

    func decoder(_ z: MLXArray) -> MLXArray {
        let blocks = SeedVR2Layout.vaeBlockChannels.count
        var y = conv(z, "decoder.conv_in", padding: [1, 1, 1])
        y = midBlock(y, "decoder.mid_block")
        for i in 0 ..< blocks {
            for r in 0 ..< SeedVR2Layout.decoderLayersPerBlock { y = resnet(y, "decoder.up_blocks.\(i).resnets.\(r)") }
            if i < blocks - 1 { y = upsample(y, "decoder.up_blocks.\(i).upsamplers.0", temporal: i < 2) }
        }
        y = SeedVR2Ops.silu(norm(y, "decoder.conv_norm_out"))
        return conv(y, "decoder.conv_out", padding: [1, 1, 1])
    }
}
