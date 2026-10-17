import Foundation
import MLX
import MLXNN

extension KleinVAE {
    /// NHWC equivalent of Flux2Downsample2D: bottom/right-only padding, then stride-2 convolution.
    public static func encoderDownsample(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let p = padded(x, widths: [.init(0), .init((0,1)), .init((0,1)), .init(0)])
        return conv2d(p, weight, stride: 2, padding: 0) + bias
    }
    /// VAEUtil.encode does NOT cast its float32 image. The first convolution and residual stream therefore stay float32;
    /// norms cast after affine, and conv_norm_out/SiLU/conv_out/quant_conv finally produce bfloat16.
    public func encodePrepared(_ image: MLXArray, trace: ((String, MLXArray) throws -> Void)? = nil) throws -> MLXArray {
        guard image.ndim == 4, image.dim(0) == 1, image.dim(1) == 3, image.dim(2) >= 16, image.dim(3) >= 16,
              image.dim(2) % 16 == 0, image.dim(3) % 16 == 0, image.dtype == .float32,
              KleinTensorValues.floats(image).allSatisfy(\.isFinite) else {
            throw KleinTransformerError.invalidInput("encoder requires finite float32 [1,3,H,W], H/W multiples of 16")
        }
        func record(_ n: String, _ a: MLXArray) throws { try trace?(n,a.transposed(0,3,1,2)) }
        var h = conv(image.transposed(0,2,3,1),"encoder.conv_in")
        try record("encoder.conv_in",h)
        for i in 0..<4 {
            for j in 0..<2 { h = resnet(h,"encoder.down_blocks.\(i).resnets.\(j)") }
            if i < 3 {
                let n = "encoder.down_blocks.\(i).downsamplers.0.conv"
                h = Self.encoderDownsample(h,weight: weights.tensor(n + ".weight"),bias: weights.tensor(n + ".bias"))
            }
            try record("encoder.down.\(i)",h)
        }
        h = resnet(h,"encoder.mid_block.resnets.0")
        h = attention(h,"encoder.mid_block.attentions.0")
        h = resnet(h,"encoder.mid_block.resnets.1"); try record("encoder.mid",h)
        h = conv(silu(norm(h,"encoder.conv_norm_out")).asType(.bfloat16),"encoder.conv_out")
        try record("encoder.conv_out",h)
        h = conv(h,"quant_conv",padding: 0); try record("encoder.quant",h)
        let mean = h.transposed(0,3,1,2)[0...,0..<32,0...,0...]
        let encoded = (mean - 0.0) * 1.0
        try trace?("encoded",encoded)
        return encoded
    }
    public func decodeStartingImagePacked(_ packed: MLXArray, trace: ((String, MLXArray) throws -> Void)? = nil) throws -> MLXArray {
        guard packed.ndim == 4, packed.dim(0) == 1, packed.dim(1) == 128, packed.dtype == .float32 else {
            throw KleinTransformerError.invalidInput("img2img decoder requires float32 packed latents")
        }
        let mean = weights.tensor("bn.running_mean").reshaped(1,128,1,1)
        let std = MLX.sqrt(weights.tensor("bn.running_var").reshaped(1,128,1,1) + KleinVAELayout.bnEps)
        let x = try Self.unpatchify(packed * std + mean)
        try trace?("unpacked",x)
        // Flux2VAE.decode: scale=1, shift=0, then post_quant_conv. Preserve the scalar operations.
        var h = conv((x / 1.0 + 0.0).transposed(0,2,3,1),"post_quant_conv",padding: 0)
        func record(_ n: String, _ a: MLXArray) throws { try trace?(n,a.transposed(0,3,1,2)) }
        try record("post_quant",h)
        h = conv(h,"decoder.conv_in"); try record("conv_in",h)
        h = resnet(h,"decoder.mid_block.resnets.0")
        h = attention(h,"decoder.mid_block.attentions.0")
        h = resnet(h,"decoder.mid_block.resnets.1"); try record("mid",h)
        for i in 0..<4 {
            for j in 0..<3 { h = resnet(h,"decoder.up_blocks.\(i).resnets.\(j)") }
            if i < 3 {
                let (b,y,x,c) = (h.dim(0),h.dim(1),h.dim(2),h.dim(3))
                h = broadcast(h[0...,0...,.newAxis,0...,.newAxis,0...],to: [b,y,2,x,2,c]).reshaped(b,y*2,x*2,c)
                h = conv(h,"decoder.up_blocks.\(i).upsamplers.0.conv")
            }
            try record("up.\(i)",h)
        }
        h = conv(silu(norm(h,"decoder.conv_norm_out")).asType(.bfloat16),"decoder.conv_out")
        let decoded = h.transposed(0,3,1,2)
        try trace?("decoded",decoded)
        return decoded
    }
}
