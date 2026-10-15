import Foundation
import MLX
import MLXFast
import MLXNN

/// Eager Klein decoder. Activations stay NHWC between layers; public inputs/outputs and trace tensors are NCHW like MFLUX.
public struct KleinVAE {
    public let weights: KleinVAEWeights
    public init(weights: KleinVAEWeights) { self.weights = weights }

    public static func unpatchify(_ x: MLXArray) throws -> MLXArray {
        guard x.ndim == 4, x.dim(1) % 4 == 0, x.size > 0 else {
            throw KleinTransformerError.invalidInput("unpatchify requires nonempty NCHW with channels divisible by four")
        }
        let (b,c,h,w) = (x.dim(0),x.dim(1),x.dim(2),x.dim(3))
        return x.reshaped(b,c/4,2,2,h,w).transposed(0,1,4,2,5,3).reshaped(b,c/4,h*2,w*2)
    }
    public static func denormalize(_ packed: MLXArray, mean: MLXArray, variance: MLXArray) throws -> MLXArray {
        guard packed.ndim == 4, packed.dim(1) == 128, packed.size > 0, packed.dtype == .bfloat16,
              mean.shape == [128], variance.shape == [128], mean.dtype == .bfloat16, variance.dtype == .bfloat16 else {
            throw KleinTransformerError.invalidInput("VAE requires BF16 [B,128,H,W] and BF16 BN statistics [128]")
        }
        let std = MLX.sqrt(variance.reshaped(1,128,1,1) + KleinVAELayout.bnEps)
        return packed * std + mean.reshaped(1,128,1,1)
    }
    public func unpack(_ packed: MLXArray, trace: ((String, MLXArray) throws -> Void)? = nil) throws -> MLXArray {
        let bn = try Self.denormalize(packed, mean: weights.tensor("bn.running_mean"), variance: weights.tensor("bn.running_var"))
        try trace?("bn", bn)
        let x = try Self.unpatchify(bn)
        try trace?("unpacked", x)
        return x
    }
    func conv(_ x: MLXArray, _ name: String, padding: Int = 1) -> MLXArray {
        conv2d(x, weights.tensor(name + ".weight"), padding: IntOrPair(padding)) + weights.tensor(name + ".bias")
    }
    /// All Klein norms compute in float32 and cast AFTER the affine transform, including resnet norms.
    func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        let (b,h,w,c) = (x.dim(0),x.dim(1),x.dim(2),x.dim(3)), g = KleinVAELayout.normGroups
        var y = x.asType(.float32).reshaped(b,-1,g,c/g).transposed(0,2,1,3).reshaped(b,g,-1)
        y = MLXFast.layerNorm(y, weight: nil, bias: nil, eps: KleinVAELayout.normEps)
        y = y.reshaped(b,g,-1,c/g).transposed(0,2,1,3).reshaped(b,h,w,c)
        return (weights.tensor(name + ".weight") * y + weights.tensor(name + ".bias")).asType(.bfloat16)
    }
    func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        quantizedMM(x, weights.tensor(name + ".weight"), scales: weights.tensor(name + ".scales"), biases: weights.tensor(name + ".biases"),
                    transpose: true, groupSize: 64, bits: 4) + weights.tensor(name + ".bias")
    }
    func resnet(_ x: MLXArray, _ n: String) -> MLXArray {
        var h = conv(silu(norm(x,n + ".norm1")),n + ".conv1")
        h = conv(silu(norm(h,n + ".norm2")),n + ".conv2")
        let residual = weights.tensors[n + ".conv_shortcut.weight"] == nil ? x : conv(x,n + ".conv_shortcut",padding: 0)
        return h + residual
    }
    /// Python `1 / mx.sqrt(q.shape[-1])`: integer input promotes to float32, independent of q.dtype.
    public static func attentionScale(channels: Int) -> Float {
        (MLXArray(Float(1)) / MLX.sqrt(MLXArray(Int32(channels)))).item(Float.self)
    }
    func attention(_ x: MLXArray, _ n: String) -> MLXArray {
        let (b,h,w,c) = (x.dim(0),x.dim(1),x.dim(2),x.dim(3)), normalized = norm(x,n + ".group_norm")
        func heads(_ p: String) -> MLXArray { linear(normalized,n + "." + p).reshaped(b,h*w,1,c).transposed(0,2,1,3) }
        let y = MLXFast.scaledDotProductAttention(queries: heads("to_q"), keys: heads("to_k"), values: heads("to_v"),
                                               scale: Self.attentionScale(channels: c), mask: .none)
        return x + linear(y.transposed(0,2,1,3).reshaped(b,h,w,c),n + ".to_out")
    }
    public func decodePacked(_ packed: MLXArray, trace: ((String, MLXArray) throws -> Void)? = nil) throws -> MLXArray {
        let x = try unpack(packed, trace: trace)
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
