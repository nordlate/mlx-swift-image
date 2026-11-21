import Foundation
import MLX
import MLXFast
import MLXNN

/// The layer primitives SeedVR2 uses, each the exact op sequence of the MLX Python module MFLUX builds, so the dtype flow (fp16
/// weights, float32 or bfloat16 activations, MLX's type promotion) is the reference's.
enum SeedVR2Ops {
    /// `nn.QuantizedLinear.__call__`: `quantized_matmul(x, weight, scales, biases, transpose=True)`, then `+ bias` if stored.
    static func quantizedLinear(_ x: MLXArray, _ w: [String: MLXArray], _ name: String) -> MLXArray {
        let y = quantizedMM(
            x, w.tensor(name + ".weight"), scales: w.tensor(name + ".scales"), biases: w.tensor(name + ".biases"), transpose: true,
            groupSize: SeedVR2Layout.groupSize, bits: SeedVR2Layout.bits, mode: .affine)
        if let bias = w[name + ".bias"] { return y + bias }
        return y
    }

    /// `nn.Linear.__call__` with a bias: `addmm(bias, x, weight.T)`.
    static func linear(_ x: MLXArray, _ w: [String: MLXArray], _ name: String) -> MLXArray {
        addMM(w.tensor(name + ".bias"), x, w.tensor(name + ".weight").T)
    }

    /// `mx.fast.rms_norm(x, weight, eps)`.
    static func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float = SeedVR2Layout.normEps) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }

    /// `TransformerBlock._rms_norm`: `mx.fast.rms_norm(x, mx.ones(x.shape[-1]), eps)`; the float32 ones promote an fp16 input.
    static func plainRMSNorm(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: .float32), eps: SeedVR2Layout.normEps)
    }

    /// `nn.GroupNorm(32, dims, eps=1e-6, pytorch_compatible=True)` over a channels-last input: normalize each group over all
    /// positions with `fast.layer_norm`, then `weight * x + bias`.
    static func groupNorm(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let shape = x.shape
        let batch = shape[0], dims = shape[shape.count - 1]
        let groups = SeedVR2Layout.groupNormGroups, groupSize = dims / groups
        var y = x.reshaped(batch, -1, groups, groupSize)
        y = y.transposed(0, 2, 1, 3).reshaped(batch, groups, -1)
        y = MLXFast.layerNorm(y, weight: nil, bias: nil, eps: SeedVR2Layout.groupNormEps)
        y = y.reshaped(batch, groups, -1, groupSize)
        y = y.transposed(0, 2, 1, 3).reshaped(shape)
        return weight * y + bias
    }

    /// `nn.silu` (compiled, shapeless, as in MLX Python).
    static func silu(_ x: MLXArray) -> MLXArray { MLXNN.silu(x) }
}
