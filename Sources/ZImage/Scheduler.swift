import Foundation
import MLX

/// Z-Image Turbo's `ModelConfig` sigma-shift constants (mflux `ModelConfig.z_image_turbo()`); there is no terminal shift.
public enum ZImageSchedule {
    public static let baseSequenceLength = 256.0
    public static let maxSequenceLength = 4096.0
    public static let baseShift = 0.5
    public static let maxShift = 1.15
}

/// What the denoising loop needs from a scheduler: the `steps + 1` float32 sigmas (ending in 0) and the Euler step.
public protocol ZImageScheduler {
    var sigmas: MLXArray { get }
    func step(noise: MLXArray, timestep: Int, latents: MLXArray) -> MLXArray
}

/// Port of mflux's `LinearScheduler`, which Z-Image Turbo uses (it does not support guidance, so mflux picks `linear`).
public struct LinearScheduler: ZImageScheduler {
    public let steps: Int
    public let width: Int
    public let height: Int
    /// `steps + 1` float32 values, ending in 0.
    public let sigmas: MLXArray
    /// `0 ..< steps` as float32: the step indices mflux calls timesteps.
    public let timesteps: MLXArray

    public init(steps: Int, width: Int, height: Int) {
        self.steps = steps
        self.width = width
        self.height = height
        // Same operation order as the reference, including the double-precision `mu` that is then rounded to float32.
        var sigmas = linspace(1.0, 1.0 / Double(steps), count: steps, dtype: .float32)
        sigmas = concatenated([sigmas.asType(.float32), MLXArray.zeros([1])])
        let m = (ZImageSchedule.maxShift - ZImageSchedule.baseShift) / (ZImageSchedule.maxSequenceLength - ZImageSchedule.baseSequenceLength)
        let b = ZImageSchedule.baseShift - m * ZImageSchedule.baseSequenceLength
        let mu = MLXArray(Float(m * Double(width) * Double(height) / 256 + b))
        let head = sigmas[..<steps]  // sigmas[:-1]
        let shifted = MLX.exp(mu) / (MLX.exp(mu) + (1 / head - 1))
        self.sigmas = concatenated([shifted, MLXArray.zeros([1])])
        self.timesteps = MLXArray(0 ..< Int32(steps)).asType(.float32)
    }

    /// `latents + noise.astype(latents.dtype) * (sigmas[t + 1] - sigmas[t]).astype(latents.dtype)`
    public func step(noise: MLXArray, timestep: Int, latents: MLXArray) -> MLXArray {
        let dt = (sigmas[timestep + 1] - sigmas[timestep]).asType(latents.dtype)
        return latents + noise.asType(latents.dtype) * dt
    }
}

/// Port of mflux 0.20's `FlowMatchEulerDiscreteScheduler` as `ZImage.generate_image` selects it for models that support
/// guidance (Z-Image Base, `requires_sigma_shift`): `Config.scheduler` builds it and then calls `set_image_seq_len`, so the
/// sigmas come from `get_timesteps_and_sigmas` with mflux's empirical `mu` (not the diffusers `shift` of 6.0).
public struct FlowMatchEulerScheduler: ZImageScheduler {
    public let steps: Int
    public let width: Int
    public let height: Int
    /// `steps + 1` float32 values, ending in 0.
    public let sigmas: MLXArray

    public init(steps: Int, width: Int, height: Int) {
        self.steps = steps
        self.width = width
        self.height = height
        let imageSeqLen = (height / 16) * (width / 16)
        let mu = Self.empiricalMu(imageSeqLen: imageSeqLen, steps: steps)
        let linear = linspace(1.0, 1.0 / Double(steps), count: steps, dtype: .float32)
        // `mx.exp(mu) / (mx.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))` with `sigma_power` 1.0; `mu` is a Python float.
        let expMu = MLX.exp(MLXArray(Float(mu)))
        let shifted = expMu / (expMu + MLX.pow(1 / linear - 1, Float(1)))
        self.sigmas = concatenated([shifted, MLXArray.zeros([1], dtype: shifted.dtype)], axis: 0)
    }

    /// `_compute_empirical_mu`, in double precision as Python computes it.
    static func empiricalMu(imageSeqLen: Int, steps: Int) -> Double {
        let (a1, b1) = (8.73809524e-05, 1.89833333)
        let (a2, b2) = (0.00016927, 0.45666666)
        let length = Double(imageSeqLen)
        if imageSeqLen > 4300 { return a2 * length + b2 }
        let m200 = a2 * length + b2
        let m10 = a1 * length + b1
        let a = (m200 - m10) / 190.0
        let b = m200 - 200.0 * a
        return a * Double(steps) + b
    }

    /// `latents + dt * noise` with `dt = (sigmas[t + 1] - sigmas[t])` and both operands in the latents' dtype.
    public func step(noise: MLXArray, timestep: Int, latents: MLXArray) -> MLXArray {
        let dt = (sigmas[timestep + 1] - sigmas[timestep]).asType(latents.dtype)
        return latents + dt * noise.asType(latents.dtype)
    }
}
