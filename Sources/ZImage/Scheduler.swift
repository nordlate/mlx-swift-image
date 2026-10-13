import Foundation
import MLX

/// Z-Image Turbo's `ModelConfig` sigma-shift constants (mflux `ModelConfig.z_image_turbo()`); there is no terminal shift.
public enum ZImageSchedule {
    public static let baseSequenceLength = 256.0
    public static let maxSequenceLength = 4096.0
    public static let baseShift = 0.5
    public static let maxShift = 1.15
}

/// Port of mflux's `LinearScheduler`, which Z-Image Turbo uses (it does not support guidance, so mflux picks `linear`).
public struct LinearScheduler {
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
