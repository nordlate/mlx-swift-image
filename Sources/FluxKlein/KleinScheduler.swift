import Foundation
import MLX

/// Port of mflux's `FlowMatchEulerDiscreteScheduler` as the worker drives Klein 4B:
/// `Config(model_config=flux2_klein_4b, guidance=1.0, scheduler="flow_match_euler_discrete")`.
///
/// `ModelConfig.flux2_klein_4b().requires_sigma_shift` is `True`, so `Config.scheduler` calls `set_image_seq_len((H/16) * (W/16))` right
/// after constructing the scheduler. That replaces the constructor's schedule (the legacy terminal-stretch path, which is NOT used)
/// with `get_timesteps_and_sigmas`: float32 `linspace(1, 1/n, n)`, shifted exponentially by the empirical `mu` (a Python double, rounded
/// to float32 when it meets `mx.exp`), then `timesteps = sigmas * 1000` and a trailing 0 sigma.
public struct KleinScheduler {
    public static let numTrainTimesteps = 1000
    /// The step counts the worker supports for Klein (1 through 8); nothing else is supported.
    public static let offeredSteps = 1...8

    public let steps: Int
    public let imageSeqLen: Int
    /// The empirical shift parameter as the Python double (before the float32 rounding the array math applies).
    public let mu: Double
    /// `steps + 1` float32 values ending in 0.
    public let sigmas: MLXArray
    /// `steps` float32 values, `sigmas[:-1] * 1000`.
    public let timesteps: MLXArray

    public init(steps: Int, height: Int, width: Int) throws {
        guard Self.offeredSteps.contains(steps) else {
            throw KleinTransformerError.invalidInput("steps \(steps) is not one of the offered \(Self.offeredSteps)")
        }
        let grid = try KleinLatent.grid(height: height, width: width)
        self.steps = steps
        self.imageSeqLen = grid.height * grid.width
        self.mu = Self.empiricalMu(imageSeqLen: imageSeqLen, numSteps: steps)
        let linear = linspace(1.0, 1.0 / Double(steps), count: steps, dtype: .float32)
        // `mx.exp(mu) / (mx.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))`, sigma_power 1.0: Python scalars are float32 weak scalars.
        let one = MLXArray(Float(1))
        let expMu = MLX.exp(MLXArray(Float(mu)))
        let shifted = expMu / (expMu + MLX.pow(one / linear - one, MLXArray(Float(1.0))))
        self.timesteps = shifted * MLXArray(Float(Self.numTrainTimesteps))
        self.sigmas = concatenated([shifted, MLXArray.zeros([1], dtype: shifted.dtype)], axis: 0)
        eval(sigmas, timesteps)
    }

    /// `_compute_empirical_mu` in double precision, exactly as written.
    public static func empiricalMu(imageSeqLen: Int, numSteps: Int) -> Double {
        let a1 = 8.73809524e-05, b1 = 1.89833333
        let a2 = 0.00016927, b2 = 0.45666666
        let seq = Double(imageSeqLen), steps = Double(numSteps)
        if imageSeqLen > 4300 { return a2 * seq + b2 }
        let m200 = a2 * seq + b2
        let m10 = a1 * seq + b1
        let a = (m200 - m10) / 190.0
        let b = m200 - 200.0 * a
        return a * steps + b
    }

    /// The step the worker takes, compiled shapeless exactly as mflux's module-level `_step`:
    /// `latents + (s1 - s2).astype(latents.dtype) * noise.astype(latents.dtype)` with `s1 = sigmas[t + 1]`, `s2 = sigmas[t]`.
    /// Compiling matters: the compiled graph keeps its own bfloat16 rounding points, which an eager version would not reproduce.
    static let compiledStep: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { (inputs: [MLXArray]) -> [MLXArray] in
        let (noise, latents, s1, s2) = (inputs[0], inputs[1], inputs[2], inputs[3])
        let dt = (s1 - s2).asType(latents.dtype)
        return [latents + dt * noise.asType(latents.dtype)]
    }

    /// `FlowMatchEulerDiscreteScheduler.step`; `timestep` must index a step of this schedule.
    public func step(noise: MLXArray, timestep: Int, latents: MLXArray) throws -> MLXArray {
        guard (0..<steps).contains(timestep) else { throw KleinTransformerError.invalidInput("timestep index \(timestep) outside 0..<\(steps)") }
        return try Self.step(noise: noise, latents: latents, nextSigma: sigmas[timestep + 1], sigma: sigmas[timestep])
    }

    /// The step from explicit sigma scalars (the isolated check feeds the reference's own sigmas).
    public static func step(noise: MLXArray, latents: MLXArray, nextSigma: MLXArray, sigma: MLXArray) throws -> MLXArray {
        guard noise.shape == latents.shape, latents.size > 0,
              [DType.bfloat16, .float16, .float32].contains(latents.dtype),
              [DType.bfloat16, .float16, .float32].contains(noise.dtype),
              nextSigma.ndim == 0, sigma.ndim == 0, nextSigma.dtype == .float32, sigma.dtype == .float32 else {
            throw KleinTransformerError.invalidInput("scheduler step requires matching nonempty floating tensors and float32 scalar sigmas")
        }
        return compiledStep([noise, latents, nextSigma, sigma])[0]
    }
}
