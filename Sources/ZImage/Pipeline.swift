import Foundation
import MLX
import MLXImageCore

/// One text-to-image job for Z-Image Turbo.
public struct GenerationRequest: Equatable, Sendable {
    /// Z-Image Turbo's supported step range (4...12).
    public static let stepRange = 4 ... 12

    public var prompt: String
    public var seed: UInt64
    public var width: Int
    public var height: Int
    public var steps: Int

    public init(prompt: String, seed: UInt64, width: Int, height: Int, steps: Int) {
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.steps = steps
    }

    /// The latent is `[16, 1, H/8, W/8]` and the transformer patches it by 2, so both sides must be multiples of 16.
    public func validate() throws {
        guard Self.stepRange.contains(steps) else {
            throw GenerationError.invalid("steps \(steps) outside the supported range \(Self.stepRange)")
        }
        guard width >= 16, height >= 16, width % 16 == 0, height % 16 == 0 else {
            throw GenerationError.invalid("size \(width)x\(height) must be positive multiples of 16")
        }
    }
}

public enum GenerationError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String { if case .invalid(let m) = self { m } else { "" } }
}

/// How the transformer's prediction is evaluated inside the denoising loop. mflux's production path is compiled
/// (`ZImage._predict` is `mx.compile(predict)` except on M1/M2); eager is the retained cross-check.
public enum PredictionMode: String, Sendable {
    case compiled
    case eager
}

public struct GeneratedImage {
    /// The sampler's final latents `[16, 1, H/8, W/8]` (bfloat16).
    public let latents: MLXArray
    /// The decoded VAE tensor `[1, 3, H, W]`.
    public let decoded: MLXArray
    /// `[H, W, 3]` uint8.
    public let pixels: MLXArray
}

/// The denoising loop over the transformer: seeded latents, the linear scheduler and one prediction per step. It needs only
/// the transformer, so the per-step latent check can run it without the text encoder or the VAE.
public final class ZImageDenoiser {
    public let transformer: ZImageTransformer
    private lazy var compiledPredict = transformer.compiledForward()

    public init(transformer: ZImageTransformer) { self.transformer = transformer }

    /// One prediction, `ZImage._predict`'s equivalent: compiled (the production path) or eager.
    public func predict(latents: MLXArray, timestep: MLXArray, captionFeatures: MLXArray, mode: PredictionMode) -> MLXArray {
        switch mode {
        case .compiled: compiledPredict(latents, timestep, captionFeatures)
        case .eager: transformer.forward(latents: latents, timestep: timestep, captionFeatures: captionFeatures)
        }
    }

    /// `1 - sigma[t]` as `[1]` float32, as the generation loop computes the timestep.
    static func timestep(_ scheduler: LinearScheduler, step t: Int) -> MLXArray {
        let sigma = scheduler.sigmas[t].reshaped(1)
        return MLX.ones(like: sigma) - sigma
    }

    /// `ZImage.generate_image`'s loop: for each step `timestep = 1 - sigma[t]`, `noise = predict(...)`, `latents =
    /// scheduler.step(...)`, evaluated every step. `onStep(t, latents)` sees the state after step `t` (0-based).
    public func denoise(
        request: GenerationRequest, captionFeatures: MLXArray, mode: PredictionMode = .compiled,
        onStep: (Int, MLXArray) throws -> Void = { _, _ in }
    ) throws -> MLXArray {
        try request.validate()
        let scheduler = LinearScheduler(steps: request.steps, width: request.width, height: request.height)
        var latents = ZImageLatentCreator.createNoise(seed: request.seed, height: request.height, width: request.width)
        for t in 0 ..< request.steps {
            let timestep = Self.timestep(scheduler, step: t)
            let noise = predict(latents: latents, timestep: timestep, captionFeatures: captionFeatures, mode: mode)
            latents = scheduler.step(noise: noise, timestep: t, latents: latents)
            eval(latents)
            try onStep(t, latents)
        }
        return latents
    }
}

/// The complete native text-to-image pipeline: tokenizer, text encoder, seeded latents, the denoising loop over the
/// transformer, VAE decoding and image conversion. Everything loads from the existing q4 snapshot, read-only.
public final class ZImageGenerator {
    public let tokenizer: QwenTokenizer
    public let textEncoder: TextEncoder
    public let denoiser: ZImageDenoiser
    public let vae: ZImageVAE

    public init(modelDirectory: URL) throws {
        tokenizer = try QwenTokenizer.load(modelDirectory: modelDirectory)
        textEncoder = TextEncoder(weights: try TextEncoderWeights.load(modelDirectory: modelDirectory))
        denoiser = ZImageDenoiser(transformer: ZImageTransformer(weights: try TransformerWeights.load(modelDirectory: modelDirectory)))
        vae = ZImageVAE(weights: try VAEWeights.load(modelDirectory: modelDirectory))
    }

    /// `PromptEncoder.encode_prompt`: the caption features `[valid, 2560]`.
    public func encodePrompt(_ prompt: String) throws -> MLXArray {
        let embedding = try textEncoder.encode(tokenizer.tokenize(prompt))
        eval(embedding)
        return embedding
    }

    /// Prompt to pixels.
    public func generate(
        _ request: GenerationRequest, mode: PredictionMode = .compiled, onStep: (Int, MLXArray) throws -> Void = { _, _ in }
    ) throws -> GeneratedImage {
        try request.validate()
        let caption = try encodePrompt(request.prompt)
        let latents = try denoiser.denoise(request: request, captionFeatures: caption, mode: mode, onStep: onStep)
        let decoded = vae.decode(samplerLatents: latents)
        let pixels = ImageOutput.pixels(decoded: decoded)
        eval(decoded, pixels)
        return GeneratedImage(latents: latents, decoded: decoded, pixels: pixels)
    }
}
