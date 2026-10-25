import Foundation
import MLX
import MLXImageCore

/// The two Z-Image checkpoints this package runs. They share the transformer, text encoder, VAE and weight loading; they differ
/// in the scheduler and in classifier-free guidance, which mflux selects per `ModelConfig`.
public enum ZImageVariant: String, Sendable {
    /// Z-Image Turbo: distilled, `LinearScheduler`, no guidance.
    case turbo
    /// Z-Image Base: `FlowMatchEulerScheduler`, classifier-free guidance with a negative prompt.
    case base

    /// The supported step range: 4...12 for Turbo; 1...50 for Base (the model card recommends 28...50).
    public var stepRange: ClosedRange<Int> {
        switch self {
        case .turbo: 4 ... 12
        case .base: 1 ... 50
        }
    }

    public var supportsGuidance: Bool { self == .base }
}

/// One text-to-image job for Z-Image Turbo or Z-Image Base.
public struct GenerationRequest: Equatable, Sendable {
    /// Z-Image Turbo's supported step range (4...12); see `ZImageVariant.stepRange`.
    public static let stepRange = ZImageVariant.turbo.stepRange

    public var prompt: String
    public var seed: UInt64
    public var width: Int
    public var height: Int
    public var steps: Int
    /// Z-Image Base only. Guidance above 1.0 runs a conditional and a negative-prompt pass per step and combines them as mflux
    /// does, `noise + guidance * (noise - negativeNoise)`; at 1.0 or below there is one pass and no negative prompt (mflux's
    /// default is 0.0). The model card recommends 3.0...5.0. Turbo rejects a non-zero value and a non-blank negative prompt.
    public var guidance: Float
    /// Z-Image Base only, used when guidance is above 1.0 (a non-blank negative prompt with guidance of 1.0 or below is rejected
    /// by `validate`, since it would be ignored). A nil or blank prompt is encoded as a single space, as mflux does.
    public var negativePrompt: String?

    public init(
        prompt: String, seed: UInt64, width: Int, height: Int, steps: Int, guidance: Float = 0, negativePrompt: String? = nil
    ) {
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.negativePrompt = negativePrompt
    }

    /// Classifier-free guidance runs only above 1.0 (mflux's `guidance <= 1.0` returns no negative encodings).
    public var usesGuidance: Bool { guidance > 1.0 }

    /// mflux's negative text: the prompt, or a single space when it is nil or blank.
    public var effectiveNegativePrompt: String {
        if let negativePrompt, !negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return negativePrompt }
        return " "
    }

    /// The latent is `[16, 1, H/8, W/8]` and the transformer patches it by 2, so both sides must be multiples of 16.
    public func validate(for variant: ZImageVariant = .turbo) throws {
        guard variant.stepRange.contains(steps) else {
            throw GenerationError.invalid("steps \(steps) outside the supported range \(variant.stepRange) for Z-Image \(variant.rawValue)")
        }
        guard guidance.isFinite, guidance >= 0 else {
            throw GenerationError.invalid("guidance \(guidance) must be a finite value >= 0")
        }
        let hasNegativePrompt = negativePrompt?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
        if !variant.supportsGuidance, guidance != 0 || hasNegativePrompt {
            throw GenerationError.invalid("Z-Image Turbo does not support guidance or a non-blank negative prompt")
        }
        if variant.supportsGuidance, !usesGuidance, hasNegativePrompt {
            throw GenerationError.invalid("a negative prompt needs guidance above 1.0 (guidance \(guidance) would ignore it)")
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

/// The denoising loop over the transformer: seeded latents, the variant's scheduler and one prediction per step (two with
/// guidance, combined as mflux does). It needs only the transformer, so the per-step latent check can run it without the text
/// encoder or the VAE.
public final class ZImageDenoiser {
    public let transformer: ZImageTransformer
    public let variant: ZImageVariant
    private lazy var compiledPredict = transformer.compiledForward()
    /// The compiled guided prediction for the most recent guidance value only, so stepping through values does not keep every graph.
    private var compiledGuidedPredict: (guidance: Float, run: (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray)?

    public init(transformer: ZImageTransformer, variant: ZImageVariant = .turbo) {
        self.transformer = transformer
        self.variant = variant
    }

    /// The scheduler mflux picks: `linear` for Turbo, `flow_match_euler_discrete` for Base.
    public func scheduler(for request: GenerationRequest) -> any ZImageScheduler {
        switch variant {
        case .turbo: LinearScheduler(steps: request.steps, width: request.width, height: request.height)
        case .base: FlowMatchEulerScheduler(steps: request.steps, width: request.width, height: request.height)
        }
    }

    /// One prediction, `ZImage._predict`'s equivalent: compiled (the production path) or eager.
    public func predict(latents: MLXArray, timestep: MLXArray, captionFeatures: MLXArray, mode: PredictionMode) -> MLXArray {
        switch mode {
        case .compiled: compiledPredict(latents, timestep, captionFeatures)
        case .eager: transformer.forward(latents: latents, timestep: timestep, captionFeatures: captionFeatures)
        }
    }

    /// mflux's guidance combination: `noise + guidance * (noise - negativeNoise)`.
    public static func combine(noise: MLXArray, negativeNoise: MLXArray, guidance: Float) -> MLXArray {
        noise + guidance * (noise - negativeNoise)
    }

    /// One guided prediction, `ZImage._predict` with negative encodings: `noise + guidance * (noise - negativeNoise)`. The
    /// compiled path wraps both passes and the combination in one graph, as mflux's `mx.compile(predict)` does; `guidance` is a
    /// constant of the graph, so a change of value recompiles.
    public func predictGuided(
        latents: MLXArray, timestep: MLXArray, captionFeatures: MLXArray, negativeCaptionFeatures: MLXArray, guidance: Float,
        mode: PredictionMode
    ) -> MLXArray {
        switch mode {
        case .compiled:
            if compiledGuidedPredict?.guidance.bitPattern != guidance.bitPattern {
                transformer.materializeWeights()
                let transformer = transformer
                let compiled = compile { (inputs: [MLXArray]) -> [MLXArray] in
                    let noise = transformer.forward(latents: inputs[0], timestep: inputs[1], captionFeatures: inputs[2])
                    let negative = transformer.forward(latents: inputs[0], timestep: inputs[1], captionFeatures: inputs[3])
                    return [Self.combine(noise: noise, negativeNoise: negative, guidance: guidance)]
                }
                compiledGuidedPredict = (guidance, { compiled([$0, $1, $2, $3])[0] })
            }
            return compiledGuidedPredict!.run(latents, timestep, captionFeatures, negativeCaptionFeatures)
        case .eager:
            let noise = transformer.forward(latents: latents, timestep: timestep, captionFeatures: captionFeatures)
            let negative = transformer.forward(latents: latents, timestep: timestep, captionFeatures: negativeCaptionFeatures)
            return Self.combine(noise: noise, negativeNoise: negative, guidance: guidance)
        }
    }

    /// `1 - sigma[t]` as `[1]` float32, as the generation loop computes the timestep.
    public static func timestep(_ scheduler: any ZImageScheduler, step t: Int) -> MLXArray {
        let sigma = scheduler.sigmas[t].reshaped(1)
        return MLX.ones(like: sigma) - sigma
    }

    /// `ZImage.generate_image`'s loop: for each step `timestep = 1 - sigma[t]`, `noise = predict(...)`, `latents =
    /// scheduler.step(...)`, evaluated every step. `onStep(t, latents)` sees the state after step `t` (0-based). A guided request
    /// (Base, guidance above 1.0) needs `negativeCaptionFeatures`.
    public func denoise(
        request: GenerationRequest, captionFeatures: MLXArray, negativeCaptionFeatures: MLXArray? = nil,
        mode: PredictionMode = .compiled, onStep: (Int, MLXArray) throws -> Void = { _, _ in }
    ) throws -> MLXArray {
        try request.validate(for: variant)
        let guided = request.usesGuidance
        if guided, negativeCaptionFeatures == nil {
            throw GenerationError.invalid("guidance \(request.guidance) needs negative caption features")
        }
        let scheduler = scheduler(for: request)
        var latents = ZImageLatentCreator.createNoise(seed: request.seed, height: request.height, width: request.width)
        for t in 0 ..< request.steps {
            let timestep = Self.timestep(scheduler, step: t)
            let noise: MLXArray
            if guided, let negativeCaptionFeatures {
                noise = predictGuided(
                    latents: latents, timestep: timestep, captionFeatures: captionFeatures,
                    negativeCaptionFeatures: negativeCaptionFeatures, guidance: request.guidance, mode: mode)
            } else {
                noise = predict(latents: latents, timestep: timestep, captionFeatures: captionFeatures, mode: mode)
            }
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
    public let variant: ZImageVariant

    public init(modelDirectory: URL, variant: ZImageVariant = .turbo) throws {
        self.variant = variant
        tokenizer = try QwenTokenizer.load(modelDirectory: modelDirectory)
        textEncoder = TextEncoder(weights: try TextEncoderWeights.load(modelDirectory: modelDirectory))
        denoiser = ZImageDenoiser(
            transformer: ZImageTransformer(weights: try TransformerWeights.load(modelDirectory: modelDirectory)), variant: variant)
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
        try request.validate(for: variant)
        let caption = try encodePrompt(request.prompt)
        let negative = request.usesGuidance ? try encodePrompt(request.effectiveNegativePrompt) : nil
        let latents = try denoiser.denoise(
            request: request, captionFeatures: caption, negativeCaptionFeatures: negative, mode: mode, onStep: onStep)
        let decoded = vae.decode(samplerLatents: latents)
        let pixels = ImageOutput.pixels(decoded: decoded)
        eval(decoded, pixels)
        return GeneratedImage(latents: latents, decoded: decoded, pixels: pixels)
    }
}
