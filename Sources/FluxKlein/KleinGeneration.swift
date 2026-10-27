import Foundation
import MLX
import MLXImageCore

public struct KleinGenerationRequest: Sendable {
    public let prompt: String
    public let width: Int, height: Int
    public let seed: UInt64
    public let steps: Int
    public let guidance: Double
    public let scheduler: String
    public let startingImage: URL?
    public let tiling: Bool
    public init(prompt: String, width: Int, height: Int, seed: UInt64, steps: Int,
                guidance: Double = 1, scheduler: String = "flow_match_euler_discrete", startingImage: URL? = nil, tiling: Bool = false) {
        self.prompt = prompt; self.width = width; self.height = height; self.seed = seed; self.steps = steps
        self.guidance = guidance; self.scheduler = scheduler; self.startingImage = startingImage; self.tiling = tiling
    }
    public func validate() throws {
        guard guidance == 1, scheduler == "flow_match_euler_discrete", startingImage == nil, !tiling else {
            throw KleinTransformerError.invalidInput("only untiled text-to-image, guidance=1, flow_match_euler_discrete is supported")
        }
        guard KleinScheduler.offeredSteps.contains(steps), !prompt.isEmpty else { throw KleinTransformerError.invalidInput("nonempty prompt and steps 1...8 required") }
        _ = try KleinLatent.grid(height: height,width: width)
    }
}
public struct KleinGenerationOutput {
    public let latents: MLXArray, packed: MLXArray, decoded: MLXArray, pixels: MLXArray
    public func writePNG(to url: URL) throws {
        try KleinGenerationPaths.requireNew(url)
        try ImageOutput.encodePNG(pixels: pixels).write(to: url,options: .withoutOverwriting)
    }
}
/// One reusable compiled prediction closure per generator. No VAE encoder, previews or VAE compilation.
public final class KleinGenerator {
    let encode: (String) throws -> KleinPromptEncoding
    public let tokenSummary: (String) -> (valid: Int, untruncated: Int)
    let predict: (MLXArray,MLXArray,MLXArray,MLXArray,MLXArray) -> MLXArray
    public let vae: KleinVAE
    /// The loaded weights (the arrays the encoder, transformer and VAE already hold). The parity harness reads them to compute digests.
    public let weights: (text: KleinTextWeights, transformer: KleinTransformerWeights, vae: KleinVAEWeights)
    public let variant: KleinVariant
    /// Loads a q4 `mflux-save` snapshot of `variant` (FLUX.2 Klein 4B by default). The snapshot is validated against the variant's layout
    /// and pins before any tensor is read; a snapshot of the other size is refused.
    public init(modelDirectory: URL, variant: KleinVariant = .klein4B) throws {
        self.variant = variant
        let tokenizer = try KleinTokenizer.load(modelDirectory: modelDirectory, variant: variant)
        let text = try KleinTextWeights.load(modelDirectory: modelDirectory, variant: variant), encoder = KleinTextEncoder(weights: text)
        let transformer = try KleinTransformerWeights.load(modelDirectory: modelDirectory, variant: variant)
        let vaeWeights = try KleinVAEWeights.load(modelDirectory: modelDirectory, variant: variant)
        // No weight digests here: hashing every tensor (4.6 GB for 4B) cost about 1.8 s of every load, and only the parity harnesses (MLXImageParity) read them.
        transformer.materialize()
        predict = KleinTransformer(weights: transformer).compiledPredict()
        vae = KleinVAE(weights: vaeWeights)
        weights = (text, transformer, vaeWeights)  // the same arrays the encoder, transformer and VAE already hold: no extra memory
        tokenSummary = { prompt in
            (KleinTokenizer.tokenize(prompt,with: tokenizer).validCount,tokenizer.encode(KleinTokenizer.renderChat(prompt: prompt)).count)
        }
        encode = { try KleinPromptEncoder.encode(KleinTokenizer.tokenize($0,with: tokenizer),with: encoder) }
    }
    /// onStep receives the evaluated post-step latent (1-based). Throwing callbacks abort generation.
    public func generate(_ request: KleinGenerationRequest, onStep: ((Int,MLXArray) throws -> Void)? = nil,
                         trace: ((String,MLXArray) throws -> Void)? = nil) throws -> KleinGenerationOutput {
        try request.validate()
        let encoding = try encode(request.prompt)
        let initial = try KleinLatent.preparePacked(seed: request.seed,height: request.height,width: request.width)
        let scheduler = try KleinScheduler(steps: request.steps,height: request.height,width: request.width)
        for (n,a) in ["prompt_embeds":encoding.promptEmbeds,"text_ids":encoding.textIDs,"latent_ids":initial.ids,
                      "sigmas":scheduler.sigmas,"timesteps":scheduler.timesteps,"initial":initial.latents] { eval(a); try trace?(n,a) }
        var latents = initial.latents
        for t in 0..<request.steps {
            let noise = predict(latents,initial.ids,encoding.promptEmbeds,encoding.textIDs,scheduler.timesteps[t])
            latents = try scheduler.step(noise: noise,timestep: t,latents: latents)
            eval(latents)
            try trace?("noise.\(t)",noise); try trace?("step.\(t+1)",latents)
            try onStep?(t+1,latents)
        }
        try trace?("after_loop",latents)
        let packed = try KleinLatent.unpack(latents,height: request.height,width: request.width)
        try trace?("packed",packed)
        let decoded = try vae.decodePacked(packed) { n,a in if n == "unpacked" { try trace?(n,a) } }
        eval(decoded)
        let pixels = try KleinGenerationPixels.convert(decoded,width: request.width,height: request.height)
        eval(pixels); try trace?("decoded",decoded); try trace?("pixels",pixels)
        return KleinGenerationOutput(latents: latents,packed: packed,decoded: decoded,pixels: pixels)
    }
}
public enum KleinGenerationPixels {
    public static func convert(_ decoded: MLXArray, width: Int, height: Int) throws -> MLXArray {
        guard width > 0, height > 0, decoded.shape == [1,3,height,width], decoded.dtype == .bfloat16,
              KleinTensorValues.floats(decoded).allSatisfy(\.isFinite) else { throw KleinTransformerError.invalidInput("decoded tensor shape/dtype/nonfinite") }
        return ImageOutput.pixels(decoded: decoded)
    }
}
