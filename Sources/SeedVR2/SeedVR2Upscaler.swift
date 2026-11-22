import Foundation
import MLX
import MLXImageCore

/// One upscale: the input pixels, the output size, `softness` and the seed of the noise.
public struct SeedVR2Request: Sendable {
    /// The input as decoded 8-bit RGB, EXIF orientation already applied (MFLUX opens the file with `exif_transpose`).
    public let image: ImageOutput.RGBImage
    public let resolution: SeedVR2Resolution
    /// 0...1: the input is first downscaled by `1 + 7 * softness` (up to 8x) and scaled back, which smooths the result. 0 disables it;
    /// 0.5 is the recommended default.
    public let softness: Double
    public let seed: UInt64

    public init(image: ImageOutput.RGBImage, resolution: SeedVR2Resolution, softness: Double = 0.5, seed: UInt64) {
        self.image = image
        self.resolution = resolution
        self.softness = softness
        self.seed = seed
    }

    /// The largest output side accepted. A 4096 x 4096 output was measured; larger ones grow the transformer's activations further.
    public static let maximumOutputSide = 8192

    public func validate() throws {
        guard image.width >= 1, image.height >= 1, image.bytes.count == image.width * image.height * 3 else {
            throw SeedVR2Error.invalidRequest("the input image is empty or its byte count does not match \(image.width) x \(image.height) RGB")
        }
        guard softness.isFinite, (0.0 ... 1.0).contains(softness) else {
            throw SeedVR2Error.invalidRequest("softness \(softness) is outside 0...1")
        }
        // Bound the target shortest side before `outputSize` converts it to Int, so a huge side or factor throws instead of trapping.
        let maximum = Self.maximumOutputSide
        switch resolution {
        case .shortestSide(let side):
            guard side >= 2 else { throw SeedVR2Error.invalidRequest("the shortest side must be at least 2 pixels, got \(side)") }
            guard side <= maximum else {
                throw SeedVR2Error.invalidRequest("the shortest side \(side) is above \(maximum) pixels")
            }
        case .scale(let factor):
            guard factor.isFinite, factor > 0 else { throw SeedVR2Error.invalidRequest("the scale factor must be positive, got \(factor)") }
            guard factor * Double(min(image.width, image.height)) <= Double(maximum) else {
                throw SeedVR2Error.invalidRequest("the scale factor \(factor) would make the shortest side larger than \(maximum) pixels")
            }
        }
        let size = SeedVR2Preprocessing.outputSize(width: image.width, height: image.height, resolution: resolution)
        guard size.width >= 2, size.height >= 2 else {
            throw SeedVR2Error.invalidRequest("the output would be \(size.width) x \(size.height); both sides must be at least 2 pixels")
        }
        guard max(size.width, size.height) <= Self.maximumOutputSide else {
            throw SeedVR2Error.invalidRequest("the output would be \(size.width) x \(size.height), above \(Self.maximumOutputSide) pixels per side")
        }
    }
}

/// The upscaled image and the intermediates the parity harness compares.
public struct SeedVR2Output {
    /// `[height, width, 3]` uint8, MFLUX's `ImageUtil.to_pil` conversion of the colour-corrected image.
    public let pixels: MLXArray
    public let width: Int, height: Int
    /// The preprocessed input `[1, 3, H16, W16]` (padded to multiples of 16).
    public let input: MLXArray
    /// The VAE latent of the input `[1, 16, 1, h, w]`, the seeded noise and the denoised latent (same shape).
    public let encoded: MLXArray, noise: MLXArray, latents: MLXArray
    /// The decoded image before colour correction, cropped to the output size, `[1, 3, height, width]`.
    public let decoded: MLXArray

    public func writePNG(to url: URL) throws { try ImageOutput.writePNG(pixels: pixels, to: url) }
}

/// SeedVR2 3B super-resolution, ported from MFLUX 0.20 (`mflux.models.seedvr2.SeedVR2` with `quantize=4`): one diffusion step of the
/// NaDiT transformer conditioned on the VAE latent of the (bicubic-resized) input and a fixed positive text embedding, tiled VAE
/// encode and decode, and MFLUX's colour correction. Loading reads the fp16 files from `numz/SeedVR2_comfyUI` and quantizes the
/// linear layers to 4 bits.
///
/// Set `MLX.Memory.cacheLimit = 0` in the process before loading: freed buffers then go back to the system at once, which keeps the
/// peak during loading at the quantized model plus one fp16 matrix, and the peak while upscaling near the live arrays.
public final class SeedVR2Upscaler {
    public let weights: SeedVR2Weights
    public let transformer: SeedVR2Transformer
    public let vae: SeedVR2VAE
    /// The bundled positive text embedding `[1, 58, 5120]` fp16.
    public let positiveEmbedding: MLXArray

    /// Validates both files (sizes, tensor names, dtypes, shapes), loads them and quantizes the linear layers.
    public init(modelDirectory: URL) throws {
        positiveEmbedding = try Self.loadPositiveEmbedding()
        weights = try SeedVR2Weights.load(modelDirectory: modelDirectory)
        transformer = SeedVR2Transformer(weights: weights)
        vae = SeedVR2VAE(weights: weights)
    }

    /// `SeedVR2TextEmbeddings.load_positive()`: the bundled `pos_emb.safetensors` (`embedding`, `[58, 5120]` fp16) with a batch axis.
    public static func loadPositiveEmbedding() throws -> MLXArray {
        guard let url = Bundle.module.url(forResource: "pos_emb", withExtension: "safetensors") else {
            throw SeedVR2Error.invalidWeights("the bundled pos_emb.safetensors is missing")
        }
        guard let embedding = try loadArrays(url: url)["embedding"], embedding.shape == [58, SeedVR2Layout.txtInDim], embedding.dtype == .float16
        else { throw SeedVR2Error.invalidWeights("the bundled pos_emb.safetensors does not hold a [58, 5120] fp16 embedding") }
        eval(embedding)
        return embedding.expandedDimensions(axis: 0)
    }

    /// `SeedVR2LatentCreator.create_noise_latents`: `mx.random.normal((1, 16, 1, h, w), key=mx.random.key(seed))`, float32, with the
    /// values of the MLX Python wheel (see `SeedVR2Noise`).
    public static func noise(seed: UInt64, height: Int, width: Int) -> MLXArray {
        SeedVR2Noise.normal([1, SeedVR2Layout.latentChannels, 1, height, width], seed: seed)
    }

    /// `SeedVR2LatentCreator.create_condition`: the latent and a channel of ones.
    public static func condition(encoded: MLXArray) -> MLXArray {
        let latent = encoded.ndim == 4 ? encoded.expandedDimensions(axis: 2) : encoded
        let mask = MLXArray.ones([1, 1, 1, latent.dim(3), latent.dim(4)], dtype: .float32)
        return concatenated([latent, mask], axis: 1)
    }

    /// The `seedvr2_euler` schedule for one step: timesteps `[1000, 0]`.
    public static let timesteps: [Float] = [1000, 0]

    /// `SeedVR2EulerScheduler.step` at step 0 (`t = 1000`, `s = 0`): `x0 = latents - (t / 1000) * prediction`.
    public static func step(prediction: MLXArray, latents: MLXArray) -> MLXArray {
        let t = MLXArray(timesteps[0]) / Float(1000.0)
        return latents - t * prediction
    }

    /// The transformer's prediction for the noise and the condition (`mx.concatenate([latents, condition], axis=1)`).
    public func predict(noise: MLXArray, condition: MLXArray) -> MLXArray {
        transformer(vid: concatenated([noise, condition], axis: 1), txt: positiveEmbedding, timestep: MLXArray(Self.timesteps[0]))
    }

    /// `SeedVR2.generate_image`: preprocess, encode (tiled), one transformer step from the seeded noise, decode (tiled), crop and
    /// colour-correct against the preprocessed input.
    public func upscale(_ request: SeedVR2Request) throws -> SeedVR2Output {
        try request.validate()
        let pre = SeedVR2Preprocessing.run(request.image, resolution: request.resolution, softness: request.softness)
        let encoded = SeedVR2VAETiler.encode(pre.image, vae: vae)
        let condition = Self.condition(encoded: encoded)
        let noise = Self.noise(seed: request.seed, height: encoded.dim(-2), width: encoded.dim(-1))
        let latents = Self.step(prediction: predict(noise: noise, condition: condition), latents: noise)
        eval(latents)
        let decodedFull = SeedVR2VAETiler.decode(latents, vae: vae)
        let decoded = decodedFull[0..., 0..., 0 ..< pre.height, 0 ..< pre.width]
        let style = pre.image[0..., 0..., 0 ..< pre.height, 0 ..< pre.width]
        let corrected = SeedVR2ColorCorrection.apply(content: decoded, style: style)
        let pixels = ImageOutput.pixels(decoded: corrected)
        eval(pixels)
        return SeedVR2Output(
            pixels: pixels, width: pre.width, height: pre.height, input: pre.image, encoded: encoded, noise: noise, latents: latents,
            decoded: decoded)
    }
}
