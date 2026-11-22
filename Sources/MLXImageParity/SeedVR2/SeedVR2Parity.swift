import Foundation
import MLX
import MLXImageCore
import SeedVR2

// SeedVR2 3B against MFLUX 0.20 (`SeedVR2(quantize=4)` on the same fp16 files). The exporter (Python, outside the package) writes the
// input images, MFLUX's preprocessed tensors and seeded noise, and every stage of complete upscales into the reference directory;
// nothing generated is committed. Preprocessing (Pillow's bicubic resize, softness, padding, normalization), the seeded noise and the
// scheduler step are compared bit for bit. The VAE encode and decode and the one transformer step each start from MFLUX's own input to
// that stage and are judged by the harness's existing relative-L2 limits (VAE output 2e-3, full transformer output 2e-3). The colour
// correction (numpy on the CPU in MFLUX) is compared from MFLUX's decoded image within the harness's pixel limits. The complete native
// upscale is judged against MFLUX's own drift, as the Z-Image Base and Klein pixel rows are: the exporter also runs the job with the
// transformer under `mx.compile`, whose prediction differs from MFLUX's eager one by about as much as the native one does (the MLX
// kernel builds of the Python wheel and the Swift package round a few operations differently, see `SeedVR2Noise` and the README).

/// `reference/seedvr2_cases.json`, shared with the Python exporter.
struct SeedVR2Cases: Decodable {
    struct Input: Decodable { let name: String }
    struct Preprocess: Decodable {
        let id: String, input: String, resolution: SeedVR2Resolution, softness: Double
        enum CodingKeys: String, CodingKey { case id, input, resolution, softness }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            id = try c.decode(String.self, forKey: .id)
            input = try c.decode(String.self, forKey: .input)
            softness = try c.decode(Double.self, forKey: .softness)
            resolution = try SeedVR2Cases.resolution(c, .resolution)
        }
    }
    struct Noise: Decodable { let id: String; let seed: UInt64; let height: Int; let width: Int }
    struct Pixel: Decodable {
        let id: String, input: String, resolution: SeedVR2Resolution, softness: Double, seed: UInt64
        enum CodingKeys: String, CodingKey { case id, input, resolution, softness, seed }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            id = try c.decode(String.self, forKey: .id)
            input = try c.decode(String.self, forKey: .input)
            softness = try c.decode(Double.self, forKey: .softness)
            seed = try c.decode(UInt64.self, forKey: .seed)
            resolution = try SeedVR2Cases.resolution(c, .resolution)
        }
    }
    let inputs: [Input]
    let preprocessCases: [Preprocess]
    let noiseCases: [Noise]
    let pixelCases: [Pixel]
    /// The exporter's manifest lists the preprocess, noise and pixel cases in that order.
    var allIDs: [String] { preprocessCases.map(\.id) + noiseCases.map(\.id) + pixelCases.map(\.id) }
    enum CodingKeys: String, CodingKey {
        case inputs, preprocessCases = "preprocess_cases", noiseCases = "noise_cases", pixelCases = "pixel_cases"
    }

    /// An integer is the shortest side; a string `"<factor>x"` is MFLUX's `ScaleFactor`.
    static func resolution<K: CodingKey>(_ c: KeyedDecodingContainer<K>, _ key: K) throws -> SeedVR2Resolution {
        if let side = try? c.decode(Int.self, forKey: key) { return .shortestSide(side) }
        let text = try c.decode(String.self, forKey: key)
        guard text.lowercased().hasSuffix("x"), let factor = Double(text.dropLast()) else {
            throw DecodingError.dataCorruptedError(forKey: key, in: c, debugDescription: "resolution \(text) is neither pixels nor '<factor>x'")
        }
        return .scale(factor)
    }

    static func load(_ url: URL) throws -> SeedVR2Cases { try JSONDecoder().decode(SeedVR2Cases.self, from: Data(contentsOf: url)) }
}

public extension ComponentParity {
    /// `SeedVR2Util.preprocess_image` from the exported input PNGs, bit for bit: the `[1, 3, H16, W16]` tensor and the output size.
    static func seedVR2Preprocess(casesJSON: URL, referenceDirectory: URL, log: (String) -> Void = { _ in }) throws -> ComponentReport {
        let cases = try SeedVR2Cases.load(casesJSON)
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "seedvr2")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.allIDs, what: "seedvr2")
        var results: [ComponentCaseResult] = []
        for c in cases.preprocessCases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let input = try ImageOutput.readPNG(at: referenceDirectory.appendingPathComponent(c.input))
            let native = SeedVR2Preprocessing.run(input, resolution: c.resolution, softness: c.softness)
            let checks = [
                ComponentCheck.exact("preprocessed image", native: native.image, reference: try require(ref, "image", c.id)),
                ComponentCheck.exact("output size (h, w)", native: MLXArray([Int32(native.height), Int32(native.width)]), reference: try require(ref, "true_hw", c.id)),
            ]
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentCheck.report(component: "seedvr2-preprocess", tolerances: [:], mlxSwiftRevision: "", manifest: manifest, cases: results)
    }

    /// `SeedVR2LatentCreator.create_noise_latents`, bit for bit.
    static func seedVR2Noise(casesJSON: URL, referenceDirectory: URL, log: (String) -> Void = { _ in }) throws -> ComponentReport {
        let cases = try SeedVR2Cases.load(casesJSON)
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "seedvr2")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.allIDs, what: "seedvr2")
        var results: [ComponentCaseResult] = []
        for c in cases.noiseCases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let native = SeedVR2Upscaler.noise(seed: c.seed, height: c.height, width: c.width)
            let result = ComponentCaseResult(id: c.id, checks: [ComponentCheck.exact("noise", native: native, reference: try require(ref, "noise", c.id))])
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentCheck.report(component: "seedvr2-noise", tolerances: [:], mlxSwiftRevision: "", manifest: manifest, cases: results)
    }

    /// Complete upscales, stage by stage and end to end. Decisive: the preprocessed input, the output size, the seeded noise and the
    /// scheduler step from MFLUX's own prediction (bitwise); the tiled VAE encode of MFLUX's input and the tiled decode of MFLUX's
    /// latents (relative L2 <= 2e-3, the VAE output limit); the transformer's prediction from MFLUX's noise and condition (relative L2
    /// <= 2e-3, the full-output limit); the colour correction of MFLUX's decoded image within the harness's pixel limits (2 / 0.1
    /// levels); and the complete native upscale, whose mean abs pixel difference from MFLUX's must be at most `pixelDriftFactor` (2)
    /// times the mean abs difference between MFLUX's eager and compiled runs of the same job. SeedVR2 maps its single transformer step
    /// straight to pixels, so the 1e-3-level prediction differences of either kind move a few pixels by more than 2 levels; the
    /// harness's 2 / 0.1 level limits and the maximum differences are recorded as diagnostics.
    static func seedVR2Upscales(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, upscaler: SeedVR2Upscaler? = nil, mlxSwiftRevision: String = "",
        log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try SeedVR2Cases.load(casesJSON)
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "seedvr2")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.allIDs, what: "seedvr2")
        let upscaler = try upscaler ?? SeedVR2Upscaler(modelDirectory: modelDirectory)
        var results: [ComponentCaseResult] = []
        for c in cases.pixelCases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let (refInput, refEncoded, refNoise) = (try require(ref, "input", c.id), try require(ref, "encoded", c.id), try require(ref, "noise", c.id))
            let (refPrediction, refLatents, refDecoded) = (try require(ref, "prediction", c.id), try require(ref, "latents", c.id), try require(ref, "decoded", c.id))
            let refPixels = try require(ref, "pixels", c.id)
            let image = try ImageOutput.readPNG(at: referenceDirectory.appendingPathComponent(c.input))
            var checks: [CheckResult] = []

            // Each stage from MFLUX's own input to it.
            let pre = SeedVR2Preprocessing.run(image, resolution: c.resolution, softness: c.softness)
            checks.append(ComponentCheck.exact("preprocessed input", native: pre.image, reference: refInput))
            checks.append(ComponentCheck.exact(
                "output size (h, w)", native: MLXArray([Int32(pre.height), Int32(pre.width)]), reference: try require(ref, "true_hw", c.id)))
            checks.append(ComponentCheck.exact(
                "seeded noise", native: SeedVR2Upscaler.noise(seed: c.seed, height: refEncoded.dim(-2), width: refEncoded.dim(-1)), reference: refNoise))
            let encoded = SeedVR2VAETiler.encode(refInput, vae: upscaler.vae)
            eval(encoded)
            checks.append(ComponentCheck.tolerance("VAE encode (tiled) of MFLUX's input", native: encoded, reference: refEncoded, limits: .vaeOutput))
            let prediction = upscaler.predict(noise: refNoise, condition: SeedVR2Upscaler.condition(encoded: refEncoded))
            eval(prediction)
            checks.append(ComponentCheck.tolerance(
                "transformer step from MFLUX's noise and condition", native: prediction, reference: refPrediction, limits: .fullTransformerOutput))
            let step = SeedVR2Upscaler.step(prediction: refPrediction, latents: refNoise)
            checks.append(ComponentCheck.exact("scheduler step of MFLUX's prediction", native: step, reference: refLatents))
            let (height, width) = (refDecoded.dim(2), refDecoded.dim(3))
            let decoded = SeedVR2VAETiler.decode(refLatents, vae: upscaler.vae)[0..., 0..., 0 ..< height, 0 ..< width]
            eval(decoded)
            checks.append(ComponentCheck.tolerance("VAE decode (tiled) of MFLUX's latents", native: decoded, reference: refDecoded, limits: .vaeOutput))
            let corrected = SeedVR2ColorCorrection.apply(content: refDecoded, style: refInput[0..., 0..., 0 ..< height, 0 ..< width])
            checks.append(ComponentCheck.tolerance(
                "colour correction of MFLUX's decoded image (diagnostic)", native: corrected, reference: try require(ref, "corrected", c.id),
                limits: .vaeOutput, decisive: false))
            let correctedPixels = ImageOutput.pixels(decoded: corrected)
            checks.append(ComponentCheck.pixels(
                "colour correction of MFLUX's decoded image, in pixels", native: correctedPixels.asArray(UInt8.self),
                reference: refPixels.asArray(UInt8.self), nativeShape: correctedPixels.shape, referenceShape: refPixels.shape))

            // MFLUX's own drift: the same job with the transformer under `mx.compile`.
            let refCompiledPixels = try require(ref, "pixels_compiled", c.id)
            checks.append(ComponentCheck.tolerance(
                "transformer step: MFLUX compiled vs MFLUX eager (reference drift)", native: try require(ref, "prediction_compiled", c.id),
                reference: refPrediction, limits: .fullTransformerOutput, decisive: false))

            // The complete native upscale.
            let output = try upscaler.upscale(SeedVR2Request(image: image, resolution: c.resolution, softness: c.softness, seed: c.seed))
            checks.append(ComponentCheck.tolerance(
                "native free-running latents vs MFLUX (diagnostic)", native: output.latents, reference: refLatents,
                limits: .fullTransformerOutput, decisive: false))
            let reference = refPixels.asArray(UInt8.self)
            let strict = ComponentCheck.pixels(
                "pixels: native upscale vs MFLUX, harness limits 2 / 0.1 (diagnostic)", native: output.pixels.asArray(UInt8.self),
                reference: reference, nativeShape: output.pixels.shape, referenceShape: refPixels.shape, decisive: false)
            let drift = ComponentCheck.pixels(
                "pixels: MFLUX compiled vs MFLUX eager (reference drift)", native: refCompiledPixels.asArray(UInt8.self), reference: reference,
                nativeShape: refCompiledPixels.shape, referenceShape: refPixels.shape, decisive: false)
            var envelope = CheckResult(
                name: "pixels: native upscale vs MFLUX, mean abs within \(pixelDriftFactor)x MFLUX's own compiled-vs-eager drift", kind: "pixels",
                decisive: true, passed: false, metrics: nil, pixels: strict.pixels, detail: nil)
            if let measured = strict.pixels, let reference = drift.pixels, output.pixels.shape == refPixels.shape {
                envelope.passed = measured.meanAbsDiff <= pixelDriftFactor * reference.meanAbsDiff
                envelope.detail = String(
                    format: "native mean abs %.3f, max abs %d; MFLUX drift mean abs %.3f, max abs %d", measured.meanAbsDiff, measured.maxAbsDiff,
                    reference.meanAbsDiff, reference.maxAbsDiff)
            } else {
                envelope.detail = "shape mismatch: native \(output.pixels.shape), reference \(refPixels.shape)"
            }
            checks += [envelope, strict, drift]
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(logLine(result, pixels: true))
            for check in checks {
                let metric = check.metrics.map { m in m.relativeL2.map { String(format: "relL2 %.3e", $0) } ?? "zero-ref" }
                    ?? check.pixels.map { String(format: "max %d mean %.3f", $0.maxAbsDiff, $0.meanAbsDiff) } ?? ""
                log("  \(check.passed ? "ok  " : "FAIL") \(check.name): \(metric) \(check.detail ?? "")")
            }
            results.append(result)
        }
        return ComponentCheck.report(
            component: "seedvr2-upscales",
            tolerances: [
                "vae_output_relative_l2": ParityTolerance.vaeOutputRelativeL2,
                "full_transformer_output_relative_l2": ParityTolerance.fullTransformerOutputRelativeL2,
                "pixel_drift_factor": pixelDriftFactor,
            ],
            mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }
}
