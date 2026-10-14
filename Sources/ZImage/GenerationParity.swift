import Foundation
import MLX

// Slice 4 checks: per-step latents of the real denoising loop, the VAE weights, isolated VAE decoding from reference
// latents, and the complete pipeline's PNG. Same machinery as slice 2 and 3: each reads only its own reference directory.
//
// Owner decision (slice 4, resolving the slice 3 finding): wherever a prediction is involved, the decisive comparison is
// native compiled against MFLUX's production compiled path (`ZImage._predict`, `mx.compile(predict)`). The eager comparisons
// are retained and recorded but never decide. The VAE has no compiled production path, so its comparison is production
// against production (eager against eager).

/// The explicit case lists of `reference/generation_cases.json`, shared with the Python exporters.
struct GenerationCases: Decodable {
    struct Case: Decodable {
        let id: String
        let seed: UInt64
        let width: Int
        let height: Int
        let steps: Int
        let prompt: String
        let latentSource: String?
        enum CodingKeys: String, CodingKey { case id, seed, width, height, steps, prompt, latentSource = "latent_source" }
    }
    let denoiseCases: [Case]
    let vaeCases: [Case]
    let pipelineCases: [Case]
    enum CodingKeys: String, CodingKey { case denoiseCases = "denoise_cases", vaeCases = "vae_cases", pipelineCases = "pipeline_cases" }

    static func load(_ url: URL) throws -> GenerationCases { try JSONDecoder().decode(GenerationCases.self, from: Data(contentsOf: url)) }
}

public extension ComponentParity {
    // MARK: Per-step latents of the real denoising loop

    /// The free-running denoising loop (seeded noise, the real transformer, the linear scheduler) against MFLUX's real
    /// `ZImage.generate_image` loop, checkpointed after every step, from the reference's own caption embedding.
    ///
    /// Decisive, every step: native compiled prediction (as the production path) vs MFLUX production compiled, relative L2
    /// <= 5e-3 (the predeclared free-running limit). Retained, never deciding: native eager vs MFLUX eager, native eager vs
    /// MFLUX compiled, MFLUX eager vs MFLUX compiled, and each step taken alone from the reference's state (isolated), which
    /// localizes a divergence.
    static func denoise(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try GenerationCases.load(casesJSON).denoiseCases
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "denoise")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.map(\.id), what: "denoise")
        let denoiser = ZImageDenoiser(transformer: ZImageTransformer(weights: try TransformerWeights.load(modelDirectory: modelDirectory)))
        let limits = ParityLimits.schedulerLatentPerStep
        var results: [ComponentCaseResult] = []
        for c in cases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let cap = try require(ref, "cap_feats", c.id)
            let request = GenerationRequest(prompt: c.prompt, seed: c.seed, width: c.width, height: c.height, steps: c.steps)
            var checks = [ComponentCheck.exact(
                "seeded noise (latents.0)", native: ZImageLatentCreator.createNoise(seed: c.seed, height: c.height, width: c.width),
                reference: try require(ref, "latents.0", c.id))]
            var compiled: [MLXArray] = []
            var eager: [MLXArray] = []
            _ = try denoiser.denoise(request: request, captionFeatures: cap, mode: .compiled) { _, latents in compiled.append(latents) }
            _ = try denoiser.denoise(request: request, captionFeatures: cap, mode: .eager) { _, latents in eager.append(latents) }
            let scheduler = LinearScheduler(steps: c.steps, width: c.width, height: c.height)
            for t in 0 ..< c.steps {
                let (mfluxCompiled, mfluxEager) = try (require(ref, "compiled.latents.\(t + 1)", c.id), require(ref, "eager.latents.\(t + 1)", c.id))
                checks.append(ComponentCheck.tolerance("step \(t): native compiled vs MFLUX compiled (production path)", native: compiled[t], reference: mfluxCompiled, limits: limits))
                checks.append(ComponentCheck.tolerance("step \(t): native eager vs MFLUX eager (retained)", native: eager[t], reference: mfluxEager, limits: limits, decisive: false))
                checks.append(ComponentCheck.tolerance("step \(t): native eager vs MFLUX compiled", native: eager[t], reference: mfluxCompiled, limits: limits, decisive: false))
                checks.append(ComponentCheck.tolerance("step \(t): MFLUX eager vs MFLUX compiled (reference drift)", native: mfluxEager, reference: mfluxCompiled, limits: limits, decisive: false))
                // Isolated: one native compiled step from the reference's compiled state before it.
                let before = try require(ref, t == 0 ? "latents.0" : "compiled.latents.\(t)", c.id)
                let noise = denoiser.predict(
                    latents: before, timestep: ZImageDenoiser.timestep(scheduler, step: t), captionFeatures: cap, mode: .compiled)
                let isolated = scheduler.step(noise: noise, timestep: t, latents: before)
                eval(isolated)
                checks.append(ComponentCheck.tolerance("step \(t): isolated native compiled step vs MFLUX compiled", native: isolated, reference: mfluxCompiled, limits: limits, decisive: false))
            }
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentCheck.report(
            component: "denoise", tolerances: ["latent_per_step_relative_l2": ParityTolerance.schedulerLatentPerStepRelativeL2],
            mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    // MARK: q4 VAE weights

    /// The loaded q4 VAE against what mflux holds after applying the same snapshot: tensor names, dtypes, shapes, the SHA-256 of
    /// every tensor's bytes, the quantization of every attention linear, and the stored quantization metadata.
    static func vaeWeights(
        modelDirectory: URL, referenceDirectory: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "vae-weights")
        let reference = try JSONDecoder().decode(WeightsManifest.self, from: Data(contentsOf: referenceDirectory.appendingPathComponent("manifest.json")))
        let loaded = try VAEWeights.load(modelDirectory: modelDirectory)
        let native = loaded.tensors
        var checks: [CheckResult] = []

        func check(_ name: String, _ problems: [String]) {
            checks.append(CheckResult(
                name: name, kind: "exact", decisive: true, passed: problems.isEmpty, metrics: nil,
                detail: problems.isEmpty ? nil : "\(problems.count) problems; first: \(problems.sorted().prefix(3).joined(separator: "; "))"))
            log("\(name): \(problems.isEmpty ? "equal" : "\(problems.count) DIFFERENCES")")
        }

        let missing = Set(reference.tensors.keys).subtracting(native.keys).map { "missing \($0)" }
        let extra = Set(native.keys).subtracting(reference.tensors.keys).map { "unexpected \($0)" }
        check("tensor names (\(reference.tensors.count) in the reference)", missing + extra)
        check("tensor dtypes and shapes", reference.tensors.compactMap { name, ref in
            guard let t = native[name] else { return nil }
            let dtype = ParityRunner.dtypeName(t.dtype)
            return dtype == ref.dtype && t.shape == ref.shape ? nil : "\(name): native \(dtype) \(t.shape), reference \(ref.dtype) \(ref.shape)"
        })
        check("tensor byte digests", reference.tensors.keys.sorted().compactMap { name in
            guard let t = native[name] else { return nil }
            return TransformerWeights.digest(t) == reference.tensors[name]!.sha256 ? nil : "\(name): SHA-256 differs"
        })

        var quantProblems: [String] = []
        if Set(reference.quantizedLinears.map(\.name)) != Set(VAELayout.quantizedLinears) { quantProblems.append("quantized linear sets differ") }
        for q in reference.quantizedLinears {
            guard let w = native[q.name + ".weight"], let s = native[q.name + ".scales"] else { quantProblems.append("\(q.name): not loaded"); continue }
            let (bits, group) = (w.dim(-1) * 32 / VAELayout.midChannels, VAELayout.midChannels / s.dim(-1))
            if bits != q.bits || group != q.groupSize || q.mode != QuantizationMode.affine.rawValue {
                quantProblems.append("\(q.name): native bits \(bits) group \(group) mode affine, reference bits \(q.bits) group \(q.groupSize) mode \(q.mode)")
            }
        }
        if !reference.unquantizedLinears.isEmpty { quantProblems.append("reference holds unquantized linears: \(reference.unquantizedLinears)") }
        check("quantization of \(reference.quantizedLinears.count) linears (bits, group size, mode)", quantProblems)

        var metadataProblems: [String] = []
        if loaded.inspection.quantizationLevel != "\(reference.storedQuantizationLevel)" { metadataProblems.append("quantization_level") }
        if loaded.inspection.mfluxVersion != reference.storedMfluxVersion { metadataProblems.append("mflux_version") }
        check("stored quantization metadata", metadataProblems)

        return ComponentCheck.report(
            component: "vae-weights", tolerances: ["names_dtypes_shapes_digests_quantization": 0], mlxSwiftRevision: mlxSwiftRevision,
            manifest: manifest, cases: [ComponentCaseResult(id: "q4-vae", checks: checks)])
    }

    // MARK: Isolated VAE decoding

    /// Decodes a known latent through the native VAE and compares with MFLUX's decoded tensor and pixels for the same latent,
    /// independent of the denoising loop (the latent is the reference's: a real MFLUX loop result, or seeded noise).
    /// The VAE is eager in production, so this is production against production. Decisive: the decoded tensor (relative L2
    /// <= 2e-3), the decoded pixels (max abs <= 2 and mean abs <= 0.1 levels), and the native image conversion applied to
    /// MFLUX's own decoded tensor (same pixel limits).
    static func vae(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try GenerationCases.load(casesJSON).vaeCases
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "vae")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.map(\.id), what: "vae")
        let vae = ZImageVAE(weights: try VAEWeights.load(modelDirectory: modelDirectory))
        var results: [ComponentCaseResult] = []
        for c in cases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let latents = try require(ref, "latents", c.id)
            let (refDecoded, refPixels) = try (require(ref, "decoded", c.id), require(ref, "pixels", c.id))
            let decoded = vae.decode(samplerLatents: latents)
            let pixels = ZImageImage.pixels(decoded: decoded)
            let pixelsFromReferenceDecoded = ZImageImage.pixels(decoded: refDecoded)
            eval(decoded, pixels, pixelsFromReferenceDecoded)
            let (nativePixels, referencePixels) = (pixels.asArray(UInt8.self), refPixels.asArray(UInt8.self))
            let checks = [
                ComponentCheck.tolerance("VAE output tensor", native: decoded, reference: refDecoded, limits: .vaeOutput),
                ComponentCheck.pixels("decoded pixels", native: nativePixels, reference: referencePixels, nativeShape: pixels.shape, referenceShape: refPixels.shape),
                ComponentCheck.pixels(
                    "image conversion of MFLUX's decoded tensor", native: pixelsFromReferenceDecoded.asArray(UInt8.self), reference: referencePixels,
                    nativeShape: pixelsFromReferenceDecoded.shape, referenceShape: refPixels.shape),
            ]
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(logLine(result, pixels: true))
            results.append(result)
        }
        return ComponentCheck.report(
            component: "vae",
            tolerances: [
                "vae_output_relative_l2": ParityTolerance.vaeOutputRelativeL2, "pixel_max_abs_levels": ParityTolerance.pixelMaxAbsLevels,
                "pixel_mean_abs_levels": ParityTolerance.pixelMeanAbsLevels,
            ], mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    // MARK: Complete pipeline

    /// The whole native pipeline (tokenizer, text encoder, seeded latents, denoising, VAE decode, PNG) from the prompt text
    /// against MFLUX's real `ZImage.generate_image`. The PNG is written by the native encoder, read back, and its decoded
    /// pixels are compared with MFLUX's pixel array. Decisive: native compiled pipeline vs MFLUX production (compiled)
    /// pipeline, max abs <= 2 and mean abs <= 0.1 levels. Retained, never deciding: the eager pipelines against each other
    /// and against the compiled reference, the final latents, and PNG structure and bytes (encoding differences are
    /// reported separately from pixels, as the plan prescribes).
    static func pipeline(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, outputDirectory: URL, mlxSwiftRevision: String,
        log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try GenerationCases.load(casesJSON).pipelineCases
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "pipeline")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.map(\.id), what: "pipeline")
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        let promptsByID = try pipelinePrompts(referenceDirectory: referenceDirectory)
        let generator = try ZImageGenerator(modelDirectory: modelDirectory)
        var results: [ComponentCaseResult] = []
        for c in cases {
            guard let prompt = promptsByID[c.id] else { throw ComponentParityError("pipeline: reference manifest has no prompt for \(c.id)") }
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let (refCompiled, refEager) = try (require(ref, "pixels_compiled", c.id), require(ref, "pixels_eager", c.id))
            let request = GenerationRequest(prompt: prompt, seed: c.seed, width: c.width, height: c.height, steps: c.steps)

            let compiled = try generator.generate(request, mode: .compiled)
            let pngURL = outputDirectory.appendingPathComponent("\(c.id).png")
            try ZImageImage.writePNG(pixels: compiled.pixels, to: pngURL)
            let decodedPNG = try ZImageImage.readPNG(at: pngURL)
            let eager = try generator.generate(request, mode: .eager)

            let shape = [c.height, c.width, 3]
            let refBytes = (compiled: refCompiled.asArray(UInt8.self), eager: refEager.asArray(UInt8.self))
            var checks = [
                ComponentCheck.pixels(
                    "PNG pixels: native compiled pipeline vs MFLUX production pipeline", native: decodedPNG.bytes, reference: refBytes.compiled,
                    nativeShape: [decodedPNG.height, decodedPNG.width, 3], referenceShape: refCompiled.shape),
                ComponentCheck.pixels(
                    "PNG round trip: written PNG vs native pixels", native: decodedPNG.bytes, reference: compiled.pixels.asArray(UInt8.self),
                    nativeShape: [decodedPNG.height, decodedPNG.width, 3], referenceShape: shape),
                ComponentCheck.pixels(
                    "pixels: native eager pipeline vs MFLUX eager pipeline (retained)", native: eager.pixels.asArray(UInt8.self),
                    reference: refBytes.eager, nativeShape: eager.pixels.shape, referenceShape: refEager.shape, decisive: false),
                ComponentCheck.pixels(
                    "pixels: native eager pipeline vs MFLUX production pipeline", native: eager.pixels.asArray(UInt8.self),
                    reference: refBytes.compiled, nativeShape: eager.pixels.shape, referenceShape: refCompiled.shape, decisive: false),
                ComponentCheck.pixels(
                    "pixels: MFLUX eager vs MFLUX production pipeline (reference drift)", native: refBytes.eager, reference: refBytes.compiled,
                    nativeShape: refEager.shape, referenceShape: refCompiled.shape, decisive: false),
            ]
            if let refLatents = ref["latents_compiled"] {
                checks.append(ComponentCheck.tolerance(
                    "final latents: native compiled vs MFLUX production", native: compiled.latents, reference: refLatents, limits: .schedulerLatentPerStep, decisive: false))
            }
            checks.append(pngStructure(native: pngURL, reference: referenceDirectory.appendingPathComponent("\(c.id).png"), width: c.width, height: c.height))
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(logLine(result, pixels: true))
            results.append(result)
        }
        return ComponentCheck.report(
            component: "pipeline",
            tolerances: ["pixel_max_abs_levels": ParityTolerance.pixelMaxAbsLevels, "pixel_mean_abs_levels": ParityTolerance.pixelMeanAbsLevels],
            mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    /// Prompt text per case ID, as the reference exporter expanded and recorded it (the single source both sides read).
    private static func pipelinePrompts(referenceDirectory: URL) throws -> [String: String] {
        struct Manifest: Decodable { struct Case: Decodable { let id: String; let promptText: String
            enum CodingKeys: String, CodingKey { case id, promptText = "prompt_text" } }
            let cases: [Case] }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: referenceDirectory.appendingPathComponent("manifest.json")))
        return Dictionary(uniqueKeysWithValues: manifest.cases.map { ($0.id, $0.promptText) })
    }

    /// PNG structure of the native file against MFLUX's (PIL) file: dimensions, bit depth, colour type; byte equality as a bonus.
    /// Diagnostic only: container differences are encoding differences, not pixel differences.
    private static func pngStructure(native: URL, reference: URL, width: Int, height: Int) -> CheckResult {
        var result = CheckResult(name: "PNG structure vs MFLUX's PNG (encoding, never decisive)", kind: "diagnostic", decisive: false, passed: true, metrics: nil, detail: nil)
        do {
            let (n, r) = try (ZImageImage.pngInfo(at: native), ZImageImage.pngInfo(at: reference))
            let same = n.width == r.width && n.height == r.height && n.bitDepth == r.bitDepth && n.colorType == r.colorType
            result.detail = "native \(n.width)x\(n.height) depth \(n.bitDepth) colour type \(n.colorType), \(n.bytes) bytes; "
                + "MFLUX \(r.width)x\(r.height) depth \(r.bitDepth) colour type \(r.colorType), \(r.bytes) bytes; "
                + "header \(same && n.width == width && n.height == height ? "equal" : "DIFFERS"); bytes \(n.sha256 == r.sha256 ? "identical" : "differ")"
        } catch {
            result.detail = "\(error)"
        }
        return result
    }

    private static func logLine(_ c: ComponentCaseResult, pixels: Bool) -> String {
        let decisive = c.checks.filter(\.decisive)
        let maxAbs = decisive.compactMap { $0.pixels?.maxAbsDiff }.max()
        let worst = decisive.compactMap { $0.metrics?.relativeL2 }.max()
        return "\(c.id): \(decisive.count) checks, \(decisive.filter { !$0.passed }.count) failed"
            + (worst.map { String(format: ", max relL2=%.3e", $0) } ?? "") + (maxAbs.map { ", pixel max abs \($0)" } ?? "") + " \(c.passed ? "PASS" : "FAIL")"
    }
}
