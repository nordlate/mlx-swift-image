import ZImage
import Foundation
import MLX
import MLXImageCore

// Z-Image Base against MFLUX 0.20 (`ModelConfig.z_image()`, `flow_match_euler_discrete`, guided `ZImage._predict`): scheduler
// sigmas bitwise, guided per-step latents of the real loop, and complete pixel jobs with and without a negative prompt. The
// limits for tensors are the harness's existing ones. Two rows differ from Turbo's: the guided prediction itself is a diagnostic (guidance
// amplifies the 1e-3 difference), and the guided pixel row is relative to MFLUX's own eager-vs-compiled drift (`pixelDriftFactor`), with the
// maximum difference recorded but not judged. The exporter (Python, outside the package) writes everything, sigmas included, to the reference
// directory; nothing generated is committed.

/// `reference/zimage_base_cases.json`, shared with the Python exporter.
struct BaseCases: Decodable {
    struct Sigma: Decodable { let id: String; let steps: Int; let width: Int; let height: Int }
    struct Job: Decodable {
        let id: String
        let seed: UInt64
        let width: Int
        let height: Int
        let steps: Int
        let guidance: Float
        let prompt: String
        let negativePrompt: String?
        enum CodingKeys: String, CodingKey { case id, seed, width, height, steps, guidance, prompt, negativePrompt = "negative_prompt" }

        var request: GenerationRequest {
            GenerationRequest(prompt: prompt, seed: seed, width: width, height: height, steps: steps, guidance: guidance, negativePrompt: negativePrompt)
        }
    }
    let sigmaCases: [Sigma]
    let stepCases: [Job]
    let pixelCases: [Job]
    /// The exporter's manifest lists the step cases, then the pixel cases.
    var allJobIDs: [String] { (stepCases + pixelCases).map(\.id) }
    enum CodingKeys: String, CodingKey { case sigmaCases = "sigma_cases", stepCases = "step_cases", pixelCases = "pixel_cases" }

    static func load(_ url: URL) throws -> BaseCases { try JSONDecoder().decode(BaseCases.self, from: Data(contentsOf: url)) }
}

/// `zimage_base_sigmas.json` in the reference export: MFLUX's sigmas as float32 bit patterns.
struct BaseSigmaFixture: Decodable {
    struct Case: Decodable {
        let id: String
        let steps: Int
        let width: Int
        let height: Int
        let sigmaBits: [UInt32]
        enum CodingKeys: String, CodingKey { case id, steps, width, height, sigmaBits = "sigma_bits" }
    }
    let cases: [Case]
    static func load(_ url: URL) throws -> BaseSigmaFixture { try JSONDecoder().decode(BaseSigmaFixture.self, from: Data(contentsOf: url)) }
}

public extension ComponentParity {
    /// The native sigmas against MFLUX's, bitwise, for every declared steps and size.
    static func baseSigmas(casesJSON: URL, referenceDirectory: URL, log: (String) -> Void = { _ in }) throws -> ComponentReport {
        let declared = try BaseCases.load(casesJSON).sigmaCases
        let fixture = try BaseSigmaFixture.load(referenceDirectory.appendingPathComponent("zimage_base_sigmas.json"))
        guard fixture.cases.map(\.id) == declared.map(\.id) else {
            throw ComponentParityError("base sigmas: fixture case list differs from zimage_base_cases.json")
        }
        var results: [ComponentCaseResult] = []
        for (c, ref) in zip(declared, fixture.cases) {
            guard (c.steps, c.width, c.height) == (ref.steps, ref.width, ref.height) else {
                throw ComponentParityError("\(c.id): fixture parameters differ from the case list")
            }
            let native = FlowMatchEulerScheduler(steps: c.steps, width: c.width, height: c.height).sigmas
            let reference = MLXArray(ref.sigmaBits.map { Float(bitPattern: $0) })
            let result = ComponentCaseResult(id: c.id, checks: [ComponentCheck.exact("sigmas", native: native, reference: reference)])
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentReport(
            component: "zimage-base-sigmas", tolerances: [:], mlxSwiftRevision: "", referenceMLXPythonVersion: nil, cases: results,
            passed: !results.isEmpty && results.allSatisfy(\.passed))
    }

    /// Guided steps of the real loop (seeded noise, the real transformer, `FlowMatchEulerScheduler`, two passes per step)
    /// against MFLUX's, from the reference's own caption and negative-caption embeddings. Decisive: sigmas and seeded noise
    /// (bitwise); the guidance combination applied to MFLUX's own two predictions (bitwise); the two unguided step-0
    /// predictions (native eager vs MFLUX eager, relative L2 <= 2e-3, the full-output limit); and every step taken alone from
    /// MFLUX's compiled state before it, native compiled vs MFLUX compiled, relative L2 <= 5e-3 (the per-step latent limit).
    /// Diagnostics: the guided prediction itself (the guidance weights multiply the unguided 1e-3-level difference by up to
    /// 1 + 2 * guidance, so its error is expected above 2e-3), the free-running loops (guidance makes the chain chaotic: MFLUX's
    /// own eager and compiled loops drift apart by the same order), and the native text embeddings.
        static func baseSteps(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try BaseCases.load(casesJSON).stepCases
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "zimage-base")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: try BaseCases.load(casesJSON).allJobIDs, what: "zimage-base")
        let generator = try ZImageGenerator(modelDirectory: modelDirectory, variant: .base)
        let denoiser = generator.denoiser
        var results: [ComponentCaseResult] = []
        for c in cases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let (cap, neg) = try (require(ref, "cap_feats", c.id), require(ref, "neg_feats", c.id))
            let request = c.request
            let scheduler = denoiser.scheduler(for: request)
            var checks = [
                ComponentCheck.exact("sigmas", native: scheduler.sigmas, reference: try require(ref, "sigmas", c.id)),
                ComponentCheck.exact(
                    "seeded noise (latents.0)", native: ZImageLatentCreator.createNoise(seed: c.seed, height: c.height, width: c.width),
                    reference: try require(ref, "latents.0", c.id)),
            ]
            let nativeCap = try generator.encodePrompt(c.prompt)
            let nativeNeg = try generator.encodePrompt(request.effectiveNegativePrompt)
            checks.append(ComponentCheck.tolerance("text embedding (diagnostic)", native: nativeCap, reference: cap, limits: .textEmbedding, decisive: false))
            checks.append(ComponentCheck.tolerance("negative text embedding (diagnostic)", native: nativeNeg, reference: neg, limits: .textEmbedding, decisive: false))

            // Step 0 predictions from the seeded noise.
            let latents0 = try require(ref, "latents.0", c.id)
            let timestep0 = ZImageDenoiser.timestep(scheduler, step: 0)
            let transformer = denoiser.transformer
            let cond = transformer.forward(latents: latents0, timestep: timestep0, captionFeatures: cap)
            let uncond = transformer.forward(latents: latents0, timestep: timestep0, captionFeatures: neg)
            let guided = denoiser.predictGuided(
                latents: latents0, timestep: timestep0, captionFeatures: cap, negativeCaptionFeatures: neg, guidance: c.guidance, mode: .compiled)
            let combined = ZImageDenoiser.combine(
                noise: try require(ref, "cond_noise.0", c.id), negativeNoise: try require(ref, "uncond_noise.0", c.id), guidance: c.guidance)
            eval(cond, uncond, guided, combined)
            checks.append(ComponentCheck.exact("guidance combination of MFLUX's own predictions vs MFLUX's guided prediction", native: combined, reference: try require(ref, "eager.noise.0", c.id)))
            checks.append(ComponentCheck.tolerance("step 0 conditional prediction: native eager vs MFLUX eager", native: cond, reference: try require(ref, "cond_noise.0", c.id), limits: .fullTransformerOutput))
            checks.append(ComponentCheck.tolerance("step 0 negative prediction: native eager vs MFLUX eager", native: uncond, reference: try require(ref, "uncond_noise.0", c.id), limits: .fullTransformerOutput))
            checks.append(ComponentCheck.tolerance("step 0 guided prediction: native compiled vs MFLUX compiled (amplified, diagnostic)", native: guided, reference: try require(ref, "compiled.noise.0", c.id), limits: .fullTransformerOutput, decisive: false))

            // Each step alone from MFLUX's compiled state before it (the isolated comparison), then the free-running loops.
            for t in 0 ..< c.steps {
                let before = try require(ref, t == 0 ? "latents.0" : "compiled.latents.\(t)", c.id)
                let noise = denoiser.predictGuided(
                    latents: before, timestep: ZImageDenoiser.timestep(scheduler, step: t), captionFeatures: cap,
                    negativeCaptionFeatures: neg, guidance: c.guidance, mode: .compiled)
                let isolated = scheduler.step(noise: noise, timestep: t, latents: before)
                eval(isolated)
                checks.append(ComponentCheck.tolerance("step \(t): isolated native compiled guided step vs MFLUX compiled", native: isolated, reference: try require(ref, "compiled.latents.\(t + 1)", c.id), limits: .schedulerLatentPerStep))
            }
            var compiled: [MLXArray] = []
            var eager: [MLXArray] = []
            _ = try denoiser.denoise(request: request, captionFeatures: cap, negativeCaptionFeatures: neg, mode: .compiled) { _, l in compiled.append(l) }
            _ = try denoiser.denoise(request: request, captionFeatures: cap, negativeCaptionFeatures: neg, mode: .eager) { _, l in eager.append(l) }
            for t in 0 ..< c.steps {
                let (mfluxCompiled, mfluxEager) = try (require(ref, "compiled.latents.\(t + 1)", c.id), require(ref, "eager.latents.\(t + 1)", c.id))
                checks.append(ComponentCheck.tolerance("step \(t): free-running native compiled vs MFLUX compiled (diagnostic)", native: compiled[t], reference: mfluxCompiled, limits: .schedulerLatentPerStep, decisive: false))
                checks.append(ComponentCheck.tolerance("step \(t): free-running native eager vs MFLUX eager (diagnostic)", native: eager[t], reference: mfluxEager, limits: .schedulerLatentPerStep, decisive: false))
                checks.append(ComponentCheck.tolerance("step \(t): MFLUX eager vs MFLUX compiled (reference drift)", native: mfluxEager, reference: mfluxCompiled, limits: .schedulerLatentPerStep, decisive: false))
            }
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentCheck.report(
            component: "zimage-base-steps",
            tolerances: [
                "latent_per_step_relative_l2": ParityTolerance.schedulerLatentPerStepRelativeL2,
                "full_transformer_output_relative_l2": ParityTolerance.fullTransformerOutputRelativeL2,
            ], mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    /// How many times MFLUX's own eager-vs-compiled mean difference the native pipeline may differ from MFLUX's compiled pixels.
    static let pixelDriftFactor = 2.0

    /// The whole native Base pipeline from the prompt texts against MFLUX's real `ZImage.generate_image`.
    ///
    /// With guidance, a pixel match within the unguided 2 / 0.1 level limits is not attainable: the guidance weights amplify
    /// bf16-level differences and the sampling chain is chaotic, so MFLUX's own eager and compiled runs of the same job differ
    /// by mean 2 to 14 levels and a maximum above 170 on the recorded cases. The decisive row is therefore relative to that
    /// measured reference drift: the native compiled pipeline's mean abs difference from MFLUX's compiled pixels must be at
    /// most `pixelDriftFactor` times MFLUX's eager-vs-compiled mean abs difference, and the shapes must match. All pixel
    /// metrics (max, mean, differing values) are recorded.
        static func basePixels(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try BaseCases.load(casesJSON).pixelCases
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "zimage-base")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: try BaseCases.load(casesJSON).allJobIDs, what: "zimage-base")
        let generator = try ZImageGenerator(modelDirectory: modelDirectory, variant: .base)
        var results: [ComponentCaseResult] = []
        for c in cases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let refPixels = try require(ref, "pixels", c.id)
            let refEager = try require(ref, "pixels_eager", c.id)
            let image = try generator.generate(c.request, mode: .compiled)
            let (native, compiledBytes, eagerBytes) = (image.pixels.asArray(UInt8.self), refPixels.asArray(UInt8.self), refEager.asArray(UInt8.self))
            let strict = ComponentCheck.pixels(
                "pixels: native compiled vs MFLUX compiled, unguided limits (diagnostic)", native: native, reference: compiledBytes,
                nativeShape: image.pixels.shape, referenceShape: refPixels.shape, decisive: false)
            let drift = ComponentCheck.pixels(
                "pixels: MFLUX eager vs MFLUX compiled (reference drift)", native: eagerBytes, reference: compiledBytes,
                nativeShape: refEager.shape, referenceShape: refPixels.shape, decisive: false)
            var envelope = CheckResult(
                name: "pixels: native compiled vs MFLUX compiled, mean abs within \(pixelDriftFactor)x MFLUX's own drift", kind: "pixels",
                decisive: true, passed: false, metrics: nil, pixels: strict.pixels, detail: nil)
            if let measured = strict.pixels, let reference = drift.pixels, image.pixels.shape == refPixels.shape {
                envelope.passed = measured.meanAbsDiff <= pixelDriftFactor * reference.meanAbsDiff
                envelope.detail = String(
                    format: "native mean abs %.3f, max abs %d; MFLUX drift mean abs %.3f, max abs %d", measured.meanAbsDiff, measured.maxAbsDiff,
                    reference.meanAbsDiff, reference.maxAbsDiff)
            } else {
                envelope.detail = "shape mismatch: native \(image.pixels.shape), reference \(refPixels.shape)"
            }
            let result = ComponentCaseResult(id: c.id, checks: [envelope, strict, drift])
            log(logLine(result, pixels: true))
            results.append(result)
        }
        return ComponentCheck.report(
            component: "zimage-base-pixels",
            tolerances: ["pixel_drift_factor": pixelDriftFactor],
            mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }
}
