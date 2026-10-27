import FluxKlein
import Foundation
import MLX
import MLXImageCore

// FLUX.2 Klein generation parity against MFLUX 0.20, for either variant (it is how Klein 9B is checked; running it for 4B checks the
// harness on the variant whose port was already pixel-identical to MFLUX 0.19 on the MLX build of its export). The Python exporter
// (outside the package) runs MFLUX's own `Flux2Klein` on the same q4 snapshot and writes, per case of
// `reference/klein_generation_cases.json`: the prompt embeddings, the sigmas, timesteps and IDs, the initial latents (seeded noise, or
// MFLUX's noised clean latents for image to image), MFLUX's clean latents with the VAE encoder on the GPU and on the CPU, MFLUX's
// compiled per-step latents and predictions, the same steps taken with MFLUX's eager prediction, and the pixels of its real
// `generate_image`, compiled and eager. Nothing generated is committed.
//
// Why some limits are relative to MFLUX's own drift: with the MLX 0.32.2 Python wheel and this package's MLX build, MFLUX's own eager
// and compiled transformer predictions on identical inputs differ by relative L2 2e-2 to 3e-2 (bf16 Klein is that sensitive to kernel
// and fusion differences), and its own VAE encoder differs by 3.5e-2 between GPU and CPU. The native pipeline is held to the fixed
// limit or to `driftFactor` times the reference's own drift on the same quantity, whichever is larger.

/// `reference/klein_generation_cases.json`, shared with the Python exporter.
struct KleinGenerationParityCases: Decodable {
    struct Job: Decodable {
        let id: String
        let seed: UInt64
        let width: Int
        let height: Int
        let steps: Int
        let prompt: String
        /// Image to image only: `image_strength`.
        let strength: Double?
    }
    let textCases: [Job]
    let imageCases: [Job]
    /// The exporter's manifest lists the text cases, then the image cases.
    var allIDs: [String] { (textCases + imageCases).map(\.id) }
    enum CodingKeys: String, CodingKey { case textCases = "text_cases", imageCases = "image_cases" }

    static func load(_ url: URL) throws -> KleinGenerationParityCases {
        try JSONDecoder().decode(KleinGenerationParityCases.self, from: Data(contentsOf: url))
    }
}

public enum KleinVariantParity {
    /// The reference export's `manifest.json` component for `variant`.
    public static func component(_ variant: KleinVariant) -> String { "\(variant.rawValue)-generation" }

    /// How many times the reference's own drift (eager vs compiled, CPU vs GPU encoder) a native difference may be.
    public static let driftFactor = 2.0

    /// The synthetic RGB-v1 starting image of an image case (`KleinStartingImageCases.algorithm`).
    static func startingRGB(width: Int, height: Int, seed: UInt64) -> [UInt8] {
        KleinStartingImageCases.rgb(KleinStartingImageCase(id: "", prompt: "", width: width, height: height, seed: seed, steps: 1, strength: 0.5))
    }

    /// Every case of `casesJSON` through the native pipeline, against the MFLUX 0.20 export in `referenceDirectory`.
    ///
    /// Decisive, per case: text IDs, latent IDs, sigmas, timesteps and (image to image) the start index and the starting image's RGB,
    /// bitwise; for text to image the seeded noise bitwise; the prompt embeddings within the text-embedding limits (relative L2 1e-3,
    /// normalized max abs 1e-2); for image to image the clean latents within relative L2 2e-3 (the `clean_initial` budget) or
    /// `driftFactor` times MFLUX's own GPU-vs-CPU encoder difference; every denoising step taken alone from MFLUX's compiled state
    /// before it (native compiled prediction and scheduler step) within relative L2 5e-3 of MFLUX's next state or `driftFactor` times
    /// MFLUX's own eager step from the same state; and the pixels of the complete native job (`pixelRows`). Diagnostics: the noised
    /// initial latents, the first prediction alone, the free-running native latents.
    public static func generation(
        variant: KleinVariant, modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String,
        log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try KleinGenerationParityCases.load(casesJSON)
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: component(variant))
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.allIDs, what: component(variant))
        let generator = try KleinGenerator(modelDirectory: modelDirectory, variant: variant)
        let predict = KleinTransformer(weights: generator.weights.transformer).compiledPredict()
        var results: [ComponentCaseResult] = []
        for c in cases.textCases + cases.imageCases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            func r(_ key: String) throws -> MLXArray { try ComponentParity.require(ref, key, c.id) }
            var checks: [CheckResult] = []

            // The complete native job, recording what the reference records.
            var traced: [String: MLXArray] = [:]
            let record: (String, MLXArray) throws -> Void = { name, array in traced[name] = array }
            let output: KleinGenerationOutput
            if let strength = c.strength {
                let image = referenceDirectory.appendingPathComponent("\(c.id).input.png")
                let prepared = try KleinPreparedStartingImage.load(image, width: c.width, height: c.height)
                var rgb = CheckResult(name: "starting image RGB (exporter PNG vs RGB-v1)", kind: "exact", decisive: true, passed: false, metrics: nil, detail: nil)
                rgb.passed = prepared.rgb == startingRGB(width: c.width, height: c.height, seed: c.seed)
                checks.append(rgb)
                let request = KleinStartingImageRequest(
                    prompt: c.prompt, width: c.width, height: c.height, seed: c.seed, steps: c.steps, image: image, strength: strength)
                output = try generator.generate(request, trace: record)
            } else {
                let request = KleinGenerationRequest(prompt: c.prompt, width: c.width, height: c.height, seed: c.seed, steps: c.steps)
                output = try generator.generate(request, trace: record)
            }
            func t(_ key: String) throws -> MLXArray {
                guard let a = traced[key] else { throw ComponentParityError("\(c.id): the native run did not record \(key)") }
                return a
            }
            let start = c.strength == nil ? 0 : Int(try r("start").item(Int32.self))
            for key in ["text_ids", "latent_ids", "sigmas", "timesteps"] {
                checks.append(ComponentCheck.exact(key, native: try t(key), reference: try r(key)))
            }
            if c.strength != nil {
                checks.append(ComponentCheck.exact("start", native: try t("start"), reference: try r("start")))
                checks.append(envelope(
                    "clean latents (VAE encoder, patchify, BN, pack)", native: try t("clean"), reference: try r("clean"),
                    referenceDrift: try r("clean_cpu"), driftName: "MFLUX encoder on the CPU vs the GPU", fixed: KleinStartingImageBudgets.clean))
                checks.append(ComponentCheck.tolerance(
                    "initial (noised clean latents, diagnostic)", native: try t("initial"), reference: try r("initial"),
                    limits: ParityLimits(relativeL2: KleinStartingImageBudgets.clean, normalizedMaxAbs: nil), decisive: false))
            } else {
                checks.append(ComponentCheck.exact("initial (seeded noise)", native: try t("initial"), reference: try r("initial")))
            }
            checks.append(ComponentCheck.tolerance("prompt embeddings", native: try t("prompt_embeds"), reference: try r("prompt_embeds"), limits: .textEmbedding))

            // Each step alone from MFLUX's compiled state before it.
            let scheduler = try KleinScheduler(steps: c.steps, height: c.height, width: c.width)
            let (embeds, textIDs, latentIDs) = (try r("prompt_embeds"), try r("text_ids"), try r("latent_ids"))
            for step in start ..< c.steps {
                let before = try r("latents.\(step)")
                let noise = predict(before, latentIDs, embeds, textIDs, scheduler.timesteps[step])
                let after = try scheduler.step(noise: noise, timestep: step, latents: before)
                eval(noise, after)
                if step == start {
                    checks.append(ComponentCheck.tolerance(
                        "step \(step) prediction from MFLUX's state (diagnostic)", native: noise, reference: try r("noise.\(step)"),
                        limits: .fullTransformerOutput, decisive: false))
                }
                checks.append(envelope(
                    "step \(step): isolated native compiled step vs MFLUX compiled", native: after, reference: try r("latents.\(step + 1)"),
                    referenceDrift: try r("eager.latents.\(step + 1)"), driftName: "MFLUX eager step vs compiled",
                    fixed: ParityTolerance.schedulerLatentPerStepRelativeL2))
                checks.append(ComponentCheck.tolerance(
                    "step \(step): free-running native vs MFLUX compiled (diagnostic)", native: try t("step.\(step + 1)"),
                    reference: try r("latents.\(step + 1)"), limits: .schedulerLatentPerStep, decisive: false))
            }

            let (refPixels, refEager) = (try r("pixels"), try r("pixels_eager"))
            checks += pixelRows(native: output.pixels, reference: refPixels, referenceEager: refEager)
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(ComponentParity.logLine(result, pixels: true))
            results.append(result)
        }
        return ComponentCheck.report(
            component: component(variant),
            tolerances: [
                "prompt_embeds_relative_l2": ParityTolerance.embeddingRelativeL2,
                "prompt_embeds_normalized_max_abs": ParityTolerance.embeddingNormalizedMaxAbs,
                "clean_initial_relative_l2": KleinStartingImageBudgets.clean,
                "latent_per_step_relative_l2": ParityTolerance.schedulerLatentPerStepRelativeL2,
                "pixels_max_abs": ParityTolerance.pixelMaxAbsLevels, "pixels_mean_abs": ParityTolerance.pixelMeanAbsLevels,
                "drift_factor": driftFactor,
            ],
            mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    /// A tensor row that passes within relative L2 `fixed`, or within `driftFactor` times the reference's own difference
    /// (`referenceDrift` against `reference`); both measurements are recorded.
    static func envelope(_ name: String, native: MLXArray, reference: MLXArray, referenceDrift: MLXArray, driftName: String, fixed: Double) -> CheckResult {
        let unlimited = ParityLimits(relativeL2: .infinity, normalizedMaxAbs: nil)
        var result = ComponentCheck.tolerance(name, native: native, reference: reference, limits: unlimited)
        let drift = ComponentCheck.tolerance(driftName, native: referenceDrift, reference: reference, limits: unlimited, decisive: false)
        guard result.detail == nil || result.detail == "bit-identical", let measured = result.metrics?.relativeL2, let own = drift.metrics?.relativeL2 else {
            result.passed = false
            return result
        }
        let limit = max(fixed, driftFactor * own)
        result.passed = measured <= limit
        result.detail = String(format: "relL2 %.3e; %@ %.3e; limit %.3e", measured, driftName, own, limit)
        return result
    }

    /// The pixel rows of one job. Decisive: the native compiled job's mean abs difference from MFLUX's compiled pixels is within the pixel
    /// limit (0.1 levels) or `driftFactor` times MFLUX's own eager-vs-compiled mean abs difference, with equal shapes. Recorded: the strict
    /// pixel-identity row (2 levels max, 0.1 mean) and MFLUX's eager-vs-compiled drift.
    static func pixelRows(native: MLXArray, reference: MLXArray, referenceEager: MLXArray) -> [CheckResult] {
        let (n, c, e) = (native.asArray(UInt8.self), reference.asArray(UInt8.self), referenceEager.asArray(UInt8.self))
        let strict = ComponentCheck.pixels(
            "pixels: native compiled vs MFLUX compiled, pixel-identity limits (diagnostic)", native: n, reference: c, nativeShape: native.shape,
            referenceShape: reference.shape, decisive: false)
        let drift = ComponentCheck.pixels(
            "pixels: MFLUX eager vs MFLUX compiled (reference drift)", native: e, reference: c, nativeShape: referenceEager.shape,
            referenceShape: reference.shape, decisive: false)
        var row = CheckResult(
            name: "pixels: native compiled vs MFLUX compiled, mean abs within \(driftFactor)x MFLUX's own drift", kind: "pixels", decisive: true,
            passed: false, metrics: nil, pixels: strict.pixels, detail: nil)
        if let measured = strict.pixels, let own = drift.pixels, native.shape == reference.shape {
            row.passed = strict.pixels?.passed == true || measured.meanAbsDiff <= driftFactor * own.meanAbsDiff
            row.detail = String(
                format: "native mean abs %.3f, max abs %d; MFLUX drift mean abs %.3f, max abs %d", measured.meanAbsDiff, measured.maxAbsDiff,
                own.meanAbsDiff, own.maxAbsDiff)
        } else {
            row.detail = "shape mismatch: native \(native.shape), reference \(reference.shape)"
        }
        return [row, strict, drift]
    }
}
