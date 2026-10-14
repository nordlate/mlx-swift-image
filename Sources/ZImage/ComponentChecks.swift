import Foundation
import MLX

/// The component checks (slice 2: latent, scheduler, weights, embeddings, first block; slice 3: middle and last block, final projection,
/// full transformer output). Each is independently runnable and returns a report whose `passed` is the verdict.
public enum ComponentParity {
    // MARK: Seeded latents

    /// Seeded initial noise against mflux's `ZImageLatentCreator.create_noise`: bitwise equal for every seed and size.
    public static func latent(
        referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let declared = try ComponentCases.load(casesJSON).latentCases
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "latent")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: declared.map(\.id), what: "latent")
        var results: [ComponentCaseResult] = []
        for c in declared {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            guard let reference = ref["noise"] else { throw ComponentParityError("\(c.id): reference has no noise") }
            let native = ZImageLatentCreator.createNoise(seed: c.seed, height: c.height, width: c.width)
            let result = ComponentCaseResult(id: c.id, checks: [ComponentCheck.exact("noise", native: native, reference: reference)])
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentCheck.report(component: "latent", tolerances: [:], mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    // MARK: Scheduler

    /// Sigmas and timesteps for every offered size and step count (bitwise), and per-step checkpoints of four chains:
    /// each step from the reference's own state (bitwise: one deterministic op on identical input) and a native free-running
    /// chain (within the predeclared relative-L2 limit), against the reference states.
    public static func scheduler(
        referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try ComponentCases.load(casesJSON)
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "scheduler")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.schedulerCases.map(\.id), what: "scheduler")
        try requireChainList(directory: referenceDirectory, declared: cases.schedulerChains.map(\.id))
        var results: [ComponentCaseResult] = []
        for c in cases.schedulerCases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let scheduler = LinearScheduler(steps: c.steps, width: c.width, height: c.height)
            let checks = try [
                ComponentCheck.exact("sigmas", native: scheduler.sigmas, reference: require(ref, "sigmas", c.id)),
                ComponentCheck.exact("timesteps", native: scheduler.timesteps, reference: require(ref, "timesteps", c.id)),
            ]
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        for c in cases.schedulerChains {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let scheduler = LinearScheduler(steps: c.steps, width: c.width, height: c.height)
            var checks = try [
                ComponentCheck.exact("sigmas", native: scheduler.sigmas, reference: require(ref, "sigmas", c.id)),
                ComponentCheck.exact("timesteps", native: scheduler.timesteps, reference: require(ref, "timesteps", c.id)),
            ]
            let noise = ZImageLatentCreator.createNoise(seed: c.seed, height: c.height, width: c.width)
            try checks.append(ComponentCheck.exact("latents.0 (seeded noise)", native: noise, reference: require(ref, "latents.0", c.id)))
            var running = noise
            for t in 0..<c.steps {
                let prediction = try require(ref, "noise.\(t)", c.id)
                let next = try require(ref, "latents.\(t + 1)", c.id)
                let isolated = try scheduler.step(noise: prediction, timestep: t, latents: require(ref, "latents.\(t)", c.id))
                running = scheduler.step(noise: prediction, timestep: t, latents: running)
                eval(isolated, running)
                checks.append(ComponentCheck.exact("isolated step \(t)", native: isolated, reference: next))
                checks.append(ComponentCheck.tolerance("free-running latents after step \(t)", native: running, reference: next, limits: .schedulerLatentPerStep))
            }
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentCheck.report(
            component: "scheduler",
            tolerances: ["sigmas_timesteps_noise": 0, "latent_per_step_relative_l2": ParityTolerance.schedulerLatentPerStepRelativeL2],
            mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    private struct ChainManifest: Decodable {
        struct IDOnly: Decodable { let id: String }
        let chains: [IDOnly]
    }

    private static func requireChainList(directory: URL, declared: [String]) throws {
        let chains = try JSONDecoder().decode(ChainManifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        guard !declared.isEmpty, chains.chains.map(\.id) == declared else {
            throw ComponentParityError("scheduler: reference chain list differs from component_cases.json")
        }
    }

    // MARK: q4 transformer weights

    private struct WeightsManifest: Decodable {
        struct Tensor: Decodable { let dtype: String; let shape: [Int]; let sha256: String }
        struct Quantized: Decodable { let name: String; let bits: Int; let groupSize: Int; let mode: String
            enum CodingKeys: String, CodingKey { case name, bits, groupSize = "group_size", mode }
        }
        let tensors: [String: Tensor]
        let quantizedLinears: [Quantized]
        let unquantizedLinears: [String]
        let storedQuantizationLevel: Int
        let storedMfluxVersion: String
        enum CodingKeys: String, CodingKey {
            case tensors, quantizedLinears = "quantized_linears", unquantizedLinears = "unquantized_linears"
            case storedQuantizationLevel = "stored_quantization_level", storedMfluxVersion = "stored_mflux_version"
        }
    }

    /// The loaded q4 transformer against what mflux holds after applying the same snapshot: tensor names, dtypes, shapes,
    /// the SHA-256 of every tensor's bytes, the quantization of every linear, and the stored quantization metadata.
    public static func weights(
        modelDirectory: URL, referenceDirectory: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "weights")
        let reference = try JSONDecoder().decode(WeightsManifest.self, from: Data(contentsOf: referenceDirectory.appendingPathComponent("manifest.json")))
        let loaded = try TransformerWeights.load(modelDirectory: modelDirectory)
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

        // Quantization as mflux's applier infers it from stored shapes, against the Python module attributes.
        let layout = Dictionary(uniqueKeysWithValues: TransformerLayout.allMatrices.map { ($0.name, $0) })
        var quantProblems: [String] = []
        if Set(reference.quantizedLinears.map(\.name)) != Set(layout.keys) { quantProblems.append("quantized linear sets differ") }
        for q in reference.quantizedLinears {
            guard let m = layout[q.name], let w = native[q.name + ".weight"], let s = native[q.name + ".scales"] else {
                quantProblems.append("\(q.name): not loaded"); continue
            }
            let (bits, group) = (w.dim(-1) * 32 / m.input, m.input / s.dim(-1))
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

        let result = ComponentCaseResult(id: "q4-transformer", checks: checks)
        return ComponentCheck.report(
            component: "weights", tolerances: ["names_dtypes_shapes_digests_quantization": 0], mlxSwiftRevision: mlxSwiftRevision,
            manifest: manifest, cases: [result])
    }

    // MARK: Patch and timestep embeddings

    private struct EmbeddingsManifest: Decodable {
        struct IDOnly: Decodable { let id: String }
        let timestepCases: [IDOnly]
        enum CodingKeys: String, CodingKey { case timestepCases = "timestep_cases" }
    }

    /// Patch, caption and timestep embeddings from the reference's isolated inputs, and the timestep embedding alone for
    /// every scheduler step of the declared cases, within the predeclared relative-L2 limit.
    public static func embeddings(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try ComponentCases.load(casesJSON)
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "embeddings")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.embeddingCases.map(\.id), what: "embeddings")
        let timesteps = try JSONDecoder().decode(EmbeddingsManifest.self, from: Data(contentsOf: referenceDirectory.appendingPathComponent("manifest.json")))
        guard timesteps.timestepCases.map(\.id) == cases.timestepCases.map(\.id), !cases.timestepCases.isEmpty else {
            throw ComponentParityError("embeddings: reference timestep case list differs from component_cases.json")
        }
        let transformer = ZImageTransformer(weights: try TransformerWeights.load(modelDirectory: modelDirectory))
        let limits = ParityLimits.patchTimestepEmbedding
        var results: [ComponentCaseResult] = []
        for c in cases.embeddingCases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let native = try transformer.embed(
                latents: require(ref, "latents", c.id), timestep: require(ref, "timestep", c.id), captionFeatures: require(ref, "cap_feats", c.id))
            let outputs: [(String, MLXArray, String)] = [
                ("image embedding", native.imageEmbedding, "x_emb"), ("caption embedding", native.captionEmbedding, "cap_emb"),
                ("image rotary table", native.imageFreqsCis, "x_freqs"), ("caption rotary table", native.captionFreqsCis, "cap_freqs"),
                ("timestep embedding", native.timestepEmbedding, "t_emb"),
            ]
            eval(outputs.map(\.1))
            let result = try ComponentCaseResult(
                id: c.id, checks: outputs.map { ComponentCheck.tolerance($0.0, native: $0.1, reference: try require(ref, $0.2, c.id), limits: limits) })
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("timesteps.safetensors"))
        for c in cases.timestepCases {
            let native = transformer.timestepEmbedding(try require(ref, "\(c.id).timestep", c.id))
            eval(native)
            let result = ComponentCaseResult(
                id: c.id, checks: [ComponentCheck.tolerance("timestep embedding", native: native, reference: try require(ref, "\(c.id).t_emb", c.id), limits: limits)])
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentCheck.report(
            component: "embeddings", tolerances: ["patch_timestep_embedding_relative_l2": ParityTolerance.patchTimestepEmbeddingRelativeL2],
            mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    // MARK: Transformer blocks

    /// `layers.0` from the reference's isolated input (the unified sequence the real forward feeds it).
    public static func block(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try ComponentCases.load(casesJSON)
        return try mainBlock(
            component: "block", layer: 0, cases: cases.blockCases, modelDirectory: modelDirectory, referenceDirectory: referenceDirectory,
            mlxSwiftRevision: mlxSwiftRevision, log: log)
    }

    /// `layers.15` from the reference's isolated input.
    public static func blockMiddle(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try ComponentCases.load(casesJSON)
        return try mainBlock(
            component: "block-middle", layer: TransformerLayout.layerCount / 2, cases: cases.blockMiddleCases, modelDirectory: modelDirectory,
            referenceDirectory: referenceDirectory, mlxSwiftRevision: mlxSwiftRevision, log: log)
    }

    /// `layers.29` from the reference's isolated input.
    public static func blockLast(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try ComponentCases.load(casesJSON)
        return try mainBlock(
            component: "block-last", layer: TransformerLayout.layerCount - 1, cases: cases.blockLastCases, modelDirectory: modelDirectory,
            referenceDirectory: referenceDirectory, mlxSwiftRevision: mlxSwiftRevision, log: log)
    }

    /// One main block from isolated reference input. The eager block output decides within the predeclared relative-L2 limit;
    /// the compiled comparisons and the intermediates are diagnostics (compiled fusion is an expected difference, recorded).
    private static func mainBlock(
        component: String, layer: Int, cases: [ComponentCases.Scene], modelDirectory: URL, referenceDirectory: URL,
        mlxSwiftRevision: String, log: (String) -> Void
    ) throws -> ComponentReport {
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: component)
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.map(\.id), what: component)
        let transformer = ZImageTransformer(weights: try TransformerWeights.load(modelDirectory: modelDirectory))
        let prefix = "layers.\(layer)."
        let compiledBlock = transformer.compiledBlock(prefix: prefix)
        var results: [ComponentCaseResult] = []
        for c in cases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            var traced: [String: MLXArray] = [:]
            let (x, mask) = try (require(ref, "x", c.id), require(ref, "attn_mask", c.id).asType(.bool))
            let (freqs, tEmb) = try (require(ref, "freqs_cis", c.id), require(ref, "t_emb", c.id))
            let output = transformer.block(
                prefix: prefix, x: x, attentionMask: mask, freqsCis: freqs, timestepEmbedding: tEmb,
                trace: { name, array in
                    if ["modulation", "attn_in", "attn_out", "after_attn", "ffn_out"].contains(name) { eval(array); traced[name] = array }
                })
            let compiled = compiledBlock(x, mask, freqs, tEmb)
            eval(output, compiled)
            var checks = try ComponentCheck.eagerAndCompiled(
                "block output", nativeEager: output, nativeCompiled: compiled, referenceEager: require(ref, "output", c.id),
                referenceCompiled: require(ref, "output_compiled", c.id), limits: .singleBlock)
            for name in ["modulation", "attn_in", "attn_out", "after_attn", "ffn_out"] {
                guard let native = traced[name], let reference = ref[name] else { continue }
                checks.append(ComponentCheck.tolerance(name, native: native, reference: reference, limits: .singleBlock, decisive: false))
            }
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentCheck.report(
            component: component, tolerances: ["single_block_relative_l2": ParityTolerance.singleBlockRelativeL2],
            mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    // MARK: Final projection

    /// The final layer (adaLN, layer norm, projection) from the reference's isolated input (what the 30th block hands it),
    /// and unpatchify of the reference's own final-layer output (bitwise).
    public static func finalProjection(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try ComponentCases.load(casesJSON).finalCases
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "final")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.map(\.id), what: "final")
        let transformer = ZImageTransformer(weights: try TransformerWeights.load(modelDirectory: modelDirectory))
        let compiledFinal = transformer.compiledFinalLayer()
        var results: [ComponentCaseResult] = []
        for c in cases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let (x, tEmb) = try (require(ref, "x", c.id), require(ref, "t_emb", c.id))
            let output = transformer.finalLayer(x, timestepEmbedding: tEmb)
            let compiled = compiledFinal(x, tEmb)
            let referenceOutput = try require(ref, "output", c.id)
            let unpatched = -ZImageTransformer.unpatchify(referenceOutput[0], size: [1, c.height / 8, c.width / 8])
            eval(output, compiled, unpatched)
            var checks = try ComponentCheck.eagerAndCompiled(
                "final projection", nativeEager: output, nativeCompiled: compiled, referenceEager: referenceOutput,
                referenceCompiled: require(ref, "output_compiled", c.id), limits: .finalProjection)
            try checks.append(ComponentCheck.exact("unpatchify of the reference final output", native: unpatched, reference: require(ref, "latent_out", c.id)))
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentCheck.report(
            component: "final", tolerances: ["final_projection_relative_l2": ParityTolerance.finalProjectionRelativeL2, "unpatchify": 0],
            mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    // MARK: Full transformer output

    /// The complete forward path (embeddings, both refiners, 30 blocks, final layer, unpatchify) from the reference's
    /// isolated inputs (seeded latents, timestep, text embedding). The eager output decides within 2e-3; the production
    /// compiled prediction path and the refiner/stack intermediates are diagnostics.
    public static func output(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ComponentReport {
        let cases = try ComponentCases.load(casesJSON).outputCases
        let manifest = try ReferenceInfo.load(directory: referenceDirectory, component: "output")
        try ComponentCheck.requireCaseList(manifest: manifest, declared: cases.map(\.id), what: "output")
        let transformer = ZImageTransformer(weights: try TransformerWeights.load(modelDirectory: modelDirectory))
        let compiledForward = transformer.compiledForward()
        var results: [ComponentCaseResult] = []
        for c in cases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let (latents, timestep, cap) = try (require(ref, "latents", c.id), require(ref, "timestep", c.id), require(ref, "cap_feats", c.id))
            var trace = ZImageTransformer.ForwardTrace()
            let output = transformer.forward(latents: latents, timestep: timestep, captionFeatures: cap, trace: { trace = $0 })
            eval(output)
            let compiled = compiledForward(latents, timestep, cap)
            eval(compiled)
            var checks = try ComponentCheck.eagerAndCompiled(
                "transformer output", nativeEager: output, nativeCompiled: compiled, referenceEager: require(ref, "output", c.id),
                referenceCompiled: require(ref, "output_compiled", c.id), limits: .fullTransformerOutput)
            for (name, native, key) in [
                ("noise refiner output", trace.noiseRefined, "x_refined"), ("context refiner output", trace.contextRefined, "cap_refined"),
                ("final layer input (after 30 blocks)", trace.finalLayerInput, "final_in"),
            ] {
                guard let native else { continue }
                eval(native)
                try checks.append(ComponentCheck.tolerance(name, native: native, reference: require(ref, key, c.id), limits: .fullTransformerOutput, decisive: false))
            }
            let result = ComponentCaseResult(id: c.id, checks: checks)
            log(ComponentCheck.logLine(result))
            results.append(result)
        }
        return ComponentCheck.report(
            component: "output", tolerances: ["full_transformer_output_relative_l2": ParityTolerance.fullTransformerOutputRelativeL2],
            mlxSwiftRevision: mlxSwiftRevision, manifest: manifest, cases: results)
    }

    static func require(_ arrays: [String: MLXArray], _ key: String, _ id: String) throws -> MLXArray {
        guard let array = arrays[key] else { throw ComponentParityError("\(id): reference has no '\(key)'") }
        return array
    }
}
