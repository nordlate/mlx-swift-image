import Foundation
import MLX
import MLXImageCore

/// Collects the named intermediates of one eager forward pass.
final class KleinTraceBox {
    var arrays: [String: MLXArray] = [:]
    func record(_ name: String, _ array: MLXArray) { arrays[name] = array }
}

/// The native side of the slice-2 check. `run` is independent: it uses the ported tokenizer and text encoder (slice 1), its own seeded
/// latents and scheduler, and the full native transformer, and never reads reference output. `runIsolated` is the diagnostic that
/// feeds the reference's captured inputs to single native components to locate an error.
public enum KleinTransformerNative {
    static func prepareOutput(_ outputDirectory: URL, modelDirectory: URL) throws {
        let fm = FileManager.default
        // Foundation may leave symlink parents unresolved when the final component does not exist.
        // Inspect every prefix, including the output itself, without following links or dropping a nonexistent tail.
        var prefix = ""
        for component in outputDirectory.path.split(separator: "/") {
            prefix += "/" + String(component)
            if (try? fm.destinationOfSymbolicLink(atPath: prefix)) != nil {
                throw KleinTransformerError.invalidInput("output path \(outputDirectory.path) contains a symbolic link at \(prefix); pass the real path")
            }
        }
        guard !KleinNativeExport.isInside(outputDirectory, modelDirectory) else {
            throw KleinTransformerError.invalidInput("output directory must not be inside the model directory")
        }
        if fm.fileExists(atPath: outputDirectory.path), !(try fm.contentsOfDirectory(atPath: outputDirectory.path)).isEmpty {
            throw KleinTransformerError.invalidInput("output directory \(outputDirectory.path) is not empty; each run writes a fresh directory")
        }
        try fm.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
    }

    /// SELF-REPORTED by this binary (a constant and an environment variable), not a build attestation; the executable hash lets the
    /// external build log, dependency verification and source hashes name this exact binary.
    static func runtimeInfo(extra: [String: String] = [:]) throws -> [String: String] {
        let env = ProcessInfo.processInfo
        let executable = try KleinNativeExport.runningExecutable()
        var info: [String: String] = [
            "self_reported_mlx_swift_revision": KleinPins.mlxSwiftRevision,
            "self_reported_build_settings": env.environment["KLEIN_BUILD_SETTINGS"] ?? "unrecorded",
            "executable_path": executable.path,
            "executable_sha256": try KleinHashing.sha256(of: executable),
            "os": env.operatingSystemVersionString,
            "default_device": "\(Device.defaultDevice())",
        ]
        info.merge(extra) { _, new in new }
        return info
    }

    static func info(_ arrays: [String: MLXArray]) -> [String: KleinTensorInfo] {
        arrays.mapValues { KleinTensorInfo(dtype: KleinDType.name($0.dtype), shape: $0.shape) }
    }

    static func manifestCase(
        _ spec: KleinTransformerCase, tokens: TokenizedPromptSummary, grid: (height: Int, width: Int), tensors: [String: MLXArray]
    ) -> KleinTransformerManifestCase {
        KleinTransformerManifestCase(
            id: spec.id, category: spec.category, prompt: spec.prompt, renderedText: KleinTokenizer.renderChat(prompt: spec.prompt),
            width: spec.width, height: spec.height, seed: Int(spec.seed), steps: spec.steps, timestepIndex: spec.timestepIndex,
            imageSeqLen: spec.imageSeqLen, gridHeight: grid.height, gridWidth: grid.width, numValid: tokens.valid,
            untruncatedCount: tokens.untruncated, tensors: info(tensors))
    }

    struct TokenizedPromptSummary { let valid: Int; let untruncated: Int }

    // MARK: Independent native run

    public static func run(modelDirectory: URL, casesJSON: URL, budgetsJSON: URL, outputDirectory: URL, log: (String) -> Void = { _ in }) throws {
        try prepareOutput(outputDirectory, modelDirectory: modelDirectory)
        let (cases, casesSHA) = try KleinTransformerCases.load(url: casesJSON)
        let budgetFailures = try KleinTransformerBudgets.check(budgetsJSON: budgetsJSON)
        guard budgetFailures.isEmpty else { throw KleinTransformerError.invalidInput("budgets file differs from the compiled limits: \(budgetFailures)") }
        let budgetsSHA = try KleinHashing.sha256(of: budgetsJSON)
        let before = try KleinTransformerPins.verifyModelFiles(modelDirectory: modelDirectory)
        log("model files match their pins (\(before.count) files)")

        let tokenizer = try KleinTokenizer.load(modelDirectory: modelDirectory)
        let textWeights = try KleinTextWeights.load(modelDirectory: modelDirectory)
        let encoder = KleinTextEncoder(weights: textWeights)
        let transformerWeights = try KleinTransformerWeights.load(modelDirectory: modelDirectory)
        let textDigests = textWeights.digests(), transformerDigests = transformerWeights.digests()
        log("loaded \(textDigests.count) text-encoder and \(transformerDigests.count) transformer tensors")
        transformerWeights.materialize()
        let transformer = KleinTransformer(weights: transformerWeights)
        let predict = transformer.compiledPredict()

        var entries: [KleinTransformerManifestCase] = []
        for spec in cases {
            let tokens = KleinTokenizer.tokenize(spec.prompt, with: tokenizer)
            let untruncated = tokenizer.encode(KleinTokenizer.renderChat(prompt: spec.prompt)).count
            let encoding = try KleinPromptEncoder.encode(tokens, with: encoder)
            let prepared = try KleinLatent.preparePacked(seed: spec.seed, height: spec.height, width: spec.width)
            let scheduler = try KleinScheduler(steps: spec.steps, height: spec.height, width: spec.width)
            let timestep = scheduler.timesteps[spec.timestepIndex]

            let box = KleinTraceBox()
            let eager = transformer.forward(
                hiddenStates: prepared.latents, encoderHiddenStates: encoding.promptEmbeds, timestep: timestep, imgIDs: prepared.ids,
                txtIDs: encoding.textIDs, trace: { box.record($0, $1) })
            eval([eager] + Array(box.arrays.values))
            let compiled = predict(prepared.latents, prepared.ids, encoding.promptEmbeds, encoding.textIDs, timestep)
            eval(compiled)

            var tensors = box.arrays
            tensors["prompt_embeds"] = encoding.promptEmbeds
            tensors["text_ids"] = encoding.textIDs
            tensors["latents"] = prepared.latents
            tensors["latent_ids"] = prepared.ids
            tensors["sigmas"] = scheduler.sigmas
            tensors["timesteps"] = scheduler.timesteps
            tensors["noise_eager"] = eager
            tensors["noise_compiled"] = compiled
            // Free-running chain of scheduler steps with synthetic (model-independent) noise.
            var latents = prepared.latents
            tensors["chain.0"] = latents
            for t in 0..<spec.steps {
                let noise = KleinTransformerTensors.chainNoise(seed: spec.seed, step: t, shape: latents.shape)
                latents = try scheduler.step(noise: noise, timestep: t, latents: latents)
                eval(latents)
                tensors["chain_noise.\(t)"] = noise
                tensors["chain.\(t + 1)"] = latents
            }
            guard Set(tensors.keys) == Set(KleinTransformerTensors.primaryKeys(spec)) else {
                throw KleinTransformerError.invalidInput("\(spec.id): produced tensors differ from the required set")
            }
            eval(Array(tensors.values))
            try save(arrays: tensors, url: outputDirectory.appendingPathComponent("\(spec.id).safetensors"))
            entries.append(manifestCase(
                spec, tokens: TokenizedPromptSummary(valid: tokens.validCount, untruncated: untruncated),
                grid: (prepared.gridHeight, prepared.gridWidth), tensors: tensors))
            log("\(spec.id): valid=\(tokens.validCount) seq=\(spec.imageSeqLen) mu=\(scheduler.mu) ok")
        }

        let controls = try KleinTransformerControls.make()
        eval(Array(controls.values))
        try save(arrays: controls, url: outputDirectory.appendingPathComponent("controls.safetensors"))
        log("controls: \(controls.count) tensors")

        let after = try KleinTransformerPins.verifyModelFiles(modelDirectory: modelDirectory)
        let manifest = KleinTransformerManifest(
            schema: 1, side: "native", casesSHA256: casesSHA, casesDigest: KleinTransformerCases.digest(of: cases), budgetsSHA256: budgetsSHA,
            runtime: try runtimeInfo(), runtimeModules: nil, modelFilesBefore: before, modelFilesAfter: after, textWeights: textDigests,
            transformerWeights: transformerDigests,
            textQuantizedModules: TextEncoderLayout.quantizedMatrices().map {
                KleinQuantizedModule(name: $0.name, bits: TextEncoderLayout.bits, groupSize: TextEncoderLayout.groupSize, mode: "affine")
            },
            transformerQuantizedModules: KleinTransformerWeights.quantizedModules(), cases: entries, controlTensors: info(controls))
        try KleinJSON.write(manifest, to: outputDirectory.appendingPathComponent("manifest.json"))
    }

    // MARK: Isolated components from the reference's captured inputs

    public static func runIsolated(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL, budgetsJSON: URL, outputDirectory: URL, log: (String) -> Void = { _ in }
    ) throws {
        try prepareOutput(outputDirectory, modelDirectory: modelDirectory)
        let (cases, casesSHA) = try KleinTransformerCases.load(url: casesJSON)
        let budgetFailures = try KleinTransformerBudgets.check(budgetsJSON: budgetsJSON)
        guard budgetFailures.isEmpty else { throw KleinTransformerError.invalidInput("budgets file differs from the compiled limits: \(budgetFailures)") }
        let referenceManifestURL = referenceDirectory.appendingPathComponent("manifest.json")
        let reference = try KleinJSON.read(KleinTransformerManifest.self, from: referenceManifestURL)
        guard reference.side == "reference", reference.casesSHA256 == casesSHA else {
            throw KleinTransformerError.invalidInput("the reference directory is not a reference run of this cases file")
        }
        let before = try KleinTransformerPins.verifyModelFiles(modelDirectory: modelDirectory)
        let weights = try KleinTransformerWeights.load(modelDirectory: modelDirectory)
        let digests = weights.digests()
        weights.materialize()
        let transformer = KleinTransformer(weights: weights)
        let predict = transformer.compiledPredict()

        var entries: [KleinTransformerManifestCase] = []
        for spec in cases {
            let url = referenceDirectory.appendingPathComponent("\(spec.id).safetensors")
            guard FileManager.default.fileExists(atPath: url.path) else { throw KleinTransformerError.invalidInput("reference tensors for \(spec.id) are missing") }
            let ref = try loadArrays(url: url)
            func need(_ name: String) throws -> MLXArray {
                guard let a = ref[name] else { throw KleinTransformerError.invalidInput("\(spec.id): reference input \(name) is missing") }
                return a
            }
            for key in KleinTransformerTensors.isolatedInputKeys(spec) { _ = try need(key) }
            var iso: [String: MLXArray] = [:]

            // Embeddings, timestep embedding, modulation and rotary tables, each from the reference's own input.
            let tembRaw = transformer.timestepEmbedding(try need("timestep_scaled"))
            iso["temb_raw"] = tembRaw
            iso["temb"] = tembRaw.asType(.bfloat16)
            iso["x_embed"] = transformer.linear(try need("latents"), "x_embedder")
            iso["ctx_embed"] = transformer.linear(try need("prompt_embeds"), "context_embedder")
            let referenceTemb = try need("temb")
            let modImg = transformer.modulation(referenceTemb, name: "double_stream_modulation_img", sets: 2)
            let modTxt = transformer.modulation(referenceTemb, name: "double_stream_modulation_txt", sets: 2)
            let modSingle = transformer.modulation(referenceTemb, name: "single_stream_modulation", sets: 1)[0]
            for (k, a) in (modImg[0] + modImg[1]).enumerated() { iso["mod_img.\(k)"] = a }
            for (k, a) in (modTxt[0] + modTxt[1]).enumerated() { iso["mod_txt.\(k)"] = a }
            for (k, a) in modSingle.enumerated() { iso["mod_single.\(k)"] = a }
            let imageRope = KleinTransformer.rotaryTables(ids: try need("latent_ids")[0])
            let textRope = KleinTransformer.rotaryTables(ids: try need("text_ids")[0])
            iso["rope_img_cos"] = imageRope.cos; iso["rope_img_sin"] = imageRope.sin
            iso["rope_txt_cos"] = textRope.cos; iso["rope_txt_sin"] = textRope.sin

            // One block at a time, from the reference's captured input, modulation and rotary tables.
            let cos = concatenated([try need("rope_txt_cos"), try need("rope_img_cos")], axis: 0)
            let sin = concatenated([try need("rope_txt_sin"), try need("rope_img_sin")], axis: 0)
            let refModImg = try [0, 3].map { base in try (0..<3).map { try need("mod_img.\(base + $0)") } }
            let refModTxt = try [0, 3].map { base in try (0..<3).map { try need("mod_txt.\(base + $0)") } }
            let refModSingle = try (0..<3).map { try need("mod_single.\($0)") }
            for i in KleinTransformerTrace.doubleBlocks {
                let (e, h) = transformer.doubleBlock(
                    i, hidden: try need("double.\(i).in_hidden"), encoder: try need("double.\(i).in_encoder"), modImg: refModImg, modTxt: refModTxt,
                    cos: cos, sin: sin)
                iso["double.\(i).out_hidden"] = h
                iso["double.\(i).out_encoder"] = e
            }
            for i in KleinTransformerTrace.singleBlocks {
                iso["single.\(i).out"] = transformer.singleBlock(i, hidden: try need("single.\(i).in"), mod: refModSingle, cos: cos, sin: sin)
            }
            iso["norm_out_out"] = transformer.normOut(try need("norm_out_in"), temb: referenceTemb)

            // The whole transformer, eager and compiled, from the reference's own latents, embeddings and timestep.
            let timestep = try need("timesteps")[spec.timestepIndex]
            iso["noise_eager"] = transformer.forward(
                hiddenStates: try need("latents"), encoderHiddenStates: try need("prompt_embeds"), timestep: timestep,
                imgIDs: try need("latent_ids"), txtIDs: try need("text_ids"))
            iso["noise_compiled"] = predict(
                try need("latents"), try need("latent_ids"), try need("prompt_embeds"), try need("text_ids"), timestep)

            // The scheduler step from the reference's own chain state, noise and sigmas, with the native compiled step.
            let sigmas = try need("sigmas")
            var stepped: [String: MLXArray] = [:]
            for t in 0..<spec.steps {
                stepped["step.\(t)"] = try KleinScheduler.step(
                    noise: try need("chain_noise.\(t)"), latents: try need("chain.\(t)"), nextSigma: sigmas[t + 1], sigma: sigmas[t])
            }
            var out = Dictionary(uniqueKeysWithValues: iso.map { ("iso." + $0.key, $0.value) })
            for (k, v) in stepped { out["iso." + k] = v }
            guard Set(out.keys) == Set(KleinTransformerTensors.isolatedKeys(spec)) else {
                throw KleinTransformerError.invalidInput("\(spec.id): isolated tensors differ from the required set")
            }
            eval(Array(out.values))
            try save(arrays: out, url: outputDirectory.appendingPathComponent("\(spec.id).safetensors"))
            let grid = try KleinLatent.grid(height: spec.height, width: spec.width)
            entries.append(manifestCase(spec, tokens: TokenizedPromptSummary(valid: 0, untruncated: 0), grid: grid, tensors: out))
            log("\(spec.id): isolated \(out.count) tensors")
        }

        let after = try KleinTransformerPins.verifyModelFiles(modelDirectory: modelDirectory)
        let manifest = KleinTransformerManifest(
            schema: 1, side: "isolated", casesSHA256: casesSHA, casesDigest: KleinTransformerCases.digest(of: cases),
            budgetsSHA256: try KleinHashing.sha256(of: budgetsJSON),
            runtime: try runtimeInfo(extra: ["reference_manifest_sha256": try KleinHashing.sha256(of: referenceManifestURL)]),
            runtimeModules: nil, modelFilesBefore: before, modelFilesAfter: after, textWeights: [:], transformerWeights: digests,
            textQuantizedModules: [], transformerQuantizedModules: KleinTransformerWeights.quantizedModules(), cases: entries, controlTensors: [:])
        try KleinJSON.write(manifest, to: outputDirectory.appendingPathComponent("manifest.json"))
    }
}
