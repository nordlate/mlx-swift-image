import Foundation
import MLX
import ZImage

/// The native side of the slice-1 check. Runs on its own: it never reads reference output, only the read-only model and the frozen
/// cases file, and writes `<id>.safetensors` per case plus `manifest.json` into a new output directory.
public enum KleinNativeExport {
    /// Names of the tensors saved per encoded case.
    public static let stateKeys: [Int: String] = [0: "hs0", 9: "hs9", 18: "hs18", 27: "hs27"]

    public static func run(modelDirectory: URL, casesJSON: URL, outputDirectory: URL, log: (String) -> Void = { _ in }) throws {
        guard outputDirectory.standardizedFileURL.path == outputDirectory.resolvingSymlinksInPath().path else {
            throw KleinError.invalidInput("output path \(outputDirectory.path) contains a symbolic link; pass the real path")
        }
        guard !isInside(outputDirectory, modelDirectory) else {
            throw KleinError.invalidInput("output directory must not be inside the model directory")
        }
        let fm = FileManager.default
        if fm.fileExists(atPath: outputDirectory.path), !(try fm.contentsOfDirectory(atPath: outputDirectory.path)).isEmpty {
            throw KleinError.invalidInput("output directory \(outputDirectory.path) is not empty; each run writes a fresh directory")
        }
        try fm.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        let (cases, casesSHA) = try KleinCases.load(url: casesJSON)
        let before = try KleinHashing.verifyModelFiles(modelDirectory: modelDirectory)
        log("model files match their pins (\(before.count) files)")

        let tokenizer = try KleinTokenizer.load(modelDirectory: modelDirectory)
        let weights = try KleinTextWeights.load(modelDirectory: modelDirectory)
        let encoder = KleinTextEncoder(weights: weights)
        let digests = weights.digests()
        log("loaded \(digests.count) text-encoder tensors")

        var entries: [KleinManifestCase] = []
        for spec in cases {
            let tokens = KleinTokenizer.tokenize(spec.prompt, with: tokenizer)
            let untruncated = spec.prompt.isEmpty ? 0 : tokenizer.encode(KleinTokenizer.renderChat(prompt: spec.prompt)).count
            // `[1, 0]` int32 for the empty prompt, `[1, 512]` int64 otherwise, as mflux's LanguageTokenizer returns them.
            let tokenDType: DType = tokens.dtype == "mlx.core.int32" ? .int32 : .int64
            var tensors: [String: MLXArray] = [
                "input_ids": KleinTextEncoder.tokenArray(tokens.inputIDs, shape: tokens.shape, dtype: tokenDType),
                "attention_mask": KleinTextEncoder.tokenArray(tokens.attentionMask, shape: tokens.shape, dtype: tokenDType),
            ]
            // Every case, including the empty prompt (zero-length tensors, as production MFLUX produces), saves all eight tensors.
            let outcome = "ok"
            let encoding = try KleinPromptEncoder.encode(tokens, with: encoder)
            for (index, key) in stateKeys { tensors[key] = encoding.states[index]! }
            tensors["prompt_embeds"] = encoding.promptEmbeds
            tensors["text_ids"] = encoding.textIDs
            eval(Array(tensors.values))
            try save(arrays: tensors, url: outputDirectory.appendingPathComponent("\(spec.id).safetensors"))
            entries.append(KleinManifestCase(
                id: spec.id, category: spec.category, prompt: spec.prompt, renderedText: KleinTokenizer.renderChat(prompt: spec.prompt),
                numValid: tokens.validCount, untruncatedCount: untruncated, idsDtype: tokens.dtype, idsShape: tokens.shape,
                maskDtype: tokens.dtype, maskShape: tokens.shape, encoderOutcome: outcome,
                tensors: tensors.mapValues { KleinTensorInfo(dtype: KleinDType.name($0.dtype), shape: $0.shape) }))
            log("\(spec.id): valid=\(tokens.validCount) \(outcome)")
        }

        let after = try KleinHashing.verifyModelFiles(modelDirectory: modelDirectory)
        // What the native encoder constructs: every quantized matrix is applied with the layout's bits, group size and affine mode.
        let quantized = KleinTextWeights.quantizedMatrices().map {
            KleinQuantizedModule(name: $0.name, bits: TextEncoderLayout.bits, groupSize: TextEncoderLayout.groupSize, mode: "affine")
        }
        let env = ProcessInfo.processInfo
        let executable = try runningExecutable()
        let executableSHA = try KleinHashing.sha256(of: executable)
        let manifest = KleinManifest(
            schema: 1, side: "native", casesSHA256: casesSHA,
            runtime: [
                // SELF-REPORTED by this binary (a constant and an environment variable), not a build attestation. The real evidence
                // is the build log, dependency verification and source hashes bound externally; the executable hash below lets that
                // binding name this exact binary.
                "self_reported_mlx_swift_revision": KleinPins.mlxSwiftRevision,
                "self_reported_build_settings": env.environment["KLEIN_BUILD_SETTINGS"] ?? "unrecorded",
                "executable_path": executable.path,
                "executable_sha256": executableSHA,
                "os": env.operatingSystemVersionString,
                "default_device": "\(Device.defaultDevice())",
            ],
            runtimeModules: nil, modelFilesBefore: before, modelFilesAfter: after, weights: digests, quantizedModules: quantized, cases: entries)
        try KleinJSON.write(manifest, to: outputDirectory.appendingPathComponent("manifest.json"))
    }
}

extension KleinNativeExport {
    /// The running binary: `CommandLine.arguments[0]` with symbolic links resolved, which must be an existing regular file.
    static func runningExecutable() throws -> URL {
        let url = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
            throw KleinError.invalidInput("cannot resolve the running executable from argv[0] \(CommandLine.arguments[0]) (run it by path)")
        }
        return url
    }

    /// True when `child` is `parent` or below it, by whole path components of the symlink-resolved paths (so `models/x-other` is not
    /// inside `models/x`, and a link into the model directory is seen for what it points to).
    static func isInside(_ child: URL, _ parent: URL) -> Bool {
        let c = child.resolvingSymlinksInPath().standardizedFileURL.pathComponents
        let p = parent.resolvingSymlinksInPath().standardizedFileURL.pathComponents
        return c.count >= p.count && Array(c[..<p.count]) == p
    }
}

public enum KleinJSON {
    public static func write<T: Encodable>(_ value: T, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try requireNewFile(url)
        try encoder.encode(value).write(to: url, options: [.withoutOverwriting])
    }

    /// Refuses a path that already exists (as a file or a link): earlier reports and manifests are evidence and are never replaced.
    public static func requireNewFile(_ url: URL) throws {
        let path = url.standardizedFileURL.path
        if (try? FileManager.default.destinationOfSymbolicLink(atPath: path)) != nil || FileManager.default.fileExists(atPath: path) {
            throw KleinError.invalidInput("\(path) already exists; existing results are never overwritten (use a new file name)")
        }
    }

    /// `requireNewFile`, and additionally no symbolic link anywhere above the file (the report could otherwise land elsewhere).
    public static func requireNewFileWithoutSymlinkParents(_ url: URL) throws {
        try requireNewFile(url)
        let parent = url.deletingLastPathComponent()
        guard parent.standardizedFileURL.path == parent.resolvingSymlinksInPath().path else {
            throw KleinError.invalidInput("\(url.path) has a symbolic link in its parent path; pass the real path")
        }
    }

    public static func read<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: url))
    }
}
