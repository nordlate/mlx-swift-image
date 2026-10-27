import CryptoKit
import Foundation
import MLXImageCore

/// An official FLUX.2 Klein bf16 diffusers snapshot that `KleinConverter` can convert, as `KleinSourceSnapshot.inspect` found it.
public struct KleinSourceSnapshot: Sendable {
    public let directory: URL
    public let variant: KleinVariant
    public let release: KleinOfficialRelease
    /// True when every weight file was hashed against the release (`inspect(..., verifyingWeightHashes: true)`). The small files
    /// (configs, indexes, tokenizer) are always hashed; `KleinConverter.convert` hashes the weight files while it reads them.
    public let weightHashesVerified: Bool

    /// Bytes of the weight files, which a conversion or a full verification reads once.
    public var weightByteCount: Int64 { release.weightFiles.reduce(0) { $0 + $1.byteCount } }

    /// Component folders whose every file must belong to the release.
    static let componentFolders = ["transformer", "text_encoder", "vae", "tokenizer", "scheduler"]

    /// Decides whether `directory` is a FLUX.2 Klein snapshot this library can convert, and which variant, without reading weights:
    ///
    /// 1. `model_index.json` names `Flux2KleinPipeline` with the Klein components and `is_distilled: true` (else `.notKlein` or, for a
    ///    base model, `.unsupportedVariant`);
    /// 2. the transformer config's depth, heads and joint attention width select the variant, and the transformer and text-encoder
    ///    configs must agree with it (`.unsupportedVariant` for an unknown size or one without conversion pins, `.invalidFile` for a
    ///    disagreeing config);
    /// 3. every required file of the official release is present (`.missingFile`) and the component folders hold nothing else
    ///    (`.unexpectedFile`; hidden files are ignored, as are other files at the top level);
    /// 4. every required file has the release's size (`.sizeMismatch`) and every small file its SHA-256 (`.hashMismatch`);
    /// 5. the weight files' safetensors headers hold exactly the tensors the conversion expects, as bf16 with the expected shapes
    ///    (`.invalidFile`).
    ///
    /// With `verifyingWeightHashes`, the weight files (34.5 GB for 9B) are hashed too; `isCancelled` is polled while hashing.
    public static func inspect(
        directory: URL, verifyingWeightHashes: Bool = false, isCancelled: (@Sendable () -> Bool)? = nil
    ) throws -> KleinSourceSnapshot {
        try inspect(directory: directory, pinnedRelease: nil, verifyingWeightHashes: verifyingWeightHashes, isCancelled: isCancelled)
    }

    /// `pinnedRelease` replaces the variant's official pins (tests use synthetic folders).
    static func inspect(
        directory: URL, pinnedRelease: KleinOfficialRelease?, verifyingWeightHashes: Bool = false, isCancelled: (@Sendable () -> Bool)? = nil
    ) throws -> KleinSourceSnapshot {
        let variant = try detectVariant(directory: directory)
        guard let release = pinnedRelease ?? KleinOfficialRelease.release(for: variant) else {
            throw KleinConversionError.unsupportedVariant("\(variant.displayName): conversion is available for \(KleinConverter.supportedVariants.map(\.displayName).joined(separator: ", "))")
        }
        try checkFileSet(directory: directory, release: release)
        for file in release.requiredFiles {
            let size = try fileSize(directory.appendingPathComponent(file.path))
            guard size == file.byteCount else { throw KleinConversionError.sizeMismatch(file: file.path, expected: file.byteCount, actual: size) }
        }
        for file in release.requiredFiles where !file.path.hasSuffix(".safetensors") {
            let actual = try KleinHashing.sha256(of: directory.appendingPathComponent(file.path))
            guard actual == file.sha256 else { throw KleinConversionError.hashMismatch(file: file.path, expected: file.sha256, actual: actual) }
        }
        try checkWeightHeaders(directory: directory, release: release, plan: KleinSnapshotPlan(variant: variant))
        if verifyingWeightHashes {
            let cancelled = isCancelled ?? { false }
            let files = release.weightFiles
            let failure = FirstError()
            // Smallest files first, so a mismatch in a small file stops the large ones early.
            let order = files.indices.sorted { files[$0].byteCount < files[$1].byteCount }
            DispatchQueue.concurrentPerform(iterations: files.count) { n in
                let file = files[order[n]]
                do {
                    let mapped = try KleinMappedFile(url: directory.appendingPathComponent(file.path))
                    var hasher = SHA256()
                    try KleinConverter.hash(mapped, 0..<mapped.byteCount, into: &hasher, isCancelled: { failure.isSet || cancelled() })
                    let actual = KleinHashing.hex(hasher.finalize())
                    if actual != file.sha256 { throw KleinConversionError.hashMismatch(file: file.path, expected: file.sha256, actual: actual) }
                } catch {
                    failure.record(cancelled() ? KleinConversionError.cancelled : error)
                }
            }
            if cancelled() { throw KleinConversionError.cancelled }
            if let error = failure.error { throw error }
        }
        return KleinSourceSnapshot(directory: directory, variant: variant, release: release, weightHashesVerified: verifyingWeightHashes)
    }

    // MARK: Steps

    static func detectVariant(directory: URL) throws -> KleinVariant {
        let indexURL = directory.appendingPathComponent("model_index.json")
        guard FileManager.default.fileExists(atPath: indexURL.path) else { throw KleinConversionError.notKlein("no model_index.json") }
        let index = try json(indexURL, "model_index.json")
        guard let className = index["_class_name"] as? String, className == "Flux2KleinPipeline" else {
            throw KleinConversionError.notKlein("model_index.json names pipeline \(index["_class_name"] as? String ?? "none"), not Flux2KleinPipeline")
        }
        let components = [("transformer", "Flux2Transformer2DModel"), ("text_encoder", "Qwen3ForCausalLM"), ("vae", "AutoencoderKLFlux2"),
                          ("tokenizer", "Qwen2TokenizerFast")]
        for (key, wanted) in components {
            guard let entry = index[key] as? [String], entry.last == wanted else {
                throw KleinConversionError.notKlein("model_index.json \(key) is \(index[key].map { "\($0)" } ?? "missing"), expected \(wanted)")
            }
        }
        guard index["is_distilled"] as? Bool == true else {
            throw KleinConversionError.unsupportedVariant("a FLUX.2 Klein base (non-distilled) model; only the distilled models are supported")
        }
        let transformerPath = "transformer/config.json", textPath = "text_encoder/config.json"
        for path in [transformerPath, textPath] where !FileManager.default.fileExists(atPath: directory.appendingPathComponent(path).path) {
            throw KleinConversionError.missingFile(path)
        }
        let t = try json(directory.appendingPathComponent(transformerPath), transformerPath)
        func int(_ d: [String: Any], _ k: String) -> Int? { (d[k] as? NSNumber)?.intValue }
        func double(_ d: [String: Any], _ k: String) -> Double? { (d[k] as? NSNumber)?.doubleValue }
        let size = (int(t, "num_layers"), int(t, "num_single_layers"), int(t, "num_attention_heads"), int(t, "joint_attention_dim"))
        guard let variant = KleinVariant.allCases.first(where: {
            let l = $0.transformer
            return size == (l.doubleBlocks, l.singleBlocks, l.heads, l.jointAttentionDim)
        }) else {
            throw KleinConversionError.unsupportedVariant("a transformer with \(size.0.map(String.init) ?? "?") double and \(size.1.map(String.init) ?? "?") single blocks, \(size.2.map(String.init) ?? "?") heads and joint attention width \(size.3.map(String.init) ?? "?")")
        }
        let transformerChecks: [(String, Bool)] = [
            ("attention_head_dim", int(t, "attention_head_dim") == KleinTransformerLayout.headDim),
            ("in_channels", int(t, "in_channels") == KleinTransformerLayout.inChannels),
            ("timestep_guidance_channels", int(t, "timestep_guidance_channels") == KleinTransformerLayout.timestepChannels),
            ("mlp_ratio", double(t, "mlp_ratio") == 3.0),
            ("rope_theta", double(t, "rope_theta") == Double(KleinTransformerLayout.ropeTheta)),
            ("axes_dims_rope", (t["axes_dims_rope"] as? [Int]) == KleinTransformerLayout.axesDims),
            ("guidance_embeds", t["guidance_embeds"] as? Bool == false),
            ("patch_size", int(t, "patch_size") == 1),
        ]
        if let bad = transformerChecks.first(where: { !$0.1 }) {
            throw KleinConversionError.invalidFile(file: transformerPath, reason: "\(bad.0) \(t[bad.0].map { "\($0)" } ?? "missing") does not match \(variant.displayName)")
        }
        let e = try json(directory.appendingPathComponent(textPath), textPath)
        let layout = variant.textEncoder
        let textChecks: [(String, Bool)] = [
            ("hidden_size", int(e, "hidden_size") == layout.hiddenSize),
            ("intermediate_size", int(e, "intermediate_size") == layout.intermediateSize),
            ("num_hidden_layers", int(e, "num_hidden_layers") == TextEncoderLayout.layerCount),
            ("num_attention_heads", int(e, "num_attention_heads") == TextEncoderLayout.attentionHeads),
            ("num_key_value_heads", int(e, "num_key_value_heads") == TextEncoderLayout.keyValueHeads),
            ("head_dim", int(e, "head_dim") == TextEncoderLayout.headDim),
            ("vocab_size", int(e, "vocab_size") == TextEncoderLayout.vocabSize),
            ("rope_theta", double(e, "rope_theta") == Double(TextEncoderLayout.ropeTheta)),
            ("model_type", e["model_type"] as? String == "qwen3"),
        ]
        if let bad = textChecks.first(where: { !$0.1 }) {
            throw KleinConversionError.invalidFile(file: textPath, reason: "\(bad.0) \(e[bad.0].map { "\($0)" } ?? "missing") does not match \(variant.displayName)")
        }
        return variant
    }

    static func checkFileSet(directory: URL, release: KleinOfficialRelease) throws {
        for file in release.requiredFiles where !FileManager.default.fileExists(atPath: directory.appendingPathComponent(file.path).path) {
            throw KleinConversionError.missingFile(file.path)
        }
        let known = Set(release.requiredFiles.map(\.path) + release.optionalFiles)
        let base = directory.standardizedFileURL.resolvingSymlinksInPath().path
        for folder in componentFolders {
            let url = directory.appendingPathComponent(folder)
            guard let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey],
                                                              options: [.skipsHiddenFiles]) else { continue }
            for case let file as URL in walker {
                guard (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { continue }
                let full = file.standardizedFileURL.resolvingSymlinksInPath().path
                let relative = full.hasPrefix(base + "/") ? String(full.dropFirst(base.count + 1)) : folder + "/" + file.lastPathComponent
                guard known.contains(relative) else { throw KleinConversionError.unexpectedFile(relative) }
            }
        }
    }

    /// Every weight file's header together must hold exactly the plan's sources plus the tensors `mflux-save` drops, as bf16 (the
    /// dropped `bn.num_batches_tracked` is int64) with the expected shapes, each tensor once.
    static func checkWeightHeaders(directory: URL, release: KleinOfficialRelease, plan: KleinSnapshotPlan) throws {
        for component in plan.components {
            var expected: [String: (dtype: String, shape: [Int])] = component.droppedSources
            for (name, shape) in component.sources { expected[name] = ("BF16", shape) }
            var seen = Set<String>()
            for file in release.weightFiles where file.path.hasPrefix(component.folder + "/") {
                let header: SafetensorsHeader
                do { header = try SafetensorsHeader.read(from: directory.appendingPathComponent(file.path)) } catch {
                    throw KleinConversionError.invalidFile(file: file.path, reason: "\(error)")
                }
                for (name, entry) in header.entries {
                    guard let want = expected[name] else {
                        throw KleinConversionError.invalidFile(file: file.path, reason: "unexpected tensor \(name)")
                    }
                    guard entry.dtype == want.dtype, entry.shape == want.shape else {
                        throw KleinConversionError.invalidFile(file: file.path, reason: "\(name) is \(entry.dtype) \(entry.shape), expected \(want.dtype) \(want.shape)")
                    }
                    guard seen.insert(name).inserted else {
                        throw KleinConversionError.invalidFile(file: file.path, reason: "\(name) is stored in more than one file")
                    }
                }
            }
            let missing = Set(expected.keys).subtracting(seen)
            guard missing.isEmpty else {
                throw KleinConversionError.invalidFile(file: component.folder, reason: "missing tensors \(missing.sorted().prefix(5))")
            }
        }
    }

    static func json(_ url: URL, _ name: String) throws -> [String: Any] {
        guard let data = try? Data(contentsOf: url), let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw KleinConversionError.invalidFile(file: name, reason: "not a JSON object")
        }
        return object
    }

    static func fileSize(_ url: URL) throws -> Int64 {
        do { return (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value ?? -1 } catch {
            throw KleinConversionError.io("cannot read the size of \(url.lastPathComponent)")
        }
    }
}
