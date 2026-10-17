import FluxKlein
import Foundation
import MLX

public struct KleinTransformerManifestCase: Codable, Equatable, Sendable {
    public var id: String
    public var category: String
    public var prompt: String
    public var renderedText: String
    public var width: Int
    public var height: Int
    public var seed: Int
    public var steps: Int
    public var timestepIndex: Int
    public var imageSeqLen: Int
    public var gridHeight: Int
    public var gridWidth: Int
    public var numValid: Int
    public var untruncatedCount: Int
    /// The tensors the producer saved to `<id>.safetensors`.
    public var tensors: [String: KleinTensorInfo]
    enum CodingKeys: String, CodingKey {
        case id, category, prompt, width, height, seed, steps, tensors
        case renderedText = "rendered_text", timestepIndex = "timestep_index", imageSeqLen = "image_seq_len", gridHeight = "grid_height"
        case gridWidth = "grid_width", numValid = "num_valid", untruncatedCount = "untruncated_count"
    }
}

/// One side's run of the slice-2 check, written by `klein-native transformer-run` (side `native`), `transformer-isolated` (side
/// `isolated`) and `klein/transformer/reference/export_klein_transformer.py` (side `reference`).
public struct KleinTransformerManifest: Codable, Sendable {
    public var schema: Int
    public var side: String
    public var casesSHA256: String
    /// SHA-256 of the canonical rendering of the cases this side actually ran (`KleinTransformerCases.digest`).
    public var casesDigest: String
    public var budgetsSHA256: String
    /// Versions and environment, free-form strings (reference: mflux/mlx/python; native: mlx-swift, OS, build settings, binary hash).
    public var runtime: [String: String]
    /// Reference only: SHA-256 of the imported `.pyc` of each audited MFLUX module (28 of them).
    public var runtimeModules: [String: String]?
    public var modelFilesBefore: [String: String]
    public var modelFilesAfter: [String: String]
    /// Loaded tensor bytes: the 905 text-encoder parameters (not for `isolated`) and the 387 transformer parameters.
    public var textWeights: [String: KleinTensorDigest]
    public var transformerWeights: [String: KleinTensorDigest]
    public var textQuantizedModules: [KleinQuantizedModule]
    public var transformerQuantizedModules: [KleinQuantizedModule]
    public var cases: [KleinTransformerManifestCase]
    /// The seeded-latent / scheduler control tensors saved to `controls.safetensors` (empty for `isolated`).
    public var controlTensors: [String: KleinTensorInfo]
    enum CodingKeys: String, CodingKey {
        case schema, side, runtime, cases
        case casesSHA256 = "cases_sha256", casesDigest = "cases_digest", budgetsSHA256 = "budgets_sha256", runtimeModules = "runtime_modules"
        case modelFilesBefore = "model_files_before", modelFilesAfter = "model_files_after", textWeights = "text_weights"
        case transformerWeights = "transformer_weights", textQuantizedModules = "text_quantized_modules"
        case transformerQuantizedModules = "transformer_quantized_modules", controlTensors = "control_tensors"
    }
}

/// The cheap scalar / RNG controls: seeded latents, pack / unpack, grid IDs for the six catalog sizes with a fixed seed, and the
/// scheduler's sigmas and timesteps for every catalog size and every offered step count. Both sides build and save them independently.
public enum KleinTransformerControls {
    public static func key(_ width: Int, _ height: Int) -> String { "\(width)x\(height)" }

    /// Every control tensor name the comparison requires.
    public static func requiredKeys() -> [String] {
        var keys: [String] = []
        for size in KleinTransformerCases.controlSizes {
            let k = key(size.width, size.height)
            keys += ["lat4d.\(k)", "packed.\(k)", "ids.\(k)", "unpacked.\(k)"]
            for n in KleinTransformerCases.controlSteps { keys += ["sigmas.\(k).n\(n)", "timesteps.\(k).n\(n)"] }
        }
        return keys
    }

    /// Expected dtype and shape of a control tensor, derived from the size and step count in its name.
    public static func expectedLayout(key name: String) -> (dtype: String, shape: [Int])? {
        let parts = name.split(separator: ".").map(String.init)
        guard parts.count >= 2, let size = KleinTransformerCases.controlSizes.first(where: { key($0.width, $0.height) == parts[1] }) else { return nil }
        let (gh, gw) = (size.height / 16, size.width / 16)
        switch parts[0] {
        case "lat4d", "unpacked":
            guard parts.count == 2 else { return nil }
            return ("mlx.core.bfloat16", [1, 128, gh, gw])
        case "packed" where parts.count == 2: return ("mlx.core.bfloat16", [1, gh * gw, 128])
        case "ids" where parts.count == 2: return ("mlx.core.int32", [1, gh * gw, 4])
        case "sigmas", "timesteps":
            guard parts.count == 3, parts[2].hasPrefix("n"), let n = Int(parts[2].dropFirst()), KleinScheduler.offeredSteps.contains(n) else { return nil }
            return ("mlx.core.float32", [parts[0] == "sigmas" ? n + 1 : n])
        default: return nil
        }
    }

    /// Builds all control tensors with the native port.
    public static func make() throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for size in KleinTransformerCases.controlSizes {
            let k = key(size.width, size.height)
            let prepared = try KleinLatent.prepare(seed: KleinTransformerCases.controlSeed, height: size.height, width: size.width)
            let packed = try KleinLatent.pack(prepared.latents)
            result["lat4d.\(k)"] = prepared.latents
            result["packed.\(k)"] = packed
            result["ids.\(k)"] = prepared.ids
            result["unpacked.\(k)"] = try KleinLatent.unpack(packed, height: size.height, width: size.width)
            for n in KleinTransformerCases.controlSteps {
                let scheduler = try KleinScheduler(steps: n, height: size.height, width: size.width)
                result["sigmas.\(k).n\(n)"] = scheduler.sigmas
                result["timesteps.\(k).n\(n)"] = scheduler.timesteps
            }
        }
        return result
    }
}
