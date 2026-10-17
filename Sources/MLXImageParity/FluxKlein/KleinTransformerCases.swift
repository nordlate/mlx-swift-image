import FluxKlein
import CryptoKit
import Foundation
import MLX

/// One frozen transformer-prediction case: the whole input of one denoising prediction (prompt, size, seed, step count) and
/// the scheduler step at which the transformer is evaluated.
public struct KleinTransformerCase: Equatable, Sendable {
    public let id: String
    public let category: String
    public let prompt: String
    public let width: Int
    public let height: Int
    public let seed: UInt64
    public let steps: Int
    public let timestepIndex: Int

    /// `(H / 16) * (W / 16)`: the packed latent sequence length.
    public var imageSeqLen: Int { (height / 16) * (width / 16) }
}

/// The frozen corpus of slice 2. The four primary cases are compiled in: `klein/transformer/cases.json` is the readable copy and
/// `check(casesJSON:)` fails when it differs. Their canonical digest is what the freeze record (`results/frozen-inputs.json`) pins
/// before the first real run; nothing is hardcoded as a hash here because the digest is derived from the table.
public enum KleinTransformerCases {
    public static let longPromptPrefix = "A very long list of lanterns: "
    public static let longPromptUnit = "lantern"
    public static let longPromptRepeat = 700

    public static let frozen: [KleinTransformerCase] = [
        KleinTransformerCase(
            id: "t1-square1024-plain", category: "plain",
            prompt: "A red bicycle leaning against a weathered brick wall, soft morning light, 35mm photograph",
            width: 1024, height: 1024, seed: 42, steps: 4, timestepIndex: 0),
        KleinTransformerCase(
            id: "t2-portrait672x896-unicode", category: "unicode",
            prompt: "Hämärä syysilta järven rannalla, sumua ja kuusimetsää. 湖畔の静かな山小屋、夕暮れの霧 Ääkköset: äöå ÄÖÅ",
            width: 672, height: 896, seed: 7, steps: 8, timestepIndex: 3),
        KleinTransformerCase(
            id: "t3-landscape1152x864-long", category: "long",
            prompt: longPromptPrefix + [String](repeating: longPromptUnit, count: longPromptRepeat).joined(separator: ", "),
            width: 1152, height: 864, seed: 31337, steps: 6, timestepIndex: 5),
        KleinTransformerCase(
            id: "t4-square768-short", category: "short", prompt: "cat",
            width: 768, height: 768, seed: 0, steps: 1, timestepIndex: 0),
    ]

    /// The six catalog sizes (`ImageSize.standardSizes`, width x height) and the offered step counts the cheap controls cover.
    public static let controlSizes: [(width: Int, height: Int)] = [(1024, 1024), (768, 768), (864, 1152), (672, 896), (1152, 864), (896, 672)]
    public static let controlSeed: UInt64 = 42
    public static let controlSteps = KleinScheduler.offeredSteps

    /// Values observed by the orchestrator in production scheduler preflight for 1024x1024 / 4 steps (recorded before this slice's
    /// native code was run): the float32 sigmas and timesteps, stated as the doubles Python printed.
    public static let knownSigmas1024Steps4: [Float] = [1, 0.9673840403556824, 0.908143937587738, 0.7671999335289001, 0]
    public static let knownTimesteps1024Steps4: [Float] = [1000, 967.384033203125, 908.1439208984375, 767.199951171875]

    /// SHA-256 over a canonical rendering of the cases (every field, the prompt by its UTF-8 SHA-256): equal tables, equal digest.
    public static func digest(of cases: [KleinTransformerCase]) -> String {
        let lines = cases.map { c in
            [c.id, c.category, "\(c.width)", "\(c.height)", "\(c.seed)", "\(c.steps)", "\(c.timestepIndex)", KleinHashing.sha256(of: Data(c.prompt.utf8))]
                .joined(separator: "|")
        }
        return KleinHashing.sha256(of: Data(lines.joined(separator: "\n").utf8))
    }

    public static var compiledDigest: String { digest(of: frozen) }

    /// Compiled-in structure rules: unique ids, offered steps, an index inside the schedule, sizes the catalog offers.
    public static func structuralFailures(_ cases: [KleinTransformerCase]) -> [String] {
        var failures: [String] = []
        let ids = cases.map(\.id)
        if Set(ids).count != ids.count { failures.append("duplicate case ids") }
        for c in cases {
            if !KleinScheduler.offeredSteps.contains(c.steps) { failures.append("\(c.id): steps \(c.steps) not offered") }
            if !(0..<max(c.steps, 1)).contains(c.timestepIndex) { failures.append("\(c.id): timestep index \(c.timestepIndex) outside the \(c.steps) steps") }
            if (try? KleinLatent.grid(height: c.height, width: c.width)) == nil { failures.append("\(c.id): size \(c.width)x\(c.height) invalid") }
            if !controlSizes.contains(where: { $0.width == c.width && $0.height == c.height }) { failures.append("\(c.id): size \(c.width)x\(c.height) is not a catalog size") }
        }
        return failures
    }

    /// Parses a cases file: slice 1's prompt forms (`prompt`, or `unit`/`repeat`/`separator`/`prefix`/`suffix`) plus `width`, `height`,
    /// `seed`, `steps` and `timestep_index` per case.
    public static func parse(_ data: Data) throws -> [KleinTransformerCase] {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any], let entries = root["cases"] as? [[String: Any]] else {
            throw KleinTransformerError.invalidInput("unreadable transformer cases file")
        }
        let prompts = try KleinCases.parse(entries)
        return try zip(entries, prompts).map { entry, spec in
            guard let width = entry["width"] as? Int, let height = entry["height"] as? Int, let seed = entry["seed"] as? Int, seed >= 0,
                  let steps = entry["steps"] as? Int, let index = entry["timestep_index"] as? Int
            else { throw KleinTransformerError.invalidInput("case \(spec.id) lacks width, height, seed, steps or timestep_index") }
            return KleinTransformerCase(id: spec.id, category: spec.category, prompt: spec.prompt, width: width, height: height, seed: UInt64(seed), steps: steps, timestepIndex: index)
        }
    }

    /// Loads the cases file and requires it to be exactly the compiled table (byte-level prompt equality, not Unicode equivalence).
    public static func load(url: URL) throws -> (cases: [KleinTransformerCase], sha256: String) {
        let data = try Data(contentsOf: url)
        let cases = try parse(data)
        let failures = differences(cases, from: frozen)
        guard failures.isEmpty else { throw KleinTransformerError.invalidInput("cases file differs from the frozen table: \(failures.joined(separator: "; "))") }
        return (cases, KleinHashing.sha256(of: data))
    }

    public static func differences(_ cases: [KleinTransformerCase], from expected: [KleinTransformerCase]) -> [String] {
        var failures: [String] = []
        if cases.map(\.id) != expected.map(\.id) { failures.append("case ids \(cases.map(\.id)) differ from \(expected.map(\.id))") }
        for (a, b) in zip(cases, expected) where a.id == b.id {
            if Array(a.prompt.utf8) != Array(b.prompt.utf8) { failures.append("\(a.id): prompt differs") }
            if a.category != b.category || a.width != b.width || a.height != b.height || a.seed != b.seed || a.steps != b.steps || a.timestepIndex != b.timestepIndex {
                failures.append("\(a.id): parameters differ")
            }
        }
        return failures
    }
}

/// Limits for slice 2, declared before any native or reference transformer result existed and compiled in: nothing a run reads can
/// loosen them. `klein/transformer/budgets.json` is the readable copy and `check(budgetsJSON:)` fails when it differs. They are the
/// rows of the plan's tolerance table: relative L2 decides (max abs is recorded as a diagnostic, never a tolerance); a reference
/// whose norm is effectively zero falls back to absolute L2 and max abs of at most `zeroReferenceAbsolute`; non-finite values, shape
/// and dtype mismatches fail.
public enum KleinTransformerBudgets {
    /// Embedding, timestep embedding, modulation parameters and rotary tables.
    public static let embeddingRelativeL2 = 1e-3
    /// One transformer block (and the final AdaLN) from the reference's own captured input.
    public static let singleBlockRelativeL2 = 1e-3
    /// The full compiled prediction: native compiled against the production path's compiled `_predict`.
    public static let fullPredictionRelativeL2 = 2e-3
    /// The free-running chain of scheduler steps with synthetic noise.
    public static let noiseChainRelativeL2 = 5e-3
    /// Slice 1's text-embedding budget for the independently re-encoded prompt embeddings (the text path is unchanged).
    public static let promptEmbedsRelativeL2 = 1e-3
    public static let promptEmbedsNormalizedMaxAbs = 1e-2
    public static let zeroReferenceAbsolute = 1e-6

    public static let embedding = ParityLimitsRow(relativeL2: embeddingRelativeL2, normalizedMaxAbs: nil)
    public static let singleBlock = ParityLimitsRow(relativeL2: singleBlockRelativeL2, normalizedMaxAbs: nil)
    public static let fullPrediction = ParityLimitsRow(relativeL2: fullPredictionRelativeL2, normalizedMaxAbs: nil)
    public static let noiseChain = ParityLimitsRow(relativeL2: noiseChainRelativeL2, normalizedMaxAbs: nil)
    public static let promptEmbeds = ParityLimitsRow(relativeL2: promptEmbedsRelativeL2, normalizedMaxAbs: promptEmbedsNormalizedMaxAbs)

    /// Compared byte for byte: everything seeded or scheduled, the token and position IDs, and the loaded weight bytes.
    public static let exactDescription = [
        "latents", "latent_ids", "sigmas", "timesteps", "text_ids", "timestep_scaled", "chain_noise.*", "chain.0",
        "controls (latents, pack/unpack, grid IDs, sigmas, timesteps)", "isolated scheduler step from reference input", "loaded weight SHA-256",
    ]

    public static var asReport: [String: Double] {
        [
            "embedding_relative_l2": embeddingRelativeL2, "single_block_relative_l2": singleBlockRelativeL2,
            "full_prediction_relative_l2": fullPredictionRelativeL2, "noise_chain_relative_l2": noiseChainRelativeL2,
            "prompt_embeds_relative_l2": promptEmbedsRelativeL2, "prompt_embeds_normalized_max_abs": promptEmbedsNormalizedMaxAbs,
            "zero_reference_absolute": zeroReferenceAbsolute,
        ]
    }

    /// The frozen file must say exactly what the compiled constants say.
    public static func check(budgetsJSON url: URL) throws -> [String] {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
              let declared = root["relative_l2"] as? [String: Double], let zero = root["zero_reference_absolute"] as? Double,
              let nonFinite = root["non_finite"] as? String, let exact = root["exact"] as? [String]
        else { return ["budgets file is not in the expected format"] }
        var failures: [String] = []
        let want: [String: Double] = [
            "embedding": embeddingRelativeL2, "single_block": singleBlockRelativeL2, "full_prediction": fullPredictionRelativeL2,
            "noise_chain": noiseChainRelativeL2, "prompt_embeds": promptEmbedsRelativeL2,
        ]
        if declared != want { failures.append("relative_l2 \(declared) differs from the compiled \(want)") }
        if root["prompt_embeds_normalized_max_abs"] as? Double != promptEmbedsNormalizedMaxAbs { failures.append("prompt_embeds_normalized_max_abs differs") }
        if zero != zeroReferenceAbsolute { failures.append("zero_reference_absolute \(zero) differs from \(zeroReferenceAbsolute)") }
        if nonFinite != "fail" { failures.append("non_finite must be \"fail\"") }
        if exact != exactDescription { failures.append("exact list differs from the compiled one") }
        return failures
    }
}

/// A relative-L2 row (optionally with a normalized max abs) of the budget table.
public struct ParityLimitsRow: Equatable, Sendable {
    public let relativeL2: Double
    public let normalizedMaxAbs: Double?
}
