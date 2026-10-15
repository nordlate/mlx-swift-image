import Foundation

/// The frozen prompt corpus (`klein/cases.json`). Both sides expand it independently; the comparison requires the expanded prompts
/// to be identical, so an expansion bug on either side cannot pass.
public struct KleinCaseSpec: Equatable, Sendable {
    public let id: String
    public let category: String
    public let prompt: String
}

public enum KleinCases {
    /// Categories the corpus must cover: plain, whitespace, empty, long/truncated, boundary, Unicode (combining, emoji, RTL, CJK) and literal specials.
    public static let requiredCategories: Set<String> = ["plain", "whitespace", "empty", "long", "boundary", "unicode", "special"]
    /// Template-rendered token counts that must each occur: just under, at and just over the 512-token limit.
    public static let requiredBoundaryCounts: Set<Int> = [511, 512, 513]

    public static func load(url: URL) throws -> (cases: [KleinCaseSpec], sha256: String) {
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any], let entries = root["cases"] as? [[String: Any]] else {
            throw KleinError.invalidInput("unreadable cases file \(url.lastPathComponent)")
        }
        let cases = try parse(entries)
        return (cases, KleinHashing.sha256(of: data))
    }

    /// `prompt`, or `prefix + separator.join(unit * repeat) + suffix`; ids must be unique and every case needs a category.
    static func parse(_ entries: [[String: Any]]) throws -> [KleinCaseSpec] {
        var seen = Set<String>()
        return try entries.map { entry in
            guard let id = entry["id"] as? String, !id.isEmpty, let category = entry["category"] as? String, !category.isEmpty else {
                throw KleinError.invalidInput("case without id or category: \(entry)")
            }
            guard seen.insert(id).inserted else { throw KleinError.invalidInput("duplicate case id \(id)") }
            if let prompt = entry["prompt"] as? String { return KleinCaseSpec(id: id, category: category, prompt: prompt) }
            guard let unit = entry["unit"] as? String, let count = entry["repeat"] as? Int, count >= 0, let separator = entry["separator"] as? String else {
                throw KleinError.invalidInput("case \(id) has neither prompt nor unit/repeat/separator")
            }
            let body = [String](repeating: unit, count: count).joined(separator: separator)
            return KleinCaseSpec(id: id, category: category, prompt: (entry["prefix"] as? String ?? "") + body + (entry["suffix"] as? String ?? ""))
        }
    }
}

public struct KleinTensorInfo: Codable, Equatable, Sendable {
    public var dtype: String
    public var shape: [Int]
}

public struct KleinQuantizedModule: Codable, Equatable, Sendable {
    public var name: String
    public var bits: Int
    public var groupSize: Int
    public var mode: String
    enum CodingKeys: String, CodingKey { case name, bits, groupSize = "group_size", mode }
}

public struct KleinManifestCase: Codable, Equatable, Sendable {
    public var id: String
    public var category: String
    public var prompt: String
    public var renderedText: String
    public var numValid: Int
    /// Token count of the rendered template before truncation (0 for the empty prompt, which bypasses the template).
    public var untruncatedCount: Int
    public var idsDtype: String
    public var idsShape: [Int]
    public var maskDtype: String
    public var maskShape: [Int]
    /// `ok` for every case (the empty prompt included, with zero-length tensors); anything else (`error: <type>: <message>`) fails the comparison.
    public var encoderOutcome: String
    /// The tensors the producer saved to `<id>.safetensors`.
    public var tensors: [String: KleinTensorInfo]
    enum CodingKeys: String, CodingKey {
        case id, category, prompt, tensors
        case renderedText = "rendered_text", numValid = "num_valid", untruncatedCount = "untruncated_count"
        case idsDtype = "ids_dtype", idsShape = "ids_shape", maskDtype = "mask_dtype", maskShape = "mask_shape"
        case encoderOutcome = "encoder_outcome"
    }
}

/// One side's run. Written by `klein-native encode` (side `native`) and by `klein/reference/export_klein_text.py` (side `reference`).
public struct KleinManifest: Codable, Sendable {
    public var schema: Int
    public var side: String
    public var casesSHA256: String
    /// Versions and environment, free-form strings (reference: mflux/mlx/python/transformers/tokenizers; native: mlx-swift, OS, build settings).
    public var runtime: [String: String]
    /// Reference only: SHA-256 of the imported `.pyc` of each audited MFLUX module.
    public var runtimeModules: [String: String]?
    public var modelFilesBefore: [String: String]
    public var modelFilesAfter: [String: String]
    /// Loaded tensor bytes per stored name: as MFLUX holds them after applying the snapshot, or as the native loader holds them.
    public var weights: [String: KleinTensorDigest]
    public var quantizedModules: [KleinQuantizedModule]
    public var cases: [KleinManifestCase]
    enum CodingKeys: String, CodingKey {
        case schema, side, runtime, weights, cases
        case casesSHA256 = "cases_sha256", runtimeModules = "runtime_modules", modelFilesBefore = "model_files_before"
        case modelFilesAfter = "model_files_after", quantizedModules = "quantized_modules"
    }
}
