import Foundation
import MLX
import MLXImageCore

// MARK: Reference artifacts

struct ReferenceManifest: Decodable {
    struct Case: Decodable {
        let id: String
        let category: String
        let prompt: String
        let renderedText: String
        let numValid: Int
        let idsDtype: String
        let idsShape: [Int]
        let maskDtype: String
        let maskShape: [Int]
        let embeddingDtype: String?
        let embeddingShape: [Int]?
        enum CodingKeys: String, CodingKey {
            case id, category, prompt
            case renderedText = "rendered_text", numValid = "num_valid", idsDtype = "ids_dtype", idsShape = "ids_shape"
            case maskDtype = "mask_dtype", maskShape = "mask_shape", embeddingDtype = "embedding_dtype", embeddingShape = "embedding_shape"
        }
    }
    let cases: [Case]
    let mlxPythonVersion: String?
    let tokenizerFiles: [String: String]
    enum CodingKeys: String, CodingKey {
        case cases, mlxPythonVersion = "mlx_python_version", tokenizerFiles = "tokenizer_files"
    }
}

public struct CaseResult: Codable, Sendable {
    public var id: String
    public var category: String
    public var tokenIDsEqual: Bool
    public var maskEqual: Bool
    public var shapesAndDtypesEqual: Bool
    public var renderedTextEqual: Bool
    public var firstTokenMismatch: String?
    public var validTokens: Int
    public var embedding: ComparisonMetrics?
    public var embeddingSkipped: String?
    public var passed: Bool
}

public struct ParityReport: Codable, Sendable {
    public var tolerances: [String: Double]
    public var mlxSwiftRevision: String
    public var referenceMLXPythonVersion: String?
    public var cases: [CaseResult]
    public var passed: Bool
}

public enum ParityRunner {
    /// Compares native tokenization and text embeddings with the reference artifacts in `referenceDirectory`.
    /// A case passes only if token IDs, masks, shapes and dtypes are exactly equal and the embedding is within tolerance.
    public static func run(
        modelDirectory: URL, referenceDirectory: URL, casesJSON: URL? = nil, mlxSwiftRevision: String, log: (String) -> Void = { _ in }
    ) throws -> ParityReport {
        let manifest = try JSONDecoder().decode(
            ReferenceManifest.self, from: Data(contentsOf: referenceDirectory.appendingPathComponent("manifest.json")))
        if let casesJSON {
            try checkCaseList(manifestIDs: manifest.cases.map(\.id), categories: Set(manifest.cases.map(\.category)), casesJSON: casesJSON)
        }
        let tokenizer = try QwenTokenizer.load(modelDirectory: modelDirectory)
        for (name, hash) in manifest.tokenizerFiles where QwenTokenizer.pinnedFileHashes[name] != hash {
            throw TokenizerError.invalid("reference was exported with a different \(name) than the pinned one")
        }
        let encoder = TextEncoder(weights: try TextEncoderWeights.load(modelDirectory: modelDirectory))
        var results: [CaseResult] = []
        for c in manifest.cases {
            let ref = try loadArrays(url: referenceDirectory.appendingPathComponent("\(c.id).safetensors"))
            let tokens = tokenizer.tokenize(c.prompt)
            let refIDs = ref["input_ids"]!.asArray(Int64.self)
            let refMask = ref["attention_mask"]!.asArray(Int64.self)
            var result = CaseResult(
                id: c.id, category: c.category, tokenIDsEqual: tokens.inputIDs == refIDs, maskEqual: tokens.attentionMask == refMask,
                shapesAndDtypesEqual: tokens.shape == c.idsShape && tokens.shape == c.maskShape && tokens.dtype == c.idsDtype
                    && tokens.dtype == c.maskDtype && tokens.validCount == c.numValid,
                renderedTextEqual: QwenTokenizer.renderChat(prompt: c.prompt) == c.renderedText || c.prompt.isEmpty,
                firstTokenMismatch: nil, validTokens: tokens.validCount, embedding: nil, embeddingSkipped: nil, passed: false)
            if let i = zip(tokens.inputIDs, refIDs).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset {
                result.firstTokenMismatch = "index \(i): native \(tokens.inputIDs[i]) reference \(refIDs[i])"
            }
            if let refEmbedding = ref["embedding"] {
                let native = try encoder.encode(tokens)
                let sameShape = native.shape == refEmbedding.shape
                let sameDtype = c.embeddingDtype == dtypeName(native.dtype) && refEmbedding.dtype == native.dtype
                result.shapesAndDtypesEqual = result.shapesAndDtypesEqual && sameShape && sameDtype
                if sameShape {
                    result.embedding = ParityMetrics.compare(
                        native: native.asType(.float32).asArray(Float.self), reference: refEmbedding.asType(.float32).asArray(Float.self))
                }
            } else if embeddingMayBeSkipped(
                prompt: c.prompt, nativeTokensEmpty: tokens.inputIDs.isEmpty, manifestShape: c.embeddingShape, manifestDtype: c.embeddingDtype)
            {
                result.embeddingSkipped = "reference has no embedding (empty prompt: the reference text encoder is not run)"
            } else {
                result.embeddingSkipped = "FAIL: reference embedding missing for a case that requires one"
            }
            result.passed = caseVerdict(result)
            results.append(result)
            let e = result.embedding
            log("\(c.id): tokens \(result.tokenIDsEqual && result.maskEqual ? "equal" : "DIFFER") valid=\(result.validTokens) "
                + "relL2=\(e?.relativeL2.map { String(format: "%.3e", $0) } ?? "n/a") "
                + "normMax=\(e?.normalizedMaxAbs.map { String(format: "%.3e", $0) } ?? "n/a") \(result.passed ? "PASS" : "FAIL")")
        }
        return ParityReport(
            tolerances: [
                "embedding_relative_l2": ParityTolerance.embeddingRelativeL2,
                "embedding_normalized_max_abs": ParityTolerance.embeddingNormalizedMaxAbs,
                "zero_reference_threshold": ParityTolerance.zeroReference,
                "zero_reference_absolute_limit": ParityTolerance.zeroFallbackAbsolute,
            ],
            mlxSwiftRevision: mlxSwiftRevision, referenceMLXPythonVersion: manifest.mlxPythonVersion, cases: results,
            passed: results.allSatisfy(\.passed))
    }

    /// The only legitimate absence of a reference embedding: an empty prompt, empty native tokens, and a manifest that
    /// itself declares no embedding. Anything else is an exporter failure and must fail the case.
    static func embeddingMayBeSkipped(prompt: String, nativeTokensEmpty: Bool, manifestShape: [Int]?, manifestDtype: String?) -> Bool {
        prompt.isEmpty && nativeTokensEmpty && manifestShape == nil && manifestDtype == nil
    }

    static func caseVerdict(_ r: CaseResult) -> Bool {
        let embeddingOK: Bool
        if let e = r.embedding { embeddingOK = e.passed } else { embeddingOK = r.embeddingSkipped?.hasPrefix("reference has no embedding") == true }
        return r.tokenIDsEqual && r.maskEqual && r.shapesAndDtypesEqual && r.renderedTextEqual && embeddingOK
    }

    /// The manifest must list exactly the case IDs of `cases.json`, in order, covering every category.
    static func checkCaseList(manifestIDs: [String], categories: Set<String>, casesJSON: URL) throws {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: casesJSON)) as? [String: Any],
              let cases = root["cases"] as? [[String: Any]] else { throw TokenizerError.invalid("unreadable cases.json") }
        let expected = cases.compactMap { $0["id"] as? String }
        guard expected == manifestIDs else {
            throw TokenizerError.invalid("reference case list \(manifestIDs) differs from cases.json \(expected)")
        }
        guard categories.isSuperset(of: ["plain", "long", "unicode"]) else {
            throw TokenizerError.invalid("reference lacks a plain, long or unicode category")
        }
    }

    static func dtypeName(_ dtype: DType) -> String {
        switch dtype {
        case .bfloat16: "mlx.core.bfloat16"
        case .float32: "mlx.core.float32"
        case .int64: "mlx.core.int64"
        case .int32: "mlx.core.int32"
        default: "mlx.core.\(dtype)"
        }
    }
}

extension ParityRunner {
    /// Diagnostic only (never decides parity): compares native float32 hidden states with the reference's, stage by stage.
    public static func diagnose(modelDirectory: URL, diagnosticFile: URL, prompt: String) throws -> [(stage: String, metrics: ComparisonMetrics)] {
        let ref = try loadArrays(url: diagnosticFile)
        let tokens = try QwenTokenizer.load(modelDirectory: modelDirectory).tokenize(prompt)
        let encoder = TextEncoder(weights: try TextEncoderWeights.load(modelDirectory: modelDirectory))
        var native: [String: MLXArray] = [:]
        _ = encoder.hiddenStates(
            inputIDs: MLXArray(tokens.inputIDs).reshaped(1, -1), attentionMask: MLXArray(tokens.attentionMask).reshaped(1, -1),
            trace: { name, array in if ref[name] != nil { eval(array); native[name] = array } })
        let layer0 = ["ln1", "q_proj", "k_proj", "v_proj", "q_norm", "q_rope", "sdpa", "attn", "resid1", "ln2", "mlp"].map { "l0.\($0)" }
        let order = ["embed"] + layer0 + (0..<TextEncoderLayout.layerCount).map { "layer\($0)" }
        // A reference key `<stage>@cpu` holds the same stage computed on the reference's CPU stream.
        let keys = order.flatMap { [$0, $0 + "@cpu"] }
        return keys.compactMap { key in
            guard let n = native[key.replacingOccurrences(of: "@cpu", with: "")], let r = ref[key] else { return nil }
            return (key, ParityMetrics.compare(native: n.asArray(Float.self), reference: r.asArray(Float.self)))
        }
    }
}
