import FluxKlein
import Foundation

/// What the text-encoding parity run pins about the snapshot and the MFLUX runtime it is compared with. Nothing here is read on the
/// inference path; the loaders only check the stored quantization metadata (`KleinPins.storedQuantizationLevel`).
public extension KleinPins {
    /// Must equal Package.swift's mlxSwiftRevision (scripts/verify-pins.sh checks the resolved checkout).
    public static let mlxSwiftRevision = "19601207e9a0de51e03ee6ec0c3c5f3784275075"

    /// The production reference runtime the measurement was taken with. The comparison fails on any other version, even if both
    /// sides agree with each other.
    public static let referenceMfluxVersion = "0.20.0"
    public static let referenceMLXVersion = "0.32.2"

    /// The frozen corpus (`klein/cases.json`): its SHA-256 and case count were recorded before any measurement. A changed corpus fails
    /// the comparison even when both sides ran the same changed file.
    public static let frozenCasesSHA256 = "d1df6e4c85a4db951f9818162875a1197a781b03b285f3365fac047c0249f162"
    public static let frozenCaseCount = 24

    /// Model-relative path to SHA-256 of every file the text-encoding slice reads. The unchanged-artifact check recomputes these
    /// before and after a run on both sides.
    public static let modelFiles: [String: String] = [
        "tokenizer/tokenizer.json": tokenizerFiles["tokenizer.json"]!,
        "tokenizer/tokenizer_config.json": tokenizerFiles["tokenizer_config.json"]!,
        "tokenizer/chat_template.jinja": tokenizerFiles["chat_template.jinja"]!,
        "text_encoder/0.safetensors": "e59ed2b6825cb8ae3d0ba1e8a43575a565d465ce4518b7d79d28020b19b639cd",
        "text_encoder/1.safetensors": "edbdea3187bc4a0ad0e1ecb7152da8e6b344669413d9e45be5e7a2244f398ee3",
        "text_encoder/model.safetensors.index.json": "7dc8598806b8bbdb0b87b9dc445ac7cf49d7acc42ae333bd0bc0252e46081ed9",
    ]

    /// SHA-256 of the runtime `.pyc` of the nine modules whose normalized bytecode was proven identical to the readable source
    /// the port was written from (`reference-code-audit.json`). The reference exporter reports the hashes of the files it imported.
    public static let runtimeModules: [String: String] = [
        "mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder": "a6a062594287647edd2f00c6f69bfc93564b1bef4c2fe9651538f646bba70f1d",
        "mflux.models.flux2.model.flux2_text_encoder.prompt_encoder": "23fd537e2b8ee3ecd1816580eaf00f86845959ae9d8fd1bec33c1916f7ebdfef",
        "mflux.models.flux2.model.flux2_text_encoder.qwen3_text_rotary_embedding": "0de67a721d425e34857b7217853c690977f90dc23d516cc0327941df8f0e6e89",
        "mflux.models.common_models.qwen3_vl.qwen3_vl_decoder_layer": "e5d388ada09e2da8bc6b7772276aaddd1d867570bad86b7b34a3da3b4b32ab18",
        "mflux.models.common_models.qwen3_vl.qwen3_vl_attention": "1ebc91b6dfcc24bedf96a9cd94486f246875917f2db00d4f347325dfd9ab03b2",
        "mflux.models.common_models.qwen3_vl.qwen3_vl_mlp": "4e123fcc0d3523906f3810850e0ae85f53a25bb2fa67bfd74d95e7c7d398ecea",
        "mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm": "20cb4db3a8877a35f236a259455af4461e585e13ee5b62ab30fea49086ae205f",
        "mflux.models.flux2.weights.flux2_weight_definition": "9435dc853e24236c10f6edf9686a0917c64b8b6ca6f582caf9684bcc9723c8d7",
        "mflux.models.flux2.flux2_initializer": "df22711f808a83849328e2af2cd0c7f96a4a232e73fd9b0d3190e5de1040c428",
    ]
}

public extension KleinHashing {
    /// Hashes every pinned model file and throws, naming the file, when one differs from its pin.
    public static func verifyModelFiles(modelDirectory: URL) throws -> [String: String] {
        var result: [String: String] = [:]
        for (path, expected) in KleinPins.modelFiles.sorted(by: { $0.key < $1.key }) {
            let digest = try sha256(of: modelDirectory.appendingPathComponent(path))
            result[path] = digest
            guard digest == expected else {
                throw KleinError.artifactChanged("\(path): SHA-256 \(digest) differs from the pinned \(expected)")
            }
        }
        return result
    }
}
