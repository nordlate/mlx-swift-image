import CryptoKit
import Foundation

/// Everything slice 1 pins about the FLUX.2 Klein q4 snapshots (the tokenizer files are the same for 4B and 9B) and the bundled MFLUX runtime it is compared with. The values
/// were measured from the installed model (`model-before.json`) and the production runtime's bytecode (`reference-code-audit.json`)
/// before any native result existed. Nothing is inherited from the Z-Image pins: Klein's `tokenizer_config.json` differs.
public enum KleinPins {
    public static let tokenizerFiles: [String: String] = [
        "tokenizer.json": "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
        "tokenizer_config.json": "779410a34a98fb2c6c925a1be6f24d632f1723473161b037f6676eb6c04886c6",
        "chat_template.jinja": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    ]

    /// Stored `quantization_level` of every shard and index. The stored `mflux_version` is per variant (`KleinVariant.storedMfluxVersion`).
    public static let storedQuantizationLevel = "4"

}

public enum KleinHashing {
    public static func hex(_ digest: SHA256.Digest) -> String { digest.map { String(format: "%02x", $0) }.joined() }

    /// Streaming SHA-256 of a file (the shards are over 2 GB).
    public static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hex(hasher.finalize())
    }

    public static func sha256(of data: Data) -> String { hex(SHA256.hash(data: data)) }
}

public enum KleinError: Error, CustomStringConvertible {
    case artifactChanged(String)
    case invalidWeights(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .artifactChanged(let m): "model artifact: \(m)"
        case .invalidWeights(let m): "Klein text encoder weights: \(m)"
        case .invalidInput(let m): "invalid input: \(m)"
        }
    }
}
