import Foundation

/// Why a folder cannot be converted, or why a conversion stopped. Paths are relative to the source folder.
public enum KleinConversionError: Error, Equatable, CustomStringConvertible {
    /// The folder is not a FLUX.2 Klein diffusers snapshot (no `model_index.json`, or another pipeline class or component).
    case notKlein(String)
    /// A FLUX.2 Klein snapshot this library cannot convert: another size or a non-distilled (base) model, or a size without conversion pins.
    case unsupportedVariant(String)
    /// A file the conversion needs is not in the folder.
    case missingFile(String)
    /// A file in a component folder (`transformer/`, `text_encoder/`, `vae/`, `tokenizer/`, `scheduler/`) that the official release does not have.
    case unexpectedFile(String)
    /// A file's size differs from the official release (found before any hashing).
    case sizeMismatch(file: String, expected: Int64, actual: Int64)
    /// A file's SHA-256 differs from the official release.
    case hashMismatch(file: String, expected: String, actual: String)
    /// A file has the expected name but its content does not match the expected structure (unreadable JSON, wrong config values,
    /// safetensors header that disagrees with the architecture).
    case invalidFile(file: String, reason: String)
    /// The output folder already exists.
    case outputExists(String)
    /// The conversion was cancelled; the staging folder was removed.
    case cancelled
    /// Reading or writing failed.
    case io(String)

    public var description: String {
        switch self {
        case .notKlein(let m): "not a FLUX.2 Klein snapshot: \(m)"
        case .unsupportedVariant(let m): "unsupported FLUX.2 Klein variant: \(m)"
        case .missingFile(let f): "missing file: \(f)"
        case .unexpectedFile(let f): "unexpected file: \(f)"
        case .sizeMismatch(let f, let e, let a): "\(f) differs from the official release (\(a) bytes, expected \(e))"
        case .hashMismatch(let f, let e, let a): "\(f) differs from the official release (SHA-256 \(a), expected \(e))"
        case .invalidFile(let f, let r): "\(f): \(r)"
        case .outputExists(let p): "output folder already exists: \(p)"
        case .cancelled: "conversion cancelled"
        case .io(let m): "I/O error: \(m)"
        }
    }
}
