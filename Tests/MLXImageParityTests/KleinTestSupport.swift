import FluxKlein
import XCTest

enum KleinTestSupport {
    /// The read-only FLUX.2 Klein 4B q4 snapshot (`KLEIN_Q4_MODEL_DIR`, or `KLEIN_MODEL_DIR`); nil when neither is set.
    static var modelDirectory: URL? {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["KLEIN_Q4_MODEL_DIR"] ?? environment["KLEIN_MODEL_DIR"], !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static func requireModel() throws -> URL {
        guard let url = modelDirectory else { throw XCTSkip("KLEIN_MODEL_DIR is not set; point it at a FLUX.2 Klein 4B q4 snapshot") }
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.appendingPathComponent("text_encoder/model.safetensors.index.json").path),
                          "KLEIN_MODEL_DIR: Klein q4 snapshot not found at \(url.path)")
        return url
    }

    /// The read-only FLUX.2 Klein 9B q4 snapshot written by `mflux-save` 0.20 (`KLEIN9B_MODEL_DIR`); nil when it is not set.
    static var model9BDirectory: URL? { directory("KLEIN9B_MODEL_DIR") }

    static func requireModel9B() throws -> URL {
        try require("KLEIN9B_MODEL_DIR", file: "text_encoder/model.safetensors.index.json", what: "a FLUX.2 Klein 9B q4 snapshot")
    }

    /// The MFLUX 0.20 generation reference export for `variant` (`KLEIN9B_REFERENCE_DIR` for 9B, `KLEIN_REFERENCE_DIR` for 4B).
    static func requireReference(_ variant: KleinVariant) throws -> URL {
        let name = variant == .klein9B ? "KLEIN9B_REFERENCE_DIR" : "KLEIN_REFERENCE_DIR"
        return try require(name, file: "manifest.json", what: "an MFLUX 0.20 \(variant.displayName) generation reference export")
    }

    private static func directory(_ name: String) -> URL? {
        guard let path = ProcessInfo.processInfo.environment[name], !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private static func require(_ name: String, file: String, what: String) throws -> URL {
        guard let url = directory(name) else { throw XCTSkip("\(name) is not set; point it at \(what)") }
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.appendingPathComponent(file).path), "\(name): \(file) not found at \(url.path)")
        return url
    }

    /// Frozen parity cases and budgets: Tests/Fixtures/klein (copies of the experiment's klein/ inputs).
    static var kleinDirectory: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/klein")
    }
}
