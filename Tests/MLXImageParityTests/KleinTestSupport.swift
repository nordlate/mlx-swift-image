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

    /// Frozen parity cases and budgets: Tests/Fixtures/klein (copies of the experiment's klein/ inputs).
    static var kleinDirectory: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/klein")
    }
}
