import Foundation
import XCTest

enum TestSupport {
    /// The read-only Z-Image Turbo q4 snapshot (`ZIMAGE_Q4_MODEL_DIR`, or `ZIMAGE_MODEL_DIR`); nil when neither is set.
    static var modelDirectory: URL? {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["ZIMAGE_Q4_MODEL_DIR"] ?? environment["ZIMAGE_MODEL_DIR"], !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static func requireModel() throws -> URL {
        guard let url = modelDirectory else { throw XCTSkip("ZIMAGE_MODEL_DIR is not set; point it at a Z-Image Turbo q4 snapshot") }
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.appendingPathComponent("text_encoder/model.safetensors.index.json").path),
                          "ZIMAGE_MODEL_DIR: q4 snapshot not found at \(url.path)")
        return url
    }

    /// Reference artifacts exported from the Python/MFLUX implementation (`ZIMAGE_REFERENCE_DIR`); the tests that read them are
    /// skipped when it is unset or the export is missing.
    static var referenceDirectory: URL? {
        guard let path = ProcessInfo.processInfo.environment["ZIMAGE_REFERENCE_DIR"], !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    /// The read-only Z-Image Base q4 snapshot (`ZIMAGE_BASE_MODEL_DIR`); nil when it is not set.
    static var baseModelDirectory: URL? {
        guard let path = ProcessInfo.processInfo.environment["ZIMAGE_BASE_MODEL_DIR"], !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static func requireBaseModel() throws -> URL {
        guard let url = baseModelDirectory else { throw XCTSkip("ZIMAGE_BASE_MODEL_DIR is not set; point it at a Z-Image Base q4 snapshot") }
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.appendingPathComponent("text_encoder/model.safetensors.index.json").path),
                          "ZIMAGE_BASE_MODEL_DIR: q4 snapshot not found at \(url.path)")
        return url
    }

    /// The Python/MFLUX Z-Image Base reference export (`ZIMAGE_BASE_REFERENCE_DIR`); nil when it is not set.
    static var baseReferenceDirectory: URL? {
        guard let path = ProcessInfo.processInfo.environment["ZIMAGE_BASE_REFERENCE_DIR"], !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static func requireBaseReference() throws -> URL {
        guard let url = baseReferenceDirectory else { throw XCTSkip("ZIMAGE_BASE_REFERENCE_DIR is not set; point it at a Z-Image Base MFLUX reference export") }
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.appendingPathComponent("manifest.json").path),
                          "ZIMAGE_BASE_REFERENCE_DIR: reference export not found at \(url.path)")
        return url
    }
}
