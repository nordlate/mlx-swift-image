import Foundation
import MLX
import XCTest
@testable import MLXImageParity
@testable import SeedVR2

/// SeedVR2 3B against MFLUX 0.20. Preprocessing and seeded noise need the reference export (`SEEDVR2_REFERENCE_DIR`); the complete
/// upscales also need the fp16 files (`SEEDVR2_MODEL_DIR`). Without them these tests skip.
final class SeedVR2ParityTests: XCTestCase {
    private static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/reference")
    private var casesJSON: URL { Self.fixtures.appendingPathComponent("seedvr2_cases.json") }

    private func requireReference() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["SEEDVR2_REFERENCE_DIR"], !path.isEmpty else {
            throw XCTSkip("SEEDVR2_REFERENCE_DIR is not set; point it at a SeedVR2 MFLUX reference export")
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.appendingPathComponent("manifest.json").path),
                          "SEEDVR2_REFERENCE_DIR: reference export not found at \(url.path)")
        return url
    }

    private func requireModel() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["SEEDVR2_MODEL_DIR"], !path.isEmpty else {
            throw XCTSkip("SEEDVR2_MODEL_DIR is not set; point it at the numz/SeedVR2_comfyUI fp16 files")
        }
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
    }

    func testDeclaredCasesCoverTheRequiredJobs() throws {
        let cases = try SeedVR2Cases.load(casesJSON)
        XCTAssertEqual(Set(cases.allIDs).count, cases.allIDs.count)
        let inputs = Set(cases.inputs.map(\.name))
        for c in cases.preprocessCases { XCTAssertTrue(inputs.contains(c.input), c.id) }
        for c in cases.pixelCases { XCTAssertTrue(inputs.contains(c.input), c.id) }
        // 512 -> 1024 and 1024 -> 2048, each at softness 0 and 0.5, with a fixed seed.
        for (input, side) in [("portrait-512.png", 1024), ("forest.png", 2048)] {
            for softness in [0.0, 0.5] {
                XCTAssertTrue(cases.pixelCases.contains { $0.input == input && $0.resolution == .shortestSide(side) && $0.softness == softness })
            }
        }
        XCTAssertTrue(cases.preprocessCases.contains { if case .scale = $0.resolution { return true } else { return false } })
    }

    func testPreprocessingEqualsMfluxBitForBit() throws {
        let reference = try requireReference()
        let report = try ComponentParity.seedVR2Preprocess(casesJSON: casesJSON, referenceDirectory: reference)
        for c in report.cases { XCTAssertTrue(c.passed, "\(c.id): \(c.checks.compactMap(\.detail))") }
        XCTAssertTrue(report.passed)
    }

    func testSeededNoiseEqualsMfluxBitForBit() throws {
        let reference = try requireReference()
        let report = try ComponentParity.seedVR2Noise(casesJSON: casesJSON, referenceDirectory: reference)
        for c in report.cases { XCTAssertTrue(c.passed, "\(c.id): \(c.checks.compactMap(\.detail))") }
        XCTAssertTrue(report.passed)
    }

    func testUpscalesMatchMflux() throws {
        let model = try requireModel()
        let reference = try requireReference()
        Memory.cacheLimit = 0
        let report = try ComponentParity.seedVR2Upscales(modelDirectory: model, referenceDirectory: reference, casesJSON: casesJSON) { print($0) }
        for c in report.cases {
            XCTAssertTrue(c.passed, "\(c.id): " + c.checks.filter { $0.decisive && !$0.passed }.map { "\($0.name) \($0.detail ?? "")" }.joined(separator: "; "))
        }
        XCTAssertTrue(report.passed)
        if let directory = ProcessInfo.processInfo.environment["SEEDVR2_PARITY_REPORT"], !directory.isEmpty {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: URL(fileURLWithPath: directory).appendingPathComponent("seedvr2-upscales.json"))
        }
    }
}
