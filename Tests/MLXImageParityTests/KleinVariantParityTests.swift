import XCTest
@testable import FluxKlein
@testable import MLXImageParity

/// FLUX.2 Klein generation against MFLUX 0.20, text to image and image to image. 9B needs its q4 snapshot (`KLEIN9B_MODEL_DIR`) and
/// reference export (`KLEIN9B_REFERENCE_DIR`); the same check for 4B needs `KLEIN_MODEL_DIR` and `KLEIN_REFERENCE_DIR`. Without them
/// these tests skip.
final class KleinVariantParityTests: XCTestCase {
    private var casesJSON: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/reference/klein_generation_cases.json")
    }

    func testDeclaredCasesAreValidRequestsForEveryVariant() throws {
        let cases = try KleinGenerationParityCases.load(casesJSON)
        XCTAssertFalse(cases.textCases.isEmpty)
        XCTAssertFalse(cases.imageCases.isEmpty)
        XCTAssertTrue(cases.textCases.contains { $0.width == 1024 && $0.height == 1024 && $0.steps == 4 }, "the default 1024² four-step job")
        XCTAssertEqual(Set(cases.allIDs).count, cases.allIDs.count)
        for c in cases.textCases {
            XCTAssertNil(c.strength)
            XCTAssertNoThrow(try KleinGenerationRequest(prompt: c.prompt, width: c.width, height: c.height, seed: c.seed, steps: c.steps).validate())
        }
        for c in cases.imageCases {
            let strength = try XCTUnwrap(c.strength)
            XCTAssertNoThrow(try KleinStartingImageRequest.start(steps: c.steps, strength: strength))
        }
        XCTAssertEqual(KleinVariantParity.component(.klein9B), "flux2-klein-9b-generation")
    }

    func testKlein9BMatchesMflux() throws {
        try check(.klein9B, model: try KleinTestSupport.requireModel9B())
    }

    func testKlein4BMatchesMflux020() throws {
        try check(.klein4B, model: try KleinTestSupport.requireModel())
    }

    private func check(_ variant: KleinVariant, model: URL) throws {
        let reference = try KleinTestSupport.requireReference(variant)
        let report = try KleinVariantParity.generation(
            variant: variant, modelDirectory: model, referenceDirectory: reference, casesJSON: casesJSON, mlxSwiftRevision: "") { print($0) }
        if let path = ProcessInfo.processInfo.environment["KLEIN_PARITY_REPORT"], !path.isEmpty {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: URL(fileURLWithPath: path).appendingPathComponent("\(variant.rawValue)-report.json"))
        }
        for c in report.cases {
            XCTAssertTrue(c.passed, "\(c.id): " + c.checks.filter { $0.decisive && !$0.passed }.map { "\($0.name) \($0.detail ?? "") \($0.metrics.map { "\($0)" } ?? "") \($0.pixels.map { "\($0)" } ?? "")" }.joined(separator: "; "))
        }
        XCTAssertTrue(report.passed)
    }
}
