import XCTest
@testable import ZImage
@testable import MLXImageParity

/// Z-Image Base against MFLUX 0.20. The sigma check needs the reference export (`ZIMAGE_BASE_REFERENCE_DIR`); the guided step tensors and the pixel
/// jobs also need the Base snapshot (`ZIMAGE_BASE_MODEL_DIR`). Without them these tests skip.
final class ZImageBaseParityTests: XCTestCase {
    private static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/reference")
    private var casesJSON: URL { Self.fixtures.appendingPathComponent("zimage_base_cases.json") }

    func testDeclaredCasesCoverTheBaseGrid() throws {
        let cases = try BaseCases.load(casesJSON)
        XCTAssertTrue(cases.sigmaCases.contains { $0.steps == 28 })
        XCTAssertTrue(cases.sigmaCases.contains { ($0.width / 16) * ($0.height / 16) > 4300 }, "the long-sequence mu branch")
        XCTAssertTrue(cases.stepCases.contains { $0.negativePrompt != nil } && cases.stepCases.contains { $0.negativePrompt == nil })
        XCTAssertTrue(cases.pixelCases.contains { $0.negativePrompt != nil } && cases.pixelCases.contains { $0.negativePrompt == nil })
        for job in cases.stepCases + cases.pixelCases {
            XCTAssertTrue(job.request.usesGuidance)
            XCTAssertNoThrow(try job.request.validate(for: .base))
        }
    }

    func testSigmasEqualMfluxBitForBit() throws {
        let reference = try TestSupport.requireBaseReference()
        let report = try ComponentParity.baseSigmas(casesJSON: casesJSON, referenceDirectory: reference)
        for c in report.cases { XCTAssertTrue(c.passed, "\(c.id): \(c.checks.compactMap(\.detail))") }
        XCTAssertTrue(report.passed)
    }

    func testGuidedStepTensorsMatchMflux() throws {
        let model = try TestSupport.requireBaseModel()
        let reference = try TestSupport.requireBaseReference()
        let report = try ComponentParity.baseSteps(modelDirectory: model, referenceDirectory: reference, casesJSON: casesJSON, mlxSwiftRevision: "")
        for c in report.cases {
            XCTAssertTrue(c.passed, "\(c.id): " + c.checks.filter { $0.decisive && !$0.passed }.map { "\($0.name) \($0.detail ?? "")" }.joined(separator: "; "))
        }
        XCTAssertTrue(report.passed)
    }

    func testGuidedPixelsMatchMflux() throws {
        let model = try TestSupport.requireBaseModel()
        let reference = try TestSupport.requireBaseReference()
        let report = try ComponentParity.basePixels(modelDirectory: model, referenceDirectory: reference, casesJSON: casesJSON, mlxSwiftRevision: "")
        for c in report.cases { XCTAssertTrue(c.passed, "\(c.id): \(c.checks.compactMap(\.pixels))") }
        XCTAssertTrue(report.passed)
    }
}
