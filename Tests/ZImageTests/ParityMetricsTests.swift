import XCTest
@testable import ZImageNative

final class ParityMetricsTests: XCTestCase {
    func testIdenticalTensorsPassWithZeroError() {
        let m = ParityMetrics.compare(native: [1, -2, 3], reference: [1, -2, 3])
        XCTAssertEqual(m.relativeL2, 0)
        XCTAssertEqual(m.normalizedMaxAbs, 0)
        XCTAssertFalse(m.zeroReferenceFallback)
        XCTAssertTrue(m.passed)
    }

    func testRelativeL2LimitIsEnforced() {
        let reference = (0..<1000).map { Float($0 % 7) - 3 }
        // A uniform relative scale error moves relative L2 and normalized max abs equally; 2e-3 breaks only the L2 limit.
        let ok = ParityMetrics.compare(native: reference.map { $0 * 1.0005 }, reference: reference)
        XCTAssertEqual(ok.relativeL2!, 5e-4, accuracy: 1e-6)
        XCTAssertTrue(ok.passed)
        let bad = ParityMetrics.compare(native: reference.map { $0 * 1.002 }, reference: reference)
        XCTAssertEqual(bad.relativeL2!, 2e-3, accuracy: 1e-6)
        XCTAssertLessThan(bad.normalizedMaxAbs!, ParityTolerance.embeddingNormalizedMaxAbs)
        XCTAssertFalse(bad.passed)
    }

    func testNormalizedMaxAbsLimitIsEnforced() {
        var reference = [Float](repeating: 0, count: 1_000_000)
        reference[0] = 1
        var native = reference
        native[1] = 0.02  // relL2 ~ 2e-2 fails anyway; check the max-abs metric itself
        let m = ParityMetrics.compare(native: native, reference: reference)
        XCTAssertEqual(m.normalizedMaxAbs!, 0.02, accuracy: 1e-7)
        XCTAssertFalse(m.passed)
    }

    func testZeroReferenceUsesAbsoluteFallback() {
        let zeros = [Float](repeating: 0, count: 8)
        let same = ParityMetrics.compare(native: zeros, reference: zeros)
        XCTAssertTrue(same.zeroReferenceFallback)
        XCTAssertNil(same.relativeL2)
        XCTAssertTrue(same.passed)
        var off = zeros
        off[3] = 1e-5
        let bad = ParityMetrics.compare(native: off, reference: zeros)
        XCTAssertTrue(bad.zeroReferenceFallback)
        XCTAssertFalse(bad.passed, "a non-zero native tensor against a zero reference must fail, not divide by zero")
        XCTAssertFalse(bad.absoluteL2.isNaN)
    }

    func testDeclaredTolerancesAreThePlanValues() {
        XCTAssertEqual(ParityTolerance.embeddingRelativeL2, 1e-3)
        XCTAssertEqual(ParityTolerance.embeddingNormalizedMaxAbs, 1e-2)
        XCTAssertEqual(ParityTolerance.zeroReference, 1e-6)
        XCTAssertEqual(ParityTolerance.zeroFallbackAbsolute, 1e-6)
    }

    private func result(skipped: String?, embedding: ComparisonMetrics? = nil) -> CaseResult {
        CaseResult(id: "x", category: "plain", tokenIDsEqual: true, maskEqual: true, shapesAndDtypesEqual: true, renderedTextEqual: true,
                   firstTokenMismatch: nil, validTokens: 5, embedding: embedding, embeddingSkipped: skipped, passed: false)
    }

    func testEmbeddingSkipOnlyAllowedForEmptyPrompt() {
        XCTAssertTrue(ParityRunner.embeddingMayBeSkipped(prompt: "", nativeTokensEmpty: true, manifestShape: nil, manifestDtype: nil))
        XCTAssertFalse(ParityRunner.embeddingMayBeSkipped(prompt: "cat", nativeTokensEmpty: false, manifestShape: nil, manifestDtype: nil))
        XCTAssertFalse(ParityRunner.embeddingMayBeSkipped(prompt: "", nativeTokensEmpty: true, manifestShape: [0, 2560], manifestDtype: nil),
                       "manifest declares an embedding the file lacks")
    }

    func testVerdictFailsForMissingEmbeddingOnNonEmptyPrompt() {
        XCTAssertFalse(ParityRunner.caseVerdict(result(skipped: "FAIL: reference embedding missing for a case that requires one")))
        XCTAssertFalse(ParityRunner.caseVerdict(result(skipped: nil)))
        XCTAssertTrue(ParityRunner.caseVerdict(result(skipped: "reference has no embedding (empty prompt: the reference text encoder is not run)")))
        XCTAssertTrue(ParityRunner.caseVerdict(result(skipped: nil, embedding: ParityMetrics.compare(native: [1], reference: [1]))))
    }

    func testCaseListMustMatchCasesJSON() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("cases-\(UUID().uuidString).json")
        try Data(#"{"cases":[{"id":"a"},{"id":"b"}]}"#.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let all: Set<String> = ["plain", "long", "unicode"]
        XCTAssertNoThrow(try ParityRunner.checkCaseList(manifestIDs: ["a", "b"], categories: all, casesJSON: url))
        XCTAssertThrowsError(try ParityRunner.checkCaseList(manifestIDs: ["a"], categories: all, casesJSON: url))
        XCTAssertThrowsError(try ParityRunner.checkCaseList(manifestIDs: ["a", "b"], categories: ["plain"], casesJSON: url))
    }
}
