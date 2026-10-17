import XCTest
import MLXImageCore
@testable import ZImage

final class ParityRunnerTests: XCTestCase {
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
