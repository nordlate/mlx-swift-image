import Foundation
import MLX
import XCTest
@testable import FluxKlein
@testable import MLXImageCore
@testable import MLXImageParity

final class KleinTensorComparisonTests: XCTestCase {
    private let limit = KleinBudgets.tensors["hs9"]!
    private func bf16(_ values: [Float], shape: [Int]) -> MLXArray { MLXArray(values).reshaped(shape).asType(.bfloat16) }

    func testIdenticalTensorsPassAndOverBudgetFails() {
        Device.withDefaultDevice(.cpu) {
            let ref = (0..<64).map { Float($0 % 5) - 2 }
            let ok = KleinCompare.floatTensor(name: "hs9", native: bf16(ref, shape: [1, 8, 8]), reference: bf16(ref, shape: [1, 8, 8]), limit: limit, expected: nil)
            XCTAssertTrue(ok.passed, "\(ok.failures)")
            XCTAssertEqual(ok.metrics?.relativeL2, 0)
            let off = KleinCompare.floatTensor(
                name: "hs9", native: bf16(ref.map { $0 * 1.05 }, shape: [1, 8, 8]), reference: bf16(ref, shape: [1, 8, 8]), limit: limit, expected: nil)
            XCTAssertFalse(off.passed)
            XCTAssertTrue(off.failures.joined().contains("over budget"))
        }
    }

    func testNonFiniteValuesFailClosedOnEitherSide() {
        Device.withDefaultDevice(.cpu) {
            let good = bf16([1, 2, 3, 4], shape: [4])
            for bad in [Float.nan, .infinity, -.infinity] {
                let poisoned = bf16([1, bad, 3, 4], shape: [4])
                XCTAssertFalse(KleinCompare.floatTensor(name: "t", native: poisoned, reference: good, limit: limit, expected: nil).passed)
                XCTAssertFalse(KleinCompare.floatTensor(name: "t", native: good, reference: poisoned, limit: limit, expected: nil).passed)
                // Both sides equally broken is still a failure, not a match.
                let both = KleinCompare.floatTensor(name: "t", native: poisoned, reference: poisoned, limit: limit, expected: nil)
                XCTAssertFalse(both.passed)
                XCTAssertEqual(both.nonFiniteCount, 2)
                XCTAssertNil(both.metrics)
            }
        }
    }

    func testShapeDTypeAndArchitectureMismatchesFail() {
        Device.withDefaultDevice(.cpu) {
            let a = bf16([1, 2, 3, 4], shape: [1, 4])
            XCTAssertFalse(KleinCompare.floatTensor(name: "t", native: a, reference: bf16([1, 2, 3, 4], shape: [4, 1]), limit: limit, expected: nil).passed)
            XCTAssertFalse(KleinCompare.floatTensor(name: "t", native: a, reference: a.asType(.float32), limit: limit, expected: nil).passed)
            XCTAssertFalse(KleinCompare.floatTensor(name: "t", native: nil, reference: a, limit: limit, expected: nil).passed)
            // Both sides agree with each other but not with the architecture (e.g. a layer-0 slice instead of [1, 512, 2560]).
            let wrong = KleinCompare.floatTensor(name: "hs9", native: a, reference: a, limit: limit, expected: KleinBudgets.expectedLayout(empty: false)["hs9"])
            XCTAssertFalse(wrong.passed)
        }
    }

    func testExactTensorsRejectAnyDifference() {
        Device.withDefaultDevice(.cpu) {
            let ids = MLXArray([0, 0, 0, 0, 0, 0, 0, 1] as [Int32]).reshaped(1, 2, 4)
            XCTAssertTrue(KleinCompare.exactTensor(name: "text_ids", native: ids, reference: ids, expected: nil).passed)
            let shifted = MLXArray([0, 0, 0, 0, 0, 0, 0, 2] as [Int32]).reshaped(1, 2, 4)
            let r = KleinCompare.exactTensor(name: "text_ids", native: shifted, reference: ids, expected: nil)
            XCTAssertFalse(r.passed)
            XCTAssertTrue(r.failures.joined().contains("first at flat index 7"))
            XCTAssertFalse(KleinCompare.exactTensor(name: "text_ids", native: ids.asType(.int64), reference: ids, expected: nil).passed, "int64 vs int32")
        }
    }

    func testAssemblyIsLayerMajorPerTokenAndWrongLayoutsAreCaught() {
        Device.withDefaultDevice(.cpu) {
            let (b, s, h) = (1, 3, 4)
            let states = (1...3).map { k in MLXArray((0..<(b * s * h)).map { Float(k * 100 + $0) }).reshaped(b, s, h) }
            let embeds = KleinTextEncoder.assemble(states)
            XCTAssertEqual(embeds.shape, [b, s, 3 * h])
            // Token 1 holds layer 1's row, then layer 2's, then layer 3's.
            XCTAssertEqual(embeds[0, 1].asArray(Float.self), [104, 105, 106, 107, 204, 205, 206, 207, 304, 305, 306, 307])
            var arrays = ["hs9": states[0], "hs18": states[1], "hs27": states[2], "prompt_embeds": embeds]
            XCTAssertEqual(KleinCompare.assemblyFailures(side: "native", arrays: arrays), [])
            // A plain reshape without the transpose, and a swapped layer order, are both rejected.
            arrays["prompt_embeds"] = stacked(states, axis: 1).reshaped(b, s, 3 * h)
            XCTAssertFalse(KleinCompare.assemblyFailures(side: "native", arrays: arrays).isEmpty)
            arrays["prompt_embeds"] = KleinTextEncoder.assemble([states[1], states[0], states[2]])
            XCTAssertFalse(KleinCompare.assemblyFailures(side: "native", arrays: arrays).isEmpty)
        }
    }
}

final class KleinReportCompletenessTests: XCTestCase {
    func testCaseListMustBeExactlyTheFrozenOne() {
        XCTAssertEqual(KleinCompare.caseListFailures(label: "r", ids: ["a", "b"], expected: ["a", "b"]), [])
        XCTAssertFalse(KleinCompare.caseListFailures(label: "r", ids: ["a"], expected: ["a", "b"]).isEmpty, "missing case")
        XCTAssertFalse(KleinCompare.caseListFailures(label: "r", ids: ["a", "b", "b"], expected: ["a", "b"]).isEmpty, "duplicate case")
        XCTAssertFalse(KleinCompare.caseListFailures(label: "r", ids: ["a", "b", "c"], expected: ["a", "b"]).isEmpty, "extra case")
        XCTAssertFalse(KleinCompare.caseListFailures(label: "r", ids: ["b", "a"], expected: ["a", "b"]).isEmpty, "reordered")
    }

    func testCoverageRequiresEveryCategoryAndTheLimitBoundary() {
        let all = KleinCases.requiredCategories
        XCTAssertEqual(KleinCompare.coverageFailures(categories: all, untruncatedCounts: [511, 512, 513, 800]), [])
        XCTAssertFalse(KleinCompare.coverageFailures(categories: all.subtracting(["unicode"]), untruncatedCounts: [511, 512, 513]).isEmpty)
        XCTAssertFalse(KleinCompare.coverageFailures(categories: all, untruncatedCounts: [511, 512]).isEmpty, "nothing over the limit")
    }

    private func passing(_ id: String) -> KleinCaseResult {
        KleinCaseResult(id: id, category: "plain", validTokens: 1, untruncatedTokens: 1, referenceEncoderOutcome: "ok", nativeEncoderOutcome: "ok",
                        tensors: [], failures: [], passed: true)
    }

    func testStatusIsNeverPassWhenIncompleteDuplicatedOrFailing() {
        let ok = [KleinCheck(name: "c", failures: [], passed: true)]
        let bad = [KleinCheck(name: "c", failures: ["x"], passed: false)]
        XCTAssertEqual(KleinCompare.status(complete: true, checks: ok, cases: [passing("a"), passing("b")], expectedCount: 2), "COMPLETE_PASS")
        XCTAssertEqual(KleinCompare.status(complete: true, checks: ok, cases: [passing("a")], expectedCount: 2), "INCOMPLETE")
        XCTAssertEqual(KleinCompare.status(complete: true, checks: ok, cases: [passing("a"), passing("a")], expectedCount: 2), "INCOMPLETE", "a duplicate cannot stand in for a case")
        XCTAssertEqual(KleinCompare.status(complete: false, checks: ok, cases: [passing("a"), passing("b")], expectedCount: 2), "INCOMPLETE")
        XCTAssertEqual(KleinCompare.status(complete: true, checks: bad, cases: [passing("a"), passing("b")], expectedCount: 2), "COMPLETE_FAIL")
        var failing = passing("b")
        failing.passed = false
        XCTAssertEqual(KleinCompare.status(complete: true, checks: ok, cases: [passing("a"), failing], expectedCount: 2), "COMPLETE_FAIL")
    }

    func testEmptyPromptLayoutIsExpectedAndZeroElementTensorsCannotHideProblems() {
        Device.withDefaultDevice(.cpu) {
            let layout = KleinBudgets.expectedLayout(empty: true)
            XCTAssertEqual(layout["input_ids"]?.dtype, "mlx.core.int32")
            XCTAssertEqual(layout["prompt_embeds"]?.shape, [1, 0, 7680])
            XCTAssertEqual(layout["text_ids"]?.shape, [1, 0, 4])
            let limit = KleinBudgets.tensors["hs9"]!
            let zero = MLXArray.zeros([1, 0, 2560], dtype: .bfloat16)
            let ok = KleinCompare.floatTensor(name: "hs9", native: zero, reference: zero, limit: limit, expected: layout["hs9"])
            XCTAssertTrue(ok.passed)
            XCTAssertEqual(ok.elementCount, 0, "a zero-element tensor proves no values and the report says so")
            XCTAssertNil(ok.metrics)
            // Missing tensors, wrong dtype, a non-empty tensor for the empty prompt and an empty one for a non-empty prompt all fail.
            XCTAssertFalse(KleinCompare.floatTensor(name: "hs9", native: nil, reference: zero, limit: limit, expected: layout["hs9"]).passed)
            XCTAssertFalse(KleinCompare.floatTensor(name: "hs9", native: zero.asType(.float32), reference: zero, limit: limit, expected: layout["hs9"]).passed)
            let full = MLXArray.zeros([1, 512, 2560], dtype: .bfloat16)
            XCTAssertFalse(KleinCompare.floatTensor(name: "hs9", native: full, reference: full, limit: limit, expected: layout["hs9"]).passed)
            XCTAssertFalse(KleinCompare.floatTensor(name: "hs9", native: zero, reference: zero, limit: limit, expected: KleinBudgets.expectedLayout(empty: false)["hs9"]).passed)
            XCTAssertFalse(KleinCompare.exactTensor(
                name: "input_ids", native: MLXArray.zeros([1, 0], dtype: .int64), reference: MLXArray.zeros([1, 0], dtype: .int32), expected: layout["input_ids"]).passed,
                           "int64 vs int32 empty token batch")
            let int32Empty = MLXArray.zeros([1, 0], dtype: .int32)
            XCTAssertTrue(KleinCompare.exactTensor(name: "input_ids", native: int32Empty, reference: int32Empty, expected: layout["input_ids"]).passed)
            // Conversions of zero-element arrays must not reach mlx-swift's unsafe byte accessors.
            XCTAssertEqual(KleinCompare.bytes(zero), Data())
            XCTAssertEqual(KleinCompare.floats(zero), [])
            XCTAssertEqual(KleinCompare.assemblyFailures(side: "native", arrays: [
                "hs9": zero, "hs18": zero, "hs27": zero, "prompt_embeds": MLXArray.zeros([1, 0, 7680], dtype: .bfloat16),
            ]), [])
            XCTAssertEqual(KleinTextEncoder.tokenArray([], shape: [1, 0], dtype: .int32).shape, [1, 0])
            XCTAssertEqual(KleinTextEncoder.arange32(0).shape, [0])
            XCTAssertEqual(KleinCompare.textIDsFailures(side: "native", ids: KleinPromptEncoder.textIDs(batch: 1, seqLen: 0)), [])
            XCTAssertFalse(KleinCompare.textIDsFailures(side: "native", ids: nil).isEmpty)
            XCTAssertFalse(KleinCompare.textIDsFailures(side: "native", ids: MLXArray([0, 0, 0, 1] as [Int32]).reshaped(1, 1, 4)).isEmpty, "position must start at 0")
        }
    }

    func testBudgetsFileMustMatchCompiledLimits() throws {
        XCTAssertEqual(try KleinBudgets.check(budgetsJSON: KleinTestSupport.kleinDirectory.appendingPathComponent("budgets.json")), [])
        let widened = """
        {"tensors": {"hs0": {"relative_l2": 1e-6, "normalized_max_abs": 1e-6}, "hs9": {"relative_l2": 0.01, "normalized_max_abs": 0.01},
        "hs18": {"relative_l2": 0.001, "normalized_max_abs": 0.01}, "hs27": {"relative_l2": 0.001, "normalized_max_abs": 0.01},
        "prompt_embeds": {"relative_l2": 0.001, "normalized_max_abs": 0.01}}, "exact": ["attention_mask", "input_ids", "text_ids"], "non_finite": "fail"}
        """
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("klein-budgets-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(widened.utf8).write(to: url)
        XCTAssertFalse(try KleinBudgets.check(budgetsJSON: url).isEmpty, "a widened limit in the file must be caught")
    }

    func testFrozenCorpusAndReferenceVersionsArePinned() throws {
        let (cases, sha) = try KleinCases.load(url: KleinTestSupport.kleinDirectory.appendingPathComponent("cases.json"))
        XCTAssertEqual(sha, KleinPins.frozenCasesSHA256, "cases.json changed after it was frozen")
        XCTAssertEqual(cases.count, KleinPins.frozenCaseCount)
        XCTAssertEqual(KleinCompare.frozenCorpusFailures(casesSHA256: sha, caseCount: cases.count), [])
        // A changed corpus fails even if both sides would agree on it; so does a changed case count.
        XCTAssertFalse(KleinCompare.frozenCorpusFailures(casesSHA256: String(repeating: "0", count: 64), caseCount: 24).isEmpty)
        XCTAssertFalse(KleinCompare.frozenCorpusFailures(casesSHA256: sha, caseCount: 23).isEmpty)
        XCTAssertEqual(KleinCompare.referenceVersionFailures(["mflux_version": "0.20.0", "mlx_version": "0.32.2"]), [])
        XCTAssertFalse(KleinCompare.referenceVersionFailures(["mflux_version": "0.20.1", "mlx_version": "0.32.2"]).isEmpty)
        XCTAssertFalse(KleinCompare.referenceVersionFailures(["mflux_version": "0.20.0", "mlx_version": "0.32.3"]).isEmpty)
        XCTAssertFalse(KleinCompare.referenceVersionFailures([:]).isEmpty)
        XCTAssertTrue(KleinCompare.isSHA256Hex(String(repeating: "ab", count: 32)))
        XCTAssertFalse(KleinCompare.isSHA256Hex(nil))
        XCTAssertFalse(KleinCompare.isSHA256Hex("abc"))
        XCTAssertFalse(KleinCompare.isSHA256Hex(String(repeating: "g", count: 64)))
    }

    func testExistingReportsAreNeverReplaced() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("klein-report-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertNoThrow(try KleinJSON.requireNewFile(url))
        try KleinJSON.write(["a": 1], to: url)
        XCTAssertThrowsError(try KleinJSON.requireNewFile(url))
        XCTAssertThrowsError(try KleinJSON.write(["a": 2], to: url))
        XCTAssertEqual(try KleinJSON.read([String: Int].self, from: url), ["a": 1], "the first report is untouched")
    }

    func testFrozenCasesParseAndCoverCategories() throws {
        let (cases, sha) = try KleinCases.load(url: KleinTestSupport.kleinDirectory.appendingPathComponent("cases.json"))
        XCTAssertEqual(sha.count, 64)
        XCTAssertEqual(Set(cases.map(\.id)).count, cases.count)
        XCTAssertTrue(KleinCases.requiredCategories.isSubset(of: Set(cases.map(\.category))))
        XCTAssertEqual(cases.first { $0.id == "empty" }?.prompt, "")
        XCTAssertEqual(cases.first { $0.id == "boundary-500" }?.prompt.split(separator: " ").count, 500)
        XCTAssertThrowsError(try KleinCases.parse([["id": "a", "category": "x", "prompt": "p"], ["id": "a", "category": "x", "prompt": "q"]]))
    }

    /// Two manifests whose case lists are short or duplicated must produce INCOMPLETE and a failing exit, whatever else matches.
    func testWholeComparisonWithMissingOrDuplicatedCasesIsIncomplete() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("klein-compare-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let ref = dir.appendingPathComponent("ref", isDirectory: true), nat = dir.appendingPathComponent("nat", isDirectory: true)
        try FileManager.default.createDirectory(at: ref, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: nat, withIntermediateDirectories: true)
        let cases = dir.appendingPathComponent("cases.json")
        try Data(#"{"cases":[{"id":"a","category":"plain","prompt":"cat"},{"id":"b","category":"empty","prompt":""}]}"#.utf8).write(to: cases)
        let sha = try KleinCases.load(url: cases).sha256
        func entry(_ id: String) -> KleinManifestCase {
            KleinManifestCase(
                id: id, category: id == "a" ? "plain" : "empty", prompt: id == "a" ? "cat" : "", renderedText: "", numValid: 0, untruncatedCount: 0,
                idsDtype: "mlx.core.int64", idsShape: [1, 512], maskDtype: "mlx.core.int64", maskShape: [1, 512], encoderOutcome: "ok", tensors: [:])
        }
        func manifest(side: String, cases list: [KleinManifestCase]) -> KleinManifest {
            KleinManifest(
                schema: 1, side: side, casesSHA256: sha, runtime: [:], runtimeModules: nil, modelFilesBefore: [:], modelFilesAfter: [:], weights: [:],
                quantizedModules: [], cases: list)
        }
        try KleinJSON.write(manifest(side: "reference", cases: [entry("a")]), to: ref.appendingPathComponent("manifest.json"))  // case b missing
        try KleinJSON.write(manifest(side: "native", cases: [entry("a"), entry("b"), entry("b")]), to: nat.appendingPathComponent("manifest.json"))  // b twice
        let report = try KleinCompare.run(
            referenceDirectory: ref, nativeDirectory: nat, casesJSON: cases,
            budgetsJSON: KleinTestSupport.kleinDirectory.appendingPathComponent("budgets.json"))
        XCTAssertEqual(report.status, "INCOMPLETE")
        XCTAssertFalse(report.passed)
        let listing = report.checks.first { $0.name == "case-list" }
        XCTAssertFalse(listing?.passed ?? true)
        XCTAssertTrue(listing?.failures.joined().contains("missing cases") ?? false)
        XCTAssertTrue(listing?.failures.joined().contains("duplicate case ids") ?? false)
        // Provenance checks fail independently: nothing was measured, nothing is claimed.
        XCTAssertFalse(report.checks.first { $0.name == "runtime-provenance" }?.passed ?? true)
        XCTAssertFalse(report.checks.first { $0.name == "model-artifacts-unchanged" }?.passed ?? true)
        XCTAssertFalse(report.checks.first { $0.name == "loaded-weight-bytes" }?.passed ?? true)
    }
}
