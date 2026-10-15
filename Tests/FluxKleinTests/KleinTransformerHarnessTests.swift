import Darwin
import Foundation
import MLX
import XCTest
@testable import KleinNative

private enum TransformerTestFiles {
    static let directory = KleinTestSupport.kleinDirectory.appendingPathComponent("transformer")
    static let cases = directory.appendingPathComponent("cases.json")
    static let budgets = directory.appendingPathComponent("budgets.json")

    static func temporaryDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("klein-transformer-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // Foundation can render /private/var as /var again; the strict output guard needs the physical path.
        guard let physicalPath = dir.path.withCString({ Darwin.realpath($0, nil) }) else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno), userInfo: [NSFilePathErrorKey: dir.path])
        }
        defer { free(physicalPath) }
        return URL(fileURLWithPath: String(cString: physicalPath), isDirectory: true)
    }
}

final class KleinTransformerTensorGateTests: XCTestCase {
    func testRelativeBudgetsAndExactBytesAreDistinctGates() {
        Device.withDefaultDevice(.cpu) {
            let reference = MLXArray([Float(1)])
            let expect = KleinTransformerTensors.Expect(dtype: "mlx.core.float32", shape: [1])
            func compare(_ value: Float, _ rule: KleinTransformerCompare.Rule) -> KleinTensorResult {
                KleinTransformerCompare.compareTensor("test", native: MLXArray([value]), reference: reference, rule: rule, expect: expect)
            }
            XCTAssertTrue(compare(1.0015, .limit(KleinTransformerBudgets.fullPrediction)).passed)
            XCTAssertFalse(compare(1.0025, .limit(KleinTransformerBudgets.fullPrediction)).passed)
            XCTAssertFalse(compare(1.0015, .limit(KleinTransformerBudgets.embedding)).passed)
            XCTAssertFalse(compare(1.0015, .limit(KleinTransformerBudgets.singleBlock)).passed)
            XCTAssertTrue(compare(1.004, .limit(KleinTransformerBudgets.noiseChain)).passed)
            XCTAssertFalse(compare(1.006, .limit(KleinTransformerBudgets.noiseChain)).passed)
            XCTAssertTrue(compare(100, .diagnostic).passed, "diagnostic magnitude does not decide parity")
            XCTAssertFalse(compare(1.000001, .exact).passed, "exact scheduler and seeded tensors have no tolerance")
            let plusZero = MLXArray([Float(0)]), minusZero = MLXArray([-Float(0)])
            XCTAssertFalse(KleinTransformerCompare.compareTensor("bytes", native: minusZero, reference: plusZero, rule: .exact, expect: expect).passed)
        }
    }

    func testMissingEmptyWrongLayoutAndNonFiniteFailEvenWhenSidesAgree() {
        Device.withDefaultDevice(.cpu) {
            let expect = KleinTransformerTensors.Expect(dtype: "mlx.core.float32", shape: [2])
            let good = MLXArray([Float(1), 2])
            func compare(_ n: MLXArray?, _ r: MLXArray?) -> KleinTensorResult {
                KleinTransformerCompare.compareTensor("test", native: n, reference: r, rule: .diagnostic, expect: expect)
            }
            XCTAssertFalse(compare(nil, good).passed)
            XCTAssertFalse(compare(good.reshaped(1, 2), good.reshaped(1, 2)).passed)
            XCTAssertFalse(compare(good.asType(.bfloat16), good.asType(.bfloat16)).passed)
            let empty = MLXArray.zeros([0])
            XCTAssertFalse(KleinTransformerCompare.compareTensor("empty", native: empty, reference: empty, rule: .exact, expect: nil).passed)
            for value in [Float.nan, .infinity, -.infinity] {
                let bad = MLXArray([Float(1), value])
                XCTAssertFalse(compare(bad, good).passed)
                XCTAssertFalse(compare(good, bad).passed)
                XCTAssertEqual(compare(bad, bad).nonFiniteCount, 2)
            }
            let temb = KleinTransformerTensors.primaryLayout(KleinTransformerCases.frozen[0])["temb_raw"]!
            XCTAssertEqual(temb.dtype, "mlx.core.float32")
            let wrongTemb = MLXArray.zeros(temb.shape, dtype: .bfloat16)
            XCTAssertFalse(KleinTransformerCompare.compareTensor("temb_raw", native: wrongTemb, reference: wrongTemb, rule: .limit(KleinTransformerBudgets.embedding), expect: temb).passed)
        }
    }

    func testZeroReferenceUsesAbsoluteL2AndScheduleSanityRejectsMatchedCorruption() {
        Device.withDefaultDevice(.cpu) {
            let zero = MLXArray([Float(0), 0])
            for (value, passes) in [(Float(0.5e-6), true), (Float(0.8e-6), false)] {
                let result = KleinTransformerCompare.compareTensor("zero", native: MLXArray([value, value]), reference: zero,
                    rule: .limit(KleinTransformerBudgets.fullPrediction), expect: nil)
                XCTAssertEqual(result.passed, passes, "absolute vector norm matters even if each coordinate is below 1e-6")
                XCTAssertEqual(result.metrics?.zeroReferenceFallback, true)
            }
            let badSchedules: [[Float]] = [[1, 0.5, 0.6, 0], [0.9, 0.5, 0], [1, 0.5, 0.01], [1, .nan, 0]]
            for sigmas in badSchedules {
                let s = MLXArray(sigmas)
                let t = MLXArray(sigmas.dropLast().map { $0 * 1000 })
                XCTAssertFalse(KleinTransformerCompare.scheduleFailures(side: "test", label: "bad", sigmas: s, timesteps: t).isEmpty)
            }
            XCTAssertFalse(KleinTransformerCompare.scheduleFailures(side: "test", label: "bad", sigmas: MLXArray([Float(1), 0]), timesteps: MLXArray([Float(999)])).isEmpty)
        }
    }
}

final class KleinTransformerFrozenInputTests: XCTestCase {
    func testCasesAndBudgetsFreezeWithoutChangingTheInputs() throws {
        let originalCases = try Data(contentsOf: TransformerTestFiles.cases)
        let originalBudgets = try Data(contentsOf: TransformerTestFiles.budgets)
        let (cases, sha) = try KleinTransformerCases.load(url: TransformerTestFiles.cases)
        XCTAssertEqual(cases, KleinTransformerCases.frozen)
        XCTAssertEqual(try KleinTransformerBudgets.check(budgetsJSON: TransformerTestFiles.budgets), [])
        let record = try KleinTransformerFreeze.record(casesJSON: TransformerTestFiles.cases, budgetsJSON: TransformerTestFiles.budgets)
        let pins = try JSONDecoder().decode([String: String].self, from: Data(record.utf8))
        XCTAssertEqual(pins["cases_sha256"], sha)
        XCTAssertEqual(pins["cases_digest"], KleinTransformerCases.compiledDigest)
        XCTAssertEqual(pins["budgets_sha256"], KleinHashing.sha256(of: originalBudgets))
        XCTAssertEqual(try Data(contentsOf: TransformerTestFiles.cases), originalCases)
        XCTAssertEqual(try Data(contentsOf: TransformerTestFiles.budgets), originalBudgets)
        for c in cases {
            XCTAssertEqual(Set(KleinTransformerTensors.primaryKeys(c)), Set(KleinTransformerTensors.primaryLayout(c).keys))
            XCTAssertEqual(Set(KleinTransformerTensors.isolatedKeys(c)), Set(KleinTransformerTensors.isolatedLayout(c).keys))
        }
        XCTAssertEqual(KleinTransformerControls.requiredKeys().count, 120)
        for key in KleinTransformerControls.requiredKeys() { XCTAssertNotNil(KleinTransformerControls.expectedLayout(key: key), key) }
        for key in ["sigmas.1024x1024.n0", "timesteps.1024x1024.n9", "ids.1024x1024.extra", "lat4d.1024x1024.extra", "packed.16x16"] {
            XCTAssertNil(KleinTransformerControls.expectedLayout(key: key), key)
        }
    }

    func testChangedCaseMissingCaseAndDuplicateCannotFreeze() throws {
        let dir = try TransformerTestFiles.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let original = try JSONSerialization.jsonObject(with: Data(contentsOf: TransformerTestFiles.cases)) as! [String: Any]
        let entries = original["cases"] as! [[String: Any]]
        var changed = entries
        changed[0]["seed"] = 43
        var invalidIndex = entries
        invalidIndex[0]["timestep_index"] = 4
        for (index, values) in [changed, Array(entries.dropLast()), entries + [entries[0]], invalidIndex].enumerated() {
            let url = dir.appendingPathComponent("cases-\(index).json")
            try JSONSerialization.data(withJSONObject: ["cases": values]).write(to: url)
            XCTAssertThrowsError(try KleinTransformerCases.load(url: url))
            XCTAssertThrowsError(try KleinTransformerFreeze.record(casesJSON: url, budgetsJSON: TransformerTestFiles.budgets))
        }
    }

    func testWidenedBudgetsAndRemovedExactGateCannotFreeze() throws {
        let dir = try TransformerTestFiles.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let original = try JSONSerialization.jsonObject(with: Data(contentsOf: TransformerTestFiles.budgets)) as! [String: Any]
        var wide = original
        var limits = wide["relative_l2"] as! [String: Double]
        limits["full_prediction"] = 0.003
        wide["relative_l2"] = limits
        var removed = original
        removed["exact"] = ["latents"]
        var finite = original
        finite["non_finite"] = "ignore"
        for (index, value) in [wide, removed, finite].enumerated() {
            let url = dir.appendingPathComponent("budgets-\(index).json")
            try JSONSerialization.data(withJSONObject: value).write(to: url)
            XCTAssertFalse(try KleinTransformerBudgets.check(budgetsJSON: url).isEmpty)
            XCTAssertThrowsError(try KleinTransformerFreeze.record(casesJSON: TransformerTestFiles.cases, budgetsJSON: url))
        }
    }
}

final class KleinTransformerReportTests: XCTestCase {
    private func manifest(side: String, cases: [KleinTransformerManifestCase], casesSHA: String, budgetsSHA: String) -> KleinTransformerManifest {
        KleinTransformerManifest(schema: 1, side: side, casesSHA256: casesSHA, casesDigest: KleinTransformerCases.compiledDigest,
            budgetsSHA256: budgetsSHA, runtime: [:], runtimeModules: nil, modelFilesBefore: [:], modelFilesAfter: [:],
            textWeights: [:], transformerWeights: [:], textQuantizedModules: [], transformerQuantizedModules: [], cases: cases, controlTensors: [:])
    }

    func testWholeComparisonCannotPassMissingDuplicatedOrWrongCorpus() throws {
        let dir = try TransformerTestFiles.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let casesSHA = try KleinHashing.sha256(of: TransformerTestFiles.cases)
        let budgetsSHA = try KleinHashing.sha256(of: TransformerTestFiles.budgets)
        let freeze = dir.appendingPathComponent("frozen.json")
        try Data(KleinTransformerFreeze.record(casesJSON: TransformerTestFiles.cases, budgetsJSON: TransformerTestFiles.budgets).utf8).write(to: freeze)
        let first = KleinTransformerCases.frozen[0]
        let entry = KleinTransformerNative.manifestCase(first, tokens: .init(valid: 20, untruncated: 20), grid: (64, 64), tensors: [:])
        for (attempt, entries) in [[], [entry, entry]].enumerated() {
            let sides = ["reference", "native", "isolated"].map { dir.appendingPathComponent("\(attempt)-\($0)") }
            for (side, output) in zip(["reference", "native", "isolated"], sides) {
                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
                try KleinJSON.write(manifest(side: side, cases: entries, casesSHA: casesSHA, budgetsSHA: budgetsSHA), to: output.appendingPathComponent("manifest.json"))
            }
            var files: [URL] = [TransformerTestFiles.cases]
            let short = dir.appendingPathComponent("short-\(attempt).json")
            try Data("{\"cases\": []}".utf8).write(to: short)
            files.append(short)
            for casesFile in files {
                let report = try Device.withDefaultDevice(.cpu) {
                    try KleinTransformerCompare.run(referenceDirectory: sides[0], nativeDirectory: sides[1], isolatedDirectory: sides[2],
                        casesJSON: casesFile, budgetsJSON: TransformerTestFiles.budgets, frozenJSON: freeze)
                }
                XCTAssertFalse(report.passed)
                XCTAssertEqual(report.status, "INCOMPLETE")
                XCTAssertEqual(report.expectedCaseIDs, KleinTransformerCases.frozen.map(\.id), "the supplied file cannot reduce required coverage")
                XCTAssertFalse(report.checks.first { $0.name == "case-list" }!.passed)
                XCTAssertFalse(report.checks.first { $0.name == "runtime-provenance" }!.passed)
                XCTAssertFalse(report.checks.first { $0.name == "model-artifacts-unchanged" }!.passed)
                if casesFile == short { XCTAssertFalse(report.checks.first { $0.name == "frozen-inputs" }!.passed) }
            }
        }
    }

    func testFailedOutputsAndReportsArePreservedIncludingSymlinkParents() throws {
        let dir = try TransformerTestFiles.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let output = dir.appendingPathComponent("attempt")
        let model = dir.appendingPathComponent("model")
        try KleinTransformerNative.prepareOutput(output, modelDirectory: model)
        let report = output.appendingPathComponent("report.json")
        try KleinJSON.write(["status": "INCOMPLETE"], to: report)
        let saved = try Data(contentsOf: report)
        XCTAssertThrowsError(try KleinTransformerNative.prepareOutput(output, modelDirectory: model))
        XCTAssertThrowsError(try KleinJSON.requireNewFileWithoutSymlinkParents(report))
        XCTAssertThrowsError(try KleinJSON.write(["status": "COMPLETE_PASS"], to: report))
        XCTAssertEqual(try Data(contentsOf: report), saved)
        XCTAssertThrowsError(try KleinTransformerNative.prepareOutput(model.appendingPathComponent("out"), modelDirectory: model))
        let link = dir.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: output)
        XCTAssertThrowsError(try KleinJSON.requireNewFileWithoutSymlinkParents(link.appendingPathComponent("new-report.json")))
        XCTAssertThrowsError(try KleinTransformerNative.prepareOutput(link.appendingPathComponent("new-run"), modelDirectory: model))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("new-run").path))
        let dangling = dir.appendingPathComponent("dangling.json")
        try FileManager.default.createSymbolicLink(at: dangling, withDestinationURL: dir.appendingPathComponent("absent"))
        XCTAssertThrowsError(try KleinJSON.requireNewFileWithoutSymlinkParents(dangling))
    }

    func testOutputGuardRejectsDirectAndDanglingSymlinksBeforeCreatingDirectories() throws {
        let dir = try TransformerTestFiles.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let fm = FileManager.default
        let model = dir.appendingPathComponent("model")
        let empty = dir.appendingPathComponent("empty")
        try fm.createDirectory(at: empty, withIntermediateDirectories: true)
        let absent = dir.appendingPathComponent("absent")
        for (name, target) in [("direct-output", empty), ("dangling-output", absent)] {
            let link = dir.appendingPathComponent(name)
            try fm.createSymbolicLink(at: link, withDestinationURL: target)
            for output in [link, link.appendingPathComponent("nested/new-run")] {
                XCTAssertThrowsError(try KleinTransformerNative.prepareOutput(output, modelDirectory: model)) { error in
                    XCTAssertTrue("\(error)".contains("symbolic link"), "\(error)")
                }
            }
            XCTAssertEqual(try fm.destinationOfSymbolicLink(atPath: link.path), target.path)
        }
        XCTAssertEqual(try fm.contentsOfDirectory(atPath: empty.path), [])
        XCTAssertFalse(fm.fileExists(atPath: absent.path))
    }
}
