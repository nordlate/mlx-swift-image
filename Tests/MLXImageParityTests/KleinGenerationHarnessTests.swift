import Darwin
import Foundation
import MLX
import XCTest
@testable import FluxKlein
@testable import MLXImageParity
@testable import MLXImageCore

private enum GenerationTestFiles {
    static var dir: URL { KleinTestSupport.kleinDirectory.appendingPathComponent("generation") }
    static var cases: URL { dir.appendingPathComponent("cases.json") }
    static var budgets: URL { dir.appendingPathComponent("budgets.json") }
    static func temp() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("klein-generation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url,withIntermediateDirectories: true)
        guard let path = url.path.withCString({ Darwin.realpath($0,nil) }) else { throw NSError(domain: NSPOSIXErrorDomain,code: Int(errno)) }
        defer { free(path) }
        return URL(fileURLWithPath: String(cString: path),isDirectory: true)
    }
}

final class KleinGenerationPixelBudgetTests: XCTestCase {
    func testBothPixelLimitsDecideAndCannotTradeOff() {
        let baseline = [UInt8](repeating: 0,count: 20)
        var a = baseline; a[0] = 2
        XCTAssertTrue(KleinGenerationPixels.budget(native: a,reference: baseline).passed)
        XCTAssertFalse(KleinGenerationPixels.budget(native: Array(a.dropLast()),reference: Array(baseline.dropLast())).passed,"mean exceeds 0.1 with max still two")
        var lowMean = [UInt8](repeating: 0,count: 100); lowMean[0] = 3
        XCTAssertFalse(KleinGenerationPixels.budget(native: lowMean,reference: [UInt8](repeating: 0,count: 100)).passed,"max exceeds two despite mean 0.03")
        XCTAssertFalse(KleinGenerationPixels.budget(native: [],reference: []).passed)
        XCTAssertFalse(KleinGenerationPixels.budget(native: [0],reference: [0,0]).passed)
    }
}

final class KleinGenerationFreezeAndPathTests: XCTestCase {
    func testFrozenPromptsSizesAndControlsReuseExactPriorCorpus() throws {
        try KleinGenerationCases.check(GenerationTestFiles.cases); try KleinGenerationBudgets.check(GenerationTestFiles.budgets)
        for (a,b) in zip(KleinGenerationCases.frozen,KleinTransformerCases.frozen) {
            XCTAssertEqual(Data(a.prompt.utf8),Data(b.prompt.utf8)); XCTAssertEqual(a.seed,b.seed)
            XCTAssertEqual(a.width,b.width); XCTAssertEqual(a.height,b.height); XCTAssertEqual(a.steps,b.steps)
        }
        let record = try KleinGenerationFreeze.record(casesJSON: GenerationTestFiles.cases,budgetsJSON: GenerationTestFiles.budgets)
        let parsed = try JSONDecoder().decode([String: String].self,from: Data(record.utf8))
        XCTAssertEqual(parsed["cases_digest"],KleinGenerationCases.digest)
        XCTAssertEqual(parsed["cases_sha256"],try KleinHashing.sha256(of: GenerationTestFiles.cases))
        XCTAssertEqual(KleinGenerationCases.vae.map(\.seed),[19,0]); XCTAssertEqual(KleinGenerationCases.allIDs.count,6)
    }
    func testChangedCorpusControlsAndBudgetsFailClosed() throws {
        let dir = try GenerationTestFiles.temp(); defer { try? FileManager.default.removeItem(at: dir) }
        let destination = dir.appendingPathComponent("changed.json")
        let original = try JSONSerialization.jsonObject(with: Data(contentsOf: GenerationTestFiles.cases)) as! [String: Any]
        for mutation in 0..<5 {
            var changed = original, cases = original["cases"] as! [[String: Any]]
            switch mutation {
            case 0: cases.removeLast()
            case 1: cases[0]["seed"] = 43
            case 2: cases[0]["prompt"] = "different"
            case 3: cases[0]["timestep_index"] = 0
            default: cases[1] = cases[0]
            }
            changed["cases"] = cases
            try JSONSerialization.data(withJSONObject: changed).write(to: destination)
            XCTAssertThrowsError(try KleinGenerationCases.check(destination))
        }
        var changed = original; changed["vae_cases"] = []
        try JSONSerialization.data(withJSONObject: changed).write(to: destination)
        XCTAssertThrowsError(try KleinGenerationCases.check(destination))
        var budget = try JSONSerialization.jsonObject(with: Data(contentsOf: GenerationTestFiles.budgets)) as! [String: Any]
        budget["pixels_mean_abs"] = 0.10001
        try JSONSerialization.data(withJSONObject: budget).write(to: destination)
        XCTAssertThrowsError(try KleinGenerationBudgets.check(destination))
    }
    func testStrictOutputRejectsExistingSymlinkAncestorsDanglingAndModelChildren() throws {
        let dir = try GenerationTestFiles.temp(); defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try KleinGenerationPaths.requireNew(dir))
        let file = dir.appendingPathComponent("exists.json"); try Data().write(to: file)
        XCTAssertThrowsError(try KleinGenerationPaths.requireNew(file))
        let link = dir.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(atPath: link.path,withDestinationPath: dir.path)
        XCTAssertThrowsError(try KleinGenerationPaths.requireNew(link.appendingPathComponent("not-created/tail/report.json")))
        let dangling = dir.appendingPathComponent("dangling")
        try FileManager.default.createSymbolicLink(atPath: dangling.path,withDestinationPath: dir.appendingPathComponent("absent").path)
        XCTAssertThrowsError(try KleinGenerationPaths.requireNew(dangling))
        XCTAssertThrowsError(try KleinGenerationPaths.requireNew(dangling.appendingPathComponent("missing/tail")))
        XCTAssertThrowsError(try KleinGenerationPaths.requireNew(dir.appendingPathComponent("model/new/output"),modelDirectory: dir.appendingPathComponent("model")))
        XCTAssertNoThrow(try KleinGenerationPaths.requireNew(dir.appendingPathComponent("new/tail/report.json")))
    }
    func testIncompleteArtifactsAlwaysEnumerateCompiledSixCasesEvenWithTamperedInput() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = try GenerationTestFiles.temp(); defer { try? FileManager.default.removeItem(at: dir) }
            let freeze = dir.appendingPathComponent("freeze.json")
            try Data(try KleinGenerationFreeze.record(casesJSON: GenerationTestFiles.cases,budgetsJSON: GenerationTestFiles.budgets).utf8).write(to: freeze)
            let bad = dir.appendingPathComponent("bad.json"); try Data("{\"cases\":[],\"vae_cases\":[]}".utf8).write(to: bad)
            let report = try KleinGenerationCompare.run(referenceDirectory: dir,nativeDirectory: dir,isolatedDirectory: dir,casesJSON: bad,budgetsJSON: GenerationTestFiles.budgets,frozenJSON: freeze)
            XCTAssertEqual(report.status,"INCOMPLETE"); XCTAssertFalse(report.passed)
            XCTAssertEqual(report.cases.map(\.id),KleinGenerationCases.allIDs)
            XCTAssertEqual(report.cases.count,6); XCTAssertTrue(report.cases.allSatisfy { !$0.passed })
            XCTAssertTrue(report.cases.allSatisfy { $0.failures.contains { $0.contains("PNG missing/invalid") } })
        }
    }
}

final class KleinGenerationRuleTests: XCTestCase {
    func testAllPredeclaredLimitsAndRuleAssignmentsDecideAtTheirBoundaries() {
        Device.withDefaultDevice(.cpu) {
            XCTAssertEqual(KleinGenerationBudgets.zero,1e-6); XCTAssertEqual(KleinGenerationBudgets.zero,ParityTolerance.zeroReference)
            XCTAssertEqual(KleinGenerationBudgets.postStep,5e-3); XCTAssertEqual(KleinGenerationBudgets.component,1e-3)
            XCTAssertEqual(KleinGenerationBudgets.completeVAE,2e-3); XCTAssertEqual(KleinGenerationBudgets.prompt,1e-3)
            XCTAssertEqual(KleinGenerationBudgets.promptMax,1e-2)
            let ref = MLXArray([Float(1)]), expect = KleinTransformerTensors.Expect(dtype: "mlx.core.float32",shape: [1])
            func compare(_ name: String,_ value: Float,synthetic: Bool = false,isolated: Bool = false) -> Bool {
                KleinTransformerCompare.compareTensor(name,native: MLXArray([value]),reference: ref,
                    rule: KleinGenerationCompare.rule(name,synthetic: synthetic,isolated: isolated),expect: expect).passed
            }
            XCTAssertTrue(compare("step.1",1.004)); XCTAssertFalse(compare("step.1",1.006))
            XCTAssertTrue(compare("packed",1.004)); XCTAssertFalse(compare("packed",1.006)); XCTAssertFalse(compare("packed",1.00001,isolated: true))
            XCTAssertTrue(compare("decoded",1.0015,isolated: true)); XCTAssertFalse(compare("decoded",1.0025,isolated: true))
            XCTAssertFalse(compare("mid",1.0015,synthetic: true,isolated: true))
            XCTAssertTrue(compare("noise.0",10)); XCTAssertTrue(compare("decoded",10))
            XCTAssertTrue(compare("unpacked",10),"independent full unpacked magnitude is diagnostic")
            XCTAssertTrue(compare("unpacked",1.0005,isolated: true))
            XCTAssertFalse(compare("unpacked",1.0015,isolated: true),"isolated full BN/unpatch keeps the 1e-3 component gate")
            for name in ["initial","latent_ids","text_ids","sigmas","timesteps","conversion"] { XCTAssertFalse(compare(name,1.00001),name) }
            let zero = MLXArray([Float(0)]), bad = MLXArray([Float(2e-6)])
            XCTAssertFalse(KleinTransformerCompare.compareTensor("zero",native: bad,reference: zero,rule: .limit(.init(relativeL2: 1,normalizedMaxAbs: nil)),expect: expect).passed)
            for a in [MLXArray([Float.nan]),MLXArray([Float.infinity]),MLXArray.zeros([2]),MLXArray([Float(1)]).asType(.bfloat16)] {
                XCTAssertFalse(KleinTransformerCompare.compareTensor("diagnostic",native: a,reference: a,rule: .diagnostic,expect: expect).passed)
            }
            XCTAssertFalse(KleinTransformerCompare.compareTensor("missing",native: nil,reference: ref,rule: .diagnostic,expect: expect).passed)
            for c in KleinGenerationCases.frozen {
                let layout = KleinGenerationTensors.layout(width: c.width,height: c.height,steps: c.steps,synthetic: false,isolated: false)
                XCTAssertEqual(layout.keys.filter { $0.hasPrefix("step.") }.count,c.steps)
                XCTAssertTrue(layout.values.allSatisfy { $0.dtype != nil && !$0.shape.contains(0) })
            }
        }
    }
}

final class KleinGenerationChainBindingTests: XCTestCase {
    func testVAEInputMustComeFromActualFinalCapturedStep() {
        Device.withDefaultDevice(.cpu) {
            let final = MLXArray((0..<256).map(Float.init)).reshaped(1,2,128).asType(.bfloat16)
            let packed = final.reshaped(1,1,2,128).transposed(0,3,1,2)
            let good = ["step.1":final,"after_loop":final,"packed":packed]
            XCTAssertTrue(KleinGenerationCompare.chainBindingFailures(good,width: 32,height: 16,steps: 1).isEmpty)
            var bad = good; bad["packed"] = packed + 1
            XCTAssertFalse(KleinGenerationCompare.chainBindingFailures(bad,width: 32,height: 16,steps: 1).isEmpty)
            bad = good; bad["after_loop"] = final + 1
            XCTAssertFalse(KleinGenerationCompare.chainBindingFailures(bad,width: 32,height: 16,steps: 1).isEmpty)
            bad = good; bad["step.1"] = nil
            XCTAssertFalse(KleinGenerationCompare.chainBindingFailures(bad,width: 32,height: 16,steps: 1).isEmpty)
        }
    }
}
