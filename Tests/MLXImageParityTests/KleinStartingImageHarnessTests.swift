import Darwin
import Foundation
import MLX
import XCTest
import MLXImageCore
@testable import FluxKlein
@testable import MLXImageParity

private enum StartingFiles {
    static let directory = KleinTestSupport.kleinDirectory.appendingPathComponent("starting-image")
    static var cases: URL { directory.appendingPathComponent("cases.json") }
    static var budgets: URL { directory.appendingPathComponent("budgets.json") }
    static func temp() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("klein-start-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url,withIntermediateDirectories: true)
        guard let path = url.path.withCString({ Darwin.realpath($0,nil) }) else { throw NSError(domain: NSPOSIXErrorDomain,code: Int(errno)) }
        defer { free(path) }
        return URL(fileURLWithPath: String(cString: path),isDirectory: true)
    }
    static func image(_ dir: URL,width: Int = 16,height: Int = 16) throws -> URL {
        let bytes = (0..<width*height*3).map { UInt8($0%256) }
        let png = try ImageOutput.encodePNG(pixels: MLXArray(bytes).reshaped(height,width,3))
        let url = dir.appendingPathComponent(UUID().uuidString+".png")
        try png.write(to: url,options: .withoutOverwriting)
        return url
    }
}

final class KleinStartingImageHostMappingCaseTests: XCTestCase {
    func testFrozenCasesCarryTheHostStartSteps() {
        XCTAssertEqual(KleinStartingImageCases.all.prefix(4).map(\.start),[2,1,7,1])
    }
}

final class KleinStartingImageHarnessTests: XCTestCase {
    func testCompiledCorpusBudgetsLayoutsAndTamperingFailClosed() throws {
        try KleinStartingImageCases.check(StartingFiles.cases); try KleinStartingImageBudgets.check(StartingFiles.budgets)
        XCTAssertEqual(KleinStartingImageCases.allIDs.count,6)
        XCTAssertEqual(KleinStartingImagePins.runtimeModules.count,56)
        XCTAssertEqual(KleinStartingImageBudgets.component,1e-3); XCTAssertEqual(KleinStartingImageBudgets.encoded,2e-3)
        for c in KleinStartingImageCases.all {
            for isolated in [false,true] {
                let layout = KleinStartingImageTensors.layout(c,isolated: isolated)
                XCTAssertEqual(layout.keys.filter { $0.hasPrefix("step.") }.count,c.tiny || isolated ? 0 : c.steps-c.start)
                XCTAssertEqual(layout.keys.filter { $0.hasPrefix("encoder.down.") }.count,c.tiny ? 4 : 0)
                if !c.tiny && !isolated { XCTAssertEqual(layout["initial"]?.dtype,"mlx.core.float32") }
            }
        }
        let dir = try StartingFiles.temp(); defer { try? FileManager.default.removeItem(at: dir) }
        let bad = dir.appendingPathComponent("bad.json")
        let original = try JSONSerialization.jsonObject(with: Data(contentsOf: StartingFiles.cases)) as! [[String: Any]]
        for mutation in 0..<5 {
            var changed = original
            switch mutation {
            case 0: changed.removeLast()
            case 1: changed[0]["strength"] = 0.5
            case 2: changed[0]["extra"] = true
            case 3: changed[1] = changed[0]
            default: changed[0]["prompt"] = "different"
            }
            try JSONSerialization.data(withJSONObject: changed).write(to: bad)
            XCTAssertThrowsError(try KleinStartingImageCases.check(bad))
        }
        var budgets = KleinStartingImageBudgets.expected; budgets["pixels_mean_abs"] = 0.10001
        try JSONSerialization.data(withJSONObject: budgets).write(to: bad)
        XCTAssertThrowsError(try KleinStartingImageBudgets.check(bad))
    }
    func testMissingAllArtifactsAndInputsAlwaysEnumeratesAllSixAndCannotPass() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = try StartingFiles.temp(); defer { try? FileManager.default.removeItem(at: dir) }
            let missing = dir.appendingPathComponent("missing.json")
            let report = try KleinStartingImageCompare.run(referenceDirectory: dir,nativeDirectory: dir,isolatedDirectory: dir,casesJSON: StartingFiles.cases,budgetsJSON: StartingFiles.budgets,fixtures: missing,frozenJSON: missing)
            XCTAssertEqual(report.status,"INCOMPLETE"); XCTAssertFalse(report.passed)
            XCTAssertEqual(report.cases.map(\.id),KleinStartingImageCases.allIDs)
            XCTAssertEqual(report.cases.count,6); XCTAssertTrue(report.cases.allSatisfy { !$0.passed })
        }
    }
    func testIndependentAndIsolatedRulesAndDiagnosticsStillRejectBadMetadata() {
        Device.withDefaultDevice(.cpu) {
            let ref = MLXArray([Float(1)]), e = KleinTransformerTensors.Expect(dtype: "mlx.core.float32",shape: [1])
            func passed(_ n: String,_ value: Float,isolated: Bool = false) -> Bool {
                KleinTransformerCompare.compareTensor(n,native: MLXArray([value]),reference: ref,rule: KleinStartingImageCompare.rule(n,isolated: isolated),expect: e).passed
            }
            XCTAssertTrue(passed("encoded",1.0015,isolated: true)); XCTAssertFalse(passed("encoded",1.0025,isolated: true))
            XCTAssertFalse(passed("same_input.clean",1.0015,isolated: true)); XCTAssertFalse(passed("encoder.mid",1.0015,isolated: true))
            XCTAssertTrue(passed("step.3",1.004)); XCTAssertFalse(passed("step.3",1.006))
            XCTAssertTrue(passed("noise.2",100)); XCTAssertTrue(passed("decoded",100))
            XCTAssertFalse(passed("decoded",1.0025,isolated: true))
            for n in ["source_rgb","prepared","seeded_noise","callback_indices","start","conversion"] { XCTAssertFalse(passed(n,1.00001),n) }
            for a in [MLXArray([Float.nan]),MLXArray([Float.infinity]),MLXArray.zeros([2]),ref.asType(.bfloat16)] {
                XCTAssertFalse(KleinTransformerCompare.compareTensor("diagnostic",native: a,reference: a,rule: .diagnostic,expect: e).passed)
            }
        }
    }
    func testPathGuardRejectsCaseAliasModelAncestorSymlinkAndOverwrite() throws {
        let dir = try StartingFiles.temp(); defer { try? FileManager.default.removeItem(at: dir) }
        let model = dir.appendingPathComponent("ModelFolder")
        try FileManager.default.createDirectory(at: model,withIntermediateDirectories: false)
        XCTAssertThrowsError(try KleinStartingImagePaths.requireNew(model.appendingPathComponent("new/tail"),model: model))
        let alias = dir.appendingPathComponent("modelfolder")
        if FileManager.default.fileExists(atPath: alias.path) { XCTAssertThrowsError(try KleinStartingImagePaths.requireNew(alias.appendingPathComponent("new/tail"),model: model)) }
        let link = dir.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(atPath: link.path,withDestinationPath: dir.appendingPathComponent("absent").path)
        XCTAssertThrowsError(try KleinStartingImagePaths.requireNew(link.appendingPathComponent("new/tail"),model: model))
        XCTAssertThrowsError(try KleinStartingImagePaths.requireNew(dir))
        XCTAssertNoThrow(try KleinStartingImagePaths.requireNew(dir.appendingPathComponent("fresh/tail"),model: model))
    }
}

final class KleinStartingImageChainTests: XCTestCase {
    func testActualEncoderCleanBlendPredictCallbackAndDecodeBindingsCannotBeSubstituted() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = KleinStartingImageCase(id: "small-chain",prompt: "cat",width: 16,height: 16,seed: 3,steps: 4,strength: 0.625)
            let rgb = KleinStartingImageCases.rgb(c), image = KleinPreparedStartingImage(width: 16,height: 16,rgb: rgb,fileSHA256: "")
            let encoded = MLXArray.zeros([1,32,2,2],dtype: .bfloat16)
            let patch = try KleinLatent.patchify(encoded), clean = try KleinLatent.pack(patch)
            let noise = try KleinLatent.preparePacked(seed: c.seed,height: 16,width: 16), schedule = try KleinScheduler(steps: 4,height: 16,width: 16)
            let initial = try KleinStartingImageMath.blend(clean: clean,noise: noise.latents,sigma: schedule.sigmas[c.start])
            var good = ["source_rgb":MLXArray(rgb).reshaped(16,16,3),"prepared":image.normalized(),"encoder_input":image.normalized(),
                        "encoded":encoded,"matched":encoded,"patched":patch,"normalized_latent":patch,"clean":clean,
                        "bn_mean":MLXArray.zeros([128],dtype: .bfloat16),"bn_var":MLXArray.ones([128],dtype: .bfloat16),
                        "seeded_noise":noise.latents,"latent_ids":noise.ids,"sigmas":schedule.sigmas,"timesteps":schedule.timesteps,
                        "start":MLXArray(Int32(2)),"callback_indices":MLXArray([Int32(3),4]),"initial":initial,"after_loop":initial,
                        "input.2":initial,"step.3":initial,"input.3":initial,"step.4":initial]
            good["packed"] = try KleinLatent.unpack(initial,height: 16,width: 16)
            good["decoded"] = MLXArray.zeros([1,3,16,16],dtype: .bfloat16)
            good["pixels"] = try KleinGenerationPixels.convert(good["decoded"]!,width: 16,height: 16)
            XCTAssertTrue(KleinStartingImageCompare.chainFailures(good,c: c,isolated: false).isEmpty)
            for name in ["prepared","encoder_input","encoded","clean","initial","input.3","after_loop","packed"] {
                var bad = good; bad[name] = good[name]! + 1
                XCTAssertFalse(KleinStartingImageCompare.chainFailures(bad,c: c,isolated: false).isEmpty,name)
            }
            var bad = good; bad["callback_indices"] = MLXArray([Int32(1),2])
            XCTAssertFalse(KleinStartingImageCompare.chainFailures(bad,c: c,isolated: false).isEmpty)
            bad = good; bad["input.3"] = nil
            XCTAssertFalse(KleinStartingImageCompare.chainFailures(bad,c: c,isolated: false).isEmpty)
            bad = good; bad["encoded"] = nil
            XCTAssertFalse(KleinStartingImageCompare.chainFailures(bad,c: c,isolated: false).isEmpty)
        }
    }
    func testMissingOrDuplicateFixtureManifestFailsBeforeNumerics() throws {
        let dir = try StartingFiles.temp(); defer { try? FileManager.default.removeItem(at: dir) }
        let path = dir.appendingPathComponent("manifest.json")
        XCTAssertThrowsError(try KleinStartingImageFixtures.check(path))
        let manifest: [String: Any] = ["schema":1,"algorithm":KleinStartingImageCases.algorithm,"cases":[]]
        try JSONSerialization.data(withJSONObject: manifest).write(to: path)
        XCTAssertThrowsError(try KleinStartingImageFixtures.check(path))
    }
}
