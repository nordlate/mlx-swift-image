import MLX
import XCTest
import MLXImageCore
@testable import ZImage
@testable import MLXImageParity

final class SliceFourToleranceTests: XCTestCase {
    /// The VAE and pixel rows of the pre-declared tolerance table; a change here is a loosening needing an explicit, recorded approval.
    func testLimitsEqualThePredeclaredTable() {
        XCTAssertEqual(ParityTolerance.vaeOutputRelativeL2, 2e-3)
        XCTAssertEqual(ParityLimits.vaeOutput.relativeL2, 2e-3)
        XCTAssertEqual(ParityTolerance.pixelMaxAbsLevels, 2)
        XCTAssertEqual(ParityTolerance.pixelMeanAbsLevels, 0.1)
        XCTAssertEqual(ParityTolerance.schedulerLatentPerStepRelativeL2, 5e-3, "the per-step latent limit is unchanged")
        XCTAssertEqual(ParityTolerance.fullTransformerOutputRelativeL2, 2e-3, "the full output limit is unchanged")
    }

    func testPixelMetricsJudgeMaxAndMeanInLevels() {
        let reference = [UInt8](repeating: 100, count: 1000)
        XCTAssertTrue(PixelMetrics.compare(native: reference, reference: reference).passed)
        var native = reference
        native[0] = 102  // max abs 2 (allowed), mean 0.002
        XCTAssertTrue(PixelMetrics.compare(native: native, reference: reference).passed)
        native[0] = 103  // max abs 3 > 2
        XCTAssertFalse(PixelMetrics.compare(native: native, reference: reference).passed)
        var wide = reference
        for i in 0 ..< 101 { wide[i] = 101 }  // max abs 1, mean 0.101 > 0.1
        let m = PixelMetrics.compare(native: wide, reference: reference)
        XCTAssertEqual(m.maxAbsDiff, 1)
        XCTAssertEqual(m.differingValues, 101)
        XCTAssertFalse(m.passed)
    }

    func testPixelCheckRejectsAShapeMismatchAndHonoursDecisiveness() {
        let a = [UInt8](repeating: 1, count: 12)
        XCTAssertFalse(ComponentCheck.pixels("p", native: a, reference: a, nativeShape: [2, 2, 3], referenceShape: [4, 1, 3]).passed)
        XCTAssertTrue(ComponentCheck.pixels("p", native: a, reference: a, nativeShape: [2, 2, 3], referenceShape: [2, 2, 3]).passed)
        let far = [UInt8](repeating: 200, count: 12)
        XCTAssertFalse(ComponentCheck.pixels("p", native: a, reference: far, nativeShape: [2, 2, 3], referenceShape: [2, 2, 3]).passed)
        let diagnostic = ComponentCheck.pixels("p", native: a, reference: far, nativeShape: [2, 2, 3], referenceShape: [2, 2, 3], decisive: false)
        XCTAssertTrue(diagnostic.passed && !diagnostic.decisive, "a diagnostic row records the metrics but cannot fail a case")
        XCTAssertEqual(diagnostic.pixels?.maxAbsDiff, 199)
    }

    /// Owner decision, slice 4: only native compiled vs MFLUX compiled decides a prediction; every eager row is a diagnostic.
    func testCompiledDecisiveRowsDecideOnlyTheCompiledComparison() {
        Device.withDefaultDevice(.cpu) {
            let reference = MLXArray([Float](repeating: 1, count: 1000))
            let drifted = MLXArray([Float](repeating: 1.1, count: 1000))  // far outside any limit
            let checks = ComponentCheck.compiledDecisive(
                "o", nativeEager: drifted, nativeCompiled: reference, referenceEager: drifted, referenceCompiled: reference, limits: .fullTransformerOutput)
            XCTAssertEqual(checks.filter(\.decisive).count, 1)
            XCTAssertTrue(checks[0].name.contains("native compiled vs MFLUX compiled"))
            XCTAssertTrue(ComponentCaseResult(id: "x", checks: checks).passed, "eager drift alone must not fail the case")
            let broken = ComponentCheck.compiledDecisive(
                "o", nativeEager: reference, nativeCompiled: drifted, referenceEager: reference, referenceCompiled: reference, limits: .fullTransformerOutput)
            XCTAssertFalse(ComponentCaseResult(id: "x", checks: broken).passed, "a compiled mismatch fails even if every eager row is exact")
        }
    }
}

final class GenerationCasesTests: XCTestCase {
    func testDeclaredGenerationCasesCoverTheDeclaredGrid() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/reference/generation_cases.json")
        let cases = try GenerationCases.load(url)
        for list in [cases.denoiseCases, cases.pipelineCases] {
            XCTAssertEqual(Set(list.map(\.steps)).isSuperset(of: [GenerationRequest.stepRange.lowerBound, GenerationRequest.stepRange.upperBound]), true)
            XCTAssertTrue(list.contains { $0.steps > GenerationRequest.stepRange.lowerBound && $0.steps < GenerationRequest.stepRange.upperBound })
            XCTAssertTrue(list.contains { $0.width == $0.height }, "square")
            XCTAssertTrue(list.contains { $0.height > $0.width }, "portrait")
            XCTAssertTrue(list.contains { $0.width > $0.height }, "landscape")
            XCTAssertEqual(Set(list.map(\.prompt)).isSuperset(of: ["plain", "long-near-limit"]), true)
            XCTAssertTrue(list.contains { $0.prompt.hasPrefix("unicode") })
            XCTAssertEqual(Set(list.map(\.seed)).count, list.count, "distinct seeds")
            for c in list { XCTAssertNoThrow(try GenerationRequest(prompt: "", seed: c.seed, width: c.width, height: c.height, steps: c.steps).validate()) }
        }
        XCTAssertEqual(Set(cases.vaeCases.map { "\($0.width)x\($0.height)" }).count, 7 - 1, "six offered sizes (the noise case repeats 1024x1024)")
        XCTAssertTrue(cases.vaeCases.contains { $0.latentSource == "seeded-noise" })
    }
}
