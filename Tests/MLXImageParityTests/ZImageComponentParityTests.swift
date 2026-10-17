import MLX
import MLXFast
import XCTest
import MLXImageCore
@testable import ZImage
@testable import MLXImageParity

final class SlicePredeclaredLimitTests: XCTestCase {
    /// The values of the pre-declared tolerance table; a change here is a loosening that needs an explicit, recorded approval.
    func testLimitsEqualThePredeclaredTable() {
        XCTAssertEqual(ParityTolerance.patchTimestepEmbeddingRelativeL2, 1e-3)
        XCTAssertEqual(ParityTolerance.singleBlockRelativeL2, 1e-3)
        XCTAssertEqual(ParityTolerance.schedulerLatentPerStepRelativeL2, 5e-3)
        XCTAssertEqual(ParityTolerance.finalProjectionRelativeL2, 1e-3)
        XCTAssertEqual(ParityTolerance.fullTransformerOutputRelativeL2, 2e-3)
        XCTAssertEqual(ParityLimits.finalProjection.relativeL2, 1e-3)
        XCTAssertEqual(ParityLimits.fullTransformerOutput.relativeL2, 2e-3)
        XCTAssertNil(ParityLimits.fullTransformerOutput.normalizedMaxAbs)
        XCTAssertEqual(ParityTolerance.embeddingRelativeL2, 1e-3)
        XCTAssertEqual(ParityTolerance.embeddingNormalizedMaxAbs, 1e-2)
        XCTAssertNil(ParityLimits.singleBlock.normalizedMaxAbs)
    }

    func testRelativeL2OnlyRowsIgnoreMaxAbs() {
        var reference = [Float](repeating: 1, count: 100_000)
        reference[0] = 100
        var native = reference
        native[1] = 1.5  // max abs / max ref = 5e-3 and relL2 ~ 1.6e-3: within text limits' max abs, over the L2 limit
        XCTAssertFalse(ParityMetrics.compare(native: native, reference: reference, limits: .singleBlock).passed)
        native[1] = 1.1  // relL2 ~ 3e-4
        XCTAssertTrue(ParityMetrics.compare(native: native, reference: reference, limits: .singleBlock).passed)
        // Text embeddings keep deciding on both metrics.
        var spike = reference
        spike[2] = 3  // max abs 2 / 100 = 2e-2 > 1e-2, relL2 ~ 6e-3 > 1e-3 (fails both)
        XCTAssertFalse(ParityMetrics.compare(native: spike, reference: reference).passed)
    }

    func testCaseVerdictNeedsADecisiveCheckAndIgnoresDiagnostics() {
        let pass = CheckResult(name: "a", kind: "exact", decisive: true, passed: true, metrics: nil, detail: nil)
        let fail = CheckResult(name: "b", kind: "exact", decisive: true, passed: false, metrics: nil, detail: nil)
        let diagnostic = CheckResult(name: "d", kind: "diagnostic", decisive: false, passed: true, metrics: nil, detail: nil)
        XCTAssertTrue(ComponentCaseResult(id: "x", checks: [pass, diagnostic]).passed)
        XCTAssertFalse(ComponentCaseResult(id: "x", checks: [pass, fail]).passed)
        XCTAssertFalse(ComponentCaseResult(id: "x", checks: [diagnostic]).passed, "a case with no decisive check cannot pass")
        XCTAssertFalse(ComponentCaseResult(id: "x", checks: []).passed)
    }
}

final class ComponentCheckTests: XCTestCase {
    func testExactComparisonDetectsASingleBitAndLayoutChanges() {
        Device.withDefaultDevice(.cpu) {
            let a = MLXArray([1, 2, 3, 4] as [Float])
            XCTAssertTrue(ComponentCheck.exact("same", native: a, reference: MLXArray([1, 2, 3, 4] as [Float])).passed)
            let off = ComponentCheck.exact("bit", native: a, reference: MLXArray([1, 2, 3, Float(bitPattern: Float(4).bitPattern + 1)]))
            XCTAssertFalse(off.passed)
            XCTAssertTrue(off.detail?.contains("1 of 4") == true)
            XCTAssertFalse(ComponentCheck.exact("dtype", native: a, reference: a.asType(.bfloat16)).passed)
            XCTAssertFalse(ComponentCheck.exact("shape", native: a, reference: a.reshaped(2, 2)).passed)
        }
    }
}

final class Slice3Tests: XCTestCase {
    func testDeclaredSlice3CaseLists() throws {
        let json = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/reference/component_cases.json")
        let cases = try ComponentCases.load(json)
        XCTAssertEqual(cases.blockMiddleCases.count, 3)
        XCTAssertEqual(cases.blockLastCases.count, 3)
        XCTAssertEqual(cases.finalCases.count, 3)
        XCTAssertEqual(cases.outputCases.count, 5)
        let ids = (cases.blockCases + cases.blockMiddleCases + cases.blockLastCases + cases.finalCases + cases.outputCases).map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count, "every case has its own reference file name")
        XCTAssertTrue(cases.outputCases.contains { $0.width != $0.height }, "a non-square size is covered")
    }

    func testUnpatchifyInvertsPatchify() {
        Device.withDefaultDevice(.cpu) {
            // 24x16 latent = 12 x 8 = 96 tokens (a multiple of 32: no padding); and 8x8 = 16 tokens padded to 32.
            for (h, w) in [(24, 16), (8, 8)] {
                let image = MLXArray(0 ..< Int32(16 * h * w)).asType(.float32).reshaped(16, 1, h, w)
                let p = ZImageTransformer.patchify(image: image, captionFeatures: MLXArray.ones([4, 2560]))
                let back = ZImageTransformer.unpatchify(p.imageTokens, size: [1, h, w])
                XCTAssertEqual(back.shape, [16, 1, h, w])
                XCTAssertTrue(ComponentCheck.bytesEqual(back, image), "\(h)x\(w): padding rows are dropped, the rest is restored")
            }
        }
    }

    func testEagerAndCompiledChecksDecideOnlyTheEagerComparison() {
        Device.withDefaultDevice(.cpu) {
            let reference = MLXArray((0 ..< 1000).map { Float($0) + 1 })
            let drifted = reference * 1.01  // relative L2 1e-2: over every limit
            let checks = ComponentCheck.eagerAndCompiled(
                "x", nativeEager: reference, nativeCompiled: drifted, referenceEager: reference, referenceCompiled: drifted, limits: .fullTransformerOutput)
            XCTAssertEqual(checks.count, 4)
            XCTAssertEqual(checks.filter(\.decisive).map(\.name), ["x"])
            XCTAssertTrue(ComponentCaseResult(id: "c", checks: checks).passed, "compiled drift is recorded, not failed")
            XCTAssertEqual(checks[2].metrics?.passed, false, "the drift is over the limit and says so")
            XCTAssertEqual(checks[2].metrics?.relativeL2 ?? 0, 1e-2, accuracy: 1e-4)
            let broken = ComponentCheck.eagerAndCompiled(
                "x", nativeEager: drifted, nativeCompiled: drifted, referenceEager: reference, referenceCompiled: drifted, limits: .fullTransformerOutput)
            XCTAssertFalse(ComponentCaseResult(id: "c", checks: broken).passed, "a native eager mismatch still fails")
        }
    }

    func testFinalLayerNormIsUnaffineLayerNorm() {
        Device.withDefaultDevice(.cpu) {
            let x = MLXArray((0 ..< 8).map { Float($0) }).reshaped(1, 2, 4)
            let y = MLXFast.layerNorm(x, eps: TransformerLayout.finalNormEps)
            XCTAssertEqual(y.mean(axis: -1).asArray(Float.self).allSatisfy { abs($0) < 1e-6 }, true)
        }
    }
}
