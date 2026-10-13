import MLX
import XCTest
@testable import ZImageNative

final class TransformerLayoutTests: XCTestCase {
    func testExpectedTensorsMatchTheSnapshotIndex() {
        let expected = TransformerLayout.expectedTensors()
        // 276 quantized linears: 270 in the 32 modulated/context blocks, x embedder, 2 final layer, cap embedder, 2 timestep.
        XCTAssertEqual(TransformerLayout.allMatrices.count, 276)
        // 4 tensors per biased and 3 per unbiased linear, plus norms, pad tokens and the caption RMSNorm weight.
        XCTAssertEqual(expected.count, 1073)
        XCTAssertEqual(expected["layers.0.attention.to_q.weight"]?.shape, [3840, 480])
        XCTAssertEqual(expected["layers.0.attention.to_q.weight"]?.dtype, "U32")
        XCTAssertEqual(expected["layers.29.feed_forward.w2.scales"]?.shape, [3840, 160])
        XCTAssertEqual(expected["layers.0.adaLN_modulation.0.bias"]?.shape, [15360])
        XCTAssertEqual(expected["all_x_embedder.2-1.weight"]?.shape, [3840, 8])
        XCTAssertEqual(expected["x_pad_token"]?.shape, [1, 3840])
        XCTAssertNil(expected["context_refiner.0.adaLN_modulation.0.weight"], "the context refiner has no modulation")
        XCTAssertNotNil(expected["noise_refiner.1.adaLN_modulation.0.weight"])
    }

    func testRealSnapshotHeadersMatchLayout() throws {
        let model = try TestSupport.requireModel()
        let (inspection, shards) = try TransformerWeights.inspect(modelDirectory: model)
        XCTAssertEqual(inspection.tensorCount, 1073)
        XCTAssertEqual(inspection.quantizedLinearCount, 276)
        XCTAssertEqual(shards, ["0.safetensors", "1.safetensors"])
        XCTAssertEqual(inspection.quantizationLevel, "4")
        XCTAssertEqual(inspection.bits, 4)
        XCTAssertEqual(inspection.groupSize, 64)
    }
}

final class SlicePredeclaredLimitTests: XCTestCase {
    /// The values of the Stage 3 table in NATIVE_MLX_PLAN.md; a change here is a loosening that needs an owner-approved record.
    func testLimitsEqualThePredeclaredTable() {
        XCTAssertEqual(ParityTolerance.patchTimestepEmbeddingRelativeL2, 1e-3)
        XCTAssertEqual(ParityTolerance.singleBlockRelativeL2, 1e-3)
        XCTAssertEqual(ParityTolerance.schedulerLatentPerStepRelativeL2, 5e-3)
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

/// Graph-building tests that run on the CPU stream, so they work under plain `swift test` (no Metal library there).
final class CPUComponentTests: XCTestCase {
    func testSchedulerShape() {
        Device.withDefaultDevice(.cpu) {
            let s = LinearScheduler(steps: 8, width: 1024, height: 1024)
            let sigmas = s.sigmas.asArray(Float.self)
            XCTAssertEqual(sigmas.count, 9)
            XCTAssertEqual(sigmas.last, 0)
            XCTAssertEqual(sigmas[0], 1, accuracy: 1e-6)
            XCTAssertEqual(zip(sigmas, sigmas.dropFirst()).filter { $0 <= $1 }.count, 0, "sigmas strictly decrease")
            XCTAssertEqual(s.timesteps.asArray(Float.self), (0..<8).map(Float.init))
            // step: latents + noise * (sigma[t+1] - sigma[t]), in the latents' dtype.
            let latents = MLXArray([Float](repeating: 1, count: 4)).asType(.bfloat16)
            let next = s.step(noise: MLXArray([Float](repeating: 2, count: 4)), timestep: 0, latents: latents)
            XCTAssertEqual(next.dtype, .bfloat16)
            XCTAssertEqual(next.asType(.float32).asArray(Float.self)[0], 1 + 2 * (sigmas[1] - sigmas[0]), accuracy: 0.02)
        }
    }

    func testNoiseShapeAndDtype() {
        Device.withDefaultDevice(.cpu) {
            let n = ZImageLatentCreator.createNoise(seed: 42, height: 672, width: 896)
            XCTAssertEqual(n.shape, [16, 1, 84, 112])
            XCTAssertEqual(n.dtype, .bfloat16)
            XCTAssertTrue(ComponentCheck.bytesEqual(n, ZImageLatentCreator.createNoise(seed: 42, height: 672, width: 896)))
            XCTAssertFalse(ComponentCheck.bytesEqual(n, ZImageLatentCreator.createNoise(seed: 43, height: 672, width: 896)))
        }
    }

    func testPatchifyGeometryAndPadding() {
        Device.withDefaultDevice(.cpu) {
            // 16x16 latent = 64 tokens (pad 0); 6 caption rows pad to 32.
            let image = MLXArray(0 ..< Int32(16 * 16 * 16)).asType(.float32).reshaped(16, 1, 16, 16)
            let caption = MLXArray.ones([6, 2560])
            let p = ZImageTransformer.patchify(image: image, captionFeatures: caption)
            XCTAssertEqual(p.imageTokens.shape, [64, 64])
            XCTAssertEqual(p.captionFeatures.shape, [32, 2560])
            XCTAssertEqual(p.imagePositionIDs[0].asArray(Int32.self), [33, 0, 0], "image ids start after the padded caption")
            XCTAssertEqual(p.imagePositionIDs[9].asArray(Int32.self), [33, 1, 1])
            XCTAssertEqual(p.captionPositionIDs[5].asArray(Int32.self), [6, 0, 0])
            XCTAssertEqual(p.captionPadMask.asType(.int32).sum().item(Int.self), 26)
            // First token = the 2x2 patch at the top-left of every channel, in (pF, pH, pW, C) order.
            let first = p.imageTokens[0].asArray(Float.self)
            XCTAssertEqual(first[0], image[0, 0, 0, 0].item(Float.self))  // c=0 at (0,0)
            XCTAssertEqual(first[1], image[1, 0, 0, 0].item(Float.self))  // c=1 at (0,0)
            XCTAssertEqual(first[16], image[0, 0, 0, 1].item(Float.self))  // next pW
            XCTAssertEqual(first[32], image[0, 0, 1, 0].item(Float.self))  // next pH

            // 8x8 latent = 16 tokens: padded to 32 by repeating the last token, pad ids are zero.
            let small = MLXArray.ones([16, 1, 8, 8])
            let q = ZImageTransformer.patchify(image: small, captionFeatures: caption)
            XCTAssertEqual(q.imageTokens.shape, [32, 64])
            XCTAssertEqual(q.imagePadMask.asType(.int32).sum().item(Int.self), 16)
            XCTAssertEqual(q.imagePositionIDs[31].asArray(Int32.self), [0, 0, 0])
        }
    }

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

    func testFrequencyEmbeddingOfZeroIsCosOnesSinZeros() {
        Device.withDefaultDevice(.cpu) {
            let e = ZImageTransformer.frequencyEmbedding(MLXArray([0] as [Float]), dim: 256, maxPeriod: 10_000)
            XCTAssertEqual(e.shape, [1, 256])
            let values = e.asArray(Float.self)
            XCTAssertEqual(values[..<128].allSatisfy { $0 == 1 }, true)
            XCTAssertEqual(values[128...].allSatisfy { $0 == 0 }, true)
        }
    }
}
