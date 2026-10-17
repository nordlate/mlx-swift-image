import MLX
import XCTest
@testable import ZImage

final class VAELayoutTests: XCTestCase {
    func testExpectedTensorsMatchTheSnapshotIndex() {
        let expected = VAELayout.expectedTensors()
        XCTAssertEqual(expected.count, 260)
        XCTAssertEqual(VAELayout.quantizedLinears.count, 8)
        XCTAssertEqual(expected["decoder.conv_in.conv.weight"]?.shape, [512, 3, 3, 16])
        XCTAssertEqual(expected["decoder.mid_block.attentions.0.to_q.weight"]?.dtype, "U32")
        XCTAssertEqual(expected["decoder.mid_block.attentions.0.to_q.weight"]?.shape, [512, 64])
        XCTAssertEqual(expected["decoder.mid_block.attentions.0.to_out.0.scales"]?.shape, [512, 8])
        XCTAssertEqual(expected["decoder.up_blocks.2.resnets.0.conv_shortcut.weight"]?.shape, [256, 1, 1, 512])
        XCTAssertEqual(expected["decoder.up_blocks.2.resnets.0.norm1.weight"]?.shape, [512])
        XCTAssertNil(expected["decoder.up_blocks.2.resnets.1.conv_shortcut.weight"], "only the first resnet of a block changes width")
        XCTAssertNil(expected["decoder.up_blocks.3.upsamplers.0.conv.weight"], "the last up block does not upsample")
        XCTAssertNotNil(expected["decoder.up_blocks.2.upsamplers.0.conv.weight"])
        XCTAssertEqual(expected["decoder.conv_out.conv.weight"]?.shape, [3, 3, 3, 128])
        XCTAssertEqual(expected["encoder.conv_out.conv2d.weight"]?.shape, [32, 3, 3, 512])
    }

    func testRealSnapshotHeadersMatchLayout() throws {
        let model = try TestSupport.requireModel()
        let (inspection, shards) = try VAEWeights.inspect(modelDirectory: model)
        XCTAssertEqual(inspection.tensorCount, 260)
        XCTAssertEqual(inspection.quantizedLinearCount, 8)
        XCTAssertEqual(shards, ["0.safetensors"])
        XCTAssertEqual(inspection.quantizationLevel, "4")
        XCTAssertEqual(inspection.bits, 4)
        XCTAssertEqual(inspection.groupSize, 64)
    }

    /// Smoke test of the whole decoder graph on the CPU stream (no Metal library under plain `swift test`): a tiny latent
    /// decodes to eight times its size with three channels, and the sampler-latent entry point unpacks `[16, 1, h, w]`.
    func testDecodeShapeOnTheRealSnapshot() throws {
        let model = try TestSupport.requireModel()
        try Device.withDefaultDevice(.cpu) {
            let vae = ZImageVAE(weights: try VAEWeights.load(modelDirectory: model))
            let decoded = vae.decode(samplerLatents: MLXArray.zeros([16, 1, 2, 4]).asType(.bfloat16))
            eval(decoded)
            XCTAssertEqual(decoded.shape, [1, 3, 16, 32])
            XCTAssertEqual(decoded.dtype, .bfloat16)
        }
    }

    func testAttentionScaleIsTheBfloat16ValueMfluxPasses() {
        Device.withDefaultDevice(.cpu) {
            // Python: float(1.0 / mx.sqrt(mx.array(512, dtype=mx.bfloat16))) == 0.044189453125 (not 1 / sqrt(512) = 0.0441942).
            XCTAssertEqual(ZImageVAE.attentionScale(channels: 512, dtype: .bfloat16), 0.044189453125)
        }
    }
}

final class SliceFourToleranceTests: XCTestCase {
    /// The VAE and pixel rows of the Stage 3 table in NATIVE_MLX_PLAN.md; a change here is a loosening needing an owner record.
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

final class GenerationRequestTests: XCTestCase {
    func testStepsAreTheOfferedRange() {
        func request(steps: Int, width: Int = 1024, height: Int = 1024) -> GenerationRequest {
            GenerationRequest(prompt: "p", seed: 1, width: width, height: height, steps: steps)
        }
        XCTAssertNoThrow(try request(steps: 4).validate())
        XCTAssertNoThrow(try request(steps: 8).validate())
        XCTAssertNoThrow(try request(steps: 12).validate())
        XCTAssertThrowsError(try request(steps: 3).validate())
        XCTAssertThrowsError(try request(steps: 13).validate())
        XCTAssertNoThrow(try request(steps: 8, width: 864, height: 1152).validate())
        XCTAssertThrowsError(try request(steps: 8, width: 1000).validate(), "latents are patched by 2 after /8: sizes are multiples of 16")
        XCTAssertThrowsError(try request(steps: 8, height: 0).validate())
    }

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

final class ImageOutputTests: XCTestCase {
    func testPixelConversionClipsAndRoundsHalfToEven() {
        Device.withDefaultDevice(.cpu) {
            // decoded x -> clip(x / 2 + 0.5, 0, 1) * 255 rounded half to even.
            let values: [Float] = [-2, -1, 0, 0.5, 1, 3]
            let decoded = MLXArray(values + values + values).reshaped(1, 3, 1, 6).asType(.bfloat16)
            let pixels = ZImageImage.pixels(decoded: decoded)
            XCTAssertEqual(pixels.shape, [1, 6, 3])
            XCTAssertEqual(pixels.dtype, .uint8)
            // x = 0 -> 127.5 -> 128 (half to even); x = 0.5 -> 0.75 -> 191.25 -> 191; clipped ends 0 and 255.
            XCTAssertEqual(pixels[0, 0..., 0].asArray(UInt8.self), [0, 0, 128, 191, 255, 255])
        }
    }

    func testPNGRoundTripIsLossless() throws {
        try Device.withDefaultDevice(.cpu) {
            let (h, w) = (5, 7)
            let source = (0 ..< h * w * 3).map { UInt8(($0 * 37 + 11) % 256) }
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("zimage-native-test-\(UUID().uuidString).png")
            defer { try? FileManager.default.removeItem(at: url) }
            try ZImageImage.writePNG(pixels: MLXArray(source).reshaped(h, w, 3).asType(.uint8), to: url)
            let image = try ZImageImage.readPNG(at: url)
            XCTAssertEqual([image.width, image.height], [w, h])
            XCTAssertEqual(image.bytes, source)
            let info = try ZImageImage.pngInfo(at: url)
            XCTAssertEqual([info.width, info.height, info.bitDepth, info.colorType], [w, h, 8, 2], "8-bit RGB without alpha, like PIL's RGB PNG")
        }
    }

    func testWritePNGRejectsOtherLayouts() throws {
        try Device.withDefaultDevice(.cpu) {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("zimage-native-test-reject.png")
            XCTAssertThrowsError(try ZImageImage.writePNG(pixels: MLXArray.zeros([4, 4, 4]).asType(.uint8), to: url))
            XCTAssertThrowsError(try ZImageImage.writePNG(pixels: MLXArray.zeros([4, 4, 3]), to: url))
        }
    }
}

/// Retained-memory fix (memory-investigation/): `compile` copies a still-lazy captured weight into every input signature's
/// graph, so the compiled entry points must evaluate the weights first. The Metal-side effect (active memory flat across
/// differently shaped jobs) is measured by zimage-retention; this checks the materialization step itself on the CPU stream.
final class WeightMaterializationTests: XCTestCase {
    func testMaterializeWeightsReadsLazilyLoadedTensors() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("zimage-lazy-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: ["w": MLXArray.zeros([2048, 2048], dtype: .float32)], url: url)  // 16 MiB
        let tensors = try loadArrays(url: url)
        let inspection = TransformerInspection(
            tensorCount: 1, quantizedLinearCount: 0, shards: [:], quantizationLevel: "4", mfluxVersion: nil, bits: 4, groupSize: 64)
        let transformer = ZImageTransformer(weights: TransformerWeights(tensors: tensors, inspection: inspection))
        let before = Memory.activeMemory
        Device.withDefaultDevice(.cpu) { transformer.materializeWeights() }
        XCTAssertGreaterThanOrEqual(Memory.activeMemory - before, 15 * 1024 * 1024, "the lazy tensor was not read by materializeWeights")
    }
}

extension WeightMaterializationTests {
    /// Guards the fix itself: building the production compiled prediction must read the lazy weights, even if it is never called.
    func testCompiledForwardReadsLazilyLoadedTensors() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("zimage-lazy-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: ["w": MLXArray.zeros([2048, 2048], dtype: .float32)], url: url)  // 16 MiB
        let inspection = TransformerInspection(
            tensorCount: 1, quantizedLinearCount: 0, shards: [:], quantizationLevel: "4", mfluxVersion: nil, bits: 4, groupSize: 64)
        let transformer = ZImageTransformer(weights: TransformerWeights(tensors: try loadArrays(url: url), inspection: inspection))
        let before = Memory.activeMemory
        Device.withDefaultDevice(.cpu) { _ = transformer.compiledForward() }
        XCTAssertGreaterThanOrEqual(Memory.activeMemory - before, 15 * 1024 * 1024, "compiledForward did not materialize the weights")
    }
}
