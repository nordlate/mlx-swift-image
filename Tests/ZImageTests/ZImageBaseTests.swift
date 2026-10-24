import MLX
import XCTest
import MLXImageCore
@testable import ZImage

final class ZImageBaseRequestTests: XCTestCase {
    private func request(steps: Int, guidance: Float = 0, negative: String? = nil) -> GenerationRequest {
        GenerationRequest(prompt: "p", seed: 1, width: 512, height: 512, steps: steps, guidance: guidance, negativePrompt: negative)
    }

    func testBaseStepsAndGuidanceValidate() {
        for steps in [1, 28, 50] { XCTAssertNoThrow(try request(steps: steps, guidance: 4, negative: "n").validate(for: .base)) }
        XCTAssertNoThrow(try request(steps: 28).validate(for: .base), "no guidance is one unguided pass, as mflux's default 0.0")
        XCTAssertThrowsError(try request(steps: 0).validate(for: .base))
        XCTAssertThrowsError(try request(steps: 51).validate(for: .base))
        XCTAssertThrowsError(try request(steps: 28, guidance: -1).validate(for: .base))
        XCTAssertThrowsError(try request(steps: 28, guidance: .nan).validate(for: .base))
    }

    func testTurboKeepsItsRangeAndRejectsGuidance() {
        XCTAssertNoThrow(try request(steps: 8).validate())
        XCTAssertThrowsError(try request(steps: 28).validate())
        XCTAssertThrowsError(try request(steps: 8, guidance: 4).validate(for: .turbo))
        XCTAssertThrowsError(try request(steps: 8, negative: "n").validate(for: .turbo))
        XCTAssertEqual(GenerationRequest.stepRange, 4 ... 12)
    }

    func testGuidanceRunsOnlyAboveOneAndBlankNegativePromptIsASpace() {
        XCTAssertFalse(request(steps: 8, guidance: 0).usesGuidance)
        XCTAssertFalse(request(steps: 8, guidance: 1).usesGuidance)
        XCTAssertTrue(request(steps: 8, guidance: 1.01).usesGuidance)
        XCTAssertEqual(request(steps: 8, guidance: 4).effectiveNegativePrompt, " ")
        XCTAssertEqual(request(steps: 8, guidance: 4, negative: "  \n").effectiveNegativePrompt, " ")
        XCTAssertEqual(request(steps: 8, guidance: 4, negative: " blurry ").effectiveNegativePrompt, " blurry ")
    }
}

final class ZImageDenoiserGuardTests: XCTestCase {
    private func denoiser(_ variant: ZImageVariant) -> ZImageDenoiser {
        let inspection = TransformerInspection(
            tensorCount: 0, quantizedLinearCount: 0, shards: [:], quantizationLevel: "4", mfluxVersion: nil, bits: 4, groupSize: 64)
        return ZImageDenoiser(transformer: ZImageTransformer(weights: TransformerWeights(tensors: [:], inspection: inspection)), variant: variant)
    }

    func testGuidedBaseNeedsNegativeFeaturesAndTurboRejectsGuidance() throws {
        try Device.withDefaultDevice(.cpu) {
            let caption = MLXArray.zeros([4, 2560])
            let guided = GenerationRequest(prompt: "p", seed: 1, width: 64, height: 64, steps: 2, guidance: 4)
            XCTAssertThrowsError(try denoiser(.base).denoise(request: guided, captionFeatures: caption))
            var turbo = guided
            turbo.steps = 8
            XCTAssertThrowsError(try denoiser(.turbo).denoise(request: turbo, captionFeatures: caption, negativeCaptionFeatures: caption))
        }
    }
}

final class FlowMatchEulerSchedulerTests: XCTestCase {
    func testEmpiricalMuFollowsMfluxsTwoBranches() {
        // 1024x1024: seq len 4096, interpolated between the 10- and 200-step lines; above 4300 tokens the 200-step line alone.
        let m200 = 0.00016927 * 4096 + 0.45666666, m10 = 8.73809524e-05 * 4096 + 1.89833333
        let a = (m200 - m10) / 190, b = m200 - 200 * a
        XCTAssertEqual(FlowMatchEulerScheduler.empiricalMu(imageSeqLen: 4096, steps: 28), a * 28 + b, accuracy: 1e-12)
        XCTAssertEqual(FlowMatchEulerScheduler.empiricalMu(imageSeqLen: 9216, steps: 28), 0.00016927 * 9216 + 0.45666666, accuracy: 1e-12)
        XCTAssertEqual(FlowMatchEulerScheduler.empiricalMu(imageSeqLen: 4300, steps: 1) != FlowMatchEulerScheduler.empiricalMu(imageSeqLen: 4301, steps: 1), true)
    }

    func testSigmasStartAtOneEndAtZeroAndDecrease() {
        Device.withDefaultDevice(.cpu) {
            for (steps, size) in [(1, 512), (28, 512), (50, 1024)] {
                let sigmas = FlowMatchEulerScheduler(steps: steps, width: size, height: size).sigmas.asArray(Float.self)
                XCTAssertEqual(sigmas.count, steps + 1)
                XCTAssertEqual(sigmas.first, 1)
                XCTAssertEqual(sigmas.last, 0)
                XCTAssertEqual(zip(sigmas, sigmas.dropFirst()).allSatisfy { $0 > $1 }, true)
            }
        }
    }

    func testTurboStillUsesTheLinearSchedulerAndBaseTheFlowMatchOne() {
        Device.withDefaultDevice(.cpu) {
            let weights = TransformerWeights(
                tensors: [:], inspection: TransformerInspection(
                    tensorCount: 0, quantizedLinearCount: 0, shards: [:], quantizationLevel: "4", mfluxVersion: nil, bits: 4, groupSize: 64))
            let request = GenerationRequest(prompt: "p", seed: 1, width: 512, height: 512, steps: 8)
            let turbo = ZImageDenoiser(transformer: ZImageTransformer(weights: weights)).scheduler(for: request)
            let base = ZImageDenoiser(transformer: ZImageTransformer(weights: weights), variant: .base).scheduler(for: request)
            XCTAssertTrue(turbo is LinearScheduler)
            XCTAssertTrue(base is FlowMatchEulerScheduler)
            XCTAssertNotEqual(turbo.sigmas.asArray(Float.self), base.sigmas.asArray(Float.self))
        }
    }
}

final class ZImageBaseModelTests: XCTestCase {
    /// The Base snapshot has Turbo's layout: the loaders read it and report the same tensor counts and q4 quantization.
    func testBaseSnapshotLoadsThroughTheSharedLoaders() throws {
        let model = try TestSupport.requireBaseModel()
        let (transformer, _) = try TransformerWeights.inspect(modelDirectory: model)
        XCTAssertEqual(transformer.bits, 4)
        XCTAssertEqual(transformer.groupSize, 64)
        let (vae, _) = try VAEWeights.inspect(modelDirectory: model)
        XCTAssertEqual(vae.tensorCount, 260)
    }

    /// A short guided job: the negative prompt changes the result, and both runs decode to the requested size.
    func testGuidanceChangesTheImage() throws {
        let model = try TestSupport.requireBaseModel()
        let generator = try ZImageGenerator(modelDirectory: model, variant: .base)
        let plain = try generator.generate(GenerationRequest(prompt: "a red bicycle", seed: 3, width: 256, height: 256, steps: 2))
        let guided = try generator.generate(
            GenerationRequest(prompt: "a red bicycle", seed: 3, width: 256, height: 256, steps: 2, guidance: 4, negativePrompt: "blurry"))
        XCTAssertEqual(plain.pixels.shape, [256, 256, 3])
        XCTAssertEqual(guided.pixels.shape, [256, 256, 3])
        XCTAssertNotEqual(plain.pixels.asArray(UInt8.self), guided.pixels.asArray(UInt8.self))
    }
}
