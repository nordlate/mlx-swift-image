import MLX
import XCTest
import MLXImageCore
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
