import XCTest
@testable import MLXImageCore

final class TextEncoderLayoutTests: XCTestCase {
    func testExpectedTensorCountMatchesSnapshotIndex() {
        // 5 top-level (embed weight/scales/biases, norm, inv_freq) + 25 per layer x 36.
        XCTAssertEqual(TextEncoderLayout.qwen3_4B.expectedTensors().count, 905)
        let q = TextEncoderLayout.qwen3_4B.expectedTensors()["layers.0.self_attn.q_proj.weight"]
        XCTAssertEqual(q?.shape, [4096, 320])
        XCTAssertEqual(q?.dtype, "U32")
        XCTAssertEqual(TextEncoderLayout.qwen3_4B.expectedTensors()["layers.35.mlp.down_proj.scales"]?.shape, [2560, 152])
        XCTAssertEqual(TextEncoderLayout.qwen3_4B.hiddenSize, 2560)
        XCTAssertEqual(TextEncoderLayout.qwen3_4B.intermediateSize, 9728)
    }

    func testQwen3_8BDiffersOnlyInWidths() {
        let small = TextEncoderLayout.qwen3_4B.expectedTensors(), large = TextEncoderLayout.qwen3_8B.expectedTensors()
        XCTAssertEqual(Set(small.keys), Set(large.keys))
        XCTAssertEqual(large["layers.35.mlp.down_proj.scales"]?.shape, [4096, 192])
        XCTAssertEqual(large["layers.0.mlp.up_proj.weight"]?.shape, [12288, 512])
        XCTAssertEqual(large["rotary_emb.inv_freq"]?.shape, small["rotary_emb.inv_freq"]?.shape)
    }
}
