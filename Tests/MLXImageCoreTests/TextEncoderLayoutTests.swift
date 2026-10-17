import XCTest
@testable import MLXImageCore

final class TextEncoderLayoutTests: XCTestCase {
    func testExpectedTensorCountMatchesSnapshotIndex() {
        // 5 top-level (embed weight/scales/biases, norm, inv_freq) + 25 per layer x 36.
        XCTAssertEqual(TextEncoderLayout.expectedTensors().count, 905)
        let q = TextEncoderLayout.expectedTensors()["layers.0.self_attn.q_proj.weight"]
        XCTAssertEqual(q?.shape, [4096, 320])
        XCTAssertEqual(q?.dtype, "U32")
        XCTAssertEqual(TextEncoderLayout.expectedTensors()["layers.35.mlp.down_proj.scales"]?.shape, [2560, 152])
    }
}
