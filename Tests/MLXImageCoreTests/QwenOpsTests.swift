import MLX
import XCTest
@testable import MLXImageCore

final class QwenOpsTests: XCTestCase {
    func testRotateHalf() {
        Device.withDefaultDevice(.cpu) {
            let x = MLXArray([1, 2, 3, 4] as [Float]).reshaped(1, 1, 1, 4)
            XCTAssertEqual(QwenOps.rotateHalf(x).asArray(Float.self), [-3, -4, 1, 2])
        }
    }

    func testQuantizedMatricesAreTheEmbeddingThenSevenPerLayer() {
        let matrices = TextEncoderLayout.qwen3_4B.quantizedMatrices()
        XCTAssertEqual(matrices.count, 1 + 36 * 7)
        XCTAssertEqual(matrices.first?.name, "embed_tokens")
        XCTAssertEqual(matrices[1].name, "layers.0.self_attn.q_proj")
        XCTAssertEqual(matrices.last?.name, "layers.35.mlp.down_proj")
    }
}
