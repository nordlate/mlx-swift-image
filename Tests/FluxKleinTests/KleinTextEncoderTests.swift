import Foundation
import MLX
import XCTest
@testable import FluxKlein
@testable import MLXImageCore

final class KleinEncoderPieceTests: XCTestCase {
    func testAdditiveMaskIsCausalPlusPaddingInTheStateDType() {
        Device.withDefaultDevice(.cpu) {
            let mask = KleinTextEncoder.additiveMask(attentionMask: MLXArray([1, 1, 0] as [Int64]).reshaped(1, 3), seqLen: 3, dtype: .bfloat16)
            XCTAssertEqual(mask.shape, [1, 1, 3, 3])
            XCTAssertEqual(mask.dtype, .bfloat16, "Klein builds the mask in the embedding dtype, not float32")
            let v = mask.asType(.float32).asArray(Float.self)
            let inf = Float.infinity
            // rows = queries; columns above the diagonal and the padded key (column 2) are -inf.
            XCTAssertEqual(v, [0, -inf, -inf, 0, 0, -inf, 0, 0, -inf])
            let single = KleinTextEncoder.additiveMask(attentionMask: MLXArray([1] as [Int64]).reshaped(1, 1), seqLen: 1, dtype: .bfloat16)
            XCTAssertEqual(single.shape, [1, 1, 1, 1])
            XCTAssertEqual(single.asType(.float32).asArray(Float.self), [0])
        }
    }

    func testRMSNormIsExplicitFloat32MathReturningTheInputDType() {
        Device.withDefaultDevice(.cpu) {
            let x = MLXArray([3, 4] as [Float]).reshaped(1, 2).asType(.bfloat16)
            let out = KleinTextEncoder.rmsNorm(x, weight: MLXArray([2, 0.5] as [Float]).asType(.bfloat16), eps: 0)
            XCTAssertEqual(out.dtype, .bfloat16)
            let scale = Float(1 / (12.5 as Float).squareRoot())
            let got = out.asType(.float32).asArray(Float.self)
            XCTAssertEqual(got[0], 2 * 3 * scale, accuracy: 0.02)
            XCTAssertEqual(got[1], 0.5 * 4 * scale, accuracy: 0.02)
            // The default eps is the model's 1e-6 for every norm, q/k norms included (Z-Image's q/k norms use 1e-5).
            XCTAssertEqual(KleinTextEncoder.rmsNormEps, 1e-6)
        }
    }

    func testRotateHalfAndPositionEmbeddings() {
        Device.withDefaultDevice(.cpu) {
            let x = MLXArray([1, 2, 3, 4] as [Float]).reshaped(1, 1, 1, 4)
            let invFreq = MLXArray((0..<64).map { Float(1) / Float(Foundation.pow(1_000_000.0, Double($0) / 64)) })
            let encoder = KleinTextEncoder(weights: KleinTextWeights(tensors: ["rotary_emb.inv_freq": invFreq]))
            let (cos, sin) = encoder.positionEmbeddings(batch: 2, seqLen: 3, dtype: .bfloat16)
            XCTAssertEqual(cos.shape, [2, 3, 128])
            XCTAssertEqual(cos.dtype, .bfloat16)
            XCTAssertEqual(sin.dtype, .bfloat16)
            let c = cos.asType(.float32)
            XCTAssertEqual(c[0, 0, 5].item(Float.self), 1)  // position 0
            XCTAssertEqual(c[0, 1, 0].item(Float.self), Float(Foundation.cos(1.0)), accuracy: 0.01)  // inv_freq[0] = 1
            XCTAssertEqual(c[0, 1, 64].item(Float.self), c[0, 1, 0].item(Float.self), "the two halves repeat the same frequencies")
            XCTAssertEqual(c[1, 2, 7].item(Float.self), c[0, 2, 7].item(Float.self))
        }
    }

    func testTextIDsAreZeroCoordinatesAndThePaddedPosition() throws {
        try Device.withDefaultDevice(.cpu) {
            let ids = KleinPromptEncoder.textIDs(batch: 2, seqLen: 5)
            XCTAssertEqual(ids.shape, [2, 5, 4])
            XCTAssertEqual(ids.dtype, .int32)
            XCTAssertEqual(ids[1].asArray(Int32.self), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4])
            // The empty prompt is supported as production MFLUX supports it: zero-length tensors, not a rejection.
            let none = KleinPromptEncoder.textIDs(batch: 1, seqLen: 0)
            XCTAssertEqual(none.shape, [1, 0, 4])
            XCTAssertEqual(none.dtype, .int32)
            let emptyStates = (0..<3).map { _ in MLXArray.zeros([1, 0, 2560], dtype: .bfloat16) }
            let emptyEmbeds = KleinTextEncoder.assemble(emptyStates)
            XCTAssertEqual(emptyEmbeds.shape, [1, 0, 7680])
            XCTAssertEqual(emptyEmbeds.dtype, .bfloat16)
            let emptyMask = KleinTextEncoder.additiveMask(attentionMask: MLXArray.zeros([1, 0], dtype: .int32), seqLen: 0, dtype: .bfloat16)
            XCTAssertEqual(emptyMask.shape, [1, 1, 0, 0])
            // A batch that is neither the padded int64 [1, 512] nor the empty int32 [1, 0] is refused before any math.
            let encoder = KleinTextEncoder(weights: KleinTextWeights(tensors: ["rotary_emb.inv_freq": MLXArray.zeros([64])]))
            XCTAssertThrowsError(try KleinPromptEncoder.encode(TokenizedPrompt(inputIDs: [1, 2], attentionMask: [1, 1], dtype: "mlx.core.int64"), with: encoder))
            XCTAssertThrowsError(try KleinPromptEncoder.encode(TokenizedPrompt(inputIDs: [], attentionMask: [], dtype: "mlx.core.int64"), with: encoder))
            XCTAssertEqual(KleinPromptEncoder.embeddingWidth, 7680)
            XCTAssertEqual(KleinTextEncoder.selectedLayers, [9, 18, 27])
        }
    }
}

final class KleinTokenizerTests: XCTestCase {
    func testChatRenderingDisablesThinkingAndDiffersFromZImage() {
        XCTAssertEqual(
            KleinTokenizer.renderChat(prompt: "hi"),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        XCTAssertNotEqual(KleinTokenizer.renderChat(prompt: "hi"), QwenTokenizer.renderChat(prompt: "hi"))
        XCTAssertTrue(KleinTokenizer.renderChat(prompt: "hi").hasPrefix(QwenTokenizer.renderChat(prompt: "hi")))
    }

    func testRealTokenizerBehavior() throws {
        let tokenizer = try KleinTokenizer.load(modelDirectory: try KleinTestSupport.requireModel())
        let plain = KleinTokenizer.tokenize("a red fox", with: tokenizer)
        XCTAssertEqual(plain.shape, [1, 512])
        XCTAssertEqual(plain.dtype, "mlx.core.int64")
        let valid = plain.validCount
        XCTAssertEqual(Array(plain.inputIDs[..<valid]), tokenizer.encode(KleinTokenizer.renderChat(prompt: "a red fox")).map(Int64.init))
        XCTAssertEqual(Array(plain.inputIDs[..<3]), tokenizer.encode("<|im_start|>user\n").map(Int64.init))
        XCTAssertEqual(Array(plain.inputIDs[(valid - 4)..<valid]), tokenizer.encode("<think>\n\n</think>\n\n").map(Int64.init), "thinking disabled block ends the sequence")
        XCTAssertEqual(plain.attentionMask, [Int64](repeating: 1, count: valid) + [Int64](repeating: 0, count: 512 - valid))
        XCTAssertTrue(plain.inputIDs[valid...].allSatisfy { $0 == QwenTokenizer.padTokenID })

        // Empty prompt: int32 [1, 0], no template; whitespace-only is not empty.
        let empty = KleinTokenizer.tokenize("", with: tokenizer)
        XCTAssertEqual(empty.shape, [1, 0])
        XCTAssertEqual(empty.dtype, "mlx.core.int32")
        XCTAssertGreaterThan(KleinTokenizer.tokenize("   ", with: tokenizer).validCount, 12)

        // Boundary and truncation: 12 template tokens around a prompt of N one-token words.
        func count(_ n: Int) -> Int { tokenizer.encode(KleinTokenizer.renderChat(prompt: [String](repeating: "word", count: n).joined(separator: " "))).count }
        XCTAssertEqual([count(499), count(500), count(501)], [511, 512, 513])
        let truncated = KleinTokenizer.tokenize([String](repeating: "word", count: 600).joined(separator: " "), with: tokenizer)
        XCTAssertEqual(truncated.validCount, 512)
        XCTAssertEqual(truncated.attentionMask, [Int64](repeating: 1, count: 512), "truncation cuts the closing template; there is no padding")

        // Unicode is NFC-normalized; literal special tokens in the prompt are tokenized as specials.
        XCTAssertEqual(tokenizer.encode("caf\u{e9}"), tokenizer.encode("cafe\u{301}"))
        XCTAssertEqual(tokenizer.encode("<|im_end|>").count, 1)
    }
}
