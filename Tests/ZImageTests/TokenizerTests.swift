import MLX
import XCTest
import MLXImageCore
@testable import ZImage

final class RealTokenizerTests: XCTestCase {
    func testTemplateTokensTruncationAndPadding() throws {
        let tok = try QwenTokenizer.load(modelDirectory: try TestSupport.requireModel())
        let out = tok.tokenize("a red fox")
        XCTAssertEqual(out.shape, [1, 512])
        XCTAssertEqual(out.dtype, "mlx.core.int64")
        XCTAssertEqual(Array(out.inputIDs.prefix(3)), [151_644, 872, 198], "<|im_start|> user \\n")
        let valid = out.validCount
        XCTAssertEqual(Array(out.inputIDs[(valid - 5)..<valid]), [151_645, 198, 151_644, 77_091, 198], "<|im_end|> \n <|im_start|> assistant \n")
        XCTAssertTrue(out.inputIDs[valid...].allSatisfy { $0 == 151_643 }, "right padding with <|endoftext|>")
        XCTAssertTrue(out.attentionMask[valid...].allSatisfy { $0 == 0 })
        let long = tok.tokenize(String(repeating: "word ", count: 600))
        XCTAssertEqual(long.validCount, 512)
        XCTAssertEqual(long.attentionMask.reduce(0, +), 512)
        XCTAssertNotEqual(long.inputIDs[511], 198, "truncation cuts the template tail like the reference")
        let empty = tok.tokenize("")
        XCTAssertEqual(empty.shape, [1, 0])
        XCTAssertEqual(empty.dtype, "mlx.core.int32")
    }

    func testPinnedFileHashIsEnforced() throws {
        let model = try TestSupport.requireModel()
        let copy = FileManager.default.temporaryDirectory.appendingPathComponent("zimage-tok-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: copy) }
        try FileManager.default.createDirectory(at: copy.appendingPathComponent("tokenizer"), withIntermediateDirectories: true)
        for name in QwenTokenizer.pinnedFileHashes.keys {
            try FileManager.default.copyItem(at: model.appendingPathComponent("tokenizer/\(name)"), to: copy.appendingPathComponent("tokenizer/\(name)"))
        }
        XCTAssertNoThrow(try QwenTokenizer.load(modelDirectory: copy))
        try Data("{% tampered %}".utf8).write(to: copy.appendingPathComponent("tokenizer/chat_template.jinja"))
        XCTAssertThrowsError(try QwenTokenizer.load(modelDirectory: copy))
    }

    /// Token IDs, masks, shapes and dtypes must equal the Python/MFLUX reference exactly for every exported case.
    func testTokensMatchReferenceExport() throws {
        let model = try TestSupport.requireModel()
        guard let reference = TestSupport.referenceDirectory else { throw XCTSkip("ZIMAGE_REFERENCE_DIR is not set; point it at a Python/MFLUX reference export") }
        let manifestURL = reference.appendingPathComponent("manifest.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: manifestURL.path), "ZIMAGE_REFERENCE_DIR: reference export not found at \(manifestURL.path)")
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
        let tok = try QwenTokenizer.load(modelDirectory: model)
        let cases = manifest["cases"] as! [[String: Any]]
        XCTAssertGreaterThanOrEqual(cases.count, 10)
        for c in cases {
            let id = c["id"] as! String
            let out = tok.tokenize(c["prompt"] as! String)
            XCTAssertEqual(QwenTokenizer.renderChat(prompt: c["prompt"] as! String), c["rendered_text"] as! String, id)
            XCTAssertEqual(out.validCount, c["num_valid"] as! Int, id)
            XCTAssertEqual(out.shape, c["ids_shape"] as! [Int], id)
            XCTAssertEqual(out.dtype, c["ids_dtype"] as! String, id)
            let ref = try loadArrays(url: reference.appendingPathComponent("\(id).safetensors"))
            XCTAssertEqual(out.inputIDs, ref["input_ids"]!.asArray(Int64.self), id)
            XCTAssertEqual(out.attentionMask, ref["attention_mask"]!.asArray(Int64.self), id)
            XCTAssertEqual(ref["attention_mask"]!.shape, out.shape, id)
        }
    }
}
