import MLX
import XCTest
@testable import ZImageNative

final class TokenizerUnitTests: XCTestCase {
    func testByteToUnicodeTable() {
        let table = QwenTokenizer.bytesToUnicode()
        XCTAssertEqual(Set(table).count, 256)
        XCTAssertEqual(table[0x20], "\u{120}")  // Ġ
        XCTAssertEqual(table[0x0A], "\u{10A}")  // Ċ
        XCTAssertEqual(table[0x41], "A")
        XCTAssertEqual(table[0xAD], "\u{143}")
    }

    /// Tiny synthetic vocabulary: merges apply in rank order, lowest rank first.
    func testBPEMergeOrderAndAddedTokens() throws {
        let t = QwenTokenizer.bytesToUnicode()
        var vocab: [String: Int] = [:]
        for (i, c) in t.enumerated() { vocab[c] = i }
        vocab["ab"] = 300; vocab["abc"] = 301; vocab["bc"] = 302
        let tok = try QwenTokenizer(
            vocab: vocab, merges: [("b", "c"), ("a", "bc"), ("a", "b")], addedTokens: [("<|x|>", 900), ("<|x|>y", 901)],
            pattern: "\\p{L}+|\\s+|[^\\s\\p{L}]+")
        XCTAssertEqual(tok.encode("abc"), [301])  // b+c first (rank 0), then a+bc
        XCTAssertEqual(tok.encode("ab"), [300])
        XCTAssertEqual(tok.encode("a bc"), [97, 32, 302], "regex pieces a, space, bc")
        XCTAssertEqual(tok.encode("a<|x|>b"), [97, 900, 98])
        XCTAssertEqual(tok.encode("<|x|>y"), [901], "leftmost-longest added token wins")
        XCTAssertEqual(tok.encode("é"), [vocab[t[0xC3]]!, vocab[t[0xA9]]!], "UTF-8 bytes of NFC é")
        XCTAssertEqual(tok.encode("e\u{301}"), tok.encode("\u{E9}"), "NFC normalization before byte-level mapping")
    }

    func testChatTemplateRendering() {
        XCTAssertEqual(QwenTokenizer.renderChat(prompt: "hi"), "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    }
}

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
        let manifestURL = TestSupport.referenceDirectory.appendingPathComponent("manifest.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: manifestURL.path), "run scripts/run-parity.sh to export the reference")
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
            let ref = try loadArrays(url: TestSupport.referenceDirectory.appendingPathComponent("\(id).safetensors"))
            XCTAssertEqual(out.inputIDs, ref["input_ids"]!.asArray(Int64.self), id)
            XCTAssertEqual(out.attentionMask, ref["attention_mask"]!.asArray(Int64.self), id)
            XCTAssertEqual(ref["attention_mask"]!.shape, out.shape, id)
        }
    }
}
