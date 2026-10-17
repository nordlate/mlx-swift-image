import XCTest
@testable import MLXImageCore

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
