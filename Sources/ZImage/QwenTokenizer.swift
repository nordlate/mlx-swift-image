import CryptoKit
import Foundation

public enum TokenizerError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self { case .invalid(let m): "tokenizer: \(m)" }
    }
}

/// Token IDs and mask as mflux's `LanguageTokenizer.tokenize` returns them for one prompt.
public struct TokenizedPrompt: Equatable, Sendable {
    public let inputIDs: [Int64]
    public let attentionMask: [Int64]
    /// mflux returns an int32 `[1, 0]` pair for an empty prompt and int64 `[1, 512]` otherwise.
    public let dtype: String
    public var shape: [Int] { [1, inputIDs.count] }
    public var validCount: Int { attentionMask.reduce(0) { $0 + Int($1) } }
}

/// Native byte-level BPE tokenizer for the Qwen2 tokenizer files shipped with Z-Image Turbo, reproducing what
/// mflux does for this model: chat template (`enable_thinking: true`, generation prompt), NFC, added-token
/// splitting, the Qwen2 split regex, byte-level mapping, BPE merges, right truncation and right padding to 512.
///
/// Only this tokenizer is supported: `load` pins the SHA-256 of `tokenizer.json`, `tokenizer_config.json` and
/// `chat_template.jinja`, because the template renderer and pad/special-token handling are specific to them.
public final class QwenTokenizer {
    public static let pinnedFileHashes: [String: String] = [
        "tokenizer.json": "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
        "tokenizer_config.json": "1cc816812993bff176eb4f7495433b736f06fba9b6e7b05cac7b4a1780650c95",
        "chat_template.jinja": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    ]
    public static let maxLength = 512
    public static let padTokenID: Int64 = 151_643  // <|endoftext|>

    private let vocab: [String: Int]
    private let mergeRanks: [String: Int]
    private let addedTokens: [(content: String, id: Int)]
    private let splitRegex: NSRegularExpression
    private static let byteToChar: [String] = bytesToUnicode()
    private var wordCache: [String: [Int]] = [:]

    /// Test seam: builds a tokenizer from already-parsed pieces (no hash pinning, no chat-template guarantee).
    init(vocab: [String: Int], merges: [(String, String)], addedTokens: [(String, Int)], pattern: String) throws {
        self.vocab = vocab
        var ranks: [String: Int] = [:]
        for (rank, pair) in merges.enumerated() { ranks[pair.0 + " " + pair.1] = rank }
        guard ranks.count == merges.count else { throw TokenizerError.invalid("duplicate merges") }
        self.mergeRanks = ranks
        self.addedTokens = addedTokens.map { (content: $0.0, id: $0.1) }
        self.splitRegex = try NSRegularExpression(pattern: pattern)
    }

    public static func load(modelDirectory: URL) throws -> QwenTokenizer {
        let dir = modelDirectory.appendingPathComponent("tokenizer")
        for (name, expected) in pinnedFileHashes {
            let data = try Data(contentsOf: dir.appendingPathComponent(name))
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard digest == expected else {
                throw TokenizerError.invalid("\(name) hash \(digest) differs from the pinned Z-Image Turbo file; the native renderer only supports the pinned template and tokenizer")
            }
        }
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("tokenizer.json"))) as! [String: Any]
        guard let normalizer = root["normalizer"] as? [String: Any], normalizer["type"] as? String == "NFC",
              let model = root["model"] as? [String: Any], model["type"] as? String == "BPE",
              let vocabJSON = model["vocab"] as? [String: Int], let mergesJSON = model["merges"] as? [[String]],
              let added = root["added_tokens"] as? [[String: Any]],
              let pre = root["pre_tokenizer"] as? [String: Any], let parts = pre["pretokenizers"] as? [[String: Any]],
              parts.count == 2, parts[0]["type"] as? String == "Split", parts[0]["behavior"] as? String == "Isolated",
              parts[1]["type"] as? String == "ByteLevel", parts[1]["add_prefix_space"] as? Bool == false,
              let pattern = (parts[0]["pattern"] as? [String: String])?["Regex"]
        else { throw TokenizerError.invalid("tokenizer.json does not have the expected NFC + Split/ByteLevel + BPE structure") }
        guard model["ignore_merges"] as? Bool == false, model["byte_fallback"] as? Bool == false,
              model["continuing_subword_prefix"] as? String == "", model["end_of_word_suffix"] as? String == "",
              model["dropout"] is NSNull, model["unk_token"] is NSNull
        else { throw TokenizerError.invalid("unsupported BPE options") }
        let merges = try mergesJSON.map { pair -> (String, String) in
            guard pair.count == 2 else { throw TokenizerError.invalid("bad merge entry") }
            return (pair[0], pair[1])
        }
        let addedPairs = try added.map { entry -> (String, Int) in
            guard let content = entry["content"] as? String, let id = entry["id"] as? Int else { throw TokenizerError.invalid("bad added token") }
            // All 26 added tokens are matched literally and not normalized, as the tokenizers library does.
            guard entry["normalized"] as? Bool == false, entry["lstrip"] as? Bool == false, entry["rstrip"] as? Bool == false,
                  entry["single_word"] as? Bool == false else { throw TokenizerError.invalid("unsupported added token flags for \(content)") }
            return (content, id)
        }
        // splitAddedTokens only probes positions holding '<'; every pinned added token starts with it.
        guard addedPairs.allSatisfy({ $0.0.hasPrefix("<") }) else { throw TokenizerError.invalid("added token not starting with '<'") }
        let tokenizer = try QwenTokenizer(vocab: vocabJSON, merges: merges, addedTokens: addedPairs, pattern: pattern)
        // Swift String equality is canonical equivalence; a collapsed key would silently change IDs.
        let rawVocabCount = ((root["model"] as? NSDictionary)?["vocab"] as? NSDictionary)?.count
        guard tokenizer.vocab.count == rawVocabCount, tokenizer.mergeRanks.count == merges.count else {
            throw TokenizerError.invalid("vocabulary or merge keys collapsed under String equality")
        }
        guard tokenizer.vocab["<|endoftext|>"] == nil, addedPairs.first(where: { $0.0 == "<|endoftext|>" })?.1 == Int(padTokenID) else {
            throw TokenizerError.invalid("pad token id mismatch")
        }
        return tokenizer
    }

    // MARK: Chat template

    /// Renders the pinned Qwen3 template for one user message with `add_generation_prompt` and `enable_thinking: true`
    /// (which adds nothing after the assistant header).
    public static func renderChat(prompt: String) -> String {
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    }

    // MARK: Tokenization

    public func tokenize(_ prompt: String) -> TokenizedPrompt {
        if prompt.isEmpty {
            // mflux special-cases an all-empty batch: int32 [1, 0] arrays, bypassing the template.
            return TokenizedPrompt(inputIDs: [], attentionMask: [], dtype: "mlx.core.int32")
        }
        var ids = encode(Self.renderChat(prompt: prompt)).map(Int64.init)
        if ids.count > Self.maxLength { ids.removeSubrange(Self.maxLength...) }
        let valid = ids.count
        ids += [Int64](repeating: Self.padTokenID, count: Self.maxLength - valid)
        let mask = [Int64](repeating: 1, count: valid) + [Int64](repeating: 0, count: Self.maxLength - valid)
        return TokenizedPrompt(inputIDs: ids, attentionMask: mask, dtype: "mlx.core.int64")
    }

    /// Token IDs of `text` without template, truncation or padding.
    public func encode(_ text: String) -> [Int] {
        var ids: [Int] = []
        for segment in splitAddedTokens(text) {
            switch segment {
            case .token(let id): ids.append(id)
            case .text(let plain): ids += encodePlain(plain)
            }
        }
        return ids
    }

    private enum Segment { case token(Int), text(String) }

    /// Leftmost-longest literal matches of the added tokens, found on the raw text before normalization.
    private func splitAddedTokens(_ text: String) -> [Segment] {
        let units = Array(text.utf8)
        let candidates = addedTokens.map { (id: $0.id, bytes: Array($0.content.utf8)) }
        var segments: [Segment] = []
        var plainStart = 0
        var i = 0
        func flush(upTo end: Int) {
            if end > plainStart { segments.append(.text(String(decoding: units[plainStart..<end], as: UTF8.self))) }
        }
        while i < units.count {
            var best: (id: Int, length: Int)?
            if units[i] == UInt8(ascii: "<") {
                for c in candidates where c.bytes.count <= units.count - i && units[i..<(i + c.bytes.count)].elementsEqual(c.bytes) {
                    if best == nil || c.bytes.count > best!.length { best = (c.id, c.bytes.count) }
                }
            }
            if let best {
                flush(upTo: i)
                segments.append(.token(best.id))
                i += best.length
                plainStart = i
            } else {
                i += 1
            }
        }
        flush(upTo: units.count)
        return segments
    }

    private func encodePlain(_ text: String) -> [Int] {
        let normalized = text.precomposedStringWithCanonicalMapping  // NFC
        let ns = normalized as NSString
        var pieces: [String] = []
        var last = 0
        for match in splitRegex.matches(in: normalized, range: NSRange(location: 0, length: ns.length)) {
            if match.range.location > last { pieces.append(ns.substring(with: NSRange(location: last, length: match.range.location - last))) }
            if match.range.length > 0 { pieces.append(ns.substring(with: match.range)) }
            last = match.range.location + match.range.length
        }
        if last < ns.length { pieces.append(ns.substring(from: last)) }
        var ids: [Int] = []
        for piece in pieces { ids += bpe(byteLevel(piece)) }
        return ids
    }

    private func byteLevel(_ piece: String) -> String {
        piece.utf8.map { Self.byteToChar[Int($0)] }.joined()
    }

    private func bpe(_ word: String) -> [Int] {
        if let cached = wordCache[word] { return cached }
        var parts = word.unicodeScalars.map { String($0) }
        while parts.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(parts.count - 1) {
                if let rank = mergeRanks[parts[i] + " " + parts[i + 1]], rank < bestRank { bestRank = rank; bestIndex = i }
            }
            if bestIndex < 0 { break }
            parts[bestIndex] += parts[bestIndex + 1]
            parts.remove(at: bestIndex + 1)
        }
        let ids = parts.map { vocab[$0]! }  // byte-level BPE covers every byte; a miss is a corrupt vocabulary
        wordCache[word] = ids
        return ids
    }

    /// GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest to U+0100 upward in byte order.
    static func bytesToUnicode() -> [String] {
        var printable: [Int] = Array(33...126) + Array(161...172) + Array(174...255)
        var table = [String](repeating: "", count: 256)
        for b in printable { table[b] = String(UnicodeScalar(UInt8(b))) }
        var extra = 0
        for b in 0..<256 where table[b].isEmpty {
            table[b] = String(UnicodeScalar(UInt32(256 + extra))!)
            extra += 1
        }
        printable.removeAll()
        return table
    }
}
