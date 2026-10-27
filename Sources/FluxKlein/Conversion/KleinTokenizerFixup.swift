import Foundation

/// The tokenizer files of the converted snapshot, which must match `KleinPins.tokenizerFiles` (the files `mflux-save` 0.20 writes).
///
/// - `chat_template.jinja` is the official file unchanged.
/// - `tokenizer.json` is the official file with four flags changed, as `Qwen2TokenizerFast.save_pretrained` under transformers 5
///   re-serializes them: the ByteLevel pre-tokenizer's `trim_offsets` becomes `true`, and the ByteLevel decoder's `add_prefix_space`,
///   `trim_offsets` and `use_regex` become `true` (4 bytes shorter). These flags affect only offsets and decoding, which the pipeline
///   does not use; the token ids are the same.
/// - `tokenizer_config.json` is the short file `save_pretrained` writes (`KleinConversionConstants.tokenizerConfig`).
enum KleinTokenizerFixup {
    static let preTokenizerOfficial = """
          {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
          }
        ]
      },
    """
    static let preTokenizerSaved = """
          {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": false
          }
        ]
      },
    """
    static let decoderOfficial = """
      "decoder": {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": false,
        "use_regex": false
      },
    """
    static let decoderSaved = """
      "decoder": {
        "type": "ByteLevel",
        "add_prefix_space": true,
        "trim_offsets": true,
        "use_regex": true
      },
    """

    /// The pinned `tokenizer.json` from the official one; throws if either block is not found exactly once or the result does not
    /// match the pin.
    static func savedTokenizerJSON(fromOfficial data: Data) throws -> Data {
        let result = try updatingFlags(data)
        let expected = KleinPins.tokenizerFiles["tokenizer.json"]!
        let actual = KleinHashing.sha256(of: result)
        guard actual == expected else {
            throw KleinConversionError.hashMismatch(file: "tokenizer/tokenizer.json (converted)", expected: expected, actual: actual)
        }
        return result
    }

    /// The two ByteLevel blocks replaced; each must occur exactly once.
    static func updatingFlags(_ data: Data) throws -> Data {
        var bytes = [UInt8](data)
        for (official, saved) in [(preTokenizerOfficial, preTokenizerSaved), (decoderOfficial, decoderSaved)] {
            let (from, to) = (Array(official.utf8), Array(saved.utf8))
            let hits = occurrences(of: from, in: bytes)
            guard hits.count == 1 else {
                throw KleinConversionError.invalidFile(file: "tokenizer/tokenizer.json", reason: "expected one ByteLevel block to update, found \(hits.count)")
            }
            bytes.replaceSubrange(hits[0]..<(hits[0] + from.count), with: to)
        }
        return Data(bytes)
    }

    /// How many bytes the fix-up adds to `tokenizer.json` (-4).
    static var byteCountChange: Int {
        [(preTokenizerOfficial, preTokenizerSaved), (decoderOfficial, decoderSaved)].reduce(0) { $0 + $1.1.utf8.count - $1.0.utf8.count }
    }

    static var tokenizerConfig: Data { Data(KleinConversionConstants.tokenizerConfig.utf8) }

    private static func occurrences(of needle: [UInt8], in haystack: [UInt8]) -> [Int] {
        guard !needle.isEmpty, haystack.count >= needle.count else { return [] }
        var hits: [Int] = []
        let first = needle[0]
        var i = 0
        let last = haystack.count - needle.count
        haystack.withUnsafeBufferPointer { h in
            needle.withUnsafeBufferPointer { n in
                while i <= last {
                    if h[i] == first, memcmp(h.baseAddress! + i, n.baseAddress!, n.count) == 0 { hits.append(i); i += n.count } else { i += 1 }
                }
            }
        }
        return hits
    }
}
