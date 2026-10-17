import XCTest
@testable import FluxKlein
@testable import MLXImageCore

enum KleinTestSupport {
    /// The read-only FLUX.2 Klein 4B q4 snapshot (`KLEIN_Q4_MODEL_DIR`, or `KLEIN_MODEL_DIR`); nil when neither is set.
    static var modelDirectory: URL? {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["KLEIN_Q4_MODEL_DIR"] ?? environment["KLEIN_MODEL_DIR"], !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static func requireModel() throws -> URL {
        guard let url = modelDirectory else { throw XCTSkip("KLEIN_MODEL_DIR is not set; point it at a FLUX.2 Klein 4B q4 snapshot") }
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.appendingPathComponent("text_encoder/model.safetensors.index.json").path),
                          "KLEIN_MODEL_DIR: Klein q4 snapshot not found at \(url.path)")
        return url
    }

    /// Frozen parity cases and budgets: Tests/Fixtures/klein (copies of the experiment's klein/ inputs).
    static var kleinDirectory: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/klein")
    }

    /// A layout-consistent index and shard headers (data ranges are irrelevant to header validation), split over two shards.
    static func validParts() -> (weightMap: [String: String], metadata: [String: String], headers: [String: SafetensorsHeader]) {
        let expected = TextEncoderLayout.expectedTensors()
        var weightMap: [String: String] = [:]
        var entries: [String: [String: SafetensorsEntry]] = ["0.safetensors": [:], "1.safetensors": [:]]
        for (name, want) in expected {
            let shard = name.hasPrefix("layers.3") || name == "norm.weight" || name == "rotary_emb.inv_freq" ? "1.safetensors" : "0.safetensors"
            weightMap[name] = shard
            entries[shard]![name] = SafetensorsEntry(dtype: want.dtype, shape: want.shape, byteRange: 0..<0)
        }
        let metadata = ["quantization_level": "4", "mflux_version": "0.19.0"]
        let headers = entries.mapValues { SafetensorsHeader(entries: $0, metadata: metadata) }
        return (weightMap, metadata, headers)
    }
}

final class KleinWeightValidationTests: XCTestCase {
    private func validate(_ parts: (weightMap: [String: String], metadata: [String: String], headers: [String: SafetensorsHeader])) throws {
        try KleinTextWeights.validate(weightMap: parts.weightMap, indexMetadata: parts.metadata, shardHeaders: parts.headers)
    }

    func testLayoutConsistentSnapshotValidates() throws {
        let parts = KleinTestSupport.validParts()
        XCTAssertEqual(parts.weightMap.count, 905)
        XCTAssertNoThrow(try validate(parts))
        XCTAssertEqual(TextEncoderLayout.quantizedMatrices().count, 1 + 36 * 7)
    }

    func testRejectsWrongQuantizationOrVersionMetadata() {
        var parts = KleinTestSupport.validParts()
        parts.metadata["quantization_level"] = "8"
        XCTAssertThrowsError(try validate(parts), "an 8-bit snapshot must not load as q4")
        parts = KleinTestSupport.validParts()
        parts.metadata["mflux_version"] = "0.18.0"
        XCTAssertThrowsError(try validate(parts))
        parts = KleinTestSupport.validParts()
        parts.headers["0.safetensors"] = SafetensorsHeader(entries: parts.headers["0.safetensors"]!.entries, metadata: ["quantization_level": "8", "mflux_version": "0.19.0"])
        XCTAssertThrowsError(try validate(parts), "a shard disagreeing with the index metadata")
    }

    func testRejectsMissingAndUnexpectedTensors() {
        var parts = KleinTestSupport.validParts()
        parts.weightMap["layers.35.mlp.down_proj.biases"] = nil
        XCTAssertThrowsError(try validate(parts))
        parts = KleinTestSupport.validParts()
        parts.weightMap["lm_head.weight"] = "0.safetensors"
        XCTAssertThrowsError(try validate(parts), "a tensor the architecture does not have")
    }

    func testRejectsTransposedOrRepackedMatrices() {
        var parts = KleinTestSupport.validParts()
        let shard = parts.weightMap["layers.0.self_attn.q_proj.weight"]!
        var entries = parts.headers[shard]!.entries
        entries["layers.0.self_attn.q_proj.weight"] = SafetensorsEntry(dtype: "U32", shape: [320, 4096], byteRange: 0..<0)  // transposed
        parts.headers[shard] = SafetensorsHeader(entries: entries, metadata: parts.metadata)
        XCTAssertThrowsError(try validate(parts), "a transposed matrix is a layout conversion")

        parts = KleinTestSupport.validParts()
        entries = parts.headers[shard]!.entries
        entries["layers.0.self_attn.q_proj.weight"] = SafetensorsEntry(dtype: "U32", shape: [4096, 640], byteRange: 0..<0)  // 8-bit packing
        parts.headers[shard] = SafetensorsHeader(entries: entries, metadata: parts.metadata)
        XCTAssertThrowsError(try validate(parts), "8-bit packed width under 4-bit metadata")

        parts = KleinTestSupport.validParts()
        entries = parts.headers[shard]!.entries
        entries["layers.0.self_attn.q_proj.scales"] = SafetensorsEntry(dtype: "BF16", shape: [4096, 80], byteRange: 0..<0)  // group size 32
        parts.headers[shard] = SafetensorsHeader(entries: entries, metadata: parts.metadata)
        XCTAssertThrowsError(try validate(parts), "group size 32 scales")
    }

    func testRejectsWrongDTypeAndShardMismatch() {
        var parts = KleinTestSupport.validParts()
        let shard = parts.weightMap["embed_tokens.scales"]!
        var entries = parts.headers[shard]!.entries
        entries["embed_tokens.scales"] = SafetensorsEntry(dtype: "F32", shape: [151_936, 40], byteRange: 0..<0)
        parts.headers[shard] = SafetensorsHeader(entries: entries, metadata: parts.metadata)
        XCTAssertThrowsError(try validate(parts), "float32 scales")

        parts = KleinTestSupport.validParts()
        parts.weightMap["norm.weight"] = "0.safetensors"  // the header of shard 1 holds it
        XCTAssertThrowsError(try validate(parts))

        parts = KleinTestSupport.validParts()
        parts.headers["2.safetensors"] = SafetensorsHeader(entries: [:], metadata: parts.metadata)
        XCTAssertThrowsError(try validate(parts), "a shard header the index does not name")
    }
}

final class KleinPinsTests: XCTestCase {
    func testKleinTokenizerPinsAreNotTheZImageOnes() {
        // tokenizer.json and chat_template.jinja are byte-identical to Z-Image's; tokenizer_config.json is not and must not be reused.
        XCTAssertEqual(KleinPins.tokenizerFiles["tokenizer.json"], QwenTokenizer.pinnedFileHashes["tokenizer.json"])
        XCTAssertEqual(KleinPins.tokenizerFiles["chat_template.jinja"], QwenTokenizer.pinnedFileHashes["chat_template.jinja"])
        XCTAssertNotEqual(KleinPins.tokenizerFiles["tokenizer_config.json"], QwenTokenizer.pinnedFileHashes["tokenizer_config.json"])
    }


    func testRealSnapshotTokenizerFilesMatchPinsAndHeadersMatchLayout() throws {
        let model = try KleinTestSupport.requireModel()
        for (name, pin) in KleinPins.tokenizerFiles {
            XCTAssertEqual(try KleinHashing.sha256(of: model.appendingPathComponent("tokenizer/" + name)), pin, name)
        }
        let (count, shards) = try KleinTextWeights.readAndValidate(modelDirectory: model)
        XCTAssertEqual(count, 905)
        XCTAssertEqual(shards, ["0.safetensors", "1.safetensors"])
    }
}
