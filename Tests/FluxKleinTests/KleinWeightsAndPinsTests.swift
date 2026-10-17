import XCTest
@testable import FluxKlein
@testable import MLXImageCore

enum KleinTestSupport {
    static var modelDirectory: URL {
        let path = ProcessInfo.processInfo.environment["KLEIN_Q4_MODEL_DIR"]
            ?? NSHomeDirectory() + "/Library/Containers/com.nordlate.taltio/Data/Library/Application Support/LocalModels/models/flux2-klein-4b-q4"
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static func requireModel() throws -> URL {
        let url = modelDirectory
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.appendingPathComponent("text_encoder/model.safetensors.index.json").path),
                          "Klein q4 snapshot not installed at \(url.path)")
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

    func testPinSetsAreComplete() {
        XCTAssertEqual(KleinPins.runtimeModules.count, 9)
        XCTAssertEqual(Set(KleinPins.modelFiles.keys).count, 6)
        XCTAssertTrue(KleinPins.modelFiles.values.allSatisfy { $0.count == 64 })
        XCTAssertTrue(KleinPins.runtimeModules.values.allSatisfy { $0.count == 64 })
        XCTAssertEqual(KleinPins.mlxSwiftRevision, "19601207e9a0de51e03ee6ec0c3c5f3784275075")
    }

    func testOutputContainmentIsByPathComponentNotStringPrefix() {
        let model = URL(fileURLWithPath: "/nonexistent-klein/models/flux2-klein-4b-q4", isDirectory: true)
        XCTAssertTrue(KleinNativeExport.isInside(model.appendingPathComponent("text_encoder"), model))
        XCTAssertTrue(KleinNativeExport.isInside(model, model))
        XCTAssertFalse(KleinNativeExport.isInside(URL(fileURLWithPath: "/nonexistent-klein/models/flux2-klein-4b-q4-out"), model), "a sibling sharing the name prefix")
    }

    func testChangedArtifactIsNamedAndRejected() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("klein-pins-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for rel in KleinPins.modelFiles.keys {
            let url = dir.appendingPathComponent(rel)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data("not the pinned bytes".utf8).write(to: url)
        }
        XCTAssertThrowsError(try KleinHashing.verifyModelFiles(modelDirectory: dir)) { error in
            XCTAssertTrue("\(error)".contains("differs from the pinned"), "\(error)")
        }
        XCTAssertThrowsError(try KleinTokenizer.load(modelDirectory: dir), "a tokenizer directory with other files must be refused")
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

/// (The audit counter is scoped to the measuring task, so this test is safe under parallel test execution.)
/// The production load path hashes no weights (slice 11: ~1.8 s of every Klein load); the harness digests stay available on demand and equal
/// the weights' own `digests()`.
final class KleinLoadDigestTests: XCTestCase {
    func testProductionInitComputesNoDigestsAndHarnessDigestsAreOnDemandAndEqual() throws {
        let model = try KleinTestSupport.requireModel()
        let (generator, loadDigested) = try KleinDigestAudit.measure { try KleinGenerator(modelDirectory: model) }
        XCTAssertEqual(loadDigested, 0, "KleinGenerator.init hashed weights")
        let ((text, transformer, vae), onDemand) = KleinDigestAudit.measure { (generator.textDigests(), generator.transformerDigests(), generator.vaeDigests()) }
        XCTAssertEqual(onDemand, text.count + transformer.count + vae.count)
        XCTAssertEqual([text.count, transformer.count, vae.count], [905, KleinTransformerLayout.tensorCount, KleinVAELayout.tensorCount])
        // Same tensors, same bytes, same SHA-256 definition as a direct load of the weights.
        XCTAssertEqual(text, try KleinTextWeights.load(modelDirectory: model).digests())
        XCTAssertEqual(transformer, try KleinTransformerWeights.load(modelDirectory: model).digests())
        XCTAssertEqual(vae, try KleinVAEWeights.load(modelDirectory: model).digests())
    }
}
