import Foundation
import XCTest
@testable import FluxKlein
@testable import MLXImageCore

final class KleinTransformerWeightValidationTests: XCTestCase {
    private struct Parts {
        var map: [String: String]
        var metadata: [String: String]
        var headers: [String: SafetensorsHeader]

        mutating func replace(_ name: String, dtype: String? = nil, shape: [Int]? = nil) {
            let shard = map[name]!
            var entries = headers[shard]!.entries
            let old = entries[name]!
            entries[name] = SafetensorsEntry(dtype: dtype ?? old.dtype, shape: shape ?? old.shape, byteRange: 0..<0)
            headers[shard] = SafetensorsHeader(entries: entries, metadata: metadata)
        }

        func validate() throws {
            try KleinTransformerWeights.validate(weightMap: map, indexMetadata: metadata, shardHeaders: headers)
        }
    }

    private func validParts() -> Parts {
        let layout = KleinTransformerLayout.klein4B.expectedTensors()
        let metadata = ["quantization_level": "4", "mflux_version": "0.19.0"]
        var map: [String: String] = [:]
        var entries: [String: [String: SafetensorsEntry]] = ["0.safetensors": [:], "1.safetensors": [:]]
        for (index, name) in layout.keys.sorted().enumerated() {
            let shard = index < 375 ? "0.safetensors" : "1.safetensors"
            map[name] = shard
            let want = layout[name]!
            entries[shard]![name] = SafetensorsEntry(dtype: want.dtype, shape: want.shape, byteRange: 0..<0)
        }
        return Parts(map: map, metadata: metadata, headers: entries.mapValues { SafetensorsHeader(entries: $0, metadata: metadata) })
    }

    func testArchitectureAndCompleteLayout() throws {
        let layout = KleinTransformerLayout.klein4B.expectedTensors()
        XCTAssertEqual(layout.count, 387)
        XCTAssertEqual(KleinTransformerLayout.klein4B.matrices().count, 109)
        XCTAssertEqual(layout["x_embedder.weight"]?.shape, [3072, 16])
        XCTAssertEqual(layout["context_embedder.scales"]?.shape, [3072, 120])
        XCTAssertEqual(layout["single_transformer_blocks.19.attn.to_qkv_mlp_proj.weight"]?.shape, [27648, 384])
        XCTAssertEqual(layout["transformer_blocks.4.attn.norm_added_k.weight"]?.dtype, "BF16")
        XCTAssertNil(layout["time_guidance_embed.guidance_linear_1.weight"])
        XCTAssertNoThrow(try validParts().validate())
    }

    func testRejectsWrongBitsGroupsTranspositionAndDType() {
        for (name, dtype, shape) in [
            ("x_embedder.weight", "U32", [3072, 32]), // 8-bit packing
            ("context_embedder.scales", "BF16", [3072, 240]), // groups of 32
            ("x_embedder.weight", "U32", [16, 3072]),
            ("transformer_blocks.4.attn.norm_added_k.weight", "F32", [128]),
            ("single_transformer_blocks.19.attn.to_out.biases", "F32", [3072, 192]),
        ] {
            var parts = validParts()
            parts.replace(name, dtype: dtype, shape: shape)
            XCTAssertThrowsError(try parts.validate(), name)
        }
    }

    func testRejectsMissingUnexpectedAndMisassignedTensors() {
        var parts = validParts()
        parts.map["proj_out.weight"] = nil
        XCTAssertThrowsError(try parts.validate())
        parts = validParts()
        parts.map["proj_out.bias"] = "0.safetensors"
        XCTAssertThrowsError(try parts.validate())
        parts = validParts()
        let name = "x_embedder.weight"
        parts.map[name] = parts.map[name] == "0.safetensors" ? "1.safetensors" : "0.safetensors"
        XCTAssertThrowsError(try parts.validate())
        parts = validParts()
        parts.headers["2.safetensors"] = SafetensorsHeader(entries: [:], metadata: parts.metadata)
        XCTAssertThrowsError(try parts.validate())
    }

    func testRejectsIndexAndShardMetadataAndWrongShardCounts() {
        for (key, value) in [("quantization_level", "8"), ("mflux_version", "0.20.0")] {
            var parts = validParts()
            parts.metadata[key] = value
            XCTAssertThrowsError(try parts.validate())
            parts = validParts()
            var wrong = parts.metadata
            wrong[key] = value
            parts.headers["1.safetensors"] = SafetensorsHeader(entries: parts.headers["1.safetensors"]!.entries, metadata: wrong)
            XCTAssertThrowsError(try parts.validate())
        }
        var parts = validParts()
        var first = parts.headers["0.safetensors"]!.entries
        var last = parts.headers["1.safetensors"]!.entries
        let moved = first.keys.sorted()[0]
        last[moved] = first.removeValue(forKey: moved)
        parts.map[moved] = "1.safetensors"
        parts.headers["0.safetensors"] = SafetensorsHeader(entries: first, metadata: parts.metadata)
        parts.headers["1.safetensors"] = SafetensorsHeader(entries: last, metadata: parts.metadata)
        XCTAssertThrowsError(try parts.validate(), "matching names with 374+13 tensors still violate the stored shard contract")
    }

    func testInstalledSnapshotHeadersValidateWithoutLoadingTensorPayloads() throws {
        let (count, shards) = try KleinTransformerWeights.readAndValidate(modelDirectory: KleinTestSupport.requireModel())
        XCTAssertEqual(count, 387)
        XCTAssertEqual(shards, ["0.safetensors", "1.safetensors"])
    }
}
