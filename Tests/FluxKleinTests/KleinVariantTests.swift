import Foundation
import XCTest
@testable import FluxKlein
@testable import MLXImageCore

/// The variant-dependent sizes, and that each variant's loaders accept its own snapshot layout and refuse the other's.
final class KleinVariantTests: XCTestCase {
    /// A layout-consistent transformer index and headers for `variant`, split as `mflux-save` splits it (sorted names stand in for
    /// parameter order; only the per-shard counts are pinned).
    private func transformerParts(_ variant: KleinVariant, mfluxVersion: String? = nil)
        -> (map: [String: String], metadata: [String: String], headers: [String: SafetensorsHeader]) {
        let layout = variant.transformer.expectedTensors()
        let metadata = ["quantization_level": "4", "mflux_version": mfluxVersion ?? variant.storedMfluxVersion]
        let counts = variant.transformerShardTensorCounts.sorted { $0.key < $1.key }
        var map: [String: String] = [:], entries: [String: [String: SafetensorsEntry]] = [:]
        var names = layout.keys.sorted()[...]
        for (shard, count) in counts {
            for name in names.prefix(count) {
                map[name] = shard
                entries[shard, default: [:]][name] = SafetensorsEntry(dtype: layout[name]!.dtype, shape: layout[name]!.shape, byteRange: 0..<0)
            }
            names = names.dropFirst(count)
        }
        return (map, metadata, entries.mapValues { SafetensorsHeader(entries: $0, metadata: metadata) })
    }

    private func textParts(_ layout: TextEncoderLayout, mfluxVersion: String)
        -> (map: [String: String], metadata: [String: String], headers: [String: SafetensorsHeader]) {
        let metadata = ["quantization_level": "4", "mflux_version": mfluxVersion]
        var map: [String: String] = [:], entries: [String: [String: SafetensorsEntry]] = [:]
        for (name, want) in layout.expectedTensors() {
            let shard = name.hasPrefix("layers.3") ? "1.safetensors" : "0.safetensors"
            map[name] = shard
            entries[shard, default: [:]][name] = SafetensorsEntry(dtype: want.dtype, shape: want.shape, byteRange: 0..<0)
        }
        return (map, metadata, entries.mapValues { SafetensorsHeader(entries: $0, metadata: metadata) })
    }

    func testKlein9BArchitecture() {
        let layout = KleinTransformerLayout.klein9B
        XCTAssertEqual(KleinVariant.klein9B.transformer, layout)
        XCTAssertEqual([layout.doubleBlocks, layout.singleBlocks, layout.heads, layout.dim, layout.jointAttentionDim], [8, 24, 32, 4096, 12288])
        XCTAssertEqual(layout.matrices().count, 153)
        XCTAssertEqual(layout.normNames().count, 80)
        XCTAssertEqual(layout.tensorCount, 539)
        let tensors = layout.expectedTensors()
        XCTAssertEqual(tensors.count, 539)
        XCTAssertEqual(tensors["x_embedder.weight"]?.shape, [4096, 16])
        XCTAssertEqual(tensors["context_embedder.weight"]?.shape, [4096, 1536])
        XCTAssertEqual(tensors["context_embedder.scales"]?.shape, [4096, 192])
        XCTAssertEqual(tensors["single_transformer_blocks.23.attn.to_qkv_mlp_proj.weight"]?.shape, [36864, 512])
        XCTAssertEqual(tensors["transformer_blocks.7.ff.linear_out.scales"]?.shape, [4096, 192])
        XCTAssertNil(tensors["transformer_blocks.8.attn.to_q.weight"])
        XCTAssertNil(tensors["single_transformer_blocks.24.attn.to_out.weight"])
        // The prompt embedding width the text encoder produces is the transformer's joint_attention_dim, for both variants.
        for variant in KleinVariant.allCases {
            XCTAssertEqual(KleinPromptEncoder.embeddingWidth(variant.textEncoder), variant.transformer.jointAttentionDim, variant.displayName)
        }
        XCTAssertEqual(KleinVariant.klein4B.transformer.tensorCount, 387)
        XCTAssertEqual(KleinVariant.klein4B.transformerShardTensorCounts.values.reduce(0, +), 387)
        XCTAssertEqual(KleinVariant.klein9B.transformerShardTensorCounts.values.reduce(0, +), 539)
    }

    func testQwen3_8BTextEncoderLayout() {
        let layout = TextEncoderLayout.qwen3_8B
        XCTAssertEqual(KleinVariant.klein9B.textEncoder, layout)
        let tensors = layout.expectedTensors()
        XCTAssertEqual(tensors.count, 905)
        XCTAssertEqual(tensors["embed_tokens.weight"]?.shape, [151_936, 512])
        XCTAssertEqual(tensors["layers.0.self_attn.q_proj.weight"]?.shape, [4096, 512])
        XCTAssertEqual(tensors["layers.0.self_attn.k_proj.weight"]?.shape, [1024, 512])
        XCTAssertEqual(tensors["layers.0.self_attn.o_proj.weight"]?.shape, [4096, 512])
        XCTAssertEqual(tensors["layers.0.mlp.down_proj.weight"]?.shape, [4096, 1536])
        XCTAssertEqual(tensors["layers.35.mlp.gate_proj.scales"]?.shape, [12288, 64])
        XCTAssertEqual(tensors["norm.weight"]?.shape, [4096])
        XCTAssertEqual(tensors["layers.0.self_attn.q_norm.weight"]?.shape, [128])
    }

    func testEachVariantAcceptsItsOwnLayoutAndRefusesTheOther() throws {
        for variant in KleinVariant.allCases {
            let other: KleinVariant = variant == .klein4B ? .klein9B : .klein4B
            let t = transformerParts(variant)
            XCTAssertNoThrow(try KleinTransformerWeights.validate(weightMap: t.map, indexMetadata: t.metadata, shardHeaders: t.headers, variant: variant))
            XCTAssertThrowsError(try KleinTransformerWeights.validate(weightMap: t.map, indexMetadata: t.metadata, shardHeaders: t.headers, variant: other))
            let wrongVersion = transformerParts(variant, mfluxVersion: other.storedMfluxVersion)
            XCTAssertThrowsError(try KleinTransformerWeights.validate(
                weightMap: wrongVersion.map, indexMetadata: wrongVersion.metadata, shardHeaders: wrongVersion.headers, variant: variant))

            let text = textParts(variant.textEncoder, mfluxVersion: variant.storedMfluxVersion)
            XCTAssertNoThrow(try KleinTextWeights.validate(weightMap: text.map, indexMetadata: text.metadata, shardHeaders: text.headers, variant: variant))
            // The other size's encoder under the right metadata is still refused on its shapes.
            let swapped = textParts(other.textEncoder, mfluxVersion: variant.storedMfluxVersion)
            XCTAssertThrowsError(try KleinTextWeights.validate(weightMap: swapped.map, indexMetadata: swapped.metadata, shardHeaders: swapped.headers, variant: variant))
        }
    }

    func testRealKlein9BSnapshotHeadersValidateAndAreRefusedAs4B() throws {
        let model = try KleinTestSupport.requireModel9B()
        for (name, pin) in KleinPins.tokenizerFiles {
            XCTAssertEqual(try KleinHashing.sha256(of: model.appendingPathComponent("tokenizer/" + name)), pin, name)
        }
        let text = try KleinTextWeights.readAndValidate(modelDirectory: model, variant: .klein9B)
        XCTAssertEqual(text.weightCount, 905)
        let transformer = try KleinTransformerWeights.readAndValidate(modelDirectory: model, variant: .klein9B)
        XCTAssertEqual(transformer.weightCount, 539)
        XCTAssertEqual(transformer.shards, ["0.safetensors", "1.safetensors", "2.safetensors"])
        XCTAssertEqual(try KleinVAEWeights.readAndValidate(modelDirectory: model, variant: .klein9B).weightCount, KleinVAELayout.tensorCount)
        XCTAssertThrowsError(try KleinTextWeights.readAndValidate(modelDirectory: model, variant: .klein4B))
        XCTAssertThrowsError(try KleinTransformerWeights.readAndValidate(modelDirectory: model, variant: .klein4B))
        XCTAssertThrowsError(try KleinGenerator(modelDirectory: model), "the default (4B) generator refuses a 9B snapshot before loading tensors")
    }

    func testRealKlein4BSnapshotIsRefusedAs9B() throws {
        let model = try KleinTestSupport.requireModel()
        XCTAssertThrowsError(try KleinTextWeights.readAndValidate(modelDirectory: model, variant: .klein9B))
        XCTAssertThrowsError(try KleinTransformerWeights.readAndValidate(modelDirectory: model, variant: .klein9B))
        XCTAssertThrowsError(try KleinGenerator(modelDirectory: model, variant: .klein9B))
    }
}
