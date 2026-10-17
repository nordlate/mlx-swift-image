import Foundation
import MLX
import MLXImageCore

/// The 387 stored tensors of Klein 4B's q4 transformer, validated against `KleinTransformerLayout` (derived from the architecture
/// constants) and loaded unchanged: no transpose, requantization or dtype conversion. Names are mflux's module paths, so nothing is mapped.
public struct KleinTransformerWeights {
    public let tensors: [String: MLXArray]

    public func tensor(_ name: String) -> MLXArray {
        guard let t = tensors[name] else { preconditionFailure("transformer tensor \(name) missing; the loader validates the full layout") }
        return t
    }

    /// Pure validation of an index and its shard headers; no tensor data is touched. Checks names (no missing, no unexpected), the
    /// exact shard set and per-shard tensor counts (375 + 12), per-tensor dtype and shape against the architecture, shard assignment,
    /// stored metadata, and the bits / group size mflux's `WeightApplier` infers from the stored shapes (packed width = in * bits / 32,
    /// scales width = in / group) for every one of the 109 matrices.
    public static func validate(weightMap: [String: String], indexMetadata: [String: String], shardHeaders: [String: SafetensorsHeader]) throws {
        func fail(_ message: String) -> KleinTransformerError { .invalidWeights(message) }
        guard indexMetadata["quantization_level"] == KleinPins.storedQuantizationLevel else {
            throw fail("index quantization_level \(indexMetadata["quantization_level"] ?? "nil"), expected \(KleinPins.storedQuantizationLevel)")
        }
        guard indexMetadata["mflux_version"] == KleinPins.storedMfluxVersion else {
            throw fail("index mflux_version \(indexMetadata["mflux_version"] ?? "nil"), expected \(KleinPins.storedMfluxVersion)")
        }
        let expected = KleinTransformerLayout.expectedTensors()
        let missing = Set(expected.keys).subtracting(weightMap.keys)
        let extra = Set(weightMap.keys).subtracting(expected.keys)
        guard missing.isEmpty, extra.isEmpty else {
            throw fail("index tensor names differ from the Flux2 transformer layout; missing \(missing.sorted().prefix(5)), unexpected \(extra.sorted().prefix(5))")
        }
        guard Set(shardHeaders.keys) == Set(weightMap.values), Set(shardHeaders.keys) == Set(KleinTransformerPins.shardTensorCounts.keys) else {
            throw fail("shard headers \(shardHeaders.keys.sorted()) do not match the index shards \(Set(weightMap.values).sorted()) and the pinned \(KleinTransformerPins.shardTensorCounts.keys.sorted())")
        }
        var seen = Set<String>()
        for (shard, header) in shardHeaders.sorted(by: { $0.key < $1.key }) {
            guard header.metadata["quantization_level"] == KleinPins.storedQuantizationLevel,
                  header.metadata["mflux_version"] == KleinPins.storedMfluxVersion else {
                throw fail("\(shard) metadata \(header.metadata), expected quantization_level \(KleinPins.storedQuantizationLevel) and mflux_version \(KleinPins.storedMfluxVersion)")
            }
            guard Set(header.entries.keys) == Set(weightMap.filter { $0.value == shard }.keys) else {
                throw fail("\(shard) tensors do not match the index assignment")
            }
            guard header.entries.count == KleinTransformerPins.shardTensorCounts[shard] else {
                throw fail("\(shard) holds \(header.entries.count) tensors, pinned \(KleinTransformerPins.shardTensorCounts[shard]!)")
            }
            for (name, entry) in header.entries {
                let want = expected[name]!
                guard entry.dtype == want.dtype, entry.shape == want.shape else {
                    throw fail("\(name): stored \(entry.dtype) \(entry.shape), expected \(want.dtype) \(want.shape)")
                }
                seen.insert(name)
            }
        }
        guard seen.count == expected.count, expected.count == KleinTransformerLayout.tensorCount else {
            throw fail("shards hold \(seen.count) tensors, expected \(expected.count)")
        }
        // Bits and group size as the applier infers them, from the stored header shapes (not the layout constants alone).
        let stored = shardHeaders.values.reduce(into: [String: SafetensorsEntry]()) { $0.merge($1.entries) { a, _ in a } }
        for matrix in KleinTransformerLayout.matrices() {
            guard let packed = stored[matrix.name + ".weight"]?.shape.last, let scales = stored[matrix.name + ".scales"]?.shape.last, scales > 0 else {
                throw fail("\(matrix.name): weight or scales missing from the shard headers")
            }
            let bits = Double(packed * 32) / Double(matrix.input), group = Double(matrix.input) / Double(scales)
            guard bits == Double(KleinTransformerLayout.bits), group == Double(KleinTransformerLayout.groupSize) else {
                throw fail("\(matrix.name): stored layout implies \(bits) bits / group size \(group), expected \(KleinTransformerLayout.bits) / \(KleinTransformerLayout.groupSize)")
            }
        }
    }

    /// Reads and validates the index and shard headers of `transformer/`; returns the tensor count and shard names.
    static func readAndValidate(modelDirectory: URL) throws -> (weightCount: Int, shards: [String]) {
        let dir = modelDirectory.appendingPathComponent("transformer")
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        guard let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String], let metadata = index["metadata"] as? [String: String]
        else { throw KleinTransformerError.invalidWeights("unreadable \(indexURL.lastPathComponent)") }
        var headers: [String: SafetensorsHeader] = [:]
        for shard in Set(weightMap.values) { headers[shard] = try SafetensorsHeader.read(from: dir.appendingPathComponent(shard)) }
        try validate(weightMap: weightMap, indexMetadata: metadata, shardHeaders: headers)
        return (weightMap.count, headers.keys.sorted())
    }

    /// Validates, then loads every tensor unchanged (lazily: data is read when first evaluated; call `materialize()` before compiling).
    public static func load(modelDirectory: URL) throws -> KleinTransformerWeights {
        let (count, shards) = try readAndValidate(modelDirectory: modelDirectory)
        var tensors: [String: MLXArray] = [:]
        for shard in shards {
            let url = modelDirectory.appendingPathComponent("transformer").appendingPathComponent(shard)
            for (name, array) in try loadArrays(url: url) {
                guard tensors.updateValue(array, forKey: name) == nil else { throw KleinTransformerError.invalidWeights("\(name) is stored in more than one shard") }
            }
        }
        guard tensors.count == count else { throw KleinTransformerError.invalidWeights("loaded \(tensors.count) tensors, expected \(count)") }
        for (name, want) in KleinTransformerLayout.expectedTensors() {
            guard let t = tensors[name], t.shape == want.shape, KleinTextWeights.safetensorsDType(t.dtype) == want.dtype else {
                throw KleinTransformerError.invalidWeights("\(name): loaded tensor differs from the stored layout")
            }
        }
        return KleinTransformerWeights(tensors: tensors)
    }

    /// Reads every tensor from the shards now. MLX's `compile` copies a still-lazy captured array into each input signature's graph
    /// (the Z-Image full-weight-copy finding), so the production-equivalent compiled prediction materializes the weights first;
    /// evaluated arrays carry no primitive and are shared by every compiled graph.
    public func materialize() {
        eval(Array(tensors.values))
    }
}
