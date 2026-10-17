import Foundation
import MLX
import MLXImageCore

public struct KleinVAEWeights {
    public let tensors: [String: MLXArray]

    public func tensor(_ name: String) -> MLXArray {
        guard let t = tensors[name] else { preconditionFailure("VAE tensor \(name) missing; the loader validates the full layout") }
        return t
    }

    public static func validate(weightMap: [String: String], indexMetadata: [String: String], shardHeaders: [String: SafetensorsHeader]) throws {
        func fail(_ message: String) -> KleinTransformerError { .invalidWeights(message) }
        guard indexMetadata["quantization_level"] == KleinPins.storedQuantizationLevel else {
            throw fail("index quantization_level \(indexMetadata["quantization_level"] ?? "nil"), expected \(KleinPins.storedQuantizationLevel)")
        }
        guard indexMetadata["mflux_version"] == KleinPins.storedMfluxVersion else {
            throw fail("index mflux_version \(indexMetadata["mflux_version"] ?? "nil"), expected \(KleinPins.storedMfluxVersion)")
        }
        let expected = KleinVAELayout.expectedTensors()
        let missing = Set(expected.keys).subtracting(weightMap.keys)
        let extra = Set(weightMap.keys).subtracting(expected.keys)
        guard missing.isEmpty, extra.isEmpty else {
            throw fail("index tensor names differ from the Flux2 VAE layout; missing \(missing.sorted().prefix(5)), unexpected \(extra.sorted().prefix(5))")
        }
        guard Set(shardHeaders.keys) == Set(weightMap.values), Set(shardHeaders.keys) == Set(KleinVAELayout.shardTensorCounts.keys) else {
            throw fail("shard headers \(shardHeaders.keys.sorted()) do not match the index shards \(Set(weightMap.values).sorted()) and the pinned \(KleinVAELayout.shardTensorCounts.keys.sorted())")
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
            guard header.entries.count == KleinVAELayout.shardTensorCounts[shard] else {
                throw fail("\(shard) holds \(header.entries.count) tensors, pinned \(KleinVAELayout.shardTensorCounts[shard]!)")
            }
            for (name, entry) in header.entries {
                let want = expected[name]!
                guard entry.dtype == want.dtype, entry.shape == want.shape else {
                    throw fail("\(name): stored \(entry.dtype) \(entry.shape), expected \(want.dtype) \(want.shape)")
                }
                seen.insert(name)
            }
        }
        guard seen.count == expected.count, expected.count == KleinVAELayout.tensorCount else {
            throw fail("shards hold \(seen.count) tensors, expected \(expected.count)")
        }
        // Bits and group size as the applier infers them, from the stored header shapes (not the layout constants alone).
        let stored = shardHeaders.values.reduce(into: [String: SafetensorsEntry]()) { $0.merge($1.entries) { a, _ in a } }
        for matrix in KleinVAELayout.matrices() {
            guard let packed = stored[matrix.name + ".weight"]?.shape.last, let scales = stored[matrix.name + ".scales"]?.shape.last, scales > 0 else {
                throw fail("\(matrix.name): weight or scales missing from the shard headers")
            }
            let bits = Double(packed * 32) / Double(matrix.input), group = Double(matrix.input) / Double(scales)
            guard bits == Double(KleinVAELayout.bits), group == Double(KleinVAELayout.groupSize) else {
                throw fail("\(matrix.name): stored layout implies \(bits) bits / group size \(group), expected \(KleinVAELayout.bits) / \(KleinVAELayout.groupSize)")
            }
        }
    }

    static func readAndValidate(modelDirectory: URL) throws -> (weightCount: Int, shards: [String]) {
        let dir = modelDirectory.appendingPathComponent("vae")
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        guard let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String], let metadata = index["metadata"] as? [String: String]
        else { throw KleinTransformerError.invalidWeights("unreadable \(indexURL.lastPathComponent)") }
        var headers: [String: SafetensorsHeader] = [:]
        for shard in Set(weightMap.values) { headers[shard] = try SafetensorsHeader.read(from: dir.appendingPathComponent(shard)) }
        try validate(weightMap: weightMap, indexMetadata: metadata, shardHeaders: headers)
        return (weightMap.count, headers.keys.sorted())
    }

    public static func load(modelDirectory: URL) throws -> KleinVAEWeights {
        let (count, shards) = try readAndValidate(modelDirectory: modelDirectory)
        var tensors: [String: MLXArray] = [:]
        for shard in shards {
            let url = modelDirectory.appendingPathComponent("vae").appendingPathComponent(shard)
            for (name, array) in try loadArrays(url: url) {
                guard tensors.updateValue(array, forKey: name) == nil else { throw KleinTransformerError.invalidWeights("\(name) is stored in more than one shard") }
            }
        }
        guard tensors.count == count else { throw KleinTransformerError.invalidWeights("loaded \(tensors.count) tensors, expected \(count)") }
        for (name, want) in KleinVAELayout.expectedTensors() {
            guard let t = tensors[name], t.shape == want.shape, KleinTextWeights.safetensorsDType(t.dtype) == want.dtype else {
                throw KleinTransformerError.invalidWeights("\(name): loaded tensor differs from the stored layout")
            }
        }
        return KleinVAEWeights(tensors: tensors)
    }

    public func materialize() {
        eval(Array(tensors.values))
    }

    public func digests() -> [String: KleinTensorDigest] {
        KleinDigestAudit.record(tensors.count)
        var result: [String: KleinTensorDigest] = [:]
        for (name, array) in tensors {
            eval(array)
            let data = KleinCompare.bytes(array)
            result[name] = KleinTensorDigest(dtype: KleinDType.name(array.dtype), shape: array.shape, bytes: data.count, sha256: KleinHashing.sha256(of: data))
        }
        return result
    }

    public static func quantizedModules() -> [KleinQuantizedModule] {
        KleinVAELayout.matrices().map {
            KleinQuantizedModule(name: $0.name, bits: KleinVAELayout.bits, groupSize: KleinVAELayout.groupSize, mode: "affine")
        }
    }
}
