import Foundation
import MLX
import MLXImageCore

/// Klein 4B's text encoder is a Qwen3 4B decoder with exactly the architecture and stored layout of Z-Image Turbo's
/// (`TextEncoderLayout.qwen3_4B`: 905 tensors, q4 group size 64, bf16 scales/biases/norms, f32 `rotary_emb.inv_freq`); Klein 9B's is the
/// Qwen3 8B decoder (`TextEncoderLayout.qwen3_8B`: the same 905 tensors, wider). The shared layout is reused, but nothing is assumed
/// from it: the Klein snapshot's own index and shard headers are validated against the variant's layout here, before any tensor is
/// read, and the Klein metadata (`quantization_level`, the variant's `mflux_version`) is pinned.
public struct KleinTextWeights {
    public let tensors: [String: MLXArray]
    public let layout: TextEncoderLayout

    public init(tensors: [String: MLXArray], layout: TextEncoderLayout = .qwen3_4B) {
        self.tensors = tensors
        self.layout = layout
    }

    public func tensor(_ name: String) -> MLXArray { tensors[name]! }

    /// Pure validation of an index and its shard headers; no tensor data is touched.
    /// Checks names (no missing, no unexpected), per-tensor dtype and shape, shard assignment, stored metadata and the bits/group size
    /// that mflux's `WeightApplier` infers from the stored shapes (packed width = in * bits / 32, scales width = in / group).
    public static func validate(
        weightMap: [String: String], indexMetadata: [String: String], shardHeaders: [String: SafetensorsHeader], variant: KleinVariant = .klein4B
    ) throws {
        func fail(_ message: String) -> KleinError { KleinError.invalidWeights(message) }
        let layout = variant.textEncoder, mfluxVersion = variant.storedMfluxVersion
        guard indexMetadata["quantization_level"] == KleinPins.storedQuantizationLevel else {
            throw fail("index quantization_level \(indexMetadata["quantization_level"] ?? "nil"), expected \(KleinPins.storedQuantizationLevel)")
        }
        guard indexMetadata["mflux_version"] == mfluxVersion else {
            throw fail("index mflux_version \(indexMetadata["mflux_version"] ?? "nil"), expected \(mfluxVersion)")
        }
        let expected = layout.expectedTensors()
        let missing = Set(expected.keys).subtracting(weightMap.keys)
        let extra = Set(weightMap.keys).subtracting(expected.keys)
        guard missing.isEmpty, extra.isEmpty else {
            throw fail("index tensor names differ from the Qwen3 layout; missing \(missing.sorted().prefix(5)), unexpected \(extra.sorted().prefix(5))")
        }
        guard Set(shardHeaders.keys) == Set(weightMap.values) else {
            throw fail("shard headers \(shardHeaders.keys.sorted()) do not match the index shards \(Set(weightMap.values).sorted())")
        }
        var seen = Set<String>()
        for (shard, header) in shardHeaders.sorted(by: { $0.key < $1.key }) {
            guard header.metadata["quantization_level"] == KleinPins.storedQuantizationLevel,
                  header.metadata["mflux_version"] == mfluxVersion else {
                throw fail("\(shard) metadata \(header.metadata), expected quantization_level \(KleinPins.storedQuantizationLevel) and mflux_version \(mfluxVersion)")
            }
            guard Set(header.entries.keys) == Set(weightMap.filter { $0.value == shard }.keys) else {
                throw fail("\(shard) tensors do not match the index assignment")
            }
            for (name, entry) in header.entries {
                let want = expected[name]!
                guard entry.dtype == want.dtype, entry.shape == want.shape else {
                    throw fail("\(name): stored \(entry.dtype) \(entry.shape), expected \(want.dtype) \(want.shape)")
                }
                seen.insert(name)
            }
        }
        guard seen.count == expected.count else { throw fail("shards hold \(seen.count) tensors, expected \(expected.count)") }
        // Bits and group size as the applier infers them, for every quantized matrix. Derived from the stored header entries
        // (the shapes just validated), so a snapshot saved at another precision cannot slip through on the layout constants alone.
        for matrix in layout.quantizedMatrices() {
            let packed = shardHeaders.values.compactMap { $0.entries[matrix.name + ".weight"] }.first!.shape.last!
            let scales = shardHeaders.values.compactMap { $0.entries[matrix.name + ".scales"] }.first!.shape.last!
            let bits = Double(packed * 32) / Double(matrix.input), group = Double(matrix.input) / Double(scales)
            guard bits == Double(TextEncoderLayout.bits), group == Double(TextEncoderLayout.groupSize) else {
                throw fail("\(matrix.name): stored layout implies \(bits) bits / group size \(group), expected \(TextEncoderLayout.bits) / \(TextEncoderLayout.groupSize)")
            }
        }
    }

    /// Reads and validates the index and shard headers of `text_encoder/` under `modelDirectory`; returns the shard names.
    static func readAndValidate(modelDirectory: URL, variant: KleinVariant = .klein4B) throws -> (weightCount: Int, shards: [String]) {
        let dir = modelDirectory.appendingPathComponent("text_encoder")
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        guard let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String], let metadata = index["metadata"] as? [String: String]
        else { throw KleinError.invalidWeights("unreadable \(indexURL.lastPathComponent)") }
        var headers: [String: SafetensorsHeader] = [:]
        for shard in Set(weightMap.values) { headers[shard] = try SafetensorsHeader.read(from: dir.appendingPathComponent(shard)) }
        try validate(weightMap: weightMap, indexMetadata: metadata, shardHeaders: headers, variant: variant)
        return (weightMap.count, headers.keys.sorted())
    }

    /// Validates, then loads every tensor unchanged (no transpose, requantization or dtype conversion), addressed by stored name.
    public static func load(modelDirectory: URL, variant: KleinVariant = .klein4B) throws -> KleinTextWeights {
        let (count, shards) = try readAndValidate(modelDirectory: modelDirectory, variant: variant)
        var tensors: [String: MLXArray] = [:]
        for shard in shards {
            let url = modelDirectory.appendingPathComponent("text_encoder").appendingPathComponent(shard)
            for (name, array) in try loadArrays(url: url) { tensors[name] = array }
        }
        guard tensors.count == count else { throw KleinError.invalidWeights("loaded \(tensors.count) tensors, expected \(count)") }
        for (name, want) in variant.textEncoder.expectedTensors() {
            guard let t = tensors[name], t.shape == want.shape, safetensorsDType(t.dtype) == want.dtype else {
                throw KleinError.invalidWeights("\(name): loaded tensor differs from the stored layout")
            }
        }
        return KleinTextWeights(tensors: tensors, layout: variant.textEncoder)
    }

    static func safetensorsDType(_ dtype: DType) -> String {
        switch dtype {
        case .bfloat16: "BF16"
        case .uint32: "U32"
        case .float32: "F32"
        default: "\(dtype)"
        }
    }
}
