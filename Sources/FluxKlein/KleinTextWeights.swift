import Foundation
import MLX
import ZImageNative

/// Klein 4B's text encoder is a Qwen3 4B decoder with exactly the architecture and stored layout of Z-Image Turbo's
/// (`TextEncoderLayout`: 905 tensors, q4 group size 64, bf16 scales/biases/norms, f32 `rotary_emb.inv_freq`). The shared layout is
/// reused, but nothing is assumed from it: the Klein snapshot's own index and shard headers are validated against it here, before
/// any tensor is read, and the Klein metadata (`quantization_level`, `mflux_version`) is pinned.
public struct KleinTextWeights {
    public let tensors: [String: MLXArray]

    public func tensor(_ name: String) -> MLXArray { tensors[name]! }

    /// The quantized matrices of the architecture: (name, out features, in features).
    static func quantizedMatrices() -> [(name: String, out: Int, input: Int)] {
        let hidden = TextEncoderLayout.hiddenSize, qDim = TextEncoderLayout.attentionHeads * TextEncoderLayout.headDim
        let kvDim = TextEncoderLayout.keyValueHeads * TextEncoderLayout.headDim, inter = TextEncoderLayout.intermediateSize
        var result: [(name: String, out: Int, input: Int)] = [("embed_tokens", TextEncoderLayout.vocabSize, hidden)]
        for layer in 0..<TextEncoderLayout.layerCount {
            let p = "layers.\(layer)."
            result += [
                (p + "self_attn.q_proj", qDim, hidden), (p + "self_attn.k_proj", kvDim, hidden), (p + "self_attn.v_proj", kvDim, hidden),
                (p + "self_attn.o_proj", hidden, qDim), (p + "mlp.gate_proj", inter, hidden), (p + "mlp.up_proj", inter, hidden),
                (p + "mlp.down_proj", hidden, inter),
            ]
        }
        return result
    }

    /// Pure validation of an index and its shard headers; no tensor data is touched.
    /// Checks names (no missing, no unexpected), per-tensor dtype and shape, shard assignment, stored metadata and the bits/group size
    /// that mflux's `WeightApplier` infers from the stored shapes (packed width = in * bits / 32, scales width = in / group).
    public static func validate(weightMap: [String: String], indexMetadata: [String: String], shardHeaders: [String: SafetensorsHeader]) throws {
        func fail(_ message: String) -> KleinError { KleinError.invalidWeights(message) }
        guard indexMetadata["quantization_level"] == KleinPins.storedQuantizationLevel else {
            throw fail("index quantization_level \(indexMetadata["quantization_level"] ?? "nil"), expected \(KleinPins.storedQuantizationLevel)")
        }
        guard indexMetadata["mflux_version"] == KleinPins.storedMfluxVersion else {
            throw fail("index mflux_version \(indexMetadata["mflux_version"] ?? "nil"), expected \(KleinPins.storedMfluxVersion)")
        }
        let expected = TextEncoderLayout.expectedTensors()
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
                  header.metadata["mflux_version"] == KleinPins.storedMfluxVersion else {
                throw fail("\(shard) metadata \(header.metadata), expected quantization_level \(KleinPins.storedQuantizationLevel) and mflux_version \(KleinPins.storedMfluxVersion)")
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
        for matrix in quantizedMatrices() {
            let packed = shardHeaders.values.compactMap { $0.entries[matrix.name + ".weight"] }.first!.shape.last!
            let scales = shardHeaders.values.compactMap { $0.entries[matrix.name + ".scales"] }.first!.shape.last!
            let bits = Double(packed * 32) / Double(matrix.input), group = Double(matrix.input) / Double(scales)
            guard bits == Double(TextEncoderLayout.bits), group == Double(TextEncoderLayout.groupSize) else {
                throw fail("\(matrix.name): stored layout implies \(bits) bits / group size \(group), expected \(TextEncoderLayout.bits) / \(TextEncoderLayout.groupSize)")
            }
        }
    }

    /// Reads and validates the index and shard headers of `text_encoder/` under `modelDirectory`; returns the shard names.
    static func readAndValidate(modelDirectory: URL) throws -> (weightCount: Int, shards: [String]) {
        let dir = modelDirectory.appendingPathComponent("text_encoder")
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        guard let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String], let metadata = index["metadata"] as? [String: String]
        else { throw KleinError.invalidWeights("unreadable \(indexURL.lastPathComponent)") }
        var headers: [String: SafetensorsHeader] = [:]
        for shard in Set(weightMap.values) { headers[shard] = try SafetensorsHeader.read(from: dir.appendingPathComponent(shard)) }
        try validate(weightMap: weightMap, indexMetadata: metadata, shardHeaders: headers)
        return (weightMap.count, headers.keys.sorted())
    }

    /// Validates, then loads every tensor unchanged (no transpose, requantization or dtype conversion), addressed by stored name.
    public static func load(modelDirectory: URL) throws -> KleinTextWeights {
        let (count, shards) = try readAndValidate(modelDirectory: modelDirectory)
        var tensors: [String: MLXArray] = [:]
        for shard in shards {
            let url = modelDirectory.appendingPathComponent("text_encoder").appendingPathComponent(shard)
            for (name, array) in try loadArrays(url: url) { tensors[name] = array }
        }
        guard tensors.count == count else { throw KleinError.invalidWeights("loaded \(tensors.count) tensors, expected \(count)") }
        for (name, want) in TextEncoderLayout.expectedTensors() {
            guard let t = tensors[name], t.shape == want.shape, safetensorsDType(t.dtype) == want.dtype else {
                throw KleinError.invalidWeights("\(name): loaded tensor differs from the stored layout")
            }
        }
        return KleinTextWeights(tensors: tensors)
    }

    static func safetensorsDType(_ dtype: DType) -> String {
        switch dtype {
        case .bfloat16: "BF16"
        case .uint32: "U32"
        case .float32: "F32"
        default: "\(dtype)"
        }
    }

    /// Per-tensor name, dtype, shape, byte count and SHA-256 of the loaded bytes (what the reference exporter reports for MFLUX's
    /// applied parameters).
    public func digests() -> [String: KleinTensorDigest] {
        KleinDigestAudit.record(tensors.count)
        var result: [String: KleinTensorDigest] = [:]
        for (name, array) in tensors {
            eval(array)
            let data = array.asData(access: .copy).data  // row-major bytes as stored, what the Python reference hashes
            result[name] = KleinTensorDigest(
                dtype: KleinDType.name(array.dtype), shape: array.shape, bytes: data.count, sha256: KleinHashing.sha256(of: data))
        }
        return result
    }
}

/// Counts the tensors hashed by `digests()` inside one `measure` scope (a test seam: the production load path must hash nothing). The count is task-local, so digests computed by
/// other tests running in parallel, or outside the scope, never reach it.
public enum KleinDigestAudit {
    public final class Scope: @unchecked Sendable {
        private let lock = NSLock()
        private var total = 0
        func add(_ count: Int) { lock.lock(); total += count; lock.unlock() }
        public var tensorCount: Int { lock.lock(); defer { lock.unlock() }; return total }
    }
    @TaskLocal static var scope: Scope?
    static func record(_ count: Int) { scope?.add(count) }
    /// Runs `body` and returns its result with the number of tensors `digests()` hashed inside it (on this task only).
    public static func measure<T>(_ body: () throws -> T) rethrows -> (result: T, tensorCount: Int) {
        let scope = Scope()
        let result = try $scope.withValue(scope, operation: body)
        return (result, scope.tensorCount)
    }
}

public struct KleinTensorDigest: Codable, Equatable, Sendable {
    public var dtype: String
    public var shape: [Int]
    public var bytes: Int
    public var sha256: String
}

public enum KleinDType {
    /// MLX Python's `str(dtype)`.
    public static func name(_ dtype: DType) -> String {
        switch dtype {
        case .bfloat16: "mlx.core.bfloat16"
        case .float32: "mlx.core.float32"
        case .uint32: "mlx.core.uint32"
        case .int64: "mlx.core.int64"
        case .int32: "mlx.core.int32"
        default: "mlx.core.\(dtype)"
        }
    }
}
