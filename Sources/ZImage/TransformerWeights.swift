import CryptoKit
import Foundation
import MLX

/// What `inspect` reports about a transformer snapshot after validation; no tensor data is needed to produce it.
public struct TransformerInspection: Codable, Equatable, Sendable {
    public let tensorCount: Int
    public let quantizedLinearCount: Int
    public let shards: [String: Int]
    public let quantizationLevel: String
    public let mfluxVersion: String?
    public let bits: Int
    public let groupSize: Int
}

/// The q4 transformer tensors of an existing Z-Image Turbo snapshot, loaded unchanged (read-only).
///
/// The names are the module paths of mflux's `ZImageTransformer` (the snapshot is mflux's own save format), so no key
/// mapping, transpose or dtype conversion happens: what the headers declare is what is loaded.
public struct TransformerWeights {
    public let tensors: [String: MLXArray]
    public let inspection: TransformerInspection

    public func tensor(_ name: String) -> MLXArray {
        guard let t = tensors[name] else { preconditionFailure("transformer tensor \(name) missing; the loader validates the full layout") }
        return t
    }

    /// Validates the index and every shard header against `TransformerLayout` without reading tensor data.
    public static func inspect(modelDirectory: URL) throws -> (inspection: TransformerInspection, shardFiles: [String]) {
        let dir = modelDirectory.appendingPathComponent("transformer")
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        guard let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String],
              let indexMeta = index["metadata"] as? [String: String]
        else { throw WeightLoadError.invalid("transformer: unreadable \(indexURL.lastPathComponent)") }
        guard indexMeta["quantization_level"] == "\(TransformerLayout.bits)" else {
            throw WeightLoadError.invalid("transformer: index quantization_level \(indexMeta["quantization_level"] ?? "nil"), expected \(TransformerLayout.bits)")
        }
        let expected = TransformerLayout.expectedTensors()
        let missing = Set(expected.keys).subtracting(weightMap.keys)
        let extra = Set(weightMap.keys).subtracting(expected.keys)
        guard missing.isEmpty, extra.isEmpty else {
            throw WeightLoadError.invalid(
                "transformer: index tensor names differ from the Z-Image layout; missing \(missing.sorted().prefix(5)), unexpected \(extra.sorted().prefix(5))")
        }

        var shardCounts: [String: Int] = [:]
        var stored: [String: SafetensorsEntry] = [:]
        var mfluxVersion: String?
        for shard in Set(weightMap.values).sorted() {
            let header = try SafetensorsHeader.read(from: dir.appendingPathComponent(shard))
            guard header.metadata["quantization_level"] == "\(TransformerLayout.bits)" else {
                throw WeightLoadError.invalid("transformer: \(shard) metadata quantization_level \(header.metadata["quantization_level"] ?? "nil")")
            }
            if let version = header.metadata["mflux_version"] {
                guard mfluxVersion == nil || mfluxVersion == version else {
                    throw WeightLoadError.invalid("transformer: shards disagree on mflux_version (\(mfluxVersion ?? "nil") vs \(version))")
                }
                mfluxVersion = version
            }
            let assigned = Set(weightMap.filter { $0.value == shard }.keys)
            guard Set(header.entries.keys) == assigned else {
                throw WeightLoadError.invalid("transformer: \(shard) tensors do not match the index assignment")
            }
            for (name, entry) in header.entries {
                let want = expected[name]!
                guard entry.dtype == want.dtype, entry.shape == want.shape else {
                    throw WeightLoadError.invalid("transformer \(name): stored \(entry.dtype) \(entry.shape), expected \(want.dtype) \(want.shape)")
                }
            }
            shardCounts[shard] = header.entries.count
            stored.merge(header.entries) { first, _ in first }
        }
        // The inference mflux's WeightApplier uses for stored quantization: bits from the packed width, group size from
        // the scales width. Here every matrix must give the layout constants, read from the shard headers' stored shapes.
        for m in TransformerLayout.allMatrices {
            guard let w = stored[m.name + ".weight"]?.shape.last, let s = stored[m.name + ".scales"]?.shape.last else {
                throw WeightLoadError.invalid("transformer \(m.name): weight or scales missing from the shard headers")
            }
            guard w * 32 / m.input == TransformerLayout.bits, m.input / s == TransformerLayout.groupSize else {
                throw WeightLoadError.invalid("transformer \(m.name): stored shapes imply bits \(w * 32 / m.input), group size \(m.input / s)")
            }
        }
        let inspection = TransformerInspection(
            tensorCount: weightMap.count, quantizedLinearCount: TransformerLayout.allMatrices.count, shards: shardCounts,
            quantizationLevel: indexMeta["quantization_level"]!, mfluxVersion: mfluxVersion,
            bits: TransformerLayout.bits, groupSize: TransformerLayout.groupSize)
        return (inspection, Set(weightMap.values).sorted())
    }

    /// Loads all transformer tensors with dtypes and shapes exactly as stored (lazily: data is read when first evaluated).
    public static func load(modelDirectory: URL) throws -> TransformerWeights {
        let (inspection, shardFiles) = try inspect(modelDirectory: modelDirectory)
        var tensors: [String: MLXArray] = [:]
        for shard in shardFiles {
            let url = modelDirectory.appendingPathComponent("transformer").appendingPathComponent(shard)
            for (name, array) in try loadArrays(url: url) { tensors[name] = array }
        }
        guard tensors.count == inspection.tensorCount else {
            throw WeightLoadError.invalid("transformer: loaded \(tensors.count) tensors, expected \(inspection.tensorCount)")
        }
        for (name, want) in TransformerLayout.expectedTensors() {
            guard let t = tensors[name], t.shape == want.shape, dtypeName(t.dtype) == want.dtype else {
                throw WeightLoadError.invalid("transformer \(name): loaded tensor differs from the stored layout")
            }
        }
        return TransformerWeights(tensors: tensors, inspection: inspection)
    }

    /// safetensors dtype name of a loaded array (only the dtypes this layout stores).
    static func dtypeName(_ dtype: DType) -> String {
        switch dtype {
        case .bfloat16: "BF16"
        case .uint32: "U32"
        case .float32: "F32"
        default: "\(dtype)"
        }
    }

    /// SHA-256 over the array's row-major bytes (what the Python reference hashes), after evaluation.
    public static func digest(_ array: MLXArray) -> String {
        SHA256.hash(data: array.asData(access: .copy).data).map { String(format: "%02x", $0) }.joined()
    }
}
