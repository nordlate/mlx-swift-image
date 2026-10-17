import Foundation
import MLX
import MLXImageCore

/// What `inspect` reports about a VAE snapshot after validation; no tensor data is needed to produce it.
public struct VAEInspection: Codable, Equatable, Sendable {
    public let tensorCount: Int
    public let quantizedLinearCount: Int
    public let shards: [String: Int]
    public let quantizationLevel: String
    public let mfluxVersion: String?
    public let bits: Int
    public let groupSize: Int
}

/// The VAE tensors of an existing Z-Image Turbo q4 snapshot, loaded unchanged (read-only), under the same rules as
/// `TransformerWeights`: the index and every shard header are validated against `VAELayout` before any tensor is read, the
/// quantization of every packed linear is derived from the stored shapes, and nothing is mapped, transposed or converted.
public struct VAEWeights {
    public let tensors: [String: MLXArray]
    public let inspection: VAEInspection

    public func tensor(_ name: String) -> MLXArray {
        guard let t = tensors[name] else { preconditionFailure("VAE tensor \(name) missing; the loader validates the full layout") }
        return t
    }

    public static func inspect(modelDirectory: URL) throws -> (inspection: VAEInspection, shardFiles: [String]) {
        let dir = modelDirectory.appendingPathComponent("vae")
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        guard let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String],
              let indexMeta = index["metadata"] as? [String: String]
        else { throw WeightLoadError.invalid("vae: unreadable \(indexURL.lastPathComponent)") }
        guard indexMeta["quantization_level"] == "\(VAELayout.bits)" else {
            throw WeightLoadError.invalid("vae: index quantization_level \(indexMeta["quantization_level"] ?? "nil"), expected \(VAELayout.bits)")
        }
        let expected = VAELayout.expectedTensors()
        let missing = Set(expected.keys).subtracting(weightMap.keys)
        let extra = Set(weightMap.keys).subtracting(expected.keys)
        guard missing.isEmpty, extra.isEmpty else {
            throw WeightLoadError.invalid(
                "vae: index tensor names differ from the Z-Image VAE layout; missing \(missing.sorted().prefix(5)), unexpected \(extra.sorted().prefix(5))")
        }

        var shardCounts: [String: Int] = [:]
        var stored: [String: SafetensorsEntry] = [:]
        var mfluxVersion: String?
        for shard in Set(weightMap.values).sorted() {
            let header = try SafetensorsHeader.read(from: dir.appendingPathComponent(shard))
            guard header.metadata["quantization_level"] == "\(VAELayout.bits)" else {
                throw WeightLoadError.invalid("vae: \(shard) metadata quantization_level \(header.metadata["quantization_level"] ?? "nil")")
            }
            if let version = header.metadata["mflux_version"] {
                guard mfluxVersion == nil || mfluxVersion == version else {
                    throw WeightLoadError.invalid("vae: shards disagree on mflux_version (\(mfluxVersion ?? "nil") vs \(version))")
                }
                mfluxVersion = version
            }
            let assigned = Set(weightMap.filter { $0.value == shard }.keys)
            guard Set(header.entries.keys) == assigned else {
                throw WeightLoadError.invalid("vae: \(shard) tensors do not match the index assignment")
            }
            for (name, entry) in header.entries {
                let want = expected[name]!
                guard entry.dtype == want.dtype, entry.shape == want.shape else {
                    throw WeightLoadError.invalid("vae \(name): stored \(entry.dtype) \(entry.shape), expected \(want.dtype) \(want.shape)")
                }
            }
            shardCounts[shard] = header.entries.count
            stored.merge(header.entries) { first, _ in first }
        }
        // mflux's applier infers bits from the packed width and the group size from the scales width.
        for name in VAELayout.quantizedLinears {
            guard let w = stored[name + ".weight"]?.shape.last, let s = stored[name + ".scales"]?.shape.last else {
                throw WeightLoadError.invalid("vae \(name): weight or scales missing from the shard headers")
            }
            let input = VAELayout.midChannels
            guard w * 32 / input == VAELayout.bits, input / s == VAELayout.groupSize else {
                throw WeightLoadError.invalid("vae \(name): stored shapes imply bits \(w * 32 / input), group size \(input / s)")
            }
        }
        let inspection = VAEInspection(
            tensorCount: weightMap.count, quantizedLinearCount: VAELayout.quantizedLinears.count, shards: shardCounts,
            quantizationLevel: indexMeta["quantization_level"]!, mfluxVersion: mfluxVersion, bits: VAELayout.bits, groupSize: VAELayout.groupSize)
        return (inspection, Set(weightMap.values).sorted())
    }

    /// Loads all VAE tensors with dtypes and shapes exactly as stored (lazily: data is read when first evaluated).
    public static func load(modelDirectory: URL) throws -> VAEWeights {
        let (inspection, shardFiles) = try inspect(modelDirectory: modelDirectory)
        var tensors: [String: MLXArray] = [:]
        for shard in shardFiles {
            let url = modelDirectory.appendingPathComponent("vae").appendingPathComponent(shard)
            for (name, array) in try loadArrays(url: url) { tensors[name] = array }
        }
        guard tensors.count == inspection.tensorCount else {
            throw WeightLoadError.invalid("vae: loaded \(tensors.count) tensors, expected \(inspection.tensorCount)")
        }
        for (name, want) in VAELayout.expectedTensors() {
            guard let t = tensors[name], t.shape == want.shape, TransformerWeights.dtypeName(t.dtype) == want.dtype else {
                throw WeightLoadError.invalid("vae \(name): loaded tensor differs from the stored layout")
            }
        }
        return VAEWeights(tensors: tensors, inspection: inspection)
    }
}
