import Foundation
import MLX

public enum WeightLoadError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self { case .invalid(let m): "text encoder weights: \(m)" }
    }
}

/// What `inspect` reports about a snapshot after validation; no tensor data is needed to produce it.
public struct TextEncoderInspection: Sendable {
    public let tensorCount: Int
    public let shards: [String: Int]
    public let quantizationLevel: String
    public let bits: Int
    public let groupSize: Int
}

/// The q4 text-encoder tensors of an existing Z-Image Turbo snapshot, loaded unchanged (read-only).
public struct TextEncoderWeights {
    public let tensors: [String: MLXArray]
    public let inspection: TextEncoderInspection

    public func tensor(_ name: String) -> MLXArray { tensors[name]! }

    /// Validates the index and every shard header against `TextEncoderLayout` without reading tensor data.
    public static func inspect(modelDirectory: URL) throws -> (inspection: TextEncoderInspection, shardFiles: [String]) {
        let dir = modelDirectory.appendingPathComponent("text_encoder")
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        guard let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String],
              let indexMeta = index["metadata"] as? [String: String]
        else { throw WeightLoadError.invalid("unreadable \(indexURL.lastPathComponent)") }
        guard indexMeta["quantization_level"] == "\(TextEncoderLayout.bits)" else {
            throw WeightLoadError.invalid("index quantization_level \(indexMeta["quantization_level"] ?? "nil"), expected \(TextEncoderLayout.bits)")
        }
        let expected = TextEncoderLayout.expectedTensors()
        let missing = Set(expected.keys).subtracting(weightMap.keys)
        let extra = Set(weightMap.keys).subtracting(expected.keys)
        guard missing.isEmpty, extra.isEmpty else {
            throw WeightLoadError.invalid("index tensor names differ from the Qwen3 layout; missing \(missing.sorted().prefix(5)), unexpected \(extra.sorted().prefix(5))")
        }

        var shardCounts: [String: Int] = [:]
        for shard in Set(weightMap.values).sorted() {
            let header = try SafetensorsHeader.read(from: dir.appendingPathComponent(shard))
            guard header.metadata["quantization_level"] == "\(TextEncoderLayout.bits)" else {
                throw WeightLoadError.invalid("\(shard) metadata quantization_level \(header.metadata["quantization_level"] ?? "nil")")
            }
            let assigned = Set(weightMap.filter { $0.value == shard }.keys)
            guard Set(header.entries.keys) == assigned else {
                throw WeightLoadError.invalid("\(shard) tensors do not match the index assignment")
            }
            for (name, entry) in header.entries {
                let want = expected[name]!
                guard entry.dtype == want.dtype, entry.shape == want.shape else {
                    throw WeightLoadError.invalid("\(name): stored \(entry.dtype) \(entry.shape), expected \(want.dtype) \(want.shape)")
                }
            }
            shardCounts[shard] = header.entries.count
        }
        // Same inference mflux's WeightApplier uses for stored quantization: bits from the packed width,
        // group size from the scales width. Here it must equal the layout constants for every matrix.
        for matrix in TextEncoderLayout.layerMatrices(0) {
            let w = expected[matrix.name + ".weight"]!.shape.last!
            let s = expected[matrix.name + ".scales"]!.shape.last!
            precondition(w * 32 / matrix.input == TextEncoderLayout.bits && matrix.input / s == TextEncoderLayout.groupSize)
        }
        let inspection = TextEncoderInspection(
            tensorCount: weightMap.count, shards: shardCounts, quantizationLevel: indexMeta["quantization_level"]!,
            bits: TextEncoderLayout.bits, groupSize: TextEncoderLayout.groupSize)
        return (inspection, Set(weightMap.values).sorted())
    }

    /// Loads all text-encoder tensors with dtypes and shapes exactly as stored.
    public static func load(modelDirectory: URL) throws -> TextEncoderWeights {
        let (inspection, shardFiles) = try inspect(modelDirectory: modelDirectory)
        var tensors: [String: MLXArray] = [:]
        for shard in shardFiles {
            let url = modelDirectory.appendingPathComponent("text_encoder").appendingPathComponent(shard)
            for (name, array) in try loadArrays(url: url) { tensors[name] = array }
        }
        guard tensors.count == inspection.tensorCount else {
            throw WeightLoadError.invalid("loaded \(tensors.count) tensors, expected \(inspection.tensorCount)")
        }
        return TextEncoderWeights(tensors: tensors, inspection: inspection)
    }
}
