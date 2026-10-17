import Foundation

/// The transformer shard layout the loader checks. The file hashes and runtime-module pins of the parity run live in MLXImageParity.
public enum KleinTransformerPins {
    /// Tensors per transformer shard (`model-before.json`): 375 + 12 = 387.
    public static let shardTensorCounts: [String: Int] = ["0.safetensors": 375, "1.safetensors": 12]
}

public enum KleinTransformerError: Error, CustomStringConvertible {
    case artifactChanged(String)
    case invalidWeights(String)
    case invalidInput(String)

    public var description: String {
        switch self {
        case .artifactChanged(let m): "model artifact: \(m)"
        case .invalidWeights(let m): "Klein transformer weights: \(m)"
        case .invalidInput(let m): "invalid input: \(m)"
        }
    }
}
