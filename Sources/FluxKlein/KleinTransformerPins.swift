import Foundation

/// The transformer shard layout the loader checks is per variant (`KleinVariant.transformerShardTensorCounts`). The file hashes and
/// runtime-module pins of the 4B parity run live in MLXImageParity.
public enum KleinTransformerPins {}

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
