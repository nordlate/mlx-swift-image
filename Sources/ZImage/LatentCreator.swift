import Foundation
import MLX

/// Port of mflux's `ZImageLatentCreator` for text-to-image: the seeded initial noise.
public enum ZImageLatentCreator {
    /// `mx.random.normal(shape=[16, 1, height // 8, width // 8], key=mx.random.key(seed)).astype(bfloat16)`
    /// (the model's precision). The Python side draws float32 normals and then casts, so the draw is the same
    /// threefry stream and only the cast rounds.
    public static func createNoise(seed: UInt64, height: Int, width: Int) -> MLXArray {
        MLXRandom.normal([TransformerLayout.inChannels, 1, height / 8, width / 8], key: MLXRandom.key(seed)).asType(.bfloat16)
    }
}
