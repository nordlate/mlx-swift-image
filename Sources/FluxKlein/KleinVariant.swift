import Foundation
import MLXImageCore

/// The FLUX.2 Klein sizes this package runs. Both are the same architecture and pipeline (same tokenizer, VAE, scheduler and numerics);
/// they differ only in the transformer's depth and width and in the Qwen3 text encoder's size, so one code path runs both.
///
/// Each variant reads a q4 `mflux-save` snapshot. The loaders pin the stored `mflux_version` metadata and the transformer's shard split
/// for each variant: the 4B snapshot was saved by MFLUX 0.19.0, the 9B snapshot by MFLUX 0.20.0 (`mflux-save --model <bf16 dir>
/// --base-model flux2-klein-9b --quantize 4`).
public enum KleinVariant: String, CaseIterable, Sendable {
    /// FLUX.2 Klein 4B: Qwen3 4B text encoder, 5 + 20 transformer blocks of width 3072.
    case klein4B = "flux2-klein-4b"
    /// FLUX.2 Klein 9B: Qwen3 8B text encoder, 8 + 24 transformer blocks of width 4096.
    case klein9B = "flux2-klein-9b"

    public var displayName: String {
        switch self {
        case .klein4B: "FLUX.2 Klein 4B"
        case .klein9B: "FLUX.2 Klein 9B"
        }
    }

    public var transformer: KleinTransformerLayout {
        switch self {
        case .klein4B: .klein4B
        case .klein9B: .klein9B
        }
    }

    public var textEncoder: TextEncoderLayout {
        switch self {
        case .klein4B: .qwen3_4B
        case .klein9B: .qwen3_8B
        }
    }

    /// The `mflux_version` stored in every shard and index of the snapshot.
    public var storedMfluxVersion: String {
        switch self {
        case .klein4B: "0.19.0"
        case .klein9B: "0.20.0"
        }
    }

    /// Tensors per transformer shard as `mflux-save` splits them (shards of at most 2 GiB, in parameter order).
    public var transformerShardTensorCounts: [String: Int] {
        switch self {
        case .klein4B: ["0.safetensors": 375, "1.safetensors": 12]
        case .klein9B: ["0.safetensors": 341, "1.safetensors": 136, "2.safetensors": 62]
        }
    }
}
