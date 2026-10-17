import Foundation

/// The architecture of Z-Image Turbo's transformer as mflux 0.18/0.20 defines it (`ZImageTransformer` defaults), and the
/// exact tensor names/dtypes/shapes the existing q4 snapshot must contain. Constants come from the mflux module
/// definitions and were checked against the snapshot headers; the loader refuses anything that differs.
public enum TransformerLayout {
    public static let dim = 3840
    public static let heads = 30
    public static let headDim = 128
    public static let layerCount = 30
    public static let refinerLayerCount = 2
    public static let feedForwardHidden = 10240  // int(dim / 3 * 8)
    public static let inChannels = 16
    public static let patchSize = 2
    public static let framePatchSize = 1
    public static let capFeatureDim = 2560
    public static let normEps: Float = 1e-5
    /// mflux's `FinalLayer` norm: `nn.LayerNorm(dim, eps=1e-6, affine=False)`.
    public static let finalNormEps: Float = 1e-6
    public static let ropeTheta: Float = 256
    public static let timestepScale: Float = 1000
    public static let axesDims = [32, 48, 48]
    public static let axesLens = [1024, 512, 512]
    /// `TimestepEmbedder(out_size=min(dim, 256), mid_size=1024)` with a 256-wide frequency embedding.
    public static let timestepFrequencySize = 256
    public static let timestepMidSize = 1024
    public static let modulationInput = 256  // min(dim, 256): width of the timestep embedding fed to every adaLN
    public static let timestepMaxPeriod: Double = 10_000
    /// Patch/frame key of the single-resolution embedder dictionaries: `"\(patch)-\(framePatch)"`.
    public static let embedderKey = "2-1"
    public static let bits = 4
    public static let groupSize = 64

    /// A 4-bit affine-quantized `nn.Linear` `[out, in]`: packed U32 `[out, in*bits/32]`, BF16 scales/biases
    /// `[out, in/group]`, plus a BF16 `bias` `[out]` when the layer has one.
    public struct Quantized {
        public let name: String
        public let out: Int
        public let input: Int
        public let bias: Bool
    }

    /// Quantized linears of one transformer/refiner block.
    static func blockMatrices(_ prefix: String, modulated: Bool) -> [Quantized] {
        var result = [
            Quantized(name: prefix + "attention.to_q", out: dim, input: dim, bias: false),
            Quantized(name: prefix + "attention.to_k", out: dim, input: dim, bias: false),
            Quantized(name: prefix + "attention.to_v", out: dim, input: dim, bias: false),
            Quantized(name: prefix + "attention.to_out.0", out: dim, input: dim, bias: false),
            Quantized(name: prefix + "feed_forward.w1", out: feedForwardHidden, input: dim, bias: false),
            Quantized(name: prefix + "feed_forward.w2", out: dim, input: feedForwardHidden, bias: false),
            Quantized(name: prefix + "feed_forward.w3", out: feedForwardHidden, input: dim, bias: false),
        ]
        if modulated { result.append(Quantized(name: prefix + "adaLN_modulation.0", out: 4 * dim, input: modulationInput, bias: true)) }
        return result
    }

    static var topLevelMatrices: [Quantized] {
        [
            Quantized(name: "all_x_embedder.\(embedderKey)", out: dim, input: framePatchSize * patchSize * patchSize * inChannels, bias: true),
            Quantized(name: "all_final_layer.\(embedderKey).adaLN_modulation.0", out: dim, input: modulationInput, bias: true),
            Quantized(name: "all_final_layer.\(embedderKey).linear", out: framePatchSize * patchSize * patchSize * inChannels, input: dim, bias: true),
            Quantized(name: "cap_embedder.1", out: dim, input: capFeatureDim, bias: true),
            Quantized(name: "t_embedder.linear1", out: timestepMidSize, input: timestepFrequencySize, bias: true),
            Quantized(name: "t_embedder.linear2", out: modulationInput, input: timestepMidSize, bias: true),
        ]
    }

    /// Block prefixes with whether they carry an adaLN modulation: noise refiner and main layers do, context refiner does not.
    static var blockPrefixes: [(prefix: String, modulated: Bool)] {
        (0..<refinerLayerCount).map { ("noise_refiner.\($0).", true) }
            + (0..<refinerLayerCount).map { ("context_refiner.\($0).", false) }
            + (0..<layerCount).map { ("layers.\($0).", true) }
    }

    /// Every quantized matrix of the model.
    public static var allMatrices: [Quantized] {
        topLevelMatrices + blockPrefixes.flatMap { blockMatrices($0.prefix, modulated: $0.modulated) }
    }

    /// Every tensor the snapshot must hold, with its stored dtype and shape.
    public static func expectedTensors() -> [String: (dtype: String, shape: [Int])] {
        var result: [String: (dtype: String, shape: [Int])] = [:]
        for m in allMatrices {
            result[m.name + ".weight"] = ("U32", [m.out, m.input * bits / 32])
            result[m.name + ".scales"] = ("BF16", [m.out, m.input / groupSize])
            result[m.name + ".biases"] = ("BF16", [m.out, m.input / groupSize])
            if m.bias { result[m.name + ".bias"] = ("BF16", [m.out]) }
        }
        result["cap_embedder.0.weight"] = ("BF16", [capFeatureDim])
        result["x_pad_token"] = ("BF16", [1, dim])
        result["cap_pad_token"] = ("BF16", [1, dim])
        for (prefix, _) in blockPrefixes {
            for norm in ["attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2"] {
                result[prefix + norm + ".weight"] = ("BF16", [dim])
            }
            for norm in ["norm_q", "norm_k"] { result[prefix + "attention." + norm + ".weight"] = ("BF16", [headDim]) }
        }
        return result
    }
}
