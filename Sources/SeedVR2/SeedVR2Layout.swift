import Foundation
import MLXImageCore

/// The SeedVR2 3B architecture as MFLUX 0.20 builds it (`SeedVR2Transformer()` and `SeedVR2VAE()` with their defaults), and the stored
/// tensors of the two fp16 files it loads (`numz/SeedVR2_comfyUI`). Every expected tensor name, dtype and shape is derived from these
/// constants; the loader refuses a file whose header differs in any of them.
public enum SeedVR2Layout {
    // MARK: Transformer (NaDiT 3B)

    public static let vidInChannels = 33  // 16 noise + 16 encoded latent + 1 mask
    public static let vidOutChannels = 16
    public static let vidDim = 2560
    public static let txtInDim = 5120
    public static let heads = 20
    public static let headDim = 128
    public static let layers = 32
    /// Blocks `0 ..< mmLayers` keep separate video and text weights; the others share one set (`.all.`).
    public static let mmLayers = 10
    public static let ropeDim = 128
    /// `RoPEModule`: `dim // 3` per axis, half of that as frequencies (21), stored per block as `attn.rope.rope.freqs`.
    public static let ropeFrequencies = (ropeDim / 3) / 2
    public static let normEps: Float = 1e-5
    public static let patch = (t: 1, h: 2, w: 2)
    public static let window = (t: 4, h: 3, w: 3)
    public static let timestepChannels = 256
    public static let embDim = 6 * vidDim
    /// `SwiGLUMLP`: `int(2 * dim * 4 / 3)` rounded up to a multiple of 256.
    public static let mlpHidden: Int = {
        let hidden = Int(Double(2 * vidDim * 4) / 3.0)
        return 256 * ((hidden + 255) / 256)
    }()

    // MARK: VAE

    public static let latentChannels = 16
    public static let vaeBlockChannels = [128, 256, 512, 512]
    public static let encoderLayersPerBlock = 2
    public static let decoderLayersPerBlock = 3
    public static let vaeScalingFactor: Float = 0.9152
    public static let vaeSpatialScale = 8
    public static let groupNormGroups = 32
    public static let groupNormEps: Float = 1e-6

    // MARK: Quantization at load (MFLUX `quantize=4`)

    public static let bits = 4
    public static let groupSize = 64

    // MARK: Files

    public static let transformerFile = "seedvr2_ema_3b_fp16.safetensors"
    public static let vaeFile = "ema_vae_fp16.safetensors"

    public struct Tensor: Equatable, Sendable {
        public let dtype: String
        public let shape: [Int]
    }

    static func f16(_ shape: Int...) -> Tensor { Tensor(dtype: "F16", shape: shape) }

    /// Whether block `i` uses the shared (`.all.`) weights.
    public static func isShared(_ block: Int) -> Bool { block >= mmLayers }

    /// The 635 tensors of `seedvr2_ema_3b_fp16.safetensors`.
    public static func transformerTensors() -> [String: Tensor] {
        var t: [String: Tensor] = [
            "vid_in.proj.weight": f16(vidDim, vidInChannels * patch.t * patch.h * patch.w), "vid_in.proj.bias": f16(vidDim),
            "txt_in.weight": f16(vidDim, txtInDim), "txt_in.bias": f16(vidDim),
            "emb_in.proj_in.weight": f16(vidDim, timestepChannels), "emb_in.proj_in.bias": f16(vidDim),
            "emb_in.proj_hid.weight": f16(vidDim, vidDim), "emb_in.proj_hid.bias": f16(vidDim),
            "emb_in.proj_out.weight": f16(embDim, vidDim), "emb_in.proj_out.bias": f16(embDim),
            "vid_out_norm.weight": f16(vidDim), "vid_out_ada.out_shift": f16(vidDim), "vid_out_ada.out_scale": f16(vidDim),
            "vid_out.proj.weight": f16(vidOutChannels * patch.t * patch.h * patch.w, vidDim),
            "vid_out.proj.bias": f16(vidOutChannels * patch.t * patch.h * patch.w),
        ]
        for i in 0 ..< layers {
            let b = "blocks.\(i)"
            let groups = isShared(i) ? ["all"] : ["vid", "txt"]
            for g in groups {
                for p in ["attn_shift", "attn_scale", "attn_gate", "mlp_shift", "mlp_scale", "mlp_gate"] { t["\(b).ada.\(g).\(p)"] = f16(vidDim) }
                t["\(b).attn.norm_q.\(g).weight"] = f16(headDim)
                t["\(b).attn.norm_k.\(g).weight"] = f16(headDim)
                t["\(b).attn.proj_qkv.\(g).weight"] = f16(3 * heads * headDim, vidDim)
                t["\(b).attn.proj_out.\(g).weight"] = f16(vidDim, heads * headDim)
                t["\(b).attn.proj_out.\(g).bias"] = f16(vidDim)
                t["\(b).mlp.\(g).proj_in.weight"] = f16(mlpHidden, vidDim)
                t["\(b).mlp.\(g).proj_in_gate.weight"] = f16(mlpHidden, vidDim)
                t["\(b).mlp.\(g).proj_out.weight"] = f16(vidDim, mlpHidden)
            }
            t["\(b).attn.rope.rope.freqs"] = f16(ropeFrequencies)
        }
        return t
    }

    /// MFLUX's `SeedVR2WeightDefinition.quantization_predicate` over the transformer: every linear layer whose input width is a
    /// multiple of 64. Only `vid_in.proj` (input 132) stays fp16.
    public static func transformerQuantizedWeights() -> Set<String> {
        Set(transformerTensors().filter { $0.value.shape.count == 2 && $0.value.shape[1] % 64 == 0 }.keys)
    }

    /// The 250 tensors of `ema_vae_fp16.safetensors` (conv weights as stored, `[out, in, kt, kh, kw]`).
    public static func vaeTensors() -> [String: Tensor] {
        var t: [String: Tensor] = [:]
        func conv(_ name: String, _ out: Int, _ inp: Int, _ k: (Int, Int, Int)) {
            t[name + ".weight"] = f16(out, inp, k.0, k.1, k.2)
            t[name + ".bias"] = f16(out)
        }
        func norm(_ name: String, _ c: Int) {
            t[name + ".weight"] = f16(c)
            t[name + ".bias"] = f16(c)
        }
        func resnet(_ name: String, _ inp: Int, _ out: Int) {
            norm(name + ".norm1", inp)
            conv(name + ".conv1", out, inp, (3, 3, 3))
            norm(name + ".norm2", out)
            conv(name + ".conv2", out, out, (3, 3, 3))
            if inp != out { conv(name + ".conv_shortcut", out, inp, (1, 1, 1)) }
        }
        func mid(_ name: String, _ c: Int) {
            resnet(name + ".resnets.0", c, c)
            resnet(name + ".resnets.1", c, c)
            let a = name + ".attentions.0"
            norm(a + ".group_norm", c)
            for l in ["to_q", "to_k", "to_v", "to_out.0"] {
                t["\(a).\(l).weight"] = f16(c, c)
                t["\(a).\(l).bias"] = f16(c)
            }
        }
        let channels = vaeBlockChannels
        conv("encoder.conv_in", channels[0], 3, (3, 3, 3))
        var output = channels[0]
        for (i, c) in channels.enumerated() {
            let input = output
            output = c
            let final = i == channels.count - 1
            for r in 0 ..< encoderLayersPerBlock { resnet("encoder.down_blocks.\(i).resnets.\(r)", r == 0 ? input : output, output) }
            if !final { conv("encoder.down_blocks.\(i).downsamplers.0.conv", output, output, (encoderTemporalDown(i) ? 3 : 1, 3, 3)) }
        }
        mid("encoder.mid_block", channels.last!)
        norm("encoder.conv_norm_out", channels.last!)
        conv("encoder.conv_out", 2 * latentChannels, channels.last!, (3, 3, 3))

        let reversed = Array(channels.reversed())
        conv("decoder.conv_in", reversed[0], latentChannels, (3, 3, 3))
        mid("decoder.mid_block", reversed[0])
        output = reversed[0]
        for (i, c) in reversed.enumerated() {
            let input = output
            output = c
            let final = i == reversed.count - 1
            for r in 0 ..< decoderLayersPerBlock { resnet("decoder.up_blocks.\(i).resnets.\(r)", r == 0 ? input : output, output) }
            if !final {
                conv("decoder.up_blocks.\(i).upsamplers.0.upscale_conv", output * decoderUpsampleFactor(i), output, (1, 1, 1))
                conv("decoder.up_blocks.\(i).upsamplers.0.conv", output, output, (3, 3, 3))
            }
        }
        norm("decoder.conv_norm_out", reversed.last!)
        conv("decoder.conv_out", 3, reversed.last!, (3, 3, 3))
        return t
    }

    /// The VAE's linear layers (the mid-block attention projections), quantized at load like the transformer's: MFLUX quantizes the
    /// whole model dict, VAE included, and its custom `CausalConv3d` and `GroupNorm` are not quantizable.
    public static func vaeQuantizedWeights() -> Set<String> {
        Set(vaeTensors().filter { $0.value.shape.count == 2 && $0.value.shape[1] % 64 == 0 }.keys)
    }

    /// `Encoder3D`: down blocks 1 and 2 also downsample in time (`temporal_down_blocks=2`).
    static func encoderTemporalDown(_ block: Int) -> Bool {
        block >= vaeBlockChannels.count - 2 - 1 && block != vaeBlockChannels.count - 1
    }

    /// `Decoder3D`: up blocks 0 and 1 also upsample in time (`temporal_up_blocks=2`): 2 x 2 x 2 channels per output channel.
    static func decoderUpsampleFactor(_ block: Int) -> Int { block < 2 ? 8 : 4 }

    // MARK: Validation

    /// Checks one file's header against the expected tensors: no missing, no unexpected, same dtype and shape.
    static func validate(header: SafetensorsHeader, expected: [String: Tensor], file: String) throws {
        let missing = Set(expected.keys).subtracting(header.entries.keys)
        let extra = Set(header.entries.keys).subtracting(expected.keys)
        guard missing.isEmpty, extra.isEmpty else {
            throw SeedVR2Error.invalidWeights(
                "\(file): tensor names differ from the SeedVR2 3B layout; missing \(missing.sorted().prefix(5)), unexpected \(extra.sorted().prefix(5))")
        }
        for (name, entry) in header.entries {
            let want = expected[name]!
            guard entry.dtype == want.dtype, entry.shape == want.shape else {
                throw SeedVR2Error.invalidWeights("\(file): \(name) is \(entry.dtype) \(entry.shape), expected \(want.dtype) \(want.shape)")
            }
        }
    }
}

/// The files `SeedVR2Upscaler` reads, as published in `numz/SeedVR2_comfyUI` at revision `09ced71023636e9bc8cdf9cdecfb2625d1e691e8`
/// (Apache 2.0, the ByteDance SeedVR2 3B weights as safetensors). The loader checks the sizes; the SHA-256 digests are for callers that
/// verify a download (hashing 7.3 GB at every load would cost seconds).
public enum SeedVR2Pins {
    public static let repository = "numz/SeedVR2_comfyUI"
    public static let revision = "09ced71023636e9bc8cdf9cdecfb2625d1e691e8"

    public struct File: Sendable {
        public let name: String
        public let bytes: Int
        public let sha256: String
    }

    public static let transformer = File(
        name: SeedVR2Layout.transformerFile, bytes: 6_783_018_808,
        sha256: "2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304")
    public static let vae = File(
        name: SeedVR2Layout.vaeFile, bytes: 501_324_814, sha256: "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1")
    public static let files = [transformer, vae]

    /// The bundled positive text embedding: ByteDance's `pos_emb.pt` from `ByteDance-Seed/SeedVR2-3B` (Apache 2.0, bf16
    /// `[58, 5120]`), converted to fp16 by MFLUX 0.20 (`pos_emb.safetensors`); every value is the bf16 value exactly.
    public static let positiveEmbeddingSHA256 = "d2e06cadb10f1f149b5aa781185b250eb60f0e64327f12efd17886cecc9c2984"
}

public enum SeedVR2Error: Error, CustomStringConvertible, Equatable {
    case invalidWeights(String)
    case invalidRequest(String)

    public var description: String {
        switch self {
        case .invalidWeights(let m): "SeedVR2 weights: \(m)"
        case .invalidRequest(let m): "SeedVR2 request: \(m)"
        }
    }
}
