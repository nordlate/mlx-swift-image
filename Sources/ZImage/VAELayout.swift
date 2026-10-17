import Foundation

/// The architecture of Z-Image Turbo's VAE as mflux defines it (`mflux.models.z_image.model.z_image_vae`), and the exact
/// tensor names/dtypes/shapes the existing q4 snapshot's `vae/` folder must contain. The snapshot is mflux's own save
/// format: convolution weights are stored in MLX's `[out, kh, kw, in]` layout, and the four attention linears of each
/// mid block are 4-bit quantized (`quantization_level` 4) while every convolution and norm is stored as BF16. The loader
/// refuses anything that differs, the encoder included, although only the decoder is evaluated by the native path.
public enum VAELayout {
    public static let latentChannels = 16
    public static let imageChannels = 3
    public static let normGroups = 32
    public static let normEps: Float = 1e-6
    public static let midChannels = 512
    public static let bits = 4
    public static let groupSize = 64
    /// `VAE.scaling_factor` and `VAE.shift_factor`: decode computes `latents / scaling + shift`.
    public static let scalingFactor: Float = 0.3611
    public static let shiftFactor: Float = 0.1159

    /// Decoder `up_blocks`: (in, out, upsample). Three resnets each.
    static let decoderBlocks: [(input: Int, output: Int, upsample: Bool)] = [(512, 512, true), (512, 512, true), (512, 256, true), (256, 128, false)]
    static let decoderResnetsPerBlock = 3
    /// Encoder `down_blocks`: (in, out, downsample). Two resnets each.
    static let encoderBlocks: [(input: Int, output: Int, downsample: Bool)] = [(128, 128, true), (128, 256, true), (256, 512, true), (512, 512, false)]
    static let encoderResnetsPerBlock = 2

    typealias Entry = (dtype: String, shape: [Int])

    private static func conv(_ result: inout [String: Entry], _ name: String, input: Int, output: Int, kernel: Int) {
        result[name + ".weight"] = ("BF16", [output, kernel, kernel, input])
        result[name + ".bias"] = ("BF16", [output])
    }

    private static func norm(_ result: inout [String: Entry], _ name: String, channels: Int) {
        result[name + ".weight"] = ("BF16", [channels])
        result[name + ".bias"] = ("BF16", [channels])
    }

    private static func resnet(_ result: inout [String: Entry], _ name: String, input: Int, output: Int) {
        norm(&result, name + ".norm1", channels: input)
        conv(&result, name + ".conv1", input: input, output: output, kernel: 3)
        norm(&result, name + ".norm2", channels: output)
        conv(&result, name + ".conv2", input: output, output: output, kernel: 3)
        if input != output { conv(&result, name + ".conv_shortcut", input: input, output: output, kernel: 1) }
    }

    /// Names of the four quantized attention linears of a mid block (`to_q`, `to_k`, `to_v`, `to_out.0`), each 512 x 512 with a bias.
    static func attentionLinears(_ prefix: String) -> [String] {
        ["to_q", "to_k", "to_v", "to_out.0"].map { prefix + ".attentions.0." + $0 }
    }

    private static func midBlock(_ result: inout [String: Entry], _ prefix: String) {
        let c = midChannels
        resnet(&result, prefix + ".resnets.0", input: c, output: c)
        resnet(&result, prefix + ".resnets.1", input: c, output: c)
        norm(&result, prefix + ".attentions.0.group_norm", channels: c)
        for linear in attentionLinears(prefix) {
            result[linear + ".weight"] = ("U32", [c, c * bits / 32])
            result[linear + ".scales"] = ("BF16", [c, c / groupSize])
            result[linear + ".biases"] = ("BF16", [c, c / groupSize])
            result[linear + ".bias"] = ("BF16", [c])
        }
    }

    /// The quantized linears of the whole VAE.
    public static var quantizedLinears: [String] { attentionLinears("encoder.mid_block") + attentionLinears("decoder.mid_block") }

    /// Every tensor the snapshot must hold, with its stored dtype and shape.
    public static func expectedTensors() -> [String: (dtype: String, shape: [Int])] {
        var r: [String: Entry] = [:]
        // Decoder (the only part the native path evaluates).
        conv(&r, "decoder.conv_in.conv", input: latentChannels, output: midChannels, kernel: 3)
        midBlock(&r, "decoder.mid_block")
        for (i, block) in decoderBlocks.enumerated() {
            for j in 0 ..< decoderResnetsPerBlock {
                resnet(&r, "decoder.up_blocks.\(i).resnets.\(j)", input: j == 0 ? block.input : block.output, output: block.output)
            }
            if block.upsample { conv(&r, "decoder.up_blocks.\(i).upsamplers.0.conv", input: block.output, output: block.output, kernel: 3) }
        }
        norm(&r, "decoder.conv_norm_out.norm", channels: decoderBlocks.last!.output)
        conv(&r, "decoder.conv_out.conv", input: decoderBlocks.last!.output, output: imageChannels, kernel: 3)
        // Encoder (validated, never evaluated here).
        conv(&r, "encoder.conv_in.conv2d", input: imageChannels, output: encoderBlocks[0].input, kernel: 3)
        for (i, block) in encoderBlocks.enumerated() {
            for j in 0 ..< encoderResnetsPerBlock {
                resnet(&r, "encoder.down_blocks.\(i).resnets.\(j)", input: j == 0 ? block.input : block.output, output: block.output)
            }
            if block.downsample { conv(&r, "encoder.down_blocks.\(i).downsamplers.0.conv", input: block.output, output: block.output, kernel: 3) }
        }
        midBlock(&r, "encoder.mid_block")
        norm(&r, "encoder.conv_norm_out.norm", channels: midChannels)
        conv(&r, "encoder.conv_out.conv2d", input: midChannels, output: 2 * latentChannels, kernel: 3)
        return r
    }
}
