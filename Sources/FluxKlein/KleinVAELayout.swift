import Foundation

/// Independently derived from Flux2Encoder/Decoder constructors. Encoder weights are validated, materialized and digested;
/// the encoder network is never executed.
public enum KleinVAELayout {
    public static let bits = 4, groupSize = 64, tensorCount = 266
    public static let shardTensorCounts = ["0.safetensors": 266]
    public static let normGroups = 32
    public static let normEps: Float = 1e-6
    public static let bnEps: Float = 1e-4
    public struct Tensor: Equatable { public var dtype: String; public var shape: [Int] }
    public struct Matrix { public let name: String; public let input: Int; public let output: Int }
    public static func matrices() -> [Matrix] {
        ["encoder", "decoder"].flatMap { side in
            ["to_q", "to_k", "to_v", "to_out"].map { Matrix(name: side + ".mid_block.attentions.0." + $0, input: 512, output: 512) }
        }
    }
    public static func expectedTensors() -> [String: Tensor] {
        var out: [String: Tensor] = [:]
        func add(_ n: String, _ shape: [Int], _ dtype: String = "BF16") { out[n] = Tensor(dtype: dtype, shape: shape) }
        func norm(_ n: String, _ c: Int) { add(n + ".weight", [c]); add(n + ".bias", [c]) }
        func conv(_ n: String, _ i: Int, _ o: Int, _ k: Int = 3) { add(n + ".weight", [o,k,k,i]); add(n + ".bias", [o]) }
        func resnet(_ n: String, _ i: Int, _ o: Int) {
            norm(n + ".norm1", i); conv(n + ".conv1", i, o)
            norm(n + ".norm2", o); conv(n + ".conv2", o, o)
            if i != o { conv(n + ".conv_shortcut", i, o, 1) }
        }
        for side in ["encoder", "decoder"] {
            let mid = side + ".mid_block"
            for j in 0..<2 { resnet(mid + ".resnets.\(j)", 512, 512) }
            norm(mid + ".attentions.0.group_norm", 512)
        }
        for m in matrices() {
            add(m.name + ".weight", [m.output, m.input * bits / 32], "U32")
            add(m.name + ".scales", [m.output, m.input / groupSize])
            add(m.name + ".biases", [m.output, m.input / groupSize]); add(m.name + ".bias", [m.output])
        }
        conv("encoder.conv_in", 3, 128)
        let enc = [128,256,512,512]
        for i in 0..<4 {
            for j in 0..<2 { resnet("encoder.down_blocks.\(i).resnets.\(j)", j == 0 && i > 0 ? enc[i-1] : enc[i], enc[i]) }
            if i < 3 { conv("encoder.down_blocks.\(i).downsamplers.0.conv", enc[i], enc[i]) }
        }
        norm("encoder.conv_norm_out", 512); conv("encoder.conv_out", 512, 64)
        conv("decoder.conv_in", 32, 512)
        let dec = [512,512,256,128]
        for i in 0..<4 {
            for j in 0..<3 { resnet("decoder.up_blocks.\(i).resnets.\(j)", j == 0 && i > 0 ? dec[i-1] : dec[i], dec[i]) }
            if i < 3 { conv("decoder.up_blocks.\(i).upsamplers.0.conv", dec[i], dec[i]) }
        }
        norm("decoder.conv_norm_out", 128); conv("decoder.conv_out", 128, 3)
        conv("quant_conv", 64, 64, 1); conv("post_quant_conv", 32, 32, 1)
        add("bn.running_mean", [128]); add("bn.running_var", [128])
        return out
    }
}
