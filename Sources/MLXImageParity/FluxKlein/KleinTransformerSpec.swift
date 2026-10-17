import FluxKlein
import Foundation
import MLX

/// The tensor names and the expected layout of everything one transformer case saves, on each side. The comparison requires exactly
/// these sets (no missing, no unexpected tensor) and pins dtype and shape from the architecture, independent of what either side says.
public enum KleinTransformerTensors {
    public struct Expect: Equatable, Sendable {
        public let dtype: String?  // nil: the dtype is not pinned (both sides must still agree)
        public let shape: [Int]
    }

    public static let sequenceLength = 512
    static let bf16 = "mlx.core.bfloat16", f32 = "mlx.core.float32", i32 = "mlx.core.int32"

    /// The synthetic noise of free-running chain step `t`: `normal(key((seed + 1) * 1000 + t))` cast to bfloat16, independent of the model.
    public static func chainNoiseSeed(seed: UInt64, step: Int) -> UInt64 { (seed + 1) * 1000 + UInt64(step) }

    public static func chainNoise(seed: UInt64, step: Int, shape: [Int]) -> MLXArray {
        MLXRandom.normal(shape, key: MLXRandom.key(chainNoiseSeed(seed: seed, step: step))).asType(.bfloat16)
    }

    /// Names the independent primary runs (reference and native) save for a case.
    public static func primaryKeys(_ c: KleinTransformerCase) -> [String] {
        ["prompt_embeds", "text_ids", "latents", "latent_ids", "sigmas", "timesteps", "noise_compiled", "noise_eager"]
            + (0...c.steps).map { "chain.\($0)" } + (0..<c.steps).map { "chain_noise.\($0)" } + KleinTransformerTrace.allKeys
    }

    public static func primaryLayout(_ c: KleinTransformerCase) -> [String: Expect] {
        let n = c.imageSeqLen, s = sequenceLength, d = KleinTransformerLayout.dim
        var layout: [String: Expect] = [
            "prompt_embeds": Expect(dtype: bf16, shape: [1, s, 7680]), "text_ids": Expect(dtype: i32, shape: [1, s, 4]),
            "latents": Expect(dtype: bf16, shape: [1, n, 128]), "latent_ids": Expect(dtype: i32, shape: [1, n, 4]),
            "sigmas": Expect(dtype: f32, shape: [c.steps + 1]), "timesteps": Expect(dtype: f32, shape: [c.steps]),
            "noise_compiled": Expect(dtype: bf16, shape: [1, n, 128]), "noise_eager": Expect(dtype: bf16, shape: [1, n, 128]),
            "timestep_scaled": Expect(dtype: bf16, shape: [1]), "temb_raw": Expect(dtype: f32, shape: [1, d]), "temb": Expect(dtype: bf16, shape: [1, d]),
            "x_embed": Expect(dtype: bf16, shape: [1, n, d]), "ctx_embed": Expect(dtype: bf16, shape: [1, s, d]),
            "rope_img_cos": Expect(dtype: f32, shape: [n, 64]), "rope_img_sin": Expect(dtype: f32, shape: [n, 64]),
            "rope_txt_cos": Expect(dtype: f32, shape: [s, 64]), "rope_txt_sin": Expect(dtype: f32, shape: [s, 64]),
            "norm_out_in": Expect(dtype: bf16, shape: [1, n, d]), "norm_out_out": Expect(dtype: bf16, shape: [1, n, d]),
        ]
        for t in 0...c.steps { layout["chain.\(t)"] = Expect(dtype: bf16, shape: [1, n, 128]) }
        for t in 0..<c.steps { layout["chain_noise.\(t)"] = Expect(dtype: bf16, shape: [1, n, 128]) }
        for k in 0..<6 { layout["mod_img.\(k)"] = Expect(dtype: bf16, shape: [1, 1, d]); layout["mod_txt.\(k)"] = Expect(dtype: bf16, shape: [1, 1, d]) }
        for k in 0..<3 { layout["mod_single.\(k)"] = Expect(dtype: bf16, shape: [1, 1, d]) }
        for i in KleinTransformerTrace.doubleBlocks {
            layout["double.\(i).in_hidden"] = Expect(dtype: bf16, shape: [1, n, d]); layout["double.\(i).out_hidden"] = Expect(dtype: bf16, shape: [1, n, d])
            layout["double.\(i).in_encoder"] = Expect(dtype: bf16, shape: [1, s, d]); layout["double.\(i).out_encoder"] = Expect(dtype: bf16, shape: [1, s, d])
        }
        for i in KleinTransformerTrace.singleBlocks {
            layout["single.\(i).in"] = Expect(dtype: bf16, shape: [1, s + n, d]); layout["single.\(i).out"] = Expect(dtype: bf16, shape: [1, s + n, d])
        }
        return layout
    }

    /// Names the native isolated run saves for a case (`iso.` + the primary name, plus one scheduler step per step).
    public static let isolatedComponentKeys: [String] =
        ["x_embed", "ctx_embed", "temb_raw", "temb"] + (0..<6).map { "mod_img.\($0)" } + (0..<6).map { "mod_txt.\($0)" }
        + (0..<3).map { "mod_single.\($0)" } + ["rope_img_cos", "rope_img_sin", "rope_txt_cos", "rope_txt_sin"]
    public static let isolatedBlockKeys: [String] =
        KleinTransformerTrace.doubleBlocks.flatMap { ["double.\($0).out_hidden", "double.\($0).out_encoder"] }
        + KleinTransformerTrace.singleBlocks.map { "single.\($0).out" } + ["norm_out_out"]

    public static func isolatedKeys(_ c: KleinTransformerCase) -> [String] {
        (isolatedComponentKeys + isolatedBlockKeys + ["noise_eager", "noise_compiled"]).map { "iso.\($0)" } + (0..<c.steps).map { "iso.step.\($0)" }
    }

    /// The reference tensors an isolated run reads as its inputs.
    public static func isolatedInputKeys(_ c: KleinTransformerCase) -> [String] {
        ["latents", "latent_ids", "prompt_embeds", "text_ids", "timesteps", "sigmas", "timestep_scaled", "temb", "norm_out_in"]
            + (0..<6).flatMap { ["mod_img.\($0)", "mod_txt.\($0)"] } + (0..<3).map { "mod_single.\($0)" }
            + ["rope_img_cos", "rope_img_sin", "rope_txt_cos", "rope_txt_sin"]
            + KleinTransformerTrace.doubleBlocks.flatMap { ["double.\($0).in_hidden", "double.\($0).in_encoder"] }
            + KleinTransformerTrace.singleBlocks.map { "single.\($0).in" }
            + (0..<c.steps).flatMap { ["chain.\($0)", "chain_noise.\($0)"] }
    }

    public static func isolatedLayout(_ c: KleinTransformerCase) -> [String: Expect] {
        let primary = primaryLayout(c)
        var layout: [String: Expect] = [:]
        for name in isolatedComponentKeys + isolatedBlockKeys + ["noise_eager", "noise_compiled"] { layout["iso." + name] = primary[name] }
        for t in 0..<c.steps { layout["iso.step.\(t)"] = primary["chain.\(t + 1)"] }
        return layout
    }
}
