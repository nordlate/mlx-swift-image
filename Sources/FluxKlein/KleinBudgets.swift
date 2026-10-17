import Foundation
import MLX
import ZImage

/// Limits for slice 1, declared before any native or reference result existed and compiled in: nothing a run reads can loosen them.
/// `klein/budgets.json` is the human-readable copy; `check(budgetsJSON:)` fails the comparison if it ever differs from these.
///
/// The selected hidden states, the assembled prompt embeddings and the embedding output use limits at least as strict as the
/// Z-Image text-embedding budget (relative L2 1e-3, normalized max abs 1e-2). The embedding output is a pure gather + dequantize
/// of stored bytes, so it gets a much tighter limit. Token IDs, masks and text IDs are compared for exact equality; weights by
/// SHA-256. Non-finite values on either side fail the tensor.
public enum KleinBudgets {
    public struct Limit: Equatable, Sendable {
        public let relativeL2: Double
        public let normalizedMaxAbs: Double
        public var parityLimits: ParityLimits { ParityLimits(relativeL2: relativeL2, normalizedMaxAbs: normalizedMaxAbs) }
    }

    public static let tensors: [String: Limit] = [
        "hs0": Limit(relativeL2: 1e-6, normalizedMaxAbs: 1e-6),
        "hs9": Limit(relativeL2: ParityTolerance.embeddingRelativeL2, normalizedMaxAbs: ParityTolerance.embeddingNormalizedMaxAbs),
        "hs18": Limit(relativeL2: ParityTolerance.embeddingRelativeL2, normalizedMaxAbs: ParityTolerance.embeddingNormalizedMaxAbs),
        "hs27": Limit(relativeL2: ParityTolerance.embeddingRelativeL2, normalizedMaxAbs: ParityTolerance.embeddingNormalizedMaxAbs),
        "prompt_embeds": Limit(relativeL2: ParityTolerance.embeddingRelativeL2, normalizedMaxAbs: ParityTolerance.embeddingNormalizedMaxAbs),
    ]

    public static let exactTensors: [String] = ["attention_mask", "input_ids", "text_ids"]

    /// Expected architecture of every compared tensor. A non-empty prompt is a padded `[1, 512]` int64 batch; the empty prompt is the
    /// `[1, 0]` int32 batch mflux's tokenizer returns, and production MFLUX runs it through the encoder with zero-length tensors
    /// (observed: outcome ok, states `[1, 0, 2560]`, embeddings `[1, 0, 7680]`, text IDs `[1, 0, 4]`). No numeric limit is involved.
    public static func expectedLayout(empty: Bool) -> [String: (dtype: String, shape: [Int])] {
        let s = empty ? 0 : 512
        let tokenDType = empty ? "mlx.core.int32" : "mlx.core.int64"
        return [
            "input_ids": (tokenDType, [1, s]), "attention_mask": (tokenDType, [1, s]),
            "hs0": ("mlx.core.bfloat16", [1, s, 2560]), "hs9": ("mlx.core.bfloat16", [1, s, 2560]),
            "hs18": ("mlx.core.bfloat16", [1, s, 2560]), "hs27": ("mlx.core.bfloat16", [1, s, 2560]),
            "prompt_embeds": ("mlx.core.bfloat16", [1, s, 7680]), "text_ids": ("mlx.core.int32", [1, s, 4]),
        ]
    }

    /// The frozen file must say exactly what the compiled constants say.
    public static func check(budgetsJSON url: URL) throws -> [String] {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
              let declared = root["tensors"] as? [String: [String: Double]], let exact = root["exact"] as? [String],
              let nonFinite = root["non_finite"] as? String
        else { return ["budgets file is not in the expected format"] }
        var failures: [String] = []
        if Set(declared.keys) != Set(tensors.keys) { failures.append("budget tensors \(declared.keys.sorted()) differ from \(tensors.keys.sorted())") }
        for (name, limit) in tensors {
            guard let d = declared[name] else { continue }
            if d["relative_l2"] != limit.relativeL2 || d["normalized_max_abs"] != limit.normalizedMaxAbs {
                failures.append("\(name): file says \(d), compiled limits are \(limit.relativeL2) / \(limit.normalizedMaxAbs)")
            }
        }
        if exact.sorted() != exactTensors { failures.append("exact tensors \(exact.sorted()) differ from \(exactTensors)") }
        if nonFinite != "fail" { failures.append("non_finite must be \"fail\"") }
        return failures
    }

    /// The compiled limits in report form.
    public static var asReport: [String: [String: Double]] {
        tensors.mapValues { ["relative_l2": $0.relativeL2, "normalized_max_abs": $0.normalizedMaxAbs] }
    }
}
