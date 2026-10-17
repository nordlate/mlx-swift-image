import Foundation

/// Stage tolerances for text embeddings, as declared in advance, in a tolerance table
/// fixed before any native result existed. They are constants here on purpose: nothing a run reads can loosen them.
public enum ParityTolerance {
    public static let embeddingRelativeL2 = 1e-3
    public static let embeddingNormalizedMaxAbs = 1e-2
    /// A reference whose norm (or max) is below this is "effectively zero": the relative form is undefined.
    public static let zeroReference = 1e-6
    /// Zero-reference fallback: absolute L2 norm and max abs difference of native vs reference, each at most this.
    public static let zeroFallbackAbsolute = 1e-6

    // Slice 2 rows, also from that table (fixed before any native result existed).
    // Relative L2 is the only deciding metric of these rows; max abs is reported as diagnostic.
    /// Patch/timestep embeddings from isolated reference input.
    public static let patchTimestepEmbeddingRelativeL2 = 1e-3
    /// A single transformer block from isolated reference input.
    public static let singleBlockRelativeL2 = 1e-3
    /// The final projection from isolated reference input.
    public static let finalProjectionRelativeL2 = 1e-3
    /// The full transformer output from isolated reference input.
    public static let fullTransformerOutputRelativeL2 = 2e-3
    /// Free-running latent state per scheduler step.
    public static let schedulerLatentPerStepRelativeL2 = 5e-3
    // Slice 4 rows, from the same table.
    /// The VAE output tensor.
    public static let vaeOutputRelativeL2 = 2e-3
    /// Decoded pixels: max abs difference in uint8 levels.
    public static let pixelMaxAbsLevels = 2.0
    /// Decoded pixels: mean abs difference in uint8 levels.
    public static let pixelMeanAbsLevels = 0.1
    // Seeded noise and scheduler sigmas/timesteps are compared for bitwise equality (no tolerance), as are tensor
    // names, dtypes, shapes and per-tensor digests of the loaded q4 weights and the quantization metadata.
}

/// The deciding limits of one comparison row. Text embeddings decide on both metrics; the slice 2 rows on relative L2 only.
public struct ParityLimits: Sendable {
    public let relativeL2: Double
    public let normalizedMaxAbs: Double?
    public init(relativeL2: Double, normalizedMaxAbs: Double?) {
        self.relativeL2 = relativeL2
        self.normalizedMaxAbs = normalizedMaxAbs
    }
    public static let textEmbedding = ParityLimits(
        relativeL2: ParityTolerance.embeddingRelativeL2, normalizedMaxAbs: ParityTolerance.embeddingNormalizedMaxAbs)
    public static let patchTimestepEmbedding = ParityLimits(
        relativeL2: ParityTolerance.patchTimestepEmbeddingRelativeL2, normalizedMaxAbs: nil)
    public static let singleBlock = ParityLimits(relativeL2: ParityTolerance.singleBlockRelativeL2, normalizedMaxAbs: nil)
    public static let finalProjection = ParityLimits(relativeL2: ParityTolerance.finalProjectionRelativeL2, normalizedMaxAbs: nil)
    public static let fullTransformerOutput = ParityLimits(
        relativeL2: ParityTolerance.fullTransformerOutputRelativeL2, normalizedMaxAbs: nil)
    public static let schedulerLatentPerStep = ParityLimits(
        relativeL2: ParityTolerance.schedulerLatentPerStepRelativeL2, normalizedMaxAbs: nil)
    public static let vaeOutput = ParityLimits(relativeL2: ParityTolerance.vaeOutputRelativeL2, normalizedMaxAbs: nil)
}

public struct ComparisonMetrics: Codable, Equatable, Sendable {
    public var relativeL2: Double?
    public var normalizedMaxAbs: Double?
    public var absoluteL2: Double
    public var maxAbsDiff: Double
    public var referenceL2: Double
    public var referenceMaxAbs: Double
    /// True when the reference was effectively zero and the absolute fallback decided the result.
    public var zeroReferenceFallback: Bool
    public var passed: Bool
}

public enum ParityMetrics {
    /// Both inputs are float32 after casting; sums accumulate in Double so the metric does not add its own rounding error.
    public static func compare(native: [Float], reference: [Float], limits: ParityLimits = .textEmbedding) -> ComparisonMetrics {
        precondition(native.count == reference.count)
        var diffSq = 0.0, refSq = 0.0, maxDiff = 0.0, maxRef = 0.0
        for (n, r) in zip(native, reference) {
            let d = Double(n) - Double(r)
            diffSq += d * d
            refSq += Double(r) * Double(r)
            maxDiff = max(maxDiff, abs(d))
            maxRef = max(maxRef, abs(Double(r)))
        }
        let absL2 = diffSq.squareRoot(), refL2 = refSq.squareRoot()
        let zero = refL2 < ParityTolerance.zeroReference || maxRef < ParityTolerance.zeroReference
        var metrics = ComparisonMetrics(
            relativeL2: nil, normalizedMaxAbs: nil, absoluteL2: absL2, maxAbsDiff: maxDiff,
            referenceL2: refL2, referenceMaxAbs: maxRef, zeroReferenceFallback: zero, passed: false)
        if zero {
            metrics.passed = absL2 <= ParityTolerance.zeroFallbackAbsolute && maxDiff <= ParityTolerance.zeroFallbackAbsolute
        } else {
            metrics.relativeL2 = absL2 / refL2
            metrics.normalizedMaxAbs = maxDiff / maxRef
            metrics.passed = metrics.relativeL2! <= limits.relativeL2
                && limits.normalizedMaxAbs.map { metrics.normalizedMaxAbs! <= $0 } ?? true
        }
        return metrics
    }
}
