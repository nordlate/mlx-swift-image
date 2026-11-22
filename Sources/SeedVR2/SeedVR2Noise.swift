import Foundation
import MLX

/// `mx.random.normal(shape, key=mx.random.key(seed))` (float32) as the MLX Python wheel computes it, bit for bit.
///
/// MLX draws `uniform(nextafter(-1, 0), 1)` and returns `sqrt(2) * erfinv(u)`. The uniform draw is the same in MLX Swift, but MLX
/// Swift compiles its Metal kernels at run time, and there `erfinv`'s `metal::log` gives the fast logarithm, while the wheel's
/// precompiled kernel gives the precise one: about 0.07 % of the values differ in the last bit. Z-Image and Klein cast their noise to
/// bfloat16, which hides this; SeedVR2 keeps float32 noise. This kernel is MLX's `erfinv` (`backend/metal/kernels/erf.h`, the same
/// coefficients and fused multiply-adds) with `metal::precise::log`, which reproduces the wheel's values exactly.
enum SeedVR2Noise {
    static let erfinvKernel = MLXFast.metalKernel(
        name: "seedvr2_erfinv_precise_log", inputNames: ["u"], outputNames: ["out"],
        source: """
            uint i = thread_position_in_grid.x;
            float a = u[i];
            float t = metal::fma(a, 0.0f - a, 1.0f);
            t = metal::precise::log(t);
            float p;
            if (metal::abs(t) > 6.125f) {
              p = 3.03697567e-10f;
              p = metal::fma(p, t, 2.93243101e-8f);
              p = metal::fma(p, t, 1.22150334e-6f);
              p = metal::fma(p, t, 2.84108955e-5f);
              p = metal::fma(p, t, 3.93552968e-4f);
              p = metal::fma(p, t, 3.02698812e-3f);
              p = metal::fma(p, t, 4.83185798e-3f);
              p = metal::fma(p, t, -2.64646143e-1f);
              p = metal::fma(p, t, 8.40016484e-1f);
            } else {
              p = 5.43877832e-9f;
              p = metal::fma(p, t, 1.43285448e-7f);
              p = metal::fma(p, t, 1.22774793e-6f);
              p = metal::fma(p, t, 1.12963626e-7f);
              p = metal::fma(p, t, -5.61530760e-5f);
              p = metal::fma(p, t, -1.47697632e-4f);
              p = metal::fma(p, t, 2.31468678e-3f);
              p = metal::fma(p, t, 1.15392581e-2f);
              p = metal::fma(p, t, -2.32015476e-1f);
              p = metal::fma(p, t, 8.86226892e-1f);
            }
            out[i] = a * p;
            """)

    static func normal(_ shape: [Int], seed: UInt64) -> MLXArray {
        let count = shape.reduce(1, *)
        let uniform = MLXRandom.uniform(low: MLXArray(Float(-1).nextUp), high: MLXArray(Float(1)), shape, key: MLXRandom.key(seed))
        let erfinv = erfinvKernel(
            [uniform.flattened()], grid: (count, 1, 1), threadGroup: (min(256, count), 1, 1), outputShapes: [[count]],
            outputDTypes: [.float32])[0]
        return (MLXArray(Float(2.0.squareRoot())) * erfinv).reshaped(shape)
    }
}
