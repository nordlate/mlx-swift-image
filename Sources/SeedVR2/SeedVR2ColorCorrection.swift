import Foundation
import MLX

/// Port of MFLUX 0.20's `SeedVR2Util.apply_color_correction` (`_lab_color_transfer_exact`, luminance weight 0.8), which runs in numpy
/// float32: the decoded image's high frequencies over the input's low frequencies (5-level wavelet), then the a and b channels
/// histogram-matched to the input in CIELAB and the lightness blended 80/20 with its matched version.
///
/// The wavelet runs as MLX element-wise ops on the GPU in numpy's order (only multiplies, adds and clips, which round identically);
/// the CIELAB conversions run on the CPU with the same float32 operations and C library functions as numpy; the histogram matching
/// ranks with a stable radix sort (numpy's stable argsort, ties in index order). numpy's 3 x 3 colour matrix product goes through
/// BLAS and may round differently in the last bit, which the pixel comparison absorbs.
public enum SeedVR2ColorCorrection {
    /// `content` (decoded) and `style` (the preprocessed input), both `[1, 3, H, W]` in `[-1, 1]`; returns `[1, 3, H, W]` float32.
    public static func apply(content: MLXArray, style: MLXArray, luminanceWeight: Float = 0.8) -> MLXArray {
        precondition(content.shape == style.shape && content.dim(0) == 1 && content.dim(1) == 3)
        let (height, width) = (content.dim(2), content.dim(3))
        let plane = height * width
        let c = content.asType(.float32), s = style.asType(.float32)
        let reconstructed = waveletReconstruction(content: c, style: s)
        eval(reconstructed, s)
        let contentLab = rgbToLab(reconstructed.asArray(Float.self), count: plane)
        let styleLab = rgbToLab(s.asArray(Float.self), count: plane)
        var lab = [Float](repeating: 0, count: 3 * plane)
        lab.withUnsafeMutableBufferPointer { out in
            contentLab.withUnsafeBufferPointer { cl in
                styleLab.withUnsafeBufferPointer { sl in
                    func channel(_ p: UnsafeBufferPointer<Float>, _ i: Int) -> UnsafeBufferPointer<Float> {
                        UnsafeBufferPointer(rebasing: p[i * plane ..< (i + 1) * plane])
                    }
                    for i in 1 ... 2 {
                        histogramMatch(source: channel(cl, i), reference: channel(sl, i), into: UnsafeMutableBufferPointer(rebasing: out[i * plane ..< (i + 1) * plane]))
                    }
                    let lightness = channel(cl, 0)
                    if luminanceWeight < 1.0 {
                        var matched = [Float](repeating: 0, count: plane)
                        matched.withUnsafeMutableBufferPointer { histogramMatch(source: lightness, reference: channel(sl, 0), into: $0) }
                        let other = Float(1.0 - Double(luminanceWeight))
                        for i in 0 ..< plane { out[i] = luminanceWeight * lightness[i] + other * matched[i] }
                    } else {
                        for i in 0 ..< plane { out[i] = lightness[i] }
                    }
                }
            }
        }
        return MLXArray(labToRGB(lab, count: plane), [1, 3, height, width])
    }

    // MARK: Wavelet (MLX)

    /// `_wavelet_blur`: a 3x3 binomial kernel with taps `radius` apart over an edge-padded image, accumulated tap by tap into zeros.
    /// The radius is capped at `min(h, w) // 8`.
    static func blur(_ image: MLXArray, radius: Int) -> MLXArray {
        let (height, width) = (image.dim(2), image.dim(3))
        let r = min(max(radius, 1), max(1, min(height, width) / 8))
        let kernel: [[Float]] = [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]
        let padded = MLX.padded(image, widths: [0, 0, [r, r], [r, r]] as [IntOrPair], mode: .edge)
        var out = MLXArray.zeros(image.shape, dtype: .float32)
        for (ky, dy) in [-1, 0, 1].enumerated() {
            let ys = r + dy * r
            for (kx, dx) in [-1, 0, 1].enumerated() {
                let xs = r + dx * r
                out = out + MLXArray(kernel[ky][kx]) * padded[0..., 0..., ys ..< ys + height, xs ..< xs + width]
            }
        }
        return out
    }

    /// `_wavelet_decomposition(levels=5)`: the summed detail of five blurs (radius 1, 2, 4, 8, 16) and the remaining low frequencies.
    static func decomposition(_ image: MLXArray) -> (high: MLXArray, low: MLXArray) {
        var high = MLXArray.zeros(image.shape, dtype: .float32)
        var current = image
        for level in 0 ..< 5 {
            let low = blur(current, radius: 1 << level)
            high = high + (current - low)
            current = low
        }
        return (high, current)
    }

    /// `_wavelet_reconstruction`: `clip(content_high + style_low, -1, 1)`.
    static func waveletReconstruction(content: MLXArray, style: MLXArray) -> MLXArray {
        let high = decomposition(content).high
        let low = decomposition(style).low
        return clip(high + low, min: Float(-1.0), max: Float(1.0))
    }

    // MARK: CIELAB (CPU)

    static let eps = 6.0 / 29.0
    static let epsF = Float(eps)
    static let eps3F = Float(eps * eps * eps)
    static let kappaF = Float(Foundation.pow(29.0 / 3.0, 3.0))

    /// `clip((x + 1) * 0.5, 0, 1)` then `_rgb_to_lab` (sRGB D65) over channel planes; returns the L, a and b planes.
    static func rgbToLab(_ planes: [Float], count: Int) -> [Float] {
        var lab = [Float](repeating: 0, count: 3 * count)
        let (m00, m01, m02): (Float, Float, Float) = (0.4124564, 0.3575761, 0.1804375)
        let (m10, m11, m12): (Float, Float, Float) = (0.2126729, 0.7151522, 0.0721750)
        let (m20, m21, m22): (Float, Float, Float) = (0.0193339, 0.1191920, 0.9503041)
        let (eps3, kappa) = (eps3F, kappaF)
        @inline(__always) func linear(_ v: Float) -> Float {
            let x = min(max((v + 1.0) * 0.5, 0), 1)
            return x > 0.04045 ? Foundation.powf((x + 0.055) / 1.055, 2.4) : x / 12.92
        }
        @inline(__always) func f(_ t: Float) -> Float { t > eps3 ? Foundation.cbrtf(t) : (kappa * t + 16.0) / 116.0 }
        planes.withUnsafeBufferPointer { p in
            lab.withUnsafeMutableBufferPointer { out in
                for i in 0 ..< count {
                    let r = linear(p[i]), g = linear(p[count + i]), b = linear(p[2 * count + i])
                    let x = (m00 * r + m01 * g + m02 * b) / 0.95047
                    let y = m10 * r + m11 * g + m12 * b
                    let z = (m20 * r + m21 * g + m22 * b) / 1.08883
                    let (fx, fy, fz) = (f(x), f(y), f(z))
                    out[i] = 116.0 * fy - 16.0
                    out[count + i] = 500.0 * (fx - fy)
                    out[2 * count + i] = 200.0 * (fy - fz)
                }
            }
        }
        return lab
    }

    /// `_lab_to_rgb`, `clip(0, 1)` and `* 2 - 1` over channel planes.
    static func labToRGB(_ lab: [Float], count: Int) -> [Float] {
        var rgb = [Float](repeating: 0, count: 3 * count)
        let (m00, m01, m02): (Float, Float, Float) = (3.2404542, -1.5371385, -0.4985314)
        let (m10, m11, m12): (Float, Float, Float) = (-0.9692660, 1.8760108, 0.0415560)
        let (m20, m21, m22): (Float, Float, Float) = (0.0556434, -0.2040259, 1.0572252)
        let (eps, kappa, gamma) = (epsF, kappaF, Float(1.0 / 2.4))
        @inline(__always) func finv(_ t: Float) -> Float { t > eps ? Foundation.powf(t, 3) : (116.0 * t - 16.0) / kappa }
        @inline(__always) func srgb(_ x: Float) -> Float {
            let v = x > 0.0031308 ? 1.055 * Foundation.powf(max(x, 0), gamma) - 0.055 : 12.92 * x
            return min(max(v, 0), 1) * 2.0 - 1.0
        }
        lab.withUnsafeBufferPointer { p in
            rgb.withUnsafeMutableBufferPointer { out in
                for i in 0 ..< count {
                    let fy = (p[i] + 16.0) / 116.0
                    let fx = p[count + i] / 500.0 + fy
                    let fz = fy - p[2 * count + i] / 200.0
                    let x = finv(fx) * 0.95047, y = finv(fy), z = finv(fz) * 1.08883
                    out[i] = srgb(m00 * x + m01 * y + m02 * z)
                    out[count + i] = srgb(m10 * x + m11 * y + m12 * z)
                    out[2 * count + i] = srgb(m20 * x + m21 * y + m22 * z)
                }
            }
        }
        return rgb
    }

    // MARK: Histogram matching

    /// `_hist_match`: each source value replaced by the reference value of the same rank.
    static func histogramMatch(source: UnsafeBufferPointer<Float>, reference: UnsafeBufferPointer<Float>, into out: UnsafeMutableBufferPointer<Float>) {
        let order = stableArgsort(source)
        let sortedReference = stableArgsort(reference)
        order.withUnsafeBufferPointer { o in
            sortedReference.withUnsafeBufferPointer { r in
                for rank in 0 ..< o.count { out[Int(o[rank])] = reference[Int(r[rank])] }
            }
        }
    }

    /// numpy's `argsort(kind="stable")` of float32 values: an LSD radix sort (two 16-bit passes, each stable) over keys that order
    /// floats as numbers. `-0.0` is keyed as `+0.0`, as numpy compares them equal.
    static func stableArgsort(_ values: UnsafeBufferPointer<Float>) -> [Int32] {
        let n = values.count
        var keys = [UInt32](repeating: 0, count: n)
        var order = [Int32](repeating: 0, count: n)
        var scratch = [Int32](repeating: 0, count: n)
        var counts = [Int](repeating: 0, count: 65537)
        keys.withUnsafeMutableBufferPointer { k in
            for i in 0 ..< n {
                let bits = (values[i] == 0 ? Float(0) : values[i]).bitPattern
                k[i] = bits & 0x8000_0000 != 0 ? ~bits : bits | 0x8000_0000
            }
        }
        order.withUnsafeMutableBufferPointer { o in for i in 0 ..< n { o[i] = Int32(i) } }
        keys.withUnsafeBufferPointer { k in
            counts.withUnsafeMutableBufferPointer { c in
                for shift: UInt32 in [0, 16] {
                    for i in 0 ..< c.count { c[i] = 0 }
                    order.withUnsafeMutableBufferPointer { o in
                        scratch.withUnsafeMutableBufferPointer { s in
                            for i in 0 ..< n { c[Int((k[Int(o[i])] >> shift) & 0xFFFF) + 1] += 1 }
                            for i in 1 ..< c.count { c[i] += c[i - 1] }
                            for i in 0 ..< n {
                                let bucket = Int((k[Int(o[i])] >> shift) & 0xFFFF)
                                s[c[bucket]] = o[i]
                                c[bucket] += 1
                            }
                        }
                    }
                    swap(&order, &scratch)
                }
            }
        }
        return order
    }
}
