import Foundation
import MLX
import MLXImageCore

/// The output size of an upscale: the shortest side in pixels (`resolution=2048`), or a factor (`resolution=ScaleFactor(2)`) that MFLUX
/// rounds down to a multiple of 16 pixels on the shortest side.
public enum SeedVR2Resolution: Equatable, Sendable {
    case shortestSide(Int)
    case scale(Double)

    /// The target shortest side for an input whose shortest side is `side` (`ScaleFactor.get_scaled_value`).
    func target(forShortestSide side: Int) -> Int {
        switch self {
        case .shortestSide(let pixels): return pixels
        case .scale(let factor):
            let value = factor * Double(side)
            return Int(value - value.truncatingRemainder(dividingBy: 16))
        }
    }
}

/// Port of MFLUX 0.20's `SeedVR2Util.preprocess_image` from the decoded RGB pixels: the output size, `softness` (a bicubic downscale by
/// up to 8x and back), Pillow's bicubic resampling (`Image.resize(..., BICUBIC)`, reproduced exactly), black padding to multiples of
/// 16, and the `[-1, 1]` tensor. EXIF orientation is the caller's: the pixels are used as given.
public enum SeedVR2Preprocessing {
    public struct Result {
        /// `[1, 3, paddedHeight, paddedWidth]` float32 in `[-1, 1]`.
        public let image: MLXArray
        /// The output size (even, before padding).
        public let width: Int, height: Int
        /// The resized RGB pixels before padding (`width * height * 3`).
        public let resized: [UInt8]
    }

    /// The output size for an input of `width` x `height`: scaled so the shortest side is the target, truncated, then rounded down to
    /// even numbers.
    public static func outputSize(width: Int, height: Int, resolution: SeedVR2Resolution) -> (width: Int, height: Int) {
        let shortest = min(width, height)
        let scale = Double(resolution.target(forShortestSide: shortest)) / Double(shortest)
        let w = Int(Double(width) * scale), h = Int(Double(height) * scale)
        return ((w / 2) * 2, (h / 2) * 2)
    }

    public static func run(_ input: ImageOutput.RGBImage, resolution: SeedVR2Resolution, softness: Double) -> Result {
        let (width, height) = outputSize(width: input.width, height: input.height, resolution: resolution)
        let factor = 1.0 + max(0.0, min(1.0, softness)) * 7.0
        var resized: ImageOutput.RGBImage
        if factor > 1.0 {
            let down = PillowBicubic.resize(
                input, width: max(2, Int(Double(width) / factor)), height: max(2, Int(Double(height) / factor)))
            resized = PillowBicubic.resize(down, width: width, height: height)
        } else {
            resized = PillowBicubic.resize(input, width: width, height: height)
        }
        let paddedWidth = width + (16 - width % 16) % 16, paddedHeight = height + (16 - height % 16) % 16
        var padded = [UInt8](repeating: 0, count: paddedWidth * paddedHeight * 3)
        for y in 0 ..< height {
            let source = y * width * 3, target = y * paddedWidth * 3
            padded.replaceSubrange(target ..< target + width * 3, with: resized.bytes[source ..< source + width * 3])
        }
        // `mx.array(np.array(img)).astype(float32) / 255.0`, clip, `* 2 - 1`, HWC to CHW, batch axis: MLX ops, as MFLUX runs them.
        var image = MLXArray(padded, [paddedHeight, paddedWidth, 3]).asType(.float32) / Float(255.0)
        image = clip(image, min: Float(0.0), max: Float(1.0))
        image = image * Float(2.0) - Float(1.0)
        image = image.transposed(2, 0, 1).expandedDimensions(axis: 0)
        return Result(image: image, width: width, height: height, resized: resized.bytes)
    }
}

/// Pillow's `ImagingResample` for 8-bit RGB with the bicubic filter (`a = -0.5`), as Pillow 12 runs it: separable, horizontal pass
/// then vertical, double-precision coefficients normalized per output pixel and converted to 22-bit fixed point, integer
/// accumulation with rounding, clamped to 0...255. Only the passes whose size changes run, and only the source rows the vertical
/// pass needs go through the horizontal one; the result is the same.
public enum PillowBicubic {
    static let precisionBits = 32 - 8 - 2

    static func filter(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
        if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0.0
    }

    /// `precompute_coeffs` and `normalize_coeffs_8bpc` for an `inSize` axis resampled to `outSize` over the whole input.
    static func coefficients(inSize: Int, outSize: Int) -> (kernelSize: Int, bounds: [(Int, Int)], weights: [Int32]) {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1.0)
        let support = 2.0 * filterScale
        let kernelSize = Int(support.rounded(.up)) * 2 + 1
        var bounds: [(Int, Int)] = []
        var weights = [Int32](repeating: 0, count: outSize * kernelSize)
        var k = [Double](repeating: 0, count: kernelSize)
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            let ss = 1.0 / filterScale
            let xmin = max(Int(center - support + 0.5), 0)
            let xmax = min(Int(center + support + 0.5), inSize) - xmin
            var ww = 0.0
            for x in 0 ..< kernelSize { k[x] = 0 }
            for x in 0 ..< xmax {
                let w = filter((Double(x + xmin) - center + 0.5) * ss)
                k[x] = w
                ww += w
            }
            for x in 0 ..< xmax where ww != 0.0 { k[x] /= ww }
            for x in 0 ..< kernelSize {
                let v = k[x] * Double(1 << precisionBits)
                weights[xx * kernelSize + x] = Int32(k[x] < 0 ? Int(-0.5 + v) : Int(0.5 + v))
            }
            bounds.append((xmin, xmax))
        }
        return (kernelSize, bounds, weights)
    }

    static func clip8(_ value: Int) -> UInt8 {
        if value >= (1 << precisionBits) << 8 { return 255 }
        if value <= 0 { return 0 }
        return UInt8(value >> precisionBits)
    }

    /// `Image.resize((width, height), Image.Resampling.BICUBIC)` of an RGB image.
    public static func resize(_ image: ImageOutput.RGBImage, width: Int, height: Int) -> ImageOutput.RGBImage {
        if width == image.width && height == image.height { return image }
        let horizontal = coefficients(inSize: image.width, outSize: width)
        var vertical = coefficients(inSize: image.height, outSize: height)
        let needHorizontal = width != image.width, needVertical = height != image.height
        let rowFirst = vertical.bounds[0].0
        let rowLast = vertical.bounds[height - 1].0 + vertical.bounds[height - 1].1

        var source = image.bytes
        var (sourceWidth, sourceHeight) = (image.width, image.height)
        if needHorizontal {
            // The horizontal pass only covers the rows the vertical pass reads; shift the vertical bounds to match.
            if needVertical { vertical.bounds = vertical.bounds.map { ($0.0 - rowFirst, $0.1) } }
            let (rowStart, rows) = needVertical ? (rowFirst, rowLast - rowFirst) : (0, image.height)
            var out = [UInt8](repeating: 0, count: width * rows * 3)
            source.withUnsafeBufferPointer { src in
                horizontal.weights.withUnsafeBufferPointer { kk in
                    for yy in 0 ..< rows {
                        let row = (yy + rowStart) * image.width * 3
                        for xx in 0 ..< width {
                            let (xmin, xmax) = horizontal.bounds[xx]
                            let k = xx * horizontal.kernelSize
                            var (s0, s1, s2) = (1 << (precisionBits - 1), 1 << (precisionBits - 1), 1 << (precisionBits - 1))
                            for x in 0 ..< xmax {
                                let p = row + (x + xmin) * 3, w = Int(kk[k + x])
                                s0 += Int(src[p]) * w
                                s1 += Int(src[p + 1]) * w
                                s2 += Int(src[p + 2]) * w
                            }
                            let o = (yy * width + xx) * 3
                            out[o] = clip8(s0)
                            out[o + 1] = clip8(s1)
                            out[o + 2] = clip8(s2)
                        }
                    }
                }
            }
            source = out
            (sourceWidth, sourceHeight) = (width, rows)
        }
        guard needVertical else { return ImageOutput.RGBImage(width: sourceWidth, height: sourceHeight, bytes: source) }
        var out = [UInt8](repeating: 0, count: sourceWidth * height * 3)
        source.withUnsafeBufferPointer { src in
            vertical.weights.withUnsafeBufferPointer { kk in
                for yy in 0 ..< height {
                    let (ymin, ymax) = vertical.bounds[yy]
                    let k = yy * vertical.kernelSize
                    for xx in 0 ..< sourceWidth {
                        var (s0, s1, s2) = (1 << (precisionBits - 1), 1 << (precisionBits - 1), 1 << (precisionBits - 1))
                        for y in 0 ..< ymax {
                            let p = ((y + ymin) * sourceWidth + xx) * 3, w = Int(kk[k + y])
                            s0 += Int(src[p]) * w
                            s1 += Int(src[p + 1]) * w
                            s2 += Int(src[p + 2]) * w
                        }
                        let o = (yy * sourceWidth + xx) * 3
                        out[o] = clip8(s0)
                        out[o + 1] = clip8(s1)
                        out[o + 2] = clip8(s2)
                    }
                }
            }
        }
        return ImageOutput.RGBImage(width: sourceWidth, height: height, bytes: out)
    }
}
