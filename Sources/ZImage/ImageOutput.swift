import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

/// mflux's decoded-tensor to image conversion and the PNG the production pipeline produces: an 8-bit RGB image (PIL mode
/// `RGB`, three interleaved channels, no alpha).
public enum ZImageImage {
    /// `ImageUtil._denormalize`, `_to_numpy` and `_numpy_to_pil`'s array step for a decoded `[1, 3, H, W]` tensor:
    /// `clip(x / 2 + 0.5, 0, 1)` in the decoder's dtype, to NHWC float32, `round(x * 255)` (half to even, as numpy rounds)
    /// to uint8. Returns `[H, W, 3]`.
    public static func pixels(decoded: MLXArray) -> MLXArray {
        let normalized = clip(decoded / 2 + 0.5, min: 0, max: 1)
        let nhwc = normalized.transposed(0, 2, 3, 1).asType(.float32)
        return MLX.round(nhwc * 255).asType(.uint8)[0]
    }

    public struct RGBImage: Equatable, Sendable {
        public let width: Int
        public let height: Int
        /// Row-major interleaved RGB, `width * height * 3` bytes.
        public let bytes: [UInt8]
    }

    public enum ImageError: Error, CustomStringConvertible {
        case invalid(String)
        public var description: String { if case .invalid(let m) = self { m } else { "" } }
    }

    /// Encodes `[H, W, 3]` uint8 pixels as an 8-bit RGB PNG. The pixel bytes are stored as given (no colour conversion).
    public static func writePNG(pixels: MLXArray, to url: URL) throws {
        guard pixels.ndim == 3, pixels.dim(2) == 3, pixels.dtype == .uint8 else {
            throw ImageError.invalid("PNG needs [H, W, 3] uint8 pixels, got \(pixels.dtype) \(pixels.shape)")
        }
        let (height, width) = (pixels.dim(0), pixels.dim(1))
        let bytes = pixels.asArray(UInt8.self)
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(
                  width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw ImageError.invalid("cannot create a PNG encoder for \(url.lastPathComponent)") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw ImageError.invalid("PNG encoding failed for \(url.lastPathComponent)") }
    }

    /// Decodes a PNG to its stored 8-bit RGB samples (no colour management: the raw decoded bytes of the image's data provider).
    /// Anything that is not an 8-bit RGB(X) image without alpha is rejected.
    public static func readPNG(at url: URL) throws -> RGBImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) == 1,
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil), let data = image.dataProvider?.data as Data?
        else { throw ImageError.invalid("cannot decode \(url.lastPathComponent)") }
        let alpha = image.alphaInfo
        let bytesPerPixel = image.bitsPerPixel / 8
        guard image.bitsPerComponent == 8, bytesPerPixel == 3 || bytesPerPixel == 4,
              alpha == .none || alpha == .noneSkipLast, image.colorSpace?.model == .rgb
        else {
            throw ImageError.invalid(
                "\(url.lastPathComponent): expected 8-bit RGB without alpha, got \(image.bitsPerComponent)-bit, \(image.bitsPerPixel) bpp, alpha \(alpha.rawValue)")
        }
        var bytes = [UInt8](repeating: 0, count: image.width * image.height * 3)
        data.withUnsafeBytes { raw in
            for y in 0 ..< image.height {
                for x in 0 ..< image.width {
                    let source = y * image.bytesPerRow + x * bytesPerPixel
                    let target = (y * image.width + x) * 3
                    bytes[target] = raw[source]
                    bytes[target + 1] = raw[source + 1]
                    bytes[target + 2] = raw[source + 2]
                }
            }
        }
        return RGBImage(width: image.width, height: image.height, bytes: bytes)
    }

    /// Structural facts about a PNG file, reported separately from pixel equality (the plan treats them as encoding differences).
    public struct PNGInfo: Codable, Equatable, Sendable {
        public let width: Int
        public let height: Int
        public let bitDepth: Int
        public let colorType: Int
        public let bytes: Int
        public let sha256: String
    }

    /// Reads the IHDR of a PNG file (width, height, bit depth, colour type) and digests the whole file.
    public static func pngInfo(at url: URL) throws -> PNGInfo {
        let data = try Data(contentsOf: url)
        let signature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        guard data.count >= 33, Array(data.prefix(8)) == signature, Array(data[12 ..< 16]) == Array("IHDR".utf8) else {
            throw ImageError.invalid("\(url.lastPathComponent) is not a PNG")
        }
        func be32(_ offset: Int) -> Int { data[offset ..< offset + 4].reduce(0) { $0 << 8 | Int($1) } }
        return PNGInfo(
            width: be32(16), height: be32(20), bitDepth: Int(data[24]), colorType: Int(data[25]), bytes: data.count,
            sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
    }
}
