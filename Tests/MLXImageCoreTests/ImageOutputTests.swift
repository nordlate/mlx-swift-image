import MLX
import XCTest
@testable import MLXImageCore

final class ImageOutputTests: XCTestCase {
    func testPixelConversionClipsAndRoundsHalfToEven() {
        Device.withDefaultDevice(.cpu) {
            // decoded x -> clip(x / 2 + 0.5, 0, 1) * 255 rounded half to even.
            let values: [Float] = [-2, -1, 0, 0.5, 1, 3]
            let decoded = MLXArray(values + values + values).reshaped(1, 3, 1, 6).asType(.bfloat16)
            let pixels = ImageOutput.pixels(decoded: decoded)
            XCTAssertEqual(pixels.shape, [1, 6, 3])
            XCTAssertEqual(pixels.dtype, .uint8)
            // x = 0 -> 127.5 -> 128 (half to even); x = 0.5 -> 0.75 -> 191.25 -> 191; clipped ends 0 and 255.
            XCTAssertEqual(pixels[0, 0..., 0].asArray(UInt8.self), [0, 0, 128, 191, 255, 255])
        }
    }

    func testPNGRoundTripIsLossless() throws {
        try Device.withDefaultDevice(.cpu) {
            let (h, w) = (5, 7)
            let source = (0 ..< h * w * 3).map { UInt8(($0 * 37 + 11) % 256) }
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("zimage-native-test-\(UUID().uuidString).png")
            defer { try? FileManager.default.removeItem(at: url) }
            try ImageOutput.writePNG(pixels: MLXArray(source).reshaped(h, w, 3).asType(.uint8), to: url)
            let image = try ImageOutput.readPNG(at: url)
            XCTAssertEqual([image.width, image.height], [w, h])
            XCTAssertEqual(image.bytes, source)
            let info = try ImageOutput.pngInfo(at: url)
            XCTAssertEqual([info.width, info.height, info.bitDepth, info.colorType], [w, h, 8, 2], "8-bit RGB without alpha, like PIL's RGB PNG")
        }
    }

    func testWritePNGRejectsOtherLayouts() throws {
        try Device.withDefaultDevice(.cpu) {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("zimage-native-test-reject.png")
            XCTAssertThrowsError(try ImageOutput.writePNG(pixels: MLXArray.zeros([4, 4, 4]).asType(.uint8), to: url))
            XCTAssertThrowsError(try ImageOutput.writePNG(pixels: MLXArray.zeros([4, 4, 3]), to: url))
        }
    }
}
