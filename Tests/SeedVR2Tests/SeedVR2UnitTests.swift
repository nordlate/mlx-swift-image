import CryptoKit
import Foundation
import MLX
import MLXImageCore
import XCTest
@testable import SeedVR2

/// Model-free checks: the stored layout, window partitioning, Pillow's bicubic resize, preprocessing sizes and padding, the tiling
/// ramps and blend, request validation and the bundled embedding.
final class SeedVR2UnitTests: XCTestCase {
    static func sha256(_ bytes: [UInt8]) -> String { SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined() }

    // MARK: Layout

    func testLayoutMatchesTheStoredFiles() {
        XCTAssertEqual(SeedVR2Layout.mlpHidden, 6912)
        XCTAssertEqual(SeedVR2Layout.ropeFrequencies, 21)
        XCTAssertEqual(SeedVR2Layout.transformerTensors().count, 635)
        XCTAssertEqual(SeedVR2Layout.vaeTensors().count, 250)
        // Every transformer linear layer but vid_in.proj (input 132), as MFLUX's predicate selects; the VAE's four attention projections
        // in the encoder and in the decoder.
        let quantized = SeedVR2Layout.transformerQuantizedWeights()
        XCTAssertEqual(quantized.count, 215)
        XCTAssertFalse(quantized.contains("vid_in.proj.weight"))
        XCTAssertTrue(quantized.isSuperset(of: ["txt_in.weight", "emb_in.proj_in.weight", "vid_out.proj.weight", "blocks.31.mlp.all.proj_out.weight"]))
        XCTAssertEqual(SeedVR2Layout.vaeQuantizedWeights().count, 8)
        XCTAssertEqual(SeedVR2Layout.vaeTensors()["encoder.down_blocks.0.downsamplers.0.conv.weight"]?.shape, [128, 128, 1, 3, 3])
        XCTAssertEqual(SeedVR2Layout.vaeTensors()["encoder.down_blocks.1.downsamplers.0.conv.weight"]?.shape, [256, 256, 3, 3, 3])
        XCTAssertEqual(SeedVR2Layout.vaeTensors()["decoder.up_blocks.1.upsamplers.0.upscale_conv.weight"]?.shape, [4096, 512, 1, 1, 1])
        XCTAssertEqual(SeedVR2Layout.vaeTensors()["decoder.up_blocks.2.upsamplers.0.upscale_conv.weight"]?.shape, [1024, 256, 1, 1, 1])
    }

    func testHeaderValidationRejectsMissingUnexpectedAndWrongTensors() throws {
        let expected = ["a.weight": SeedVR2Layout.f16(2, 64), "a.bias": SeedVR2Layout.f16(2)]
        func header(_ entries: [String: (String, [Int])]) throws -> SafetensorsHeader {
            var json: [String: Any] = [:]
            var offset = 0
            for (name, (dtype, shape)) in entries.sorted(by: { $0.key < $1.key }) {
                let bytes = shape.reduce(1, *) * 2
                json[name] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]]
                offset += bytes
            }
            return try SafetensorsHeader.parse(headerJSON: JSONSerialization.data(withJSONObject: json), dataByteCount: offset)
        }
        XCTAssertNoThrow(try SeedVR2Layout.validate(header: header(["a.weight": ("F16", [2, 64]), "a.bias": ("F16", [2])]), expected: expected, file: "f"))
        XCTAssertThrowsError(try SeedVR2Layout.validate(header: header(["a.weight": ("F16", [2, 64])]), expected: expected, file: "f"))
        XCTAssertThrowsError(try SeedVR2Layout.validate(
            header: header(["a.weight": ("F16", [2, 64]), "a.bias": ("F16", [2]), "b": ("F16", [1])]), expected: expected, file: "f"))
        XCTAssertThrowsError(try SeedVR2Layout.validate(header: header(["a.weight": ("BF16", [2, 64]), "a.bias": ("F16", [2])]), expected: expected, file: "f"))
        XCTAssertThrowsError(try SeedVR2Layout.validate(header: header(["a.weight": ("F16", [2, 128]), "a.bias": ("F16", [2])]), expected: expected, file: "f"))
    }

    func testMissingModelDirectoryIsRejectedBeforeLoading() {
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("seedvr2-empty-\(UUID().uuidString)")
        XCTAssertThrowsError(try SeedVR2Weights.validate(modelDirectory: empty)) { error in
            guard case SeedVR2Error.invalidWeights(let message) = error else { return XCTFail("\(error)") }
            XCTAssertTrue(message.contains(SeedVR2Layout.transformerFile))
        }
    }

    // MARK: Windows

    func testWindowsOfA64PatchGrid() {
        // 1024 x 1024 output: 64 x 64 patches, windows of 20 (rescaled to 60 x 60 / 3), the shifted ones offset by 10.
        let plain = SeedVR2Windows(t: 1, h: 64, w: 64, shift: false)
        XCTAssertEqual(plain.windows.count, 16)
        XCTAssertEqual(plain.windows.map(\.h), [0 ..< 20, 20 ..< 40, 40 ..< 60, 60 ..< 64].flatMap { [$0] }.repeated(4))
        let shifted = SeedVR2Windows(t: 1, h: 64, w: 64, shift: true)
        XCTAssertEqual(shifted.windows.count, 16)
        XCTAssertEqual(Array(shifted.windows.prefix(4)).map(\.h), [0 ..< 10, 10 ..< 30, 30 ..< 50, 50 ..< 64])
        for windows in [plain, shifted] {
            XCTAssertEqual(Set(windows.forward).count, 64 * 64)
            for (position, token) in windows.forward.enumerated() { XCTAssertEqual(Int(windows.reverse[Int(token)]), position) }
        }
    }

    func testWindowCountsForLargerAndNonSquareGrids() {
        XCTAssertEqual(SeedVR2Windows(t: 1, h: 128, w: 128, shift: false).windows.count, 49)
        XCTAssertEqual(SeedVR2Windows(t: 1, h: 256, w: 256, shift: false).windows.count, 169)
        // 1440 x 1080 (68 x 90 patches): scale sqrt(3600 / 6120), 52 x 69 rescaled, windows of 18 x 23.
        let grid = SeedVR2Windows(t: 1, h: 68, w: 90, shift: true)
        XCTAssertEqual(grid.windows.map(\.count).reduce(0, +), 68 * 90)
        XCTAssertEqual(grid.windows.first?.h, 0 ..< 9)
        XCTAssertEqual(grid.windows.first?.w, 0 ..< 11)
    }

    // MARK: Preprocessing

    /// Pillow 12.3 `Image.resize(..., BICUBIC)` digests of a 64 x 48 test pattern (`(i * 37) % 251`).
    func testBicubicResizeMatchesPillow() {
        let source = ImageOutput.RGBImage(width: 64, height: 48, bytes: (0 ..< 64 * 48 * 3).map { UInt8(($0 * 37) % 251) })
        let expected: [(Int, Int, String)] = [
            (150, 113, "09e5925b7c49ca5f0098aa5ed894e85a28a46a933904882221e641f4537c2f5c"),
            (17, 13, "5ee0cb790ed66fb9f25435aa8cdde2520de114fc1cf9eb72d44c5309e471e8d9"),
            (64, 100, "848b63cb4420f48f121a1b76f87572d2fd1c5d44d4d75a0bfe151bbfcf9b5d5e"),
            (200, 48, "0f3a22c2ff69832c579598b95fb69ba38dc06032238a46371ed2b8a0999f6425"),
        ]
        for (width, height, digest) in expected {
            let resized = PillowBicubic.resize(source, width: width, height: height)
            XCTAssertEqual(resized.width, width)
            XCTAssertEqual(resized.height, height)
            XCTAssertEqual(Self.sha256(resized.bytes), digest, "\(width) x \(height)")
        }
        XCTAssertEqual(PillowBicubic.resize(source, width: 64, height: 48), source)
    }

    func testOutputSizesFollowMflux() {
        XCTAssertTrue(SeedVR2Preprocessing.outputSize(width: 512, height: 512, resolution: .shortestSide(1024)) == (1024, 1024))
        XCTAssertTrue(SeedVR2Preprocessing.outputSize(width: 400, height: 300, resolution: .shortestSide(1080)) == (1440, 1080))
        XCTAssertTrue(SeedVR2Preprocessing.outputSize(width: 333, height: 517, resolution: .scale(2)) == (656, 1018))
        XCTAssertTrue(SeedVR2Preprocessing.outputSize(width: 777, height: 1001, resolution: .scale(1.5)) == (1152, 1484))
        XCTAssertTrue(SeedVR2Preprocessing.outputSize(width: 1024, height: 1024, resolution: .scale(4)) == (4096, 4096))
    }

    func testPreprocessingPadsWithBlackToMultiplesOf16() {
        let input = ImageOutput.RGBImage(width: 40, height: 30, bytes: [UInt8](repeating: 255, count: 40 * 30 * 3))
        let result = SeedVR2Preprocessing.run(input, resolution: .shortestSide(36), softness: 0.5)
        XCTAssertEqual(result.width, 48)
        XCTAssertEqual(result.height, 36)
        XCTAssertEqual(result.image.shape, [1, 3, 48, 48])
        let values = result.image.asArray(Float.self)
        XCTAssertEqual(values[(0 * 48 + 0) * 48 + 0], 1.0)  // white stays white through both bicubic passes
        XCTAssertEqual(values[(0 * 48 + 40) * 48 + 0], -1.0)  // padded row
    }

    // MARK: Tiling

    func testPinnedRampsAreTheRaisedCosine() {
        for n in [8, 64] {
            let pinned = SeedVR2VAETiler.cosRamp(n)
            XCTAssertEqual(pinned.count, n)
            XCTAssertEqual(pinned.first, 0)
            XCTAssertEqual(pinned.last, 1)
            for (i, value) in pinned.enumerated() {
                let t = Double(i) / Double(n - 1)
                XCTAssertEqual(Double(value), 0.5 - 0.5 * cos(t * Double.pi), accuracy: 1e-6)
            }
        }
    }

    func testBlendOfConstantTilesIsConstant() {
        var blend = SeedVR2VAETiler.Blend(channels: 2, height: 20, width: 20)
        let tile = MLXArray.full([1, 2, 12, 12], values: MLXArray(Float(0.25)))
        let ramp = SeedVR2VAETiler.cosRamp(8)
        for y in [0, 8] {
            for x in [0, 8] {
                blend.add(tile: tile, at: (y, x), size: (12, 12), overlap: (4, 4), ramp: ramp, lastRow: y == 8, lastColumn: x == 8)
            }
        }
        let result = blend.result().asArray(Float.self)
        XCTAssertEqual(result.count, 2 * 20 * 20)
        for value in result { XCTAssertEqual(value, 0.25, accuracy: 1e-6) }
    }

    // MARK: Requests and embedding

    func testRequestValidation() {
        let image = ImageOutput.RGBImage(width: 40, height: 30, bytes: [UInt8](repeating: 0, count: 40 * 30 * 3))
        XCTAssertNoThrow(try SeedVR2Request(image: image, resolution: .scale(2), softness: 0.5, seed: 1).validate())
        // MFLUX rounds a scale factor down to 16-pixel steps: 2 x 6 = 12 becomes 0.
        let tiny = ImageOutput.RGBImage(width: 8, height: 6, bytes: [UInt8](repeating: 0, count: 8 * 6 * 3))
        XCTAssertThrowsError(try SeedVR2Request(image: tiny, resolution: .scale(2), softness: 0.5, seed: 1).validate())
        XCTAssertNoThrow(try SeedVR2Request(image: image, resolution: .shortestSide(12), softness: 0, seed: 1).validate())
        XCTAssertThrowsError(try SeedVR2Request(image: image, resolution: .scale(2), softness: 1.5, seed: 1).validate())
        XCTAssertThrowsError(try SeedVR2Request(image: image, resolution: .scale(2), softness: .nan, seed: 1).validate())
        XCTAssertThrowsError(try SeedVR2Request(image: image, resolution: .scale(0), softness: 0, seed: 1).validate())
        XCTAssertThrowsError(try SeedVR2Request(image: image, resolution: .shortestSide(1), softness: 0, seed: 1).validate())
        XCTAssertThrowsError(try SeedVR2Request(image: image, resolution: .shortestSide(9000), softness: 0, seed: 1).validate())
        // Sizes that do not fit in Int must throw, not trap.
        XCTAssertThrowsError(try SeedVR2Request(image: image, resolution: .scale(1e20), softness: 0, seed: 1).validate())
        XCTAssertThrowsError(try SeedVR2Request(image: image, resolution: .shortestSide(Int.max), softness: 0, seed: 1).validate())
        XCTAssertThrowsError(try SeedVR2Request(image: image, resolution: .scale(.infinity), softness: 0, seed: 1).validate())
        // A 4096 shortest side (a 5460 x 4096 output) is within the limit.
        XCTAssertNoThrow(try SeedVR2Request(image: image, resolution: .shortestSide(4096), softness: 0, seed: 1).validate())
    }

    func testBundledPositiveEmbedding() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "pos_emb", withExtension: "safetensors"))
        XCTAssertEqual(Self.sha256([UInt8](try Data(contentsOf: url))), SeedVR2Pins.positiveEmbeddingSHA256)
        let embedding = try SeedVR2Upscaler.loadPositiveEmbedding()
        XCTAssertEqual(embedding.shape, [1, 58, 5120])
        XCTAssertEqual(embedding.dtype, .float16)
    }

    func testSchedulerStepIsTheDenoisedEstimate() {
        let latents = MLXArray([Float(0.5), -1.25, 2])
        let prediction = MLXArray([Float(0.25), 0.75, -1])
        XCTAssertEqual(SeedVR2Upscaler.step(prediction: prediction, latents: latents).asArray(Float.self), [0.25, -2.0, 3.0])
    }
}

private extension Array {
    func repeated(_ count: Int) -> [Element] { (0 ..< count).flatMap { _ in self } }
}
