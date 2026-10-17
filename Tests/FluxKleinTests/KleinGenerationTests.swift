import Darwin
import Foundation
import MLX
import XCTest
@testable import FluxKlein
@testable import MLXImageCore

private enum GenerationTestFiles {
    static func temp() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("klein-generation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url,withIntermediateDirectories: true)
        guard let path = url.path.withCString({ Darwin.realpath($0,nil) }) else { throw NSError(domain: NSPOSIXErrorDomain,code: Int(errno)) }
        defer { free(path) }
        return URL(fileURLWithPath: String(cString: path),isDirectory: true)
    }
}

final class KleinGenerationRequestTests: XCTestCase {
    func testSupportedControlsAndSizesValidateWithoutAllocating() throws {
        for n in 1...8 { try KleinGenerationRequest(prompt: "cat",width: 672,height: 896,seed: 0,steps: n).validate() }
        for n in [0,9,Int.max] { XCTAssertThrowsError(try KleinGenerationRequest(prompt: "cat",width: 64,height: 64,seed: 0,steps: n).validate()) }
        for n in [0,15,63,8193,Int.max] { XCTAssertThrowsError(try KleinGenerationRequest(prompt: "cat",width: n,height: 64,seed: 0,steps: 1).validate()) }
    }
    func testStartingImageTilingGuidanceAndSchedulerAreRejected() {
        let requests = [
            KleinGenerationRequest(prompt: "cat",width: 64,height: 64,seed: 0,steps: 1,startingImage: URL(fileURLWithPath: "/no-image")),
            KleinGenerationRequest(prompt: "cat",width: 64,height: 64,seed: 0,steps: 1,tiling: true),
            KleinGenerationRequest(prompt: "cat",width: 64,height: 64,seed: 0,steps: 1,guidance: 2),
            KleinGenerationRequest(prompt: "cat",width: 64,height: 64,seed: 0,steps: 1,guidance: .nan),
            KleinGenerationRequest(prompt: "cat",width: 64,height: 64,seed: 0,steps: 1,scheduler: "euler"),
        ]
        for r in requests { XCTAssertThrowsError(try r.validate()) }
    }
}

final class KleinGenerationPixelTests: XCTestCase {
    func testConversionUsesClipNHWCAndHalfEvenWithExactPNGReadback() throws {
        try Device.withDefaultDevice(.cpu) {
            let out = try GenerationTestFiles.temp(); defer { try? FileManager.default.removeItem(at: out) }
            let decoded = MLXArray([Float(-2),0,-1,1,2,0.5]).reshaped(1,3,1,2).asType(.bfloat16)
            let pixels = try KleinGenerationPixels.convert(decoded,width: 2,height: 1)
            XCTAssertEqual(pixels.asArray(UInt8.self),[0,0,255,128,255,191])
            let path = out.appendingPathComponent("image.png")
            let output = KleinGenerationOutput(latents: decoded,packed: decoded,decoded: decoded,pixels: pixels)
            try output.writePNG(to: path)
            XCTAssertEqual(try ImageOutput.readPNG(at: path).bytes,pixels.asArray(UInt8.self))
            let info = try ImageOutput.pngInfo(at: path)
            XCTAssertEqual(info.colorType,2); XCTAssertEqual(info.bitDepth,8); XCTAssertEqual(info.width,2)
            XCTAssertThrowsError(try output.writePNG(to: path))
            XCTAssertThrowsError(try ImageOutput.readPNG(at: out.appendingPathComponent("missing.png")))
        }
    }
    func testWrongShapeDTypeAndNonFiniteDecodedAreRejectedBeforePixels() throws {
        try Device.withDefaultDevice(.cpu) {
            for bad in [MLXArray.zeros([1,3,2,2]), MLXArray.zeros([1,4,2,2]).asType(.bfloat16),
                        MLXArray.full([1,3,2,2],values: MLXArray(Float.nan)).asType(.bfloat16),
                        MLXArray.full([1,3,2,2],values: MLXArray(Float.infinity)).asType(.bfloat16)] {
                XCTAssertThrowsError(try KleinGenerationPixels.convert(bad,width: 2,height: 2))
            }
            for (width,height) in [(0,2),(2,0),(0,0)] {
                let empty = MLXArray.zeros([1,3,height,width]).asType(.bfloat16)
                XCTAssertThrowsError(try KleinGenerationPixels.convert(empty,width: width,height: height))
            }
            let decoded = MLXArray.zeros([1,3,2,2]).asType(.bfloat16)
            for (width,height) in [(-1,2),(2,-1)] {
                XCTAssertThrowsError(try KleinGenerationPixels.convert(decoded,width: width,height: height))
            }
        }
    }
}
