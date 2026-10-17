import Darwin
import Foundation
import MLX
import XCTest
import MLXImageCore
@testable import FluxKlein

private enum StartingFiles {
    static let directory = KleinTestSupport.kleinDirectory.appendingPathComponent("starting-image")
    static var cases: URL { directory.appendingPathComponent("cases.json") }
    static var budgets: URL { directory.appendingPathComponent("budgets.json") }
    static func temp() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("klein-start-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url,withIntermediateDirectories: true)
        guard let path = url.path.withCString({ Darwin.realpath($0,nil) }) else { throw NSError(domain: NSPOSIXErrorDomain,code: Int(errno)) }
        defer { free(path) }
        return URL(fileURLWithPath: String(cString: path),isDirectory: true)
    }
    static func image(_ dir: URL,width: Int = 16,height: Int = 16) throws -> URL {
        let bytes = (0..<width*height*3).map { UInt8($0%256) }
        let png = try ImageOutput.encodePNG(pixels: MLXArray(bytes).reshaped(height,width,3))
        let url = dir.appendingPathComponent(UUID().uuidString+".png")
        try png.write(to: url,options: .withoutOverwriting)
        return url
    }
}
final class KleinStartingImageRequestTests: XCTestCase {
    func testAll28HostMappingsAndTinyStrengthHaveActualSkippedStepSemantics() throws {
        var count = 0
        for steps in 2...8 { for change in 1..<steps {
            let strength = try KleinStartingImageRequest.strength(steps: steps,change: change)
            let start = try KleinStartingImageRequest.start(steps: steps,strength: strength)
            XCTAssertEqual(start,steps-change); XCTAssertEqual(steps-start,change); count += 1
        } }
        XCTAssertEqual(count,28)
        XCTAssertEqual(try KleinStartingImageRequest.start(steps: 8,strength: Double.leastNonzeroMagnitude),1)
        XCTAssertEqual(try KleinStartingImageRequest.start(steps: 8,strength: 0.125.nextDown),1)
        XCTAssertEqual(try KleinStartingImageRequest.start(steps: 8,strength: 0.25.nextDown),1)
        XCTAssertEqual(try KleinStartingImageRequest.start(steps: 8,strength: 0.25),2)
        XCTAssertEqual(try KleinStartingImageRequest.start(steps: 8,strength: 1.0.nextDown),7)
        for strength in [0,-0.01,1,1.1,Double.nan,Double.infinity,-Double.infinity] {
            XCTAssertThrowsError(try KleinStartingImageRequest.start(steps: 4,strength: strength))
        }
        for steps in [0,1,9,Int.max] { XCTAssertThrowsError(try KleinStartingImageRequest.start(steps: steps,strength: 0.5)) }
        for change in [0,4,Int.max] { XCTAssertThrowsError(try KleinStartingImageRequest.strength(steps: 4,change: change)) }
    }
    func testReadablePreparedImageRequiredAndUnsupportedControlsRejected() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = try StartingFiles.temp(); defer { try? FileManager.default.removeItem(at: dir) }
            let file = try StartingFiles.image(dir)
            func request(image: URL? = nil,strength: Double = 0.5,guidance: Double = 1,scheduler: String = "flow_match_euler_discrete",tiling: Bool = false,width: Int = 16,prompt: String = "cat") -> KleinStartingImageRequest {
                .init(prompt: prompt,width: width,height: 16,seed: 0,steps: 2,image: image ?? file,strength: strength,guidance: guidance,scheduler: scheduler,tiling: tiling)
            }
            XCTAssertNoThrow(try request().validate())
            for bad in [request(image: dir.appendingPathComponent("missing")),request(image: dir),request(image: URL(string: "https://example.invalid/prepared.png")!),request(strength: .nan),request(strength: 0),request(guidance: 2),request(scheduler: "euler"),request(tiling: true),request(width: 32),request(prompt: "")] {
                XCTAssertThrowsError(try bad.validate())
            }
        }
    }
}
final class KleinPreparedPNGTests: XCTestCase {
    func testRawRGBOrderRowsAndFloat32NormalizationMatchExactSamplesAndDoNotMutatePNG() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = try StartingFiles.temp(); defer { try? FileManager.default.removeItem(at: dir) }
            let file = try StartingFiles.image(dir,width: 32,height: 16), before = try Data(contentsOf: file)
            let image = try KleinPreparedStartingImage.load(file,width: 32,height: 16)
            XCTAssertEqual(image.rgb,(0..<32*16*3).map { UInt8($0%256) })
            let normalized = image.normalized()
            XCTAssertEqual(normalized.dtype,.float32); XCTAssertEqual(normalized.shape,[1,3,16,32])
            let values = normalized.asArray(Float.self)
            for channel in 0..<3 { for pixel in 0..<512 {
                let expected = Float(2) * (Float(image.rgb[pixel*3+channel])/Float(255)) - Float(1)
                XCTAssertEqual(values[channel*512+pixel].bitPattern,expected.bitPattern)
            } }
            XCTAssertEqual(try Data(contentsOf: file),before)
            XCTAssertEqual(image.fileSHA256,KleinHashing.sha256(of: before))
            XCTAssertEqual(image.rgbSHA256,KleinHashing.sha256(of: Data(image.rgb)))
        }
    }
    func testRejectsNonPNGWrongDimensionsUnsupportedBitDepthColorInterlaceAnimationAndTruncation() throws {
        try Device.withDefaultDevice(.cpu) {
            let dir = try StartingFiles.temp(); defer { try? FileManager.default.removeItem(at: dir) }
            let file = try StartingFiles.image(dir), original = try Data(contentsOf: file)
            for mutation in 0..<9 {
                var data = original
                switch mutation {
                case 0: data[0] = 0
                case 1: data[24] = 16
                case 2: data[25] = 6 // RGBA
                case 3: data[25] = 0 // grayscale
                case 4: data[28] = 1 // Adam7 unsupported
                case 5: data = data.prefix(32)
                case 6: data.append(0)
                case 7: data.replaceSubrange(37..<41,with: Array("acTL".utf8))
                default: data[19] = 32 // wrong width in IHDR, rejected before decode
                }
                let bad = dir.appendingPathComponent("bad\(mutation).png"); try data.write(to: bad)
                XCTAssertThrowsError(try KleinPreparedStartingImage.load(bad,width: 16,height: 16),"mutation \(mutation)")
            }
            XCTAssertThrowsError(try KleinPreparedStartingImage.load(file,width: 32,height: 16))
        }
    }
}
final class KleinStartingImageMathTests: XCTestCase {
    func testDownsamplePadsOnlyBottomRightAndUsesStrideTwoInNHWC() {
        Device.withDefaultDevice(.cpu) {
            let input = MLXArray((1...24).map(Float.init)).reshaped(1,4,6,1)
            let weight = MLXArray.ones([1,3,3,1]), bias = MLXArray([Float(0)])
            let out = KleinVAE.encoderDownsample(input,weight: weight,bias: bias)
            XCTAssertEqual(out.shape,[1,2,3,1])
            XCTAssertEqual(out.asArray(Float.self),[72,90,69,102,114,82])
            // Distinct channels and rectangular axes catch layout swaps.
            let two = concatenated([input,input*10],axis: 3)
            let pick = MLXArray([Float(1),0,0,1]).reshaped(2,1,1,2)
            let point = KleinVAE.encoderDownsample(two,weight: pick,bias: MLXArray.zeros([2]))
            XCTAssertEqual(point.shape,[1,3,4,2])
            XCTAssertEqual(Array(point.asArray(Float.self).prefix(6)),[1,10,3,30,5,50])
        }
    }
    func testEvenCropThenCenterCropOrZeroPadBeforePatchAndPack() throws {
        try Device.withDefaultDevice(.cpu) {
            let one = MLXArray((0..<35).map(Float.init)).reshaped(1,1,5,7)
            let input = broadcast(one,to: [1,32,5,7]).asType(.bfloat16)
            let matched = try KleinStartingImageMath.spatial(input,height: 2,width: 4)
            XCTAssertEqual(matched.shape,[1,32,2,4])
            XCTAssertEqual(Array(matched.asArray(Float.self).prefix(8)),[8,9,10,11,15,16,17,18])
            let padded = try KleinStartingImageMath.spatial(input,height: 6,width: 8)
            let plane = Array(padded.asArray(Float.self).prefix(48))
            XCTAssertEqual(Array(plane.prefix(8)),[Float](repeating: 0,count: 8))
            XCTAssertEqual(Array(plane[8..<16]),[0,0,1,2,3,4,5,0])
            let patch = try KleinLatent.patchify(matched)
            XCTAssertEqual(Array(patch.asArray(Float.self).prefix(8)),[8,10,9,11,15,17,16,18])
            let packed = try KleinLatent.pack(patch)
            XCTAssertEqual(packed.shape,[1,2,128]); XCTAssertEqual(Array(packed.asArray(Float.self).prefix(4)),[8,9,15,16])
            XCTAssertThrowsError(try KleinStartingImageMath.spatial(MLXArray.zeros([1,32,1,2]),height: 2,width: 2))
            XCTAssertThrowsError(try KleinStartingImageMath.spatial(input,height: 3,width: 2))
        }
    }
    func testBNMathAndFloat32SigmaPromoteBlendWithoutChangingNoiseOrClean() throws {
        try Device.withDefaultDevice(.cpu) {
            let x = MLXArray.full([1,128,2,3],values: MLXArray(Float(5)),dtype: .bfloat16)
            let mean = MLXArray.full([128],values: MLXArray(Float(1)),dtype: .bfloat16)
            let variance = MLXArray.full([128],values: MLXArray(Float(4)),dtype: .bfloat16)
            let normalized = try KleinStartingImageMath.normalize(x,mean: mean,variance: variance)
            XCTAssertEqual(normalized.dtype,.bfloat16); XCTAssertEqual(normalized.asArray(Float.self),[Float](repeating: 2,count: 768))
            let clean = MLXArray([Float(1),3]).asType(.bfloat16), noise = MLXArray([Float(5),-1]).asType(.bfloat16)
            let cleanBytes = KleinTensorValues.bytes(clean), noiseBytes = KleinTensorValues.bytes(noise)
            let blend = try KleinStartingImageMath.blend(clean: clean,noise: noise,sigma: MLXArray(Float(0.25)))
            XCTAssertEqual(blend.dtype,.float32); XCTAssertEqual(blend.asArray(Float.self),[2,2])
            XCTAssertEqual(KleinTensorValues.bytes(clean),cleanBytes); XCTAssertEqual(KleinTensorValues.bytes(noise),noiseBytes)
            XCTAssertThrowsError(try KleinStartingImageMath.blend(clean: clean,noise: noise,sigma: MLXArray([Float(0.25)])))
            XCTAssertThrowsError(try KleinStartingImageMath.normalize(x.asType(.float32),mean: mean,variance: variance))
            XCTAssertThrowsError(try KleinStartingImageMath.normalize(x,mean: mean,variance: MLXArray.zeros([127])))
        }
    }
}

final class KleinStartingImageValidationTests: XCTestCase {
    func testPNGEXIFAllowsOnlyAbsentOrOneAndRejectsMalformedIFD() throws {
        let one: [UInt8] = [0x49,0x49,42,0,8,0,0,0,1,0,0x12,1,3,0,1,0,0,0,1,0,0,0,0,0,0,0]
        XCTAssertNoThrow(try KleinPreparedStartingImage.validateEXIF(one))
        var rotated = one; rotated[18] = 6
        XCTAssertThrowsError(try KleinPreparedStartingImage.validateEXIF(rotated))
        var badCount = one; badCount[8] = 2
        XCTAssertThrowsError(try KleinPreparedStartingImage.validateEXIF(badCount))
        var badOffset = one; badOffset[4] = 255
        XCTAssertThrowsError(try KleinPreparedStartingImage.validateEXIF(badOffset))
        XCTAssertThrowsError(try KleinPreparedStartingImage.validateEXIF([]))
        let absent: [UInt8] = [0x4d,0x4d,0,42,0,0,0,8,0,0,0,0,0,0]
        XCTAssertNoThrow(try KleinPreparedStartingImage.validateEXIF(absent))
    }
    func testEncoderRejectsWrongShapeTypeAndNonfiniteBeforeAccessingWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let vae = KleinVAE(weights: KleinVAEWeights(tensors: [:]))
            for bad in [MLXArray.zeros([1,3,16,16],dtype: .bfloat16),MLXArray.zeros([1,4,16,16]),MLXArray.zeros([1,3,16,15]),
                        MLXArray.zeros([3,16,16]),MLXArray.full([1,3,16,16],values: MLXArray(Float.nan)),
                        MLXArray.full([1,3,16,16],values: MLXArray(Float.infinity))] {
                XCTAssertThrowsError(try vae.encodePrepared(bad))
            }
            XCTAssertThrowsError(try vae.decodeStartingImagePacked(MLXArray.zeros([1,128,1,1],dtype: .bfloat16)))
        }
    }
}
