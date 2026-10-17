import Foundation
import MLX
import XCTest
@testable import FluxKlein

final class KleinTransformerMathTests: XCTestCase {
    func testPackUnpackPreservesChannelAndSpatialOrder() throws {
        try Device.withDefaultDevice(.cpu) {
            // Two channels over a 2x3 grid: token rows must interleave channels, not spatial positions.
            let x = MLXArray((0..<12).map { Float($0) }).reshaped(1, 2, 2, 3)
            let packed = try KleinLatent.pack(x)
            XCTAssertEqual(packed.shape, [1, 6, 2])
            XCTAssertEqual(packed.asArray(Float.self), [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])
            let unpacked = try KleinLatent.unpack(packed, height: 32, width: 48)
            XCTAssertEqual(unpacked.shape, x.shape)
            XCTAssertEqual(KleinTensorValues.bytes(unpacked), KleinTensorValues.bytes(x))
            XCTAssertThrowsError(try KleinLatent.unpack(packed, height: 48, width: 48))
            XCTAssertThrowsError(try KleinLatent.pack(MLXArray.zeros([2, 3])))
            XCTAssertThrowsError(try KleinLatent.unpack(x, height: 31, width: 48))
        }
    }

    func testPatchifyUsesAdjacentTwoByTwoPixelsAndSqueezesOnlySingletonFrame() throws {
        try Device.withDefaultDevice(.cpu) {
            let x = MLXArray((0..<16).map { Float($0) }).reshaped(1, 1, 4, 4)
            let expected: [Float] = [0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15]
            let patched = try KleinLatent.patchify(x)
            XCTAssertEqual(patched.shape, [1, 4, 2, 2])
            XCTAssertEqual(patched.asArray(Float.self), expected)
            XCTAssertEqual(try KleinLatent.patchify(x.reshaped(1, 1, 1, 4, 4)).asArray(Float.self), expected)
            XCTAssertThrowsError(try KleinLatent.patchify(MLXArray.zeros([1, 1, 3, 4])))
            XCTAssertThrowsError(try KleinLatent.patchify(MLXArray.zeros([1, 1, 2, 4, 4])))
        }
    }

    func testGridIDsAreRowMajorAndBatchIndependent() throws {
        try Device.withDefaultDevice(.cpu) {
            let ids = try KleinLatent.gridIDs(batch: 2, height: 2, width: 3, tCoord: 7)
            XCTAssertEqual(ids.shape, [2, 6, 4])
            XCTAssertEqual(ids.dtype, .int32)
            let expected: [Int32] = [7, 0, 0, 0, 7, 0, 1, 0, 7, 0, 2, 0, 7, 1, 0, 0, 7, 1, 1, 0, 7, 1, 2, 0]
            XCTAssertEqual(ids[0].asArray(Int32.self), expected)
            XCTAssertEqual(ids[1].asArray(Int32.self), expected)
            XCTAssertThrowsError(try KleinLatent.gridIDs(batch: 0, height: 2, width: 3))
            XCTAssertThrowsError(try KleinLatent.gridIDs(batch: 1, height: -1, width: 3))
            XCTAssertThrowsError(try KleinLatent.gridIDs(batch: 1, height: 2, width: 3, tCoord: Int.max))
            XCTAssertThrowsError(try KleinLatent.prepare(seed: 0, height: 32, width: 48, batch: 0))
            for size in [0, 15, 17, 8193] { XCTAssertThrowsError(try KleinLatent.grid(height: size, width: 32)) }
            let tiny = try KleinLatent.preparePacked(seed: 0, height: 32, width: 48)
            XCTAssertEqual(tiny.latents.shape, [1, 6, 128])
            XCTAssertEqual(tiny.latents.dtype, .bfloat16)
            XCTAssertEqual(tiny.ids.shape, [1, 6, 4])
        }
    }

    func testSchedulerShapesAllOfferedStepsAndRejectsInvalidIndices() throws {
        try Device.withDefaultDevice(.cpu) {
            for n in 1...8 {
                let scheduler = try KleinScheduler(steps: n, height: 32, width: 48)
                XCTAssertEqual(scheduler.imageSeqLen, 6)
                XCTAssertEqual(scheduler.sigmas.shape, [n + 1])
                XCTAssertEqual(scheduler.timesteps.shape, [n])
                XCTAssertEqual(scheduler.sigmas.dtype, .float32)
            }
            for n in [0, 9, -1] { XCTAssertThrowsError(try KleinScheduler(steps: n, height: 32, width: 48)) }
            XCTAssertThrowsError(try KleinScheduler(steps: 4, height: 33, width: 48))
            let scheduler = try KleinScheduler(steps: 1, height: 32, width: 48)
            let x = MLXArray([2, 4] as [Float]).asType(.bfloat16)
            let noise = MLXArray([1, 3] as [Float])
            let stepped = try scheduler.step(noise: noise, timestep: 0, latents: x)
            XCTAssertEqual(stepped.dtype, .bfloat16)
            XCTAssertEqual(stepped.asType(.float32).asArray(Float.self), [1, 1])
            for index in [-1, 1, Int.max] { XCTAssertThrowsError(try scheduler.step(noise: noise, timestep: index, latents: x)) }
            XCTAssertThrowsError(try scheduler.step(noise: noise.reshaped(1, 2), timestep: 0, latents: x))
            XCTAssertThrowsError(try KleinScheduler.step(noise: noise, latents: x, nextSigma: MLXArray([Float(0)]), sigma: MLXArray(Float(1))))
            XCTAssertThrowsError(try scheduler.step(noise: noise, timestep: 0, latents: x.asType(.int32)))
        }
    }

    func testRotaryUsesFourAxesAndInterleavedPairs() {
        Device.withDefaultDevice(.cpu) {
            let ids = MLXArray([0, 1, 2, 3] as [Int32]).reshaped(1, 4)
            let rope = KleinTransformer.rotaryTables(ids: ids)
            XCTAssertEqual(rope.cos.shape, [1, 64])
            XCTAssertEqual(rope.sin.dtype, .float32)
            for axis in 0..<4 {
                XCTAssertEqual(rope.cos[0, axis * 16].item(Float.self), Float(cos(Double(axis))), accuracy: 1e-6)
                XCTAssertEqual(rope.sin[0, axis * 16].item(Float.self), Float(sin(Double(axis))), accuracy: 1e-6)
            }
            // A quarter turn gives (-odd, even), unlike Qwen's half-vector rotation.
            let q = MLXArray([1, 2, 3, 4] as [Float]).reshaped(1, 1, 1, 4).asType(.bfloat16)
            let (qr, kr) = KleinTransformer.applyRope(q, q, cos: MLXArray.zeros([1, 2]), sin: MLXArray.ones([1, 2]))
            XCTAssertEqual(qr.dtype, .bfloat16)
            XCTAssertEqual(qr.asType(.float32).asArray(Float.self), [-2, 1, -4, 3])
            XCTAssertEqual(KleinTensorValues.bytes(qr), KleinTensorValues.bytes(kr))
            let frequencies = KleinTransformer.timestepFrequencies(MLXArray([Float(0)]))
            XCTAssertEqual(frequencies.shape, [1, 256])
            XCTAssertEqual(frequencies.dtype, .float32)
            XCTAssertEqual(frequencies.asArray(Float.self), [Float](repeating: 1, count: 128) + [Float](repeating: 0, count: 128))
        }
    }
}
