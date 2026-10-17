import Foundation
import MLX
import XCTest
@testable import FluxKlein
@testable import MLXImageCore

final class KleinVAEValidationTests: XCTestCase {
    let metadata = ["quantization_level":"4","mflux_version":"0.19.0"]
    func parts() -> (map: [String: String],entries: [String: SafetensorsEntry]) {
        let entries = KleinVAELayout.expectedTensors().mapValues { SafetensorsEntry(dtype: $0.dtype,shape: $0.shape,byteRange: 0..<0) }
        return (entries.mapValues { _ in "0.safetensors" },entries)
    }
    func validate(_ map: [String: String],_ entries: [String: SafetensorsEntry],index: [String: String]? = nil,shard: [String: String]? = nil) throws {
        try KleinVAEWeights.validate(weightMap: map,indexMetadata: index ?? metadata,
                                     shardHeaders: ["0.safetensors":SafetensorsHeader(entries: entries,metadata: shard ?? metadata)])
    }
    func testArchitectureHasFullEncoderAndDecoderWithStoredMLXConvolutions() throws {
        let l = KleinVAELayout.expectedTensors(), p = parts()
        XCTAssertEqual(l.count,266)
        XCTAssertEqual(l.filter { $0.key.hasPrefix("encoder.") }.count,114)
        XCTAssertEqual(l.filter { $0.key.hasPrefix("decoder.") }.count,146)
        XCTAssertEqual(KleinVAELayout.matrices().count,8)
        XCTAssertEqual(l["encoder.conv_out.weight"]?.shape,[64,3,3,512])
        XCTAssertEqual(l["decoder.conv_in.weight"]?.shape,[512,3,3,32])
        XCTAssertEqual(l["decoder.up_blocks.2.resnets.0.conv_shortcut.weight"]?.shape,[256,1,1,512])
        XCTAssertEqual(l["post_quant_conv.weight"]?.shape,[32,1,1,32])
        XCTAssertEqual(l["quant_conv.weight"]?.shape,[64,1,1,64])
        XCTAssertEqual(l["decoder.mid_block.attentions.0.to_out.weight"]?.shape,[512,64])
        XCTAssertEqual(l["encoder.mid_block.attentions.0.to_q.scales"]?.shape,[512,8])
        XCTAssertEqual(l["decoder.mid_block.attentions.0.to_out.bias"]?.dtype,"BF16")
        XCTAssertEqual(l["bn.running_var"]?.shape,[128])
        XCTAssertNil(l["decoder.mid_block.attentions.0.to_out.0.weight"])
        XCTAssertNil(l["decoder.conv_in.conv.weight"])
        try validate(p.map,p.entries)
    }
    func testWrongPackingGroupsConvLayoutAndDTypesAreRejected() {
        for (name,dtype,shape) in [
            ("decoder.mid_block.attentions.0.to_q.weight","U32",[512,128]),
            ("encoder.mid_block.attentions.0.to_k.scales","BF16",[512,16]),
            ("decoder.conv_in.weight","BF16",[512,32,3,3]),
            ("decoder.conv_norm_out.weight","F32",[128]),
            ("bn.running_mean","F32",[128]),
            ("quant_conv.weight","BF16",[32,1,1,32]),
        ] {
            var p = parts(); p.entries[name] = SafetensorsEntry(dtype: dtype,shape: shape,byteRange: 0..<0)
            XCTAssertThrowsError(try validate(p.map,p.entries),name)
        }
    }
    func testMissingExtraShardAndMetadataAreRejected() {
        var p = parts(); p.map["encoder.conv_in.weight"] = nil
        XCTAssertThrowsError(try validate(p.map,p.entries))
        p = parts(); p.map["unused"] = "0.safetensors"; p.entries["unused"] = p.entries["bn.running_mean"]
        XCTAssertThrowsError(try validate(p.map,p.entries))
        p = parts(); p.map["bn.running_var"] = "1.safetensors"
        XCTAssertThrowsError(try validate(p.map,p.entries))
        for (key,value) in [("quantization_level","8"),("mflux_version","0.20.0")] {
            var bad = metadata; bad[key] = value; let p = parts()
            XCTAssertThrowsError(try validate(p.map,p.entries,index: bad))
            XCTAssertThrowsError(try validate(p.map,p.entries,shard: bad))
        }
    }
    func testActualInstalled266TensorHeaderWithoutEvaluatingEncoder() throws {
        let model = try KleinTestSupport.requireModel()
        let result = try KleinVAEWeights.readAndValidate(modelDirectory: model)
        XCTAssertEqual(result.weightCount,266); XCTAssertEqual(result.shards,["0.safetensors"])
        XCTAssertEqual(KleinGenerationPins.modelFiles.count,11); XCTAssertEqual(KleinGenerationPins.runtimeModules.count,53)
    }
}

final class KleinVAEMathTests: XCTestCase {
    func testUnpatchSpatialChannelOrderUsesEachTwoByTwoPatch() throws {
        try Device.withDefaultDevice(.cpu) {
            let values = Array(Int32(0)..<Int32(48))
            let out = try KleinVAE.unpatchify(MLXArray(values).reshaped(1,8,2,3))
            var expected: [Int32] = []
            // Independent coordinate address calculation, without using reshape/transpose or the inverse native patchify.
            for channel in 0..<2 { for y in 0..<4 { for x in 0..<6 {
                let sourceChannel = channel*4 + (y%2)*2 + x%2
                expected.append(values[(sourceChannel*2 + y/2)*3 + x/2])
            } } }
            XCTAssertEqual(out.shape,[1,2,4,6]); XCTAssertEqual(out.asArray(Int32.self),expected)
            XCTAssertThrowsError(try KleinVAE.unpatchify(MLXArray.zeros([1,3,2,2])))
        }
    }
    func testBNUses128IndependentChannelStatisticsBeforeUnpatch() throws {
        try Device.withDefaultDevice(.cpu) {
            let mean = MLXArray((0..<128).map(Float.init)).asType(.bfloat16)
            let variance = MLXArray.full([128],values: MLXArray(Float(4))).asType(.bfloat16)
            let packed = broadcast(MLXArray([Float(0),1]).reshaped(1,1,1,2),to: [1,128,1,2]).asType(.bfloat16)
            let bn = try KleinVAE.denormalize(packed,mean: mean,variance: variance)
            let expected = (0..<128).flatMap { [Float($0),Float($0+2)] }
            XCTAssertEqual(bn.dtype,.bfloat16); XCTAssertEqual(bn.asType(.float32).asArray(Float.self),expected)
            let plain = try KleinVAE.unpatchify(bn)
            XCTAssertEqual(Array(plain.asType(.float32).asArray(Float.self).prefix(8)),[0,1,2,3,2,3,4,5])
            XCTAssertThrowsError(try KleinVAE.denormalize(packed.asType(.float32),mean: mean,variance: variance))
            XCTAssertThrowsError(try KleinVAE.denormalize(packed,mean: mean[0..<32],variance: variance))
        }
    }
    func testKleinNormAffineReturnsBF16AndAttentionScaleUsesFloat32() {
        Device.withDefaultDevice(.cpu) {
            let weights = KleinVAEWeights(tensors: ["n.weight":MLXArray.full([32],values: MLXArray(Float(2))).asType(.bfloat16),
                                                   "n.bias":MLXArray.full([32],values: MLXArray(Float(3))).asType(.bfloat16)])
            let input = MLXArray((0..<64).map { Float($0%32) + ($0 < 32 ? 0 : 2) }).reshaped(1,1,2,32).asType(.bfloat16)
            let out = KleinVAE(weights: weights).norm(input,"n")
            XCTAssertEqual(out.dtype,.bfloat16)
            XCTAssertEqual(out.asType(.float32).asArray(Float.self),[Float](repeating: 1,count: 32)+[Float](repeating: 5,count: 32))
            let scale = KleinVAE.attentionScale(channels: 512)
            XCTAssertEqual(scale,1/Float(512).squareRoot(),accuracy: 1e-8)
            let bf16Scale = (MLXArray(Float(1)).asType(.bfloat16) / MLX.sqrt(MLXArray(Float(512)).asType(.bfloat16))).asType(.float32).item(Float.self)
            XCTAssertNotEqual(scale,bf16Scale)
        }
    }
}
