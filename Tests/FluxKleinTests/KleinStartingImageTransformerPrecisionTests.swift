import Foundation
import MLX
import XCTest
@testable import KleinNative

final class KleinStartingImageTransformerPrecisionTests: XCTestCase {
    /// Independent scalar RMS calculation; values straddle BF16 input rounding so casting before normalization is detectable.
    private func normalized(_ values: [Float],width: Int,weight: [Float]) -> [Float] {
        stride(from: 0,to: values.count,by: width).flatMap { offset -> [Float] in
            let row = Array(values[offset..<offset+width])
            let meanSquare = row.reduce(0.0) { $0 + Double($1)*Double($1) } / Double(width)
            let divisor = sqrt(meanSquare + Double(KleinTransformerLayout.attentionNormEps))
            return row.enumerated().map { Float(Double($0.element)/divisor*Double(weight[$0.offset])) }
        }
    }

    func testFloat32SingleQKCastAfterNormalizationBeforeRoPEWithoutCastingValues() {
        Device.withDefaultDevice(.cpu) {
            let names = ["single_transformer_blocks.0.attn.norm_q","single_transformer_blocks.0.attn.norm_k"]
            let transformer = KleinTransformer(weights: .init(tensors: Dictionary(uniqueKeysWithValues: names.map { ($0+".weight",MLXArray.ones([4],dtype: .bfloat16)) })))
            let queries: [Float] = [1.003,1.004,0.001,0.001, -1.003,1.004,-0.001,0.001]
            let keys: [Float] = [1.004,1.003,0.001,-0.001, 1.004,-1.003,0.001,0.001]
            let projectedQ = MLXArray(queries).reshaped(1,1,2,4), projectedK = MLXArray(keys).reshaped(1,1,2,4)
            let projectedV = MLXArray([Float(1.003),-0.003,2.001,0.007,3.004,-1.001,0.009,4.003]).reshaped(1,1,2,4)
            let originalQ = KleinCompare.bytes(projectedQ), originalK = KleinCompare.bytes(projectedK), originalV = KleinCompare.bytes(projectedV)
            let q = transformer.qkNorm(projectedQ,names[0],outputDType: .bfloat16)
            let k = transformer.qkNorm(projectedK,names[1],outputDType: .bfloat16)
            for (actual,input,name) in [(q,queries,names[0]),(k,keys,names[1])] {
                let expected = MLXArray(normalized(input,width: 4,weight: [1,1,1,1])).reshaped(1,1,2,4).asType(.bfloat16)
                XCTAssertEqual(actual.dtype,.bfloat16)
                XCTAssertEqual(KleinCompare.bytes(actual),KleinCompare.bytes(expected))
                let prematurelyRounded = transformer.qkNorm(MLXArray(input).reshaped(1,1,2,4).asType(.bfloat16),name,outputDType: .bfloat16)
                XCTAssertNotEqual(KleinCompare.bytes(actual),KleinCompare.bytes(prematurelyRounded),"normalization must consume float32 projections")
            }
            let cos = MLXArray([Float(0.6),0.8,0.8,0.6]).reshaped(2,2), sin = MLXArray([Float(0.8),0.6,0.6,0.8]).reshaped(2,2)
            let (rotatedQ,rotatedK) = KleinTransformer.applyRope(q,k,cos: cos,sin: sin)
            XCTAssertEqual(rotatedQ.dtype,.bfloat16); XCTAssertEqual(rotatedK.dtype,.bfloat16)
            // Scalar float32 rotations start from the already rounded BF16 normalized values.
            for (actual,input) in [(rotatedQ,q),(rotatedK,k)] {
                let floats = input.asArray(Float.self), cs = cos.asArray(Float.self), ss = sin.asArray(Float.self)
                var expected: [Float] = []
                for pair in 0..<4 {
                    expected.append(floats[2*pair]*cs[pair] + (-floats[2*pair+1])*ss[pair])
                    expected.append(floats[2*pair+1]*cs[pair] + floats[2*pair]*ss[pair])
                }
                XCTAssertEqual(KleinCompare.bytes(actual),KleinCompare.bytes(MLXArray(expected).reshaped(1,1,2,4).asType(.bfloat16)))
            }
            XCTAssertEqual(projectedV.dtype,.float32)
            XCTAssertEqual(KleinCompare.bytes(projectedQ),originalQ); XCTAssertEqual(KleinCompare.bytes(projectedK),originalK)
            XCTAssertEqual(KleinCompare.bytes(projectedV),originalV)
        }
    }

    func testBF16SingleNormAndRoPEAreBitwiseIdenticalToPreviousDtypeRestoration() {
        Device.withDefaultDevice(.cpu) {
            let transformer = KleinTransformer(weights: .init(tensors: ["norm.weight":MLXArray([Float(0.75),1.25,1.5,2]).asType(.bfloat16)]))
            let projected = MLXArray([Float(1.003),-2.007,0.031,3.009, -4.007,0.019,2.031,1.009]).reshaped(1,1,2,4).asType(.bfloat16)
            let previous = transformer.qkNorm(projected,"norm") // Original projection-dtype restoration.
            let corrected = transformer.qkNorm(projected,"norm",outputDType: .bfloat16)
            XCTAssertEqual(previous.dtype,.bfloat16); XCTAssertEqual(corrected.dtype,.bfloat16)
            XCTAssertEqual(KleinCompare.bytes(corrected),KleinCompare.bytes(previous))
            let cos = MLXArray([Float(0.6),0.8,0.8,0.6]).reshaped(2,2), sin = MLXArray([Float(0.8),0.6,0.6,0.8]).reshaped(2,2)
            let oldRope = KleinTransformer.applyRope(previous,previous,cos: cos,sin: sin)
            let newRope = KleinTransformer.applyRope(corrected,corrected,cos: cos,sin: sin)
            XCTAssertEqual(KleinCompare.bytes(oldRope.0),KleinCompare.bytes(newRope.0))
            XCTAssertEqual(KleinCompare.bytes(oldRope.1),KleinCompare.bytes(newRope.1))
        }
    }

    func testJointProcessQKVRestoresActualFloat32ProjectionDtypeAndPreservesBF16Path() {
        Device.withDefaultDevice(.cpu) {
            // Tiny synthetic affine-q4 projections: 64 input features, the real 24x128 head layout. No model files or forward pass.
            let channels = KleinTransformerLayout.dim
            var tensors: [String: MLXArray] = [:]
            for name in ["q","k","v"] {
                tensors[name+".weight"] = MLXArray.full([channels,8],values: MLXArray(UInt32(0x11111111)),dtype: .uint32)
                tensors[name+".scales"] = MLXArray.full([channels,1],values: MLXArray(Float(0.25)),dtype: .bfloat16)
                tensors[name+".biases"] = MLXArray.full([channels,1],values: MLXArray(Float(-0.5)),dtype: .bfloat16)
            }
            for name in ["norm_q","norm_k"] { tensors[name+".weight"] = MLXArray.ones([128],dtype: .bfloat16) }
            let transformer = KleinTransformer(weights: .init(tensors: tensors))
            let values = MLXArray((0..<64).map { Float($0)*0.003+0.001 }).reshaped(1,1,64)
            for dtype in [DType.float32,.bfloat16] {
                let result = transformer.processQKV(values.asType(dtype),q: "q",k: "k",v: "v",normQ: "norm_q",normK: "norm_k")
                for array in [result.0,result.1,result.2] {
                    XCTAssertEqual(array.shape,[1,24,1,128]); XCTAssertEqual(array.dtype,dtype)
                }
                let expectedProjection = -Float(0.25) * values.asType(dtype).asArray(Float.self).reduce(0,+)
                XCTAssertEqual(result.2[0,0,0,0].item(Float.self),expectedProjection,accuracy: dtype == .float32 ? 1e-5 : 0.01)
                for array in [result.0,result.1] { XCTAssertEqual(array[0,0,0,0].item(Float.self),-1,accuracy: 1e-3) }
            }
        }
    }
}
