import XCTest
@testable import ZImage

final class TextEncoderWeightsTests: XCTestCase {
    func testRealSnapshotHeadersMatchLayout() throws {
        let model = try TestSupport.requireModel()
        let (inspection, shards) = try TextEncoderWeights.inspect(modelDirectory: model)
        XCTAssertEqual(inspection.tensorCount, 905)
        XCTAssertEqual(shards, ["0.safetensors", "1.safetensors"])
        XCTAssertEqual(inspection.bits, 4)
        XCTAssertEqual(inspection.groupSize, 64)
    }
}
