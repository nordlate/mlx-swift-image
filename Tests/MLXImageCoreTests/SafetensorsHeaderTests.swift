import XCTest
@testable import MLXImageCore

final class SafetensorsHeaderTests: XCTestCase {
    func testParsesConsistentHeader() throws {
        let json = TestSupport.safetensorsHeaderJSON(
            [("a", "BF16", [2, 3], 0..<12), ("b", "U32", [4], 12..<28)], metadata: ["quantization_level": "4"])
        let header = try SafetensorsHeader.parse(headerJSON: json, dataByteCount: 28)
        XCTAssertEqual(header.entries["a"], SafetensorsEntry(dtype: "BF16", shape: [2, 3], byteRange: 0..<12))
        XCTAssertEqual(header.entries["b"]?.byteRange, 12..<28)
        XCTAssertEqual(header.metadata["quantization_level"], "4")
    }

    func testRejectsShapeByteMismatch() {
        let json = TestSupport.safetensorsHeaderJSON([("a", "BF16", [2, 3], 0..<10)])
        XCTAssertThrowsError(try SafetensorsHeader.parse(headerJSON: json, dataByteCount: 10))
    }

    func testRejectsGapAndOverrun() {
        let gap = TestSupport.safetensorsHeaderJSON([("a", "U32", [1], 4..<8)])
        XCTAssertThrowsError(try SafetensorsHeader.parse(headerJSON: gap, dataByteCount: 8))
        let overrun = TestSupport.safetensorsHeaderJSON([("a", "U32", [2], 0..<8)])
        XCTAssertThrowsError(try SafetensorsHeader.parse(headerJSON: overrun, dataByteCount: 4))
        let short = TestSupport.safetensorsHeaderJSON([("a", "U32", [1], 0..<4)])
        XCTAssertThrowsError(try SafetensorsHeader.parse(headerJSON: short, dataByteCount: 8), "trailing bytes not covered by a tensor")
    }

    func testRejectsUnknownDtype() {
        let json = TestSupport.safetensorsHeaderJSON([("a", "F8_E4M3", [1], 0..<1)])
        XCTAssertThrowsError(try SafetensorsHeader.parse(headerJSON: json, dataByteCount: 1))
    }

    func testReadsHeaderFromFile() throws {
        let json = TestSupport.safetensorsHeaderJSON([("w", "F32", [2], 0..<8)])
        var file = withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) }
        file.append(json)
        file.append(Data(count: 8))
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("zimage-test-\(UUID().uuidString).safetensors")
        try file.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(try SafetensorsHeader.read(from: url).entries["w"]?.shape, [2])
        try file.dropLast(1).write(to: url)
        XCTAssertThrowsError(try SafetensorsHeader.read(from: url))
    }
}
