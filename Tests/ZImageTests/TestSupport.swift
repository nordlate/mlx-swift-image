import Foundation
import XCTest

enum TestSupport {
    /// The read-only q4 snapshot; tests that need it are skipped when it is not installed.
    static var modelDirectory: URL {
        let path = ProcessInfo.processInfo.environment["ZIMAGE_Q4_MODEL_DIR"]
            ?? NSHomeDirectory() + "/Library/Containers/com.nordlate.taltio/Data/Library/Application Support/LocalModels/models/z-image-turbo-q4"
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    static func requireModel() throws -> URL {
        let url = modelDirectory
        try XCTSkipUnless(FileManager.default.fileExists(atPath: url.appendingPathComponent("text_encoder/model.safetensors.index.json").path),
                          "q4 snapshot not installed at \(url.path)")
        return url
    }

    /// Reference artifacts written by scripts/run-parity.sh (ignored build output).
    static var referenceDirectory: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("build/reference")
    }

    /// Builds a safetensors file image: 8-byte little-endian header length, JSON header, data.
    static func safetensorsHeaderJSON(_ entries: [(String, String, [Int], Range<Int>)], metadata: [String: String]? = nil) -> Data {
        var root: [String: Any] = [:]
        for (name, dtype, shape, range) in entries {
            root[name] = ["dtype": dtype, "shape": shape, "data_offsets": [range.lowerBound, range.upperBound]] as [String: Any]
        }
        if let metadata { root["__metadata__"] = metadata }
        return try! JSONSerialization.data(withJSONObject: root)
    }
}
