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

    /// Reference artifacts written by the experiment's scripts/run-parity.sh (ignored build output); ZIMAGE_REFERENCE_DIR overrides the
    /// default `<repo>/Experiments/native-mlx-zimage/build/reference`.
    static var referenceDirectory: URL {
        if let path = ProcessInfo.processInfo.environment["ZIMAGE_REFERENCE_DIR"] {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Experiments/native-mlx-zimage/build/reference", isDirectory: true)
    }
}
