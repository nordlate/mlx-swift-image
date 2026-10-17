import Foundation

enum TestSupport {
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
