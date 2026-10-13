import Foundation

/// One tensor entry of a safetensors header: dtype name as stored (`BF16`, `U32`, `F32`, ...), shape and byte range.
public struct SafetensorsEntry: Equatable, Sendable {
    public let dtype: String
    public let shape: [Int]
    public let byteRange: Range<Int>
}

public struct SafetensorsHeader: Sendable {
    public let entries: [String: SafetensorsEntry]
    public let metadata: [String: String]
}

public enum SafetensorsError: Error, CustomStringConvertible {
    case truncated(String)
    case malformedHeader(String)
    case inconsistent(String)

    public var description: String {
        switch self {
        case .truncated(let m): "safetensors truncated: \(m)"
        case .malformedHeader(let m): "safetensors header malformed: \(m)"
        case .inconsistent(let m): "safetensors inconsistent: \(m)"
        }
    }
}

extension SafetensorsHeader {
    /// Bytes per element for the dtypes this model family stores.
    static let elementSize: [String: Int] = ["BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "I64": 8, "U8": 1]

    /// Reads and validates only the header of a safetensors file; tensor data is not read.
    public static func read(from url: URL) throws -> SafetensorsHeader {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let fileSize = Int(try handle.seekToEnd())
        try handle.seek(toOffset: 0)
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw SafetensorsError.truncated("\(url.lastPathComponent): no length prefix")
        }
        let length = prefix.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
        guard length > 0, length <= 64 << 20, 8 + length <= fileSize else {
            throw SafetensorsError.truncated("\(url.lastPathComponent): header length \(length) exceeds file size \(fileSize)")
        }
        guard let json = try handle.read(upToCount: length), json.count == length else {
            throw SafetensorsError.truncated("\(url.lastPathComponent): short header read")
        }
        return try parse(headerJSON: json, dataByteCount: fileSize - 8 - length, name: url.lastPathComponent)
    }

    /// Parses header JSON and checks that dtype/shape/byte ranges agree and tile the data section exactly.
    public static func parse(headerJSON: Data, dataByteCount: Int, name: String = "header") throws -> SafetensorsHeader {
        guard let root = try JSONSerialization.jsonObject(with: headerJSON) as? [String: Any] else {
            throw SafetensorsError.malformedHeader("\(name): not a JSON object")
        }
        var entries: [String: SafetensorsEntry] = [:]
        var metadata: [String: String] = [:]
        for (key, value) in root {
            if key == "__metadata__" {
                guard let dict = value as? [String: String] else {
                    throw SafetensorsError.malformedHeader("\(name): __metadata__ must map strings to strings")
                }
                metadata = dict
                continue
            }
            guard let dict = value as? [String: Any], let dtype = dict["dtype"] as? String,
                  let shape = dict["shape"] as? [Int], let offsets = dict["data_offsets"] as? [Int], offsets.count == 2
            else { throw SafetensorsError.malformedHeader("\(name): bad entry \(key)") }
            guard let size = elementSize[dtype] else {
                throw SafetensorsError.malformedHeader("\(name): \(key) has unsupported dtype \(dtype)")
            }
            guard offsets[0] >= 0, offsets[0] <= offsets[1], offsets[1] <= dataByteCount else {
                throw SafetensorsError.inconsistent("\(name): \(key) byte range \(offsets) outside data section of \(dataByteCount) bytes")
            }
            let expected = shape.reduce(1, *) * size
            guard offsets[1] - offsets[0] == expected else {
                throw SafetensorsError.inconsistent("\(name): \(key) shape \(shape) \(dtype) needs \(expected) bytes, range holds \(offsets[1] - offsets[0])")
            }
            entries[key] = SafetensorsEntry(dtype: dtype, shape: shape, byteRange: offsets[0]..<offsets[1])
        }
        var cursor = 0
        for (key, entry) in entries.sorted(by: { $0.value.byteRange.lowerBound < $1.value.byteRange.lowerBound }) {
            guard entry.byteRange.lowerBound == cursor else {
                throw SafetensorsError.inconsistent("\(name): \(key) leaves a gap or overlap at byte \(cursor)")
            }
            cursor = entry.byteRange.upperBound
        }
        guard cursor == dataByteCount else {
            throw SafetensorsError.inconsistent("\(name): tensors cover \(cursor) of \(dataByteCount) data bytes")
        }
        return SafetensorsHeader(entries: entries, metadata: metadata)
    }
}
