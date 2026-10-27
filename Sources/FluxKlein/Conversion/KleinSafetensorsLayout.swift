import Cmlx
import Foundation
import MLX
import MLXImageCore

/// Where each tensor of one output shard goes, and the exact header bytes, so the shard can be written tensor by tensor with
/// positioned writes and still come out byte-identical to `mx.save_safetensors` (MLX 0.32's `save_safetensors`).
///
/// MLX writes the header as compact JSON with sorted keys (nlohmann::json) and the tensor data in the iteration order of the
/// `std::unordered_map` the arrays are passed in. That order depends only on the keys and the order they were inserted (the order of
/// the Python dict, which is mflux's parameter order), so it is reproduced by inserting the same keys, in the same order, into the
/// same C++ map type through mlx-c and saving placeholder scalars: `dataOrder(keys:scratch:)`.
struct KleinShardLayout: Equatable {
    struct Entry: Equatable {
        let name: String
        let dtype: String
        let shape: [Int]
        /// Offset within the data section.
        let dataOffset: Int
        let byteCount: Int
    }

    /// Entries keyed by name.
    let entries: [String: Entry]
    /// The 8-byte little-endian header length followed by the header JSON.
    let headerBytes: [UInt8]
    let fileByteCount: Int

    var dataStart: Int { headerBytes.count }

    func fileOffset(of name: String) -> Int { dataStart + entries[name]!.dataOffset }

    /// Lays out `tensors` (in insertion order) as MLX would save them with `metadata`.
    init(tensors: [KleinPlannedTensor], metadata: [String: String], scratch: URL) throws {
        let order = try Self.dataOrder(keys: tensors.map(\.name), scratch: scratch)
        let byName = Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, $0) })
        var offset = 0
        var entries: [String: Entry] = [:]
        for name in order {
            let t = byName[name]!
            entries[name] = Entry(name: name, dtype: t.dtype, shape: t.shape, dataOffset: offset, byteCount: t.byteCount)
            offset += t.byteCount
        }
        self.entries = entries
        let json = Self.headerJSON(entries: Array(entries.values), metadata: metadata)
        var header = withUnsafeBytes(of: UInt64(json.count).littleEndian, Array.init)
        header += json
        headerBytes = header
        fileByteCount = header.count + offset
    }

    /// nlohmann::json's `dump()` of MLX's header object: keys in byte order, no whitespace, `__metadata__` sorted with the tensors.
    static func headerJSON(entries: [Entry], metadata: [String: String]) -> [UInt8] {
        func less(_ a: String, _ b: String) -> Bool { Array(a.utf8).lexicographicallyPrecedes(Array(b.utf8)) }
        var objects: [(key: String, value: String)] = entries.map { e in
            let shape = e.shape.map(String.init).joined(separator: ",")
            return (e.name, "{\"data_offsets\":[\(e.dataOffset),\(e.dataOffset + e.byteCount)],\"dtype\":\"\(e.dtype)\",\"shape\":[\(shape)]}")
        }
        let meta = metadata.sorted { less($0.key, $1.key) }.map { "\(jsonString($0.key)):\(jsonString($0.value))" }.joined(separator: ",")
        objects.append(("__metadata__", metadata.isEmpty ? "null" : "{\(meta)}"))
        objects.sort { less($0.key, $1.key) }
        return Array(("{" + objects.map { "\(jsonString($0.key)):\($0.value)" }.joined(separator: ",") + "}").utf8)
    }

    /// The names here are plain ASCII; quotes, backslashes and control characters are escaped as nlohmann does.
    static func jsonString(_ s: String) -> String {
        var out = "\""
        for scalar in s.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            case _ where scalar.value < 0x20: out += String(format: "\\u%04x", scalar.value)
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out + "\""
    }

    /// The order in which MLX 0.32's `save_safetensors` writes the data of `keys` inserted in this order. Saves placeholder scalars
    /// (4 bytes each) to a file in `scratch` through mlx-c and reads the order back from its header. Evaluates only already
    /// contiguous scalars, so no GPU kernel runs.
    static func dataOrder(keys: [String], scratch: URL) throws -> [String] {
        guard Set(keys).count == keys.count else { throw KleinConversionError.io("duplicate tensor names in a shard") }
        let url = scratch.appendingPathComponent(".order-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let placeholder = MLXArray(Int32(0))
        eval(placeholder)
        let map = mlx_map_string_to_array_new()
        defer { mlx_map_string_to_array_free(map) }
        let metadata = mlx_map_string_to_string_new()
        defer { mlx_map_string_to_string_free(metadata) }
        mlx_map_string_to_string_insert(metadata, "order", "probe")
        for key in keys { mlx_map_string_to_array_insert(map, key, placeholder.ctx) }
        guard mlx_save_safetensors(url.path, map, metadata) == 0 else { throw KleinConversionError.io("cannot write \(url.path)") }
        let header = try SafetensorsHeader.read(from: url)
        let order = header.entries.sorted { $0.value.byteRange.lowerBound < $1.value.byteRange.lowerBound }.map(\.key)
        guard order.count == keys.count else { throw KleinConversionError.io("placeholder shard holds \(order.count) of \(keys.count) tensors") }
        return order
    }
}

/// `model.safetensors.index.json` as mflux writes it: `json.dump(..., indent=2)` of the metadata and the weight map, the weight map
/// in shard order and, within a shard, in parameter order; no trailing newline.
enum KleinIndexJSON {
    static func render(metadata: [(String, String)], weightMap: [(String, String)]) -> [UInt8] {
        let s = KleinShardLayout.jsonString
        func object(_ pairs: [(String, String)], indent: String) -> String {
            guard !pairs.isEmpty else { return "{}" }
            return "{\n" + pairs.map { "\(indent)  \(s($0.0)): \(s($0.1))" }.joined(separator: ",\n") + "\n\(indent)}"
        }
        let text = "{\n  \"metadata\": \(object(metadata, indent: "  ")),\n  \"weight_map\": \(object(weightMap, indent: "  "))\n}"
        return Array(text.utf8)
    }
}
