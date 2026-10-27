import Darwin
import Foundation
import MLXImageCore

/// A read-only memory map of a whole safetensors file with its parsed header. Tensor bytes are read straight from the map, so a
/// tensor costs no copy of the file and pages are read from disk once, when they are first touched.
final class KleinMappedFile {
    let url: URL
    let byteCount: Int
    let header: SafetensorsHeader
    /// Offset of the data section (8-byte length prefix plus the header JSON).
    let dataOffset: Int
    let base: UnsafeRawPointer
    private let descriptor: Int32

    init(url: URL) throws {
        self.url = url
        header = try SafetensorsHeader.read(from: url)
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw KleinConversionError.io("cannot open \(url.lastPathComponent): \(String(cString: strerror(errno)))") }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size > 8 else {
            close(fd)
            throw KleinConversionError.io("cannot stat \(url.lastPathComponent)")
        }
        byteCount = Int(info.st_size)
        guard let map = mmap(nil, byteCount, PROT_READ, MAP_PRIVATE, fd, 0), map != MAP_FAILED else {
            close(fd)
            throw KleinConversionError.io("cannot map \(url.lastPathComponent): \(String(cString: strerror(errno)))")
        }
        madvise(map, byteCount, MADV_SEQUENTIAL)
        base = UnsafeRawPointer(map)
        descriptor = fd
        let headerLength = Int(base.loadUnaligned(as: UInt64.self).littleEndian)
        dataOffset = 8 + headerLength
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), byteCount)
        close(descriptor)
    }

    /// Absolute file range of a tensor's bytes.
    func fileRange(of name: String) -> Range<Int>? {
        guard let entry = header.entries[name] else { return nil }
        return (dataOffset + entry.byteRange.lowerBound)..<(dataOffset + entry.byteRange.upperBound)
    }

    func bytes(_ range: Range<Int>) -> UnsafeRawBufferPointer {
        UnsafeRawBufferPointer(start: base + range.lowerBound, count: range.count)
    }

    /// Tells the kernel the pages of `range` are no longer needed, so a large conversion does not keep the whole input resident.
    func release(_ range: Range<Int>) {
        let page = Int(getpagesize())
        let start = range.lowerBound / page * page
        guard range.upperBound > start else { return }
        madvise(UnsafeMutableRawPointer(mutating: base + start), range.upperBound - start, MADV_DONTNEED)
    }
}
