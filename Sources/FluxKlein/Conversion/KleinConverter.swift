import CryptoKit
import Darwin
import Foundation
import MLXImageCore

/// How far a conversion is: source weight bytes read, hashed and converted.
public struct KleinConversionProgress: Sendable, Equatable {
    public let completedBytes: Int64
    public let totalBytes: Int64
    public var fractionCompleted: Double { totalBytes > 0 ? Double(completedBytes) / Double(totalBytes) : 0 }
}

public struct KleinConversionResult: Sendable {
    public let variant: KleinVariant
    public let outputDirectory: URL
    /// Bytes written (equals `KleinConverter.requiredBytes(for:)`).
    public let outputByteCount: Int64
    /// The Hugging Face repository and commit whose files were converted (every file matched its pinned SHA-256).
    public let sourceRepository: String
    public let sourceRevision: String
    public let duration: TimeInterval
}

/// Converts an official FLUX.2 Klein bf16 diffusers snapshot (as downloaded from Hugging Face) into the q4 snapshot the loaders read,
/// byte for byte what `mflux-save --base-model <variant> --quantize 4` (MFLUX 0.20) writes from the same files: the same tensor names,
/// dtypes, shapes, shard split, data order, metadata and tokenizer files.
///
/// Every linear layer and the token embedding are quantized on the CPU (`KleinQ4Quantizer`, which reproduces MLX's Metal kernel
/// exactly); no GPU work is done. Weights stream tensor by tensor from memory-mapped source files, which are hashed in the same pass,
/// so each source byte is read once. Output goes to a hidden staging folder next to `outputDirectory` and is renamed into place only
/// when every file has been written, synced, every source hash matched and the loaders' header checks pass; on cancellation or any
/// failure the staging folder is removed. The source folder is only read.
public enum KleinConverter {
    /// The variants with pinned official releases.
    public static let supportedVariants: [KleinVariant] = [.klein9B]

    /// Exact bytes the converted snapshot occupies (9B: 9,541,191,231). A conversion needs this much free space on the output volume.
    public static func requiredBytes(for variant: KleinVariant) throws -> Int64 {
        guard supportedVariants.contains(variant) else { throw KleinConversionError.unsupportedVariant("\(variant.displayName): no conversion pins") }
        let scratch = FileManager.default.temporaryDirectory
        var total: Int64 = 0
        for layout in try shardLayouts(plan: KleinSnapshotPlan(variant: variant), scratch: scratch) {
            total += layout.shards.reduce(0) { $0 + Int64($1.fileByteCount) } + Int64(layout.index.count)
        }
        let release = KleinOfficialRelease.release(for: variant)!
        for name in ["tokenizer/tokenizer.json", "tokenizer/chat_template.jinja"] {
            total += release.requiredFiles.first { $0.path == name }!.byteCount
        }
        return total + Int64(KleinTokenizerFixup.byteCountChange + KleinTokenizerFixup.tokenizerConfig.count)
    }

    /// Validates `sourceDirectory` (`KleinSourceSnapshot.inspect`), converts it and moves the result to `outputDirectory`, which must
    /// not exist. `progress` is called from worker threads, one call at a time; `isCancelled` is polled throughout (every few MB).
    @discardableResult
    public static func convert(
        sourceDirectory: URL, outputDirectory: URL,
        progress: (@Sendable (KleinConversionProgress) -> Void)? = nil,
        isCancelled: (@Sendable () -> Bool)? = nil
    ) throws -> KleinConversionResult {
        try convert(sourceDirectory: sourceDirectory, outputDirectory: outputDirectory, pinnedRelease: nil, progress: progress, isCancelled: isCancelled)
    }

    /// `pinnedRelease` replaces the variant's official pins (tests use synthetic folders).
    static func convert(
        sourceDirectory: URL, outputDirectory: URL, pinnedRelease: KleinOfficialRelease?,
        progress: (@Sendable (KleinConversionProgress) -> Void)?, isCancelled: (@Sendable () -> Bool)?
    ) throws -> KleinConversionResult {
        let started = Date()
        let cancelled: @Sendable () -> Bool = isCancelled ?? { false }
        let source = try KleinSourceSnapshot.inspect(directory: sourceDirectory, pinnedRelease: pinnedRelease)
        let fileManager = FileManager.default
        guard !fileManager.fileExists(atPath: outputDirectory.path) else { throw KleinConversionError.outputExists(outputDirectory.path) }
        let parent = outputDirectory.deletingLastPathComponent()
        do { try fileManager.createDirectory(at: parent, withIntermediateDirectories: true) } catch {
            throw KleinConversionError.io("cannot create \(parent.path): \(error.localizedDescription)")
        }
        let staging = parent.appendingPathComponent(".\(outputDirectory.lastPathComponent).converting-\(UUID().uuidString)")
        do { try fileManager.createDirectory(at: staging, withIntermediateDirectories: false) } catch {
            throw KleinConversionError.io("cannot create \(staging.path): \(error.localizedDescription)")
        }
        do {
            let byteCount = try convert(source: source, staging: staging, progress: progress, isCancelled: cancelled)
            if cancelled() { throw KleinConversionError.cancelled }
            do { try fileManager.moveItem(at: staging, to: outputDirectory) } catch {
                if fileManager.fileExists(atPath: outputDirectory.path) { throw KleinConversionError.outputExists(outputDirectory.path) }
                throw KleinConversionError.io("cannot move the converted snapshot into place: \(error.localizedDescription)")
            }
            return KleinConversionResult(variant: source.variant, outputDirectory: outputDirectory, outputByteCount: byteCount,
                                         sourceRepository: source.release.repository, sourceRevision: source.release.revision,
                                         duration: Date().timeIntervalSince(started))
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
    }

    /// `convert` for Swift concurrency: runs on a background thread, and cancelling the task cancels the conversion.
    @discardableResult
    public static func convert(
        sourceDirectory: URL, outputDirectory: URL, progress: (@Sendable (KleinConversionProgress) -> Void)? = nil
    ) async throws -> KleinConversionResult {
        try await convert(sourceDirectory: sourceDirectory, outputDirectory: outputDirectory, pinnedRelease: nil, progress: progress)
    }

    static func convert(
        sourceDirectory: URL, outputDirectory: URL, pinnedRelease: KleinOfficialRelease?, progress: (@Sendable (KleinConversionProgress) -> Void)?
    ) async throws -> KleinConversionResult {
        let flag = CancellationFlag()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    continuation.resume(with: Result {
                        try convert(sourceDirectory: sourceDirectory, outputDirectory: outputDirectory, pinnedRelease: pinnedRelease,
                                    progress: progress, isCancelled: { flag.isSet })
                    })
                }
            }
        } onCancel: {
            flag.set()
        }
    }

    // MARK: Conversion

    struct ComponentLayout {
        let plan: KleinComponentPlan
        let shards: [KleinShardLayout]
        /// Tensor name to shard index.
        let shardOf: [String: Int]
        let index: [UInt8]
    }

    static func shardLayouts(plan: KleinSnapshotPlan, scratch: URL) throws -> [ComponentLayout] {
        let version = plan.variant.storedMfluxVersion, level = KleinPins.storedQuantizationLevel
        let metadata = ["quantization_level": level, "mflux_version": version]
        return try plan.components.map { component in
            var shards: [KleinShardLayout] = []
            var shardOf: [String: Int] = [:]
            var weightMap: [(String, String)] = []
            for (number, indices) in component.shards.enumerated() {
                let tensors = indices.map { component.tensors[$0] }
                shards.append(try KleinShardLayout(tensors: tensors, metadata: metadata, scratch: scratch))
                for t in tensors {
                    shardOf[t.name] = number
                    weightMap.append((t.name, "\(number).safetensors"))
                }
            }
            let index = KleinIndexJSON.render(metadata: [("quantization_level", level), ("mflux_version", version)], weightMap: weightMap)
            return ComponentLayout(plan: component, shards: shards, shardOf: shardOf, index: index)
        }
    }

    /// Writes the whole snapshot into `staging`; returns the bytes written.
    static func convert(
        source: KleinSourceSnapshot, staging: URL, progress: (@Sendable (KleinConversionProgress) -> Void)?, isCancelled: @escaping @Sendable () -> Bool
    ) throws -> Int64 {
        let plan = KleinSnapshotPlan(variant: source.variant)
        let layouts = try shardLayouts(plan: plan, scratch: staging)
        var written: Int64 = 0

        // Create every shard with its header and full length; tensors are then written in place, in any order.
        var outputs: [String: [KleinOutputFile]] = [:]
        defer { outputs.values.joined().forEach { $0.close() } }
        for layout in layouts {
            let folder = staging.appendingPathComponent(layout.plan.folder)
            try createDirectory(folder)
            outputs[layout.plan.folder] = try layout.shards.enumerated().map { number, shard in
                let file = try KleinOutputFile(url: folder.appendingPathComponent("\(number).safetensors"), length: shard.fileByteCount)
                try file.write(shard.headerBytes, at: 0)
                written += Int64(shard.fileByteCount)
                return file
            }
            try write(Data(layout.index), to: folder.appendingPathComponent("model.safetensors.index.json"))
            written += Int64(layout.index.count)
        }

        // Targets: for each source tensor, where its outputs go.
        var targets: [String: [Target]] = [:]
        for layout in layouts {
            for tensor in layout.plan.tensors {
                let shard = layout.shardOf[tensor.name]!
                let target = Target(tensor: tensor, file: outputs[layout.plan.folder]![shard], offset: layout.shards[shard].fileOffset(of: tensor.name))
                if case .constant(let bytes) = tensor.recipe {
                    try target.file.write(bytes, at: target.offset)
                } else {
                    targets[tensor.recipe.source!, default: []].append(target)
                }
            }
        }

        // Stream every weight file once: hash it in order and convert each tensor as its bytes pass.
        let destinations = targets
        let weightFiles = source.release.weightFiles
        let total = source.weightByteCount
        let reporter = ProgressReporter(total: total, callback: progress)
        let failure = FirstError()
        let stop: @Sendable () -> Bool = { failure.isSet || isCancelled() }
        DispatchQueue.concurrentPerform(iterations: weightFiles.count) { i in
            do {
                let pin = weightFiles[i]
                let component = pin.path.split(separator: "/").first.map(String.init)!
                let mapped = try KleinMappedFile(url: source.directory.appendingPathComponent(pin.path))
                var hasher = SHA256()
                var cursor = 0
                let entries = mapped.header.entries.sorted { $0.value.byteRange.lowerBound < $1.value.byteRange.lowerBound }
                for (name, _) in entries {
                    let range = mapped.fileRange(of: name)!
                    try hash(mapped, cursor..<range.upperBound, into: &hasher, isCancelled: stop)
                    cursor = range.upperBound
                    let sourceTargets = destinations[name] ?? []
                    guard !sourceTargets.isEmpty || plan.components.first(where: { $0.folder == component })!.droppedSources[name] != nil else {
                        throw KleinConversionError.invalidFile(file: pin.path, reason: "tensor \(name) has no destination")
                    }
                    try produce(source: mapped, range: range, shape: mapped.header.entries[name]!.shape, targets: sourceTargets, stop: stop)
                    mapped.release(range)
                    reporter.add(Int64(range.count))
                }
                try hash(mapped, cursor..<mapped.byteCount, into: &hasher, isCancelled: stop)
                reporter.add(Int64(mapped.byteCount - (entries.last.map { mapped.fileRange(of: $0.key)!.upperBound } ?? 0)) + Int64(mapped.dataOffset))
                let actual = KleinHashing.hex(hasher.finalize())
                guard actual == pin.sha256 else { throw KleinConversionError.hashMismatch(file: pin.path, expected: pin.sha256, actual: actual) }
            } catch {
                failure.record(isCancelled() ? KleinConversionError.cancelled : error)
            }
        }
        if isCancelled() { throw KleinConversionError.cancelled }
        if let error = failure.error { throw error }

        // Tokenizer files.
        let tokenizer = staging.appendingPathComponent("tokenizer")
        try createDirectory(tokenizer)
        let officialTokenizer = try Data(contentsOf: source.directory.appendingPathComponent("tokenizer/tokenizer.json"))
        let tokenizerFiles: [(String, Data)] = [
            ("tokenizer_config.json", KleinTokenizerFixup.tokenizerConfig),
            ("tokenizer.json", try KleinTokenizerFixup.savedTokenizerJSON(fromOfficial: officialTokenizer)),
            ("chat_template.jinja", try Data(contentsOf: source.directory.appendingPathComponent("tokenizer/chat_template.jinja"))),
        ]
        for (name, data) in tokenizerFiles {
            guard KleinHashing.sha256(of: data) == KleinPins.tokenizerFiles[name] else {
                throw KleinConversionError.hashMismatch(file: "tokenizer/\(name) (converted)", expected: KleinPins.tokenizerFiles[name]!, actual: KleinHashing.sha256(of: data))
            }
            try write(data, to: tokenizer.appendingPathComponent(name))
            written += Int64(data.count)
        }

        for file in outputs.values.joined() { try file.sync() }
        if isCancelled() { throw KleinConversionError.cancelled }

        // The loaders' own header checks, before the snapshot is moved into place.
        do {
            _ = try KleinTransformerWeights.readAndValidate(modelDirectory: staging, variant: source.variant)
            _ = try KleinTextWeights.readAndValidate(modelDirectory: staging, variant: source.variant)
            _ = try KleinVAEWeights.readAndValidate(modelDirectory: staging, variant: source.variant)
        } catch {
            throw KleinConversionError.io("the converted snapshot fails the loader checks: \(error)")
        }
        return written
    }

    struct Target {
        let tensor: KleinPlannedTensor
        let file: KleinOutputFile
        let offset: Int
    }

    /// Rows per quantization block: about 8 MB of bf16 input.
    static func blockRows(columns: Int) -> Int { max(1, (8 << 20) / (columns * 2)) }

    static func produce(source: KleinMappedFile, range: Range<Int>, shape: [Int], targets: [Target], stop: @Sendable () -> Bool) throws {
        let bytes = source.bytes(range)
        for target in targets {
            switch target.tensor.recipe {
            case .copy:
                try target.file.write(bytes, at: target.offset)
            case .transposedConv:
                let out = transposeConv(bytes.bindMemory(to: UInt16.self), shape: shape)
                try out.withUnsafeBytes { try target.file.write($0, at: target.offset) }
            case .quantized, .constant:
                break  // quantized parts are written together below
            }
        }
        func part(_ wanted: KleinTensorRecipe.QuantizedPart) -> Target? {
            targets.first { if case .quantized(_, let p) = $0.tensor.recipe { p == wanted } else { false } }
        }
        guard let weight = part(.weight), let scales = part(.scales), let biases = part(.biases) else { return }
        let (rows, columns) = (shape[0], shape[1])
        let groupsPerRow = columns / KleinQ4Quantizer.groupSize, wordsPerRow = columns / KleinQ4Quantizer.valuesPerWord
        let step = blockRows(columns: columns)
        let blocks = (rows + step - 1) / step
        let failure = FirstError()
        let src = bytes.baseAddress!.assumingMemoryBound(to: UInt16.self)
        DispatchQueue.concurrentPerform(iterations: blocks) { block in
            guard !failure.isSet, !stop() else { return }
            let r0 = block * step, r1 = min(rows, r0 + step), n = r1 - r0
            var w = [UInt32](repeating: 0, count: n * wordsPerRow)
            var s = [UInt16](repeating: 0, count: n * groupsPerRow)
            var b = [UInt16](repeating: 0, count: n * groupsPerRow)
            w.withUnsafeMutableBufferPointer { wp in s.withUnsafeMutableBufferPointer { sp in b.withUnsafeMutableBufferPointer { bp in
                KleinQ4Quantizer.quantizeGroups(source: src + r0 * columns, groupCount: n * groupsPerRow,
                                                weight: wp.baseAddress!, scales: sp.baseAddress!, biases: bp.baseAddress!)
            } } }
            do {
                try w.withUnsafeBytes { try weight.file.write($0, at: weight.offset + r0 * wordsPerRow * 4) }
                try s.withUnsafeBytes { try scales.file.write($0, at: scales.offset + r0 * groupsPerRow * 2) }
                try b.withUnsafeBytes { try biases.file.write($0, at: biases.offset + r0 * groupsPerRow * 2) }
            } catch {
                failure.record(error)
            }
        }
        if let error = failure.error { throw error }
        if stop() { throw KleinConversionError.cancelled }
    }

    /// `[out, in, kh, kw]` to `[out, kh, kw, in]` (16-bit elements).
    static func transposeConv(_ src: UnsafeBufferPointer<UInt16>, shape: [Int]) -> [UInt16] {
        let (o, i, kh, kw) = (shape[0], shape[1], shape[2], shape[3])
        var out = [UInt16](repeating: 0, count: o * i * kh * kw)
        for a in 0..<o { for c in 0..<i { for y in 0..<kh { for x in 0..<kw {
            out[((a * kh + y) * kw + x) * i + c] = src[((a * i + c) * kh + y) * kw + x]
        } } } }
        return out
    }

    /// Hashes `range` of the map in 8 MB slices, polling `isCancelled` between slices.
    static func hash(_ file: KleinMappedFile, _ range: Range<Int>, into hasher: inout SHA256, isCancelled: () -> Bool) throws {
        var start = range.lowerBound
        let slice = 8 << 20
        while start < range.upperBound {
            if isCancelled() { throw KleinConversionError.cancelled }
            let end = min(range.upperBound, start + slice)
            hasher.update(bufferPointer: file.bytes(start..<end))
            start = end
        }
    }

    static func createDirectory(_ url: URL) throws {
        do { try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true) } catch {
            throw KleinConversionError.io("cannot create \(url.lastPathComponent): \(error.localizedDescription)")
        }
    }

    static func write(_ data: Data, to url: URL) throws {
        do { try data.write(to: url) } catch { throw KleinConversionError.io("cannot write \(url.lastPathComponent): \(error.localizedDescription)") }
    }
}

// MARK: Support

/// A preallocated output file written with positioned writes (safe from several threads at once).
final class KleinOutputFile: @unchecked Sendable {
    let url: URL
    private var descriptor: Int32

    init(url: URL, length: Int) throws {
        self.url = url
        descriptor = open(url.path, O_CREAT | O_EXCL | O_RDWR, 0o644)
        guard descriptor >= 0 else { throw KleinConversionError.io("cannot create \(url.lastPathComponent): \(String(cString: strerror(errno)))") }
        guard ftruncate(descriptor, off_t(length)) == 0 else {
            throw KleinConversionError.io("cannot size \(url.lastPathComponent): \(String(cString: strerror(errno)))")
        }
    }

    func write(_ bytes: [UInt8], at offset: Int) throws { try bytes.withUnsafeBytes { try write($0, at: offset) } }

    func write(_ buffer: UnsafeRawBufferPointer, at offset: Int) throws {
        var done = 0
        while done < buffer.count {
            let n = pwrite(descriptor, buffer.baseAddress! + done, buffer.count - done, off_t(offset + done))
            if n < 0 {
                if errno == EINTR { continue }
                throw KleinConversionError.io("cannot write \(url.lastPathComponent): \(String(cString: strerror(errno)))")
            }
            done += n
        }
    }

    func sync() throws {
        guard fsync(descriptor) == 0 else { throw KleinConversionError.io("cannot sync \(url.lastPathComponent): \(String(cString: strerror(errno)))") }
    }

    func close() {
        if descriptor >= 0 { Darwin.close(descriptor); descriptor = -1 }
    }

    deinit { close() }
}

final class FirstError: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Error?
    var error: Error? { lock.withLock { stored } }
    var isSet: Bool { lock.withLock { stored != nil } }
    func record(_ error: Error) { lock.withLock { if stored == nil { stored = error } } }
}

final class CancellationFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.withLock { value } }
    func set() { lock.withLock { value = true } }
}

/// Serializes progress callbacks and limits them to one per 0.1 % (plus the last).
final class ProgressReporter: @unchecked Sendable {
    private let lock = NSLock()
    private let total: Int64
    private let callback: (@Sendable (KleinConversionProgress) -> Void)?
    private var completed: Int64 = 0
    private var reported: Int64 = -1

    init(total: Int64, callback: (@Sendable (KleinConversionProgress) -> Void)?) {
        self.total = total
        self.callback = callback
    }

    func add(_ bytes: Int64) {
        lock.withLock {
            completed += bytes
            guard let callback, completed == total || completed - reported >= max(1, total / 1000) else { return }
            reported = completed
            callback(KleinConversionProgress(completedBytes: completed, totalBytes: total))
        }
    }
}
