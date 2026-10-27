import Darwin
import MLX
import XCTest
@testable import FluxKlein
@testable import MLXImageCore

/// Converts the official FLUX.2 Klein 9B bf16 release and compares the result with the snapshot `mflux-save` 0.20 wrote from the same
/// files. Opt in with `KLEIN9B_BF16_DIR` (the official folder, read only) and `KLEIN9B_MODEL_DIR` (the MFLUX q4 snapshot, read only);
/// the output goes under `KLEIN_CONVERSION_OUTPUT_DIR` (default: the temporary folder) and is removed afterwards unless
/// `KLEIN_CONVERSION_KEEP=1`.
final class KleinConverterIntegrationTests: XCTestCase {
    func testConvertsOfficial9BByteForByteAndGeneratesTheSameImage() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let sourcePath = environment["KLEIN9B_BF16_DIR"], !sourcePath.isEmpty else {
            throw XCTSkip("KLEIN9B_BF16_DIR is not set; point it at the official FLUX.2 Klein 9B bf16 download")
        }
        let reference = try KleinTestSupport.requireModel9B()
        let source = URL(fileURLWithPath: sourcePath, isDirectory: true)
        let parent = URL(fileURLWithPath: environment["KLEIN_CONVERSION_OUTPUT_DIR"] ?? NSTemporaryDirectory(), isDirectory: true)
        let output = parent.appendingPathComponent("flux2-klein-9b-q4-converted-\(UUID().uuidString.prefix(8))")
        defer { if environment["KLEIN_CONVERSION_KEEP"] != "1" { try? FileManager.default.removeItem(at: output) } }

        let sampler = FootprintSampler()
        sampler.start()
        let progressLog = ProgressLog()
        let result = try KleinConverter.convert(sourceDirectory: source, outputDirectory: output, progress: { progressLog.record($0) })
        let peak = sampler.stop()
        let required = try KleinConverter.requiredBytes(for: .klein9B)
        print(String(format: "CONVERSION 9B: %.1f s, peak phys_footprint %.2f GiB (baseline %.2f GiB), %lld bytes written, required %lld, %d progress calls",
                     result.duration, Double(peak.peak) / Double(1 << 30), Double(peak.baseline) / Double(1 << 30),
                     result.outputByteCount, required, progressLog.count))
        XCTAssertEqual(result.outputByteCount, required)
        XCTAssertEqual(progressLog.last?.fractionCompleted, 1)

        // Same files, same bytes; on a difference, name the differing tensors.
        let files = try relativeFiles(reference)
        XCTAssertEqual(try relativeFiles(output), files)
        var differing: [String] = []
        for file in files {
            let a = reference.appendingPathComponent(file), b = output.appendingPathComponent(file)
            if try KleinHashing.sha256(of: a) == KleinHashing.sha256(of: b) { continue }
            differing.append(file)
            guard file.hasSuffix(".safetensors") else { continue }
            let (ha, hb) = (try KleinMappedFile(url: a), try KleinMappedFile(url: b))
            XCTAssertEqual(Set(ha.header.entries.keys), Set(hb.header.entries.keys), file)
            for name in ha.header.entries.keys.sorted() {
                guard let ra = ha.fileRange(of: name), let rb = hb.fileRange(of: name) else { continue }
                let (x, y) = (ha.bytes(ra), hb.bytes(rb))
                let (ea, eb) = (ha.header.entries[name]!, hb.header.entries[name]!)
                if ea.dtype != eb.dtype || ea.shape != eb.shape {
                    differing.append("\(file) \(name): dtype or shape")
                } else if x.count != y.count || memcmp(x.baseAddress!, y.baseAddress!, x.count) != 0 {
                    differing.append("\(file) \(name): data")
                }
            }
        }
        XCTAssertEqual(differing, [], "files or tensors that differ from the MFLUX snapshot")
        print("CONVERSION 9B: \(files.count) files compared, \(differing.isEmpty ? "all byte-identical" : "\(differing.count) differences")")

        // One image from each snapshot, same request.
        let request = KleinGenerationRequest(prompt: "a lighthouse on a rocky coast at dusk, oil painting", width: 512, height: 512, seed: 42, steps: 4)
        func pixels(_ directory: URL) throws -> [UInt8] {
            let generator = try KleinGenerator(modelDirectory: directory, variant: .klein9B)
            let image = try generator.generate(request).pixels
            eval(image)
            return image.asArray(UInt8.self)
        }
        let converted = try pixels(output)
        Memory.clearCache()
        let mflux = try pixels(reference)
        let diffs = zip(converted, mflux).map { abs(Int($0) - Int($1)) }
        print("CONVERSION 9B generation 512x512 seed 42: \(diffs.filter { $0 != 0 }.count) of \(diffs.count) channel values differ, max \(diffs.max() ?? 0)")
        XCTAssertEqual(converted, mflux)
    }

    private func relativeFiles(_ directory: URL) throws -> [String] {
        let base = directory.standardizedFileURL.resolvingSymlinksInPath().path
        let walker = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])!
        return walker.compactMap { item -> String? in
            guard let url = item as? URL, (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true else { return nil }
            return String(url.standardizedFileURL.resolvingSymlinksInPath().path.dropFirst(base.count + 1))
        }.sorted()
    }
}

final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var count = 0
    private(set) var last: KleinConversionProgress?
    func record(_ p: KleinConversionProgress) {
        lock.withLock {
            if let last { XCTAssertGreaterThanOrEqual(p.completedBytes, last.completedBytes) }
            count += 1
            last = p
        }
    }
}

/// Samples the process's `phys_footprint` every 20 ms.
final class FootprintSampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private var running = false
    private var baseline: UInt64 = 0

    static func footprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return status == KERN_SUCCESS ? info.phys_footprint : 0
    }

    func start() {
        baseline = Self.footprint()
        lock.withLock { running = true; peak = baseline }
        Thread.detachNewThread { [self] in
            while lock.withLock({ running }) {
                let now = Self.footprint()
                lock.withLock { peak = max(peak, now) }
                usleep(20_000)
            }
        }
    }

    func stop() -> (peak: UInt64, baseline: UInt64) {
        lock.withLock {
            running = false
            peak = max(peak, Self.footprint())
            return (peak, baseline)
        }
    }
}
