import CryptoKit
import Darwin
import Foundation
import MLX
import MLXImageCore
import XCTest
@testable import SeedVR2

enum SeedVR2TestSupport {
    /// The read-only folder with `seedvr2_ema_3b_fp16.safetensors` and `ema_vae_fp16.safetensors` (`SEEDVR2_MODEL_DIR`).
    static func requireModel() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["SEEDVR2_MODEL_DIR"], !path.isEmpty else {
            throw XCTSkip("SEEDVR2_MODEL_DIR is not set; point it at the numz/SeedVR2_comfyUI fp16 files")
        }
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: url.appendingPathComponent(SeedVR2Layout.transformerFile).path),
            "SEEDVR2_MODEL_DIR: \(SeedVR2Layout.transformerFile) not found at \(url.path)")
        return url
    }

    /// The process's peak physical footprint (`ri_lifetime_max_phys_footprint`), as the MFLUX measurements read it.
    static func peakFootprint() -> UInt64 {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        return status == 0 ? info.ri_lifetime_max_phys_footprint : 0
    }
}

/// Needs `SEEDVR2_MODEL_DIR`: the stored files against the pins and the layout, and a small upscale end to end.
final class SeedVR2ModelTests: XCTestCase {
    func testStoredFilesMatchThePinsAndTheLayout() throws {
        let model = try SeedVR2TestSupport.requireModel()
        XCTAssertNoThrow(try SeedVR2Weights.validate(modelDirectory: model))
        for file in SeedVR2Pins.files {
            let handle = try FileHandle(forReadingFrom: model.appendingPathComponent(file.name))
            defer { try? handle.close() }
            var hasher = SHA256()
            while let chunk = try handle.read(upToCount: 64 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
            XCTAssertEqual(hasher.finalize().map { String(format: "%02x", $0) }.joined(), file.sha256, file.name)
        }
    }

    func testSmallUpscaleRunsAndIsDeterministic() throws {
        let model = try SeedVR2TestSupport.requireModel()
        Memory.cacheLimit = 0
        let upscaler = try SeedVR2Upscaler(modelDirectory: model)
        XCTAssertEqual(upscaler.weights.transformer["txt_in.weight"]?.dtype, .uint32)
        XCTAssertEqual(upscaler.weights.transformer["vid_in.proj.weight"]?.dtype, .float16)
        XCTAssertEqual(upscaler.weights.vae["encoder.mid_block.attentions.0.to_q.scales"]?.dtype, .float16)
        XCTAssertEqual(upscaler.weights.vae["encoder.conv_in.weight"]?.shape, [128, 3, 3, 3, 3])
        // A smooth 44 x 60 gradient at 2x: MFLUX's ScaleFactor rounds the shortest side down to a multiple of 16 (88 -> 80), so the
        // output is 80 x 108 (60 * 80 / 44 = 109.1, truncated and made even), padded to 80 x 112.
        let (width, height) = (44, 60)
        var bytes: [UInt8] = []
        for y in 0 ..< height { for x in 0 ..< width { bytes += [UInt8(x * 5), UInt8(y * 4), UInt8((x + y) * 2)] } }
        let request = SeedVR2Request(image: .init(width: width, height: height, bytes: bytes), resolution: .scale(2), softness: 0.5, seed: 3)
        let first = try upscaler.upscale(request)
        XCTAssertEqual(first.pixels.shape, [108, 80, 3])
        XCTAssertEqual(first.input.shape, [1, 3, 112, 80])
        XCTAssertEqual(first.latents.shape, [1, 16, 1, 14, 10])
        let second = try upscaler.upscale(request)
        XCTAssertEqual(first.pixels.asArray(UInt8.self), second.pixels.asArray(UInt8.self))
        // The colour correction carries the input's low frequencies: the mean colour stays close to the input's.
        let mean = first.pixels.asType(.float32).mean(axes: [0, 1]).asArray(Float.self)
        let inputMean = MLXArray(bytes, [height, width, 3]).asType(.float32).mean(axes: [0, 1]).asArray(Float.self)
        for c in 0 ..< 3 { XCTAssertEqual(mean[c], inputMean[c], accuracy: 12) }
    }
}

/// Opt-in measurement (`SEEDVR2_MEASURE=<input png>:<shortest side>:<softness>`), one job per process for the lifetime peak, with
/// `Memory.cacheLimit = 0`: load time and peak after load (the quantization at load), then the upscale's time and the process peak.
/// Prints one JSON line; run it in a release build (`swift test -c release -Xswiftc -enable-testing --filter SeedVR2MeasurementTests`).
final class SeedVR2MeasurementTests: XCTestCase {
    func testMeasureOneUpscale() throws {
        guard let spec = ProcessInfo.processInfo.environment["SEEDVR2_MEASURE"], !spec.isEmpty else {
            throw XCTSkip("SEEDVR2_MEASURE is not set (<input png>:<shortest side>:<softness>)")
        }
        let model = try SeedVR2TestSupport.requireModel()
        let parts = spec.split(separator: ":").map(String.init)
        guard parts.count == 3, let side = Int(parts[1]), let softness = Double(parts[2]) else { return XCTFail("bad SEEDVR2_MEASURE \(spec)") }
        Memory.cacheLimit = 0
        let image = try ImageOutput.readPNG(at: URL(fileURLWithPath: (parts[0] as NSString).expandingTildeInPath))
        let began = Date()
        let upscaler = try SeedVR2Upscaler(modelDirectory: model)
        let loadSeconds = Date().timeIntervalSince(began)
        let peakAfterLoad = SeedVR2TestSupport.peakFootprint()
        let request = SeedVR2Request(image: image, resolution: .shortestSide(side), softness: softness, seed: 42)
        var phases: [String: Double] = [:]
        if ProcessInfo.processInfo.environment["SEEDVR2_MEASURE_PHASES"] == "1" {
            // The lifetime peak after each stage of `upscale`, run stage by stage (diagnostic; the timed run below is separate).
            func mark(_ name: String) { phases[name] = Double(SeedVR2TestSupport.peakFootprint()) / 1e9 }
            let pre = SeedVR2Preprocessing.run(request.image, resolution: request.resolution, softness: request.softness)
            eval(pre.image); mark("1-preprocess")
            let encoded = SeedVR2VAETiler.encode(pre.image, vae: upscaler.vae)
            eval(encoded); mark("2-encode")
            let noise = SeedVR2Upscaler.noise(seed: request.seed, height: encoded.dim(-2), width: encoded.dim(-1))
            let latents = SeedVR2Upscaler.step(
                prediction: upscaler.predict(noise: noise, condition: SeedVR2Upscaler.condition(encoded: encoded)), latents: noise)
            eval(latents); mark("3-transformer")
            let decoded = SeedVR2VAETiler.decode(latents, vae: upscaler.vae)[0..., 0..., 0 ..< pre.height, 0 ..< pre.width]
            eval(decoded); mark("4-decode")
            let corrected = SeedVR2ColorCorrection.apply(content: decoded, style: pre.image[0..., 0..., 0 ..< pre.height, 0 ..< pre.width])
            eval(ImageOutput.pixels(decoded: corrected)); mark("5-colour-correction")
        }
        let started = Date()
        let output = try upscaler.upscale(request)
        let seconds = Date().timeIntervalSince(started)
        let peak = SeedVR2TestSupport.peakFootprint()
        let result: [String: Any] = [
            "input": parts[0], "output": "\(output.width)x\(output.height)", "softness": softness, "loadSeconds": loadSeconds,
            "peakAfterLoadGB": Double(peakAfterLoad) / 1e9, "seconds": seconds, "peakGB": Double(peak) / 1e9,
            "mlxPeakGB": Double(Memory.peakMemory) / 1e9, "phasePeaksGB": phases,
        ]
        print("SEEDVR2_MEASUREMENT " + String(data: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]), encoding: .utf8)!)
        if let out = ProcessInfo.processInfo.environment["SEEDVR2_MEASURE_OUTPUT"], !out.isEmpty {
            try output.writePNG(to: URL(fileURLWithPath: out))
        }
    }
}
