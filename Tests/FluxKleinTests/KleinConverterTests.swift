import CryptoKit
import MLX
import XCTest
@testable import FluxKlein
@testable import MLXImageCore

/// The converter's parts on small synthetic data; no model files needed. The end-to-end check against MFLUX's output is
/// `KleinConverterIntegrationTests`.
final class KleinConverterTests: XCTestCase {
    // MARK: Quantization

    /// Values 0...15 repeated: min 0, max 15, so scale -1, bias 15 and each value stored as 15 - w.
    func testQuantizerHandWorkedGroup() {
        let values = (0..<64).map { KleinQ4Quantizer.floatToBF16(Float($0 % 16)) }
        let out = KleinQ4Quantizer.quantize(bf16: values, rows: 1, columns: 64)
        XCTAssertEqual(out.weight, [0x89AB_CDEF, 0x0123_4567, 0x89AB_CDEF, 0x0123_4567, 0x89AB_CDEF, 0x0123_4567, 0x89AB_CDEF, 0x0123_4567])
        XCTAssertEqual(out.scales, [KleinQ4Quantizer.floatToBF16(-1)])
        XCTAssertEqual(out.biases, [KleinQ4Quantizer.floatToBF16(15)])
    }

    /// The Metal kernel starts the group maximum at 0, so an all-negative group spans [w_min, 0]: -1...-16 gets scale 16/15 and -1
    /// is stored as 14. MLX's CPU quantizer would use the true maximum (-1), scale 1, and store 15.
    func testQuantizerAllNegativeGroupFollowsTheMetalKernel() {
        let values = (0..<64).map { KleinQ4Quantizer.floatToBF16(-Float(1 + $0 % 16)) }
        let out = KleinQ4Quantizer.quantize(bf16: values, rows: 1, columns: 64)
        XCTAssertEqual(out.weight[0] & 0xF, 14)
        XCTAssertEqual(out.biases, [KleinQ4Quantizer.floatToBF16(-16)])
        XCTAssertEqual(out.scales, [KleinQ4Quantizer.floatToBF16(Float(-16) / Float(-15))])
    }

    /// Byte-identical to MLX's Metal `affine_quantize` (what `mflux-save` runs) on random rows and on edge-case groups: all negative,
    /// constant, zero, a single outlier, and values on rounding ties.
    func testQuantizerMatchesTheMLXMetalKernel() {
        var generator = SystemRandomNumberGenerator()
        let columns = 256
        var rows: [[Float]] = (0..<24).map { r in (0..<columns).map { _ in Float.random(in: -1...1, using: &generator) * Float(r + 1) * 0.01 } }
        rows.append((0..<columns).map { -Float(1 + $0 % 16) })
        rows.append([Float](repeating: 0.375, count: columns))
        rows.append([Float](repeating: -0.375, count: columns))
        rows.append([Float](repeating: 0, count: columns))
        rows.append((0..<columns).map { $0 % 64 == 5 ? 3 : 0.001 })
        rows.append((0..<columns).map { Float($0 % 31) * 0.5 - 7.5 })
        let bf16 = rows.flatMap { $0.map(KleinQ4Quantizer.floatToBF16) }
        let swift = KleinQ4Quantizer.quantize(bf16: bf16, rows: rows.count, columns: columns)
        let array = bf16.withUnsafeBytes { MLXArray(Data($0), [rows.count, columns], dtype: .bfloat16) }
        let (w, s, b) = quantized(array, groupSize: 64, bits: 4, stream: .gpu)
        eval(w, s, b!)
        XCTAssertEqual(swift.weight, w.asArray(UInt32.self))
        XCTAssertEqual(swift.scales.withUnsafeBytes { Data($0) }, s.asData(access: .copy).data)
        XCTAssertEqual(swift.biases.withUnsafeBytes { Data($0) }, b!.asData(access: .copy).data)
    }

    func testBF16RoundsToNearestEven() {
        XCTAssertEqual(KleinQ4Quantizer.floatToBF16(1), 0x3F80)
        XCTAssertEqual(KleinQ4Quantizer.floatToBF16(Float(bitPattern: 0x3F80_8000)), 0x3F80)  // tie, even below
        XCTAssertEqual(KleinQ4Quantizer.floatToBF16(Float(bitPattern: 0x3F81_8000)), 0x3F82)  // tie, even above
        XCTAssertEqual(KleinQ4Quantizer.floatToBF16(Float(bitPattern: 0x3F80_8001)), 0x3F81)
        XCTAssertEqual(KleinQ4Quantizer.bf16ToFloat(0xC170), -15)
    }

    func testConvolutionTransposeMatchesMLX() {
        let shape = [4, 3, 2, 5]
        let values = (0..<shape.reduce(1, *)).map { UInt16($0) }
        let swift = values.withUnsafeBufferPointer { KleinConverter.transposeConv($0, shape: shape) }
        let mlx = MLXArray(values.map(Int32.init), shape).transposed(0, 2, 3, 1)
        XCTAssertEqual(swift.map(Int32.init), mlx.asArray(Int32.self))
    }

    // MARK: Plan and shards

    func testPlanStoresExactlyTheLoaderLayouts() {
        for variant in KleinVariant.allCases {
            let plan = KleinSnapshotPlan(variant: variant)
            XCTAssertEqual(plan.components.map(\.folder), ["vae", "transformer", "text_encoder"])
            func stored(_ folder: String) -> [String: [String]] {
                Dictionary(uniqueKeysWithValues: plan.components.first { $0.folder == folder }!.tensors.map { ($0.name, [$0.dtype] + $0.shape.map(String.init)) })
            }
            func expected(_ d: [String: (dtype: String, shape: [Int])]) -> [String: [String]] { d.mapValues { [$0.dtype] + $0.shape.map(String.init) } }
            XCTAssertEqual(stored("transformer"), expected(variant.transformer.expectedTensors()), variant.rawValue)
            XCTAssertEqual(stored("text_encoder"), expected(variant.textEncoder.expectedTensors()), variant.rawValue)
            XCTAssertEqual(stored("vae"), KleinVAELayout.expectedTensors().mapValues { [$0.dtype] + $0.shape.map(String.init) })
            let transformer = plan.components[1]
            let counts = transformer.shards.enumerated().map { ("\($0.offset).safetensors", $0.element.count) }
            XCTAssertEqual(Dictionary(uniqueKeysWithValues: counts), variant.transformerShardTensorCounts, variant.rawValue)
            XCTAssertEqual(plan.components[0].shards.map(\.count), [KleinVAELayout.tensorCount])
        }
        XCTAssertEqual(KleinSnapshotPlan(variant: .klein9B).components[2].shards.map(\.count), [422, 483])
    }

    func testPlanParameterOrderAnchors() {
        let plan = KleinSnapshotPlan(variant: .klein9B)
        let text = plan.components[2].tensors.map(\.name)
        XCTAssertEqual(Array(text.prefix(5)), ["embed_tokens.weight", "embed_tokens.scales", "embed_tokens.biases", "layers.0.input_layernorm.weight", "layers.0.self_attn.q_proj.weight"])
        XCTAssertEqual(Array(text.suffix(2)), ["norm.weight", "rotary_emb.inv_freq"])
        let transformer = plan.components[1].tensors.map(\.name)
        XCTAssertEqual(transformer.first, "time_guidance_embed.linear_1.weight")
        XCTAssertEqual(Array(transformer.suffix(3)), ["proj_out.weight", "proj_out.scales", "proj_out.biases"])
        let vae = plan.components[0].tensors.map(\.name)
        XCTAssertEqual(Array(vae.prefix(4)), ["encoder.conv_in.weight", "encoder.conv_in.bias", "encoder.down_blocks.0.resnets.0.norm1.bias", "encoder.down_blocks.0.resnets.0.norm1.weight"])
        XCTAssertEqual(Array(vae.suffix(2)), ["bn.running_mean", "bn.running_var"])
        let attention = vae.filter { $0.hasPrefix("decoder.mid_block.attentions.0.to_q") }
        XCTAssertEqual(attention, ["weight", "scales", "biases", "bias"].map { "decoder.mid_block.attentions.0.to_q." + $0 })
    }

    func testShardSplitFollowsMflux() {
        XCTAssertEqual(KleinComponentPlan.split(byteCounts: [1, 1, 1], limit: 2), [[0, 1], [2]])
        XCTAssertEqual(KleinComponentPlan.split(byteCounts: [2, 2], limit: 2), [[0], [1]])
        XCTAssertEqual(KleinComponentPlan.split(byteCounts: [1, 2, 1], limit: 3), [[0, 1], [2]])
        // mflux checks before the first tensor too: one tensor over the limit leaves an empty first shard.
        XCTAssertEqual(KleinComponentPlan.split(byteCounts: [3], limit: 2), [[], [0]])
        XCTAssertEqual(KleinComponentPlan.split(byteCounts: [], limit: 2), [])
    }

    func testRequiredBytes() throws {
        XCTAssertEqual(try KleinConverter.requiredBytes(for: .klein9B), 9_541_191_231)
        XCTAssertThrowsError(try KleinConverter.requiredBytes(for: .klein4B)) { XCTAssertTrue("\($0)".contains("unsupported")) }
    }

    // MARK: Files and metadata

    /// A shard laid out and written tensor by tensor is byte-identical to MLX's own `save_safetensors` of the same arrays, inserted in
    /// the same order: header JSON, metadata, data order and offsets.
    func testShardLayoutMatchesMLXSave() throws {
        let directory = try temporaryDirectory()
        var arrays: [String: MLXArray] = [:]
        for i in 0..<40 {
            switch i % 3 {
            case 0: arrays["layers.\(i).weight"] = MLXArray((0..<(8 * (i + 1))).map { UInt32($0 * 7 + i) }, [i + 1, 8])
            case 1: arrays["layers.\(i).scales"] = MLXArray((0..<(i + 2)).map { Float($0) / 3 }, [i + 2]).asType(.bfloat16)
            default: arrays["rotary_\(i).inv_freq"] = MLXArray((0..<4).map { Float($0) + Float(i) }, [2, 2])
            }
        }
        let metadata = ["quantization_level": "4", "mflux_version": "0.20.0"]
        let reference = directory.appendingPathComponent("mlx.safetensors")
        try save(arrays: arrays, metadata: metadata, url: reference)
        // `save` inserts in the dictionary's iteration order, which `keys` repeats.
        let tensors = arrays.keys.map { name in
            let a = arrays[name]!
            let dtype = a.dtype == .bfloat16 ? "BF16" : a.dtype == .uint32 ? "U32" : "F32"
            return KleinPlannedTensor(name: name, dtype: dtype, shape: a.shape, recipe: .copy(source: name))
        }
        let layout = try KleinShardLayout(tensors: tensors, metadata: metadata, scratch: directory)
        let url = directory.appendingPathComponent("ours.safetensors")
        let file = try KleinOutputFile(url: url, length: layout.fileByteCount)
        try file.write(layout.headerBytes, at: 0)
        for (name, array) in arrays { try file.write([UInt8](array.asData(access: .copy).data), at: layout.fileOffset(of: name)) }
        file.close()
        XCTAssertEqual(try Data(contentsOf: url), try Data(contentsOf: reference))
    }

    func testIndexJSONMatchesPythonJSONDump() {
        let text = String(decoding: KleinIndexJSON.render(metadata: [("quantization_level", "4"), ("mflux_version", "0.20.0")],
                                                        weightMap: [("b.weight", "0.safetensors"), ("a.weight", "1.safetensors")]), as: UTF8.self)
        XCTAssertEqual(text, """
        {
          "metadata": {
            "quantization_level": "4",
            "mflux_version": "0.20.0"
          },
          "weight_map": {
            "b.weight": "0.safetensors",
            "a.weight": "1.safetensors"
          }
        }
        """)
    }

    func testTokenizerConfigIsThePinnedFile() {
        XCTAssertEqual(KleinHashing.sha256(of: KleinTokenizerFixup.tokenizerConfig), KleinPins.tokenizerFiles["tokenizer_config.json"])
    }

    func testTokenizerFlagUpdate() throws {
        let body = "{\n  \"pre_tokenizer\": {\n    \"pretokenizers\": [\n" + KleinTokenizerFixup.preTokenizerOfficial
            + "\n  \"post_processor\": {},\n" + KleinTokenizerFixup.decoderOfficial + "\n  \"model\": {}\n}"
        let updated = try KleinTokenizerFixup.updatingFlags(Data(body.utf8))
        XCTAssertEqual(updated.count - body.utf8.count, KleinTokenizerFixup.byteCountChange)
        XCTAssertEqual(KleinTokenizerFixup.byteCountChange, -4)
        let text = String(decoding: updated, as: UTF8.self)
        XCTAssertTrue(text.contains(KleinTokenizerFixup.preTokenizerSaved) && text.contains(KleinTokenizerFixup.decoderSaved))
        XCTAssertFalse(text.contains(KleinTokenizerFixup.preTokenizerOfficial) || text.contains(KleinTokenizerFixup.decoderOfficial))

        XCTAssertThrowsError(try KleinTokenizerFixup.updatingFlags(Data("{}".utf8))) { assertInvalidFile($0) }
        XCTAssertThrowsError(try KleinTokenizerFixup.updatingFlags(Data((body + body).utf8))) { assertInvalidFile($0) }
        XCTAssertThrowsError(try KleinTokenizerFixup.savedTokenizerJSON(fromOfficial: Data(body.utf8))) {
            guard case KleinConversionError.hashMismatch = $0 else { return XCTFail("\($0)") }
        }
    }

    /// The stored `inv_freq` is `1 / 1e6 ** (2i / 128)` as MLX evaluates it in float32; within 2 ulp of the libm value.
    func testInvFreqConstant() {
        let bits = KleinConversionConstants.qwen3InvFreqBits
        XCTAssertEqual(bits.count, 64)
        XCTAssertEqual(KleinConversionConstants.qwen3InvFreq.count, 256)
        for (i, b) in bits.enumerated() {
            let exact = Float(1.0 / pow(1_000_000.0, Double(2 * i) / 128))
            XCTAssertLessThanOrEqual(abs(Int64(b) - Int64(exact.bitPattern)), 2, "inv_freq[\(i)]")
        }
    }

    // MARK: Validation

    func testInspectAcceptsTheSyntheticRelease() throws {
        let fixture = try SyntheticSource()
        let snapshot = try KleinSourceSnapshot.inspect(directory: fixture.directory, pinnedRelease: fixture.release)
        XCTAssertEqual(snapshot.variant, .klein9B)
        XCTAssertFalse(snapshot.weightHashesVerified)
    }

    func testInspectRefusesWhatIsNotKlein() throws {
        let fixture = try SyntheticSource()
        try fixture.writeJSON("model_index.json", fixture.modelIndex.merging(["_class_name": "QwenImagePipeline"]) { $1 })
        assertError(try fixture.inspect(), .notKlein)
        try FileManager.default.removeItem(at: fixture.url("model_index.json"))
        assertError(try fixture.inspect(), .notKlein)
        try fixture.writeJSON("model_index.json", fixture.modelIndex.merging(["vae": ["diffusers", "AutoencoderKL"]]) { $1 })
        assertError(try fixture.inspect(), .notKlein)
    }

    func testInspectRefusesUnsupportedVariants() throws {
        let fixture = try SyntheticSource()
        try fixture.writeJSON("model_index.json", fixture.modelIndex.merging(["is_distilled": false]) { $1 })
        assertError(try fixture.inspect(), .unsupportedVariant)
        try fixture.writeJSON("model_index.json", fixture.modelIndex)
        try fixture.writeJSON("transformer/config.json", fixture.transformerConfig.merging(["num_layers": 6]) { $1 })
        assertError(try fixture.inspect(), .unsupportedVariant)
        // Klein 4B is recognized but has no official pins to convert from.
        try fixture.writeJSON("transformer/config.json", fixture.transformerConfig.merging(
            ["num_layers": 5, "num_single_layers": 20, "num_attention_heads": 24, "joint_attention_dim": 7680]) { $1 })
        try fixture.writeJSON("text_encoder/config.json", fixture.textConfig.merging(["hidden_size": 2560, "intermediate_size": 9728]) { $1 })
        XCTAssertEqual(try KleinSourceSnapshot.detectVariant(directory: fixture.directory), .klein4B)
        assertError(try KleinSourceSnapshot.inspect(directory: fixture.directory), .unsupportedVariant)
    }

    func testInspectRefusesInconsistentConfigs() throws {
        let fixture = try SyntheticSource()
        try fixture.writeJSON("text_encoder/config.json", fixture.textConfig.merging(["hidden_size": 2560]) { $1 })
        assertError(try fixture.inspect(), .invalidFile)
        try fixture.writeJSON("text_encoder/config.json", fixture.textConfig)
        try fixture.writeJSON("transformer/config.json", fixture.transformerConfig.merging(["guidance_embeds": true]) { $1 })
        assertError(try fixture.inspect(), .invalidFile)
    }

    func testInspectRefusesMissingAndUnexpectedFiles() throws {
        let fixture = try SyntheticSource()
        try FileManager.default.removeItem(at: fixture.url("tokenizer/chat_template.jinja"))
        assertError(try fixture.inspect(), .missingFile)
        try Data("x".utf8).write(to: fixture.url("tokenizer/chat_template.jinja"))
        try fixture.refreshPins()
        try Data().write(to: fixture.url("transformer/extra.safetensors"))
        assertError(try fixture.inspect(), .unexpectedFile)
        try FileManager.default.removeItem(at: fixture.url("transformer/extra.safetensors"))
        // Hidden files, optional release files and other top-level files are fine.
        try Data().write(to: fixture.url("transformer/.DS_Store"))
        try Data().write(to: fixture.url("README.md"))
        try Data().write(to: fixture.url("notes.txt"))
        XCTAssertNoThrow(try fixture.inspect())
    }

    func testInspectRefusesChangedFiles() throws {
        let fixture = try SyntheticSource()
        let template = fixture.url("tokenizer/chat_template.jinja")
        let original = try Data(contentsOf: template)
        try (original + Data("!".utf8)).write(to: template)
        assertError(try fixture.inspect(), .sizeMismatch)
        var changed = original
        changed[0] ^= 1
        try changed.write(to: template)
        assertError(try fixture.inspect(), .hashMismatch)
        try original.write(to: template)
        XCTAssertNoThrow(try fixture.inspect())
        // The weight files' hashes are checked only on request (or while converting).
        assertError(try KleinSourceSnapshot.inspect(directory: fixture.directory, pinnedRelease: fixture.release.withWeightHash("vae", "00"),
                                                    verifyingWeightHashes: true), .hashMismatch)
    }

    func testInspectRefusesWeightHeadersThatDisagree() throws {
        let fixture = try SyntheticSource()
        try fixture.writeWeights(component: "vae") { sources in
            var s = sources
            s["decoder.conv_in.weight"] = [512, 32, 1, 3]
            return s
        }
        try fixture.refreshPins()
        assertError(try fixture.inspect(), .invalidFile)
        try fixture.writeWeights(component: "vae") { $0.merging(["extra.weight": [4]]) { $1 } }
        try fixture.refreshPins()
        assertError(try fixture.inspect(), .invalidFile)
    }

    // MARK: Conversion: output, cancellation and failure

    func testConvertRefusesAnExistingOutput() throws {
        let fixture = try SyntheticSource()
        let output = try temporaryDirectory()
        assertError(try KleinConverter.convert(sourceDirectory: fixture.directory, outputDirectory: output, pinnedRelease: fixture.release,
                                               progress: nil, isCancelled: nil), .outputExists)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: output.path), [])
    }

    func testCancelRemovesStagingAndLeavesTheSourceUntouched() throws {
        let fixture = try SyntheticSource()
        let before = try fixture.fingerprint()
        let parent = try temporaryDirectory()
        let output = parent.appendingPathComponent("klein")
        let flag = CancellationFlag()
        let calls = ProgressLog()
        assertError(try KleinConverter.convert(sourceDirectory: fixture.directory, outputDirectory: output, pinnedRelease: fixture.release,
                                               progress: { calls.record($0); flag.set() }, isCancelled: { flag.isSet }), .cancelled)
        XCTAssertGreaterThan(calls.count, 0)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: parent.path), [], "staging folder left behind")
        XCTAssertEqual(try fixture.fingerprint(), before)
    }

    func testCancellingTheAsyncConversionTaskStopsIt() async throws {
        let fixture = try SyntheticSource()
        let parent = try temporaryDirectory()
        let output = parent.appendingPathComponent("klein")
        let started = CancellationFlag()
        let release = fixture.release
        let task = Task {
            try await KleinConverter.convert(sourceDirectory: fixture.directory, outputDirectory: output, pinnedRelease: release,
                                             progress: { _ in started.set() })
        }
        while !started.isSet { try await Task.sleep(nanoseconds: 1_000_000) }
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("expected cancellation")
        } catch {
            XCTAssertEqual(error as? KleinConversionError, .cancelled)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: parent.path), [])
    }

    func testWeightHashMismatchFailsTheConversionAndRemovesStaging() throws {
        let fixture = try SyntheticSource()
        let parent = try temporaryDirectory()
        let output = parent.appendingPathComponent("klein")
        XCTAssertThrowsError(try KleinConverter.convert(sourceDirectory: fixture.directory, outputDirectory: output,
                                                        pinnedRelease: fixture.release.withWeightHash("vae", "00"), progress: nil, isCancelled: nil)) {
            guard case KleinConversionError.hashMismatch(let file, _, _) = $0 else { return XCTFail("\($0)") }
            XCTAssertTrue(file.hasPrefix("vae/"), file)
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: parent.path), [])
    }
}

// MARK: Helpers

enum ErrorKind { case notKlein, unsupportedVariant, missingFile, unexpectedFile, sizeMismatch, hashMismatch, invalidFile, outputExists, cancelled }

func assertError<T>(_ expression: @autoclosure () throws -> T, _ kind: ErrorKind, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertThrowsError(try expression(), file: file, line: line) { error in
        let actual: ErrorKind?
        switch error as? KleinConversionError {
        case .notKlein: actual = .notKlein
        case .unsupportedVariant: actual = .unsupportedVariant
        case .missingFile: actual = .missingFile
        case .unexpectedFile: actual = .unexpectedFile
        case .sizeMismatch: actual = .sizeMismatch
        case .hashMismatch: actual = .hashMismatch
        case .invalidFile: actual = .invalidFile
        case .outputExists: actual = .outputExists
        case .cancelled: actual = .cancelled
        default: actual = nil
        }
        XCTAssertEqual(actual, kind, "\(error)", file: file, line: line)
    }
}

func assertInvalidFile(_ error: Error, file: StaticString = #filePath, line: UInt = #line) {
    guard case KleinConversionError.invalidFile = error else { return XCTFail("\(error)", file: file, line: line) }
}

func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("klein-converter-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

/// A folder with the official 9B structure: real config values, tiny tokenizer files, and sparse weight files whose headers hold
/// every source tensor (all zeros; they take no disk space). `release` pins its own files, with placeholder weight hashes.
final class SyntheticSource {
    let directory: URL
    private(set) var release: KleinOfficialRelease
    private static let weightPaths = ["transformer": "transformer/diffusion_pytorch_model.safetensors",
                                      "text_encoder": "text_encoder/model.safetensors", "vae": "vae/diffusion_pytorch_model.safetensors"]

    let modelIndex: [String: Any] = [
        "_class_name": "Flux2KleinPipeline", "is_distilled": true,
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"], "text_encoder": ["transformers", "Qwen3ForCausalLM"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"], "transformer": ["diffusers", "Flux2Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLFlux2"],
    ]
    let transformerConfig: [String: Any] = [
        "_class_name": "Flux2Transformer2DModel", "attention_head_dim": 128, "axes_dims_rope": [32, 32, 32, 32], "eps": 1e-6,
        "guidance_embeds": false, "in_channels": 128, "joint_attention_dim": 12288, "mlp_ratio": 3.0, "num_attention_heads": 32,
        "num_layers": 8, "num_single_layers": 24, "patch_size": 1, "rope_theta": 2000, "timestep_guidance_channels": 256,
    ]
    let textConfig: [String: Any] = [
        "model_type": "qwen3", "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151_936, "rope_theta": 1_000_000,
    ]

    init() throws {
        directory = try temporaryDirectory()
        release = KleinOfficialRelease.klein9B
        try writeJSON("model_index.json", modelIndex)
        try writeJSON("transformer/config.json", transformerConfig)
        try writeJSON("text_encoder/config.json", textConfig)
        try writeJSON("vae/config.json", ["_class_name": "AutoencoderKLFlux2"])
        try writeJSON("transformer/diffusion_pytorch_model.safetensors.index.json", ["weight_map": [String: String]()])
        try writeJSON("text_encoder/model.safetensors.index.json", ["weight_map": [String: String]()])
        try FileManager.default.createDirectory(at: url("tokenizer"), withIntermediateDirectories: true)
        try Data("{\"model\": {}}".utf8).write(to: url("tokenizer/tokenizer.json"))
        try Data("{{ messages }}".utf8).write(to: url("tokenizer/chat_template.jinja"))
        for component in Self.weightPaths.keys { try writeWeights(component: component) { $0 } }
        try refreshPins()
    }

    deinit { try? FileManager.default.removeItem(at: directory) }

    func url(_ path: String) -> URL { directory.appendingPathComponent(path) }

    func inspect() throws -> KleinSourceSnapshot { try KleinSourceSnapshot.inspect(directory: directory, pinnedRelease: release) }

    func writeJSON(_ path: String, _ object: Any) throws {
        try FileManager.default.createDirectory(at: url(path).deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: url(path))
    }

    /// Writes one sparse safetensors file holding the component's source tensors (bf16; `bn.num_batches_tracked` int64).
    func writeWeights(component: String, edit: ([String: [Int]]) -> [String: [Int]]) throws {
        let plan = KleinSnapshotPlan(variant: .klein9B).components.first { $0.folder == component }!
        var shapes = plan.sources
        for (name, entry) in plan.droppedSources { shapes[name] = entry.shape }
        shapes = edit(shapes)
        var header: [String: Any] = ["__metadata__": ["format": "pt"]]
        var offset = 0
        for name in shapes.keys.sorted() {
            let dtype = plan.droppedSources[name]?.dtype ?? "BF16"
            let bytes = shapes[name]!.reduce(1, *) * (dtype == "I64" ? 8 : 2)
            header[name] = ["dtype": dtype, "shape": shapes[name]!, "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while json.count % 8 != 0 { json.append(0x20) }
        let path = url(Self.weightPaths[component]!)
        try? FileManager.default.removeItem(at: path)
        let file = try KleinOutputFile(url: path, length: 8 + json.count + offset)
        try file.write(withUnsafeBytes(of: UInt64(json.count).littleEndian, Array.init) + [UInt8](json), at: 0)
        file.close()
    }

    /// Pins the current files: sizes of all, SHA-256 of the small ones, a placeholder hash for the weights.
    func refreshPins() throws {
        var files: [KleinOfficialRelease.File] = []
        let small = ["model_index.json", "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors.index.json",
                     "text_encoder/config.json", "text_encoder/model.safetensors.index.json", "vae/config.json",
                     "tokenizer/tokenizer.json", "tokenizer/chat_template.jinja"]
        for path in small + Self.weightPaths.values.sorted() {
            let size = try KleinSourceSnapshot.fileSize(url(path))
            let hash = path.hasSuffix(".safetensors") ? String(repeating: "0", count: 64) : try KleinHashing.sha256(of: url(path))
            files.append(.init(path: path, byteCount: size, sha256: hash))
        }
        release = KleinOfficialRelease(variant: .klein9B, repository: "synthetic", revision: "0", requiredFiles: files,
                                       optionalFiles: KleinOfficialRelease.klein9B.optionalFiles)
    }

    /// Names, sizes and modification dates of every file, to show a conversion left the source alone.
    func fingerprint() throws -> [String] {
        let walker = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey])!
        return try walker.compactMap { item -> String? in
            guard let u = item as? URL else { return nil }
            let v = try u.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            return "\(u.path) \(v.fileSize ?? -1) \(v.contentModificationDate?.timeIntervalSince1970 ?? 0)"
        }.sorted()
    }
}

extension KleinOfficialRelease {
    func withWeightHash(_ component: String, _ hash: String) -> KleinOfficialRelease {
        KleinOfficialRelease(variant: variant, repository: repository, revision: revision,
                             requiredFiles: requiredFiles.map { $0.path.hasPrefix(component + "/") && $0.path.hasSuffix(".safetensors")
                                 ? .init(path: $0.path, byteCount: $0.byteCount, sha256: hash) : $0 },
                             optionalFiles: optionalFiles)
    }
}
