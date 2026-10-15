import Foundation
import MLX
import ZImageNative

public struct KleinDiagnostic: Codable, Sendable {
    public var name: String
    /// Relative L2, max abs and the rest, recorded; a diagnostic never fails on its magnitude.
    public var metrics: ComparisonMetrics?
    /// Structural failures only (missing tensor, dtype, shape, non-finite values).
    public var failures: [String]
}

public struct KleinTransformerCaseResult: Codable, Sendable {
    public var id: String
    public var category: String
    public var imageSeqLen: Int
    public var tensors: [KleinTensorResult]
    public var diagnostics: [KleinDiagnostic]
    public var failures: [String]
    public var passed: Bool
}

public struct KleinTransformerReport: Codable, Sendable {
    /// `COMPLETE_PASS`, `COMPLETE_FAIL` (everything was compared and something differs) or `INCOMPLETE` (a case, tensor, file or
    /// control the comparison requires is missing, duplicated or unreadable). Only `COMPLETE_PASS` is a pass.
    public var status: String
    public var passed: Bool
    public var expectedCaseIDs: [String]
    public var casesSHA256: String
    public var casesDigest: String
    public var budgetsSHA256: String
    public var budgets: [String: Double]
    public var frozenInputsSHA256: String
    public var referenceManifestSHA256: String
    public var nativeManifestSHA256: String
    public var isolatedManifestSHA256: String
    public var referenceTensorFilesSHA256: [String: String]
    public var nativeTensorFilesSHA256: [String: String]
    public var isolatedTensorFilesSHA256: [String: String]
    public var referenceRuntime: [String: String]
    public var nativeRuntime: [String: String]
    public var isolatedRuntime: [String: String]
    public var referenceRuntimeModules: [String: String]
    public var checks: [KleinCheck]
    public var controls: [KleinTensorResult]
    public var cases: [KleinTransformerCaseResult]
}

public enum KleinTransformerCompare {
    enum Rule {
        case exact
        case limit(ParityLimitsRow)
        case diagnostic
    }

    // MARK: One tensor

    static func isFloating(_ array: MLXArray) -> Bool { [DType.bfloat16, DType.float16, DType.float32].contains(array.dtype) }

    /// Missing tensors, a dtype or shape disagreement (between the sides or with the architecture), a zero-element tensor and any
    /// non-finite value on either side fail closed; metrics are computed only when both sides are well formed.
    static func compareTensor(
        _ name: String, native: MLXArray?, reference: MLXArray?, rule: Rule, expect: KleinTransformerTensors.Expect?
    ) -> KleinTensorResult {
        var r = KleinTensorResult(name: name, dtype: nil, shape: nil, exact: nil, metrics: nil, nonFiniteCount: 0, failures: [], passed: false)
        guard let native, let reference else {
            r.failures.append("missing: native \(native == nil ? "absent" : "present"), reference \(reference == nil ? "absent" : "present")")
            return r
        }
        r.dtype = KleinDType.name(native.dtype)
        r.shape = native.shape
        if native.dtype != reference.dtype { r.failures.append("dtype: native \(native.dtype) reference \(reference.dtype)") }
        if native.shape != reference.shape { r.failures.append("shape: native \(native.shape) reference \(reference.shape)") }
        if let expect {
            for (side, array) in [("native", native), ("reference", reference)] {
                if let dtype = expect.dtype, KleinDType.name(array.dtype) != dtype { r.failures.append("\(side) is \(KleinDType.name(array.dtype)), expected \(dtype)") }
                if array.shape != expect.shape { r.failures.append("\(side) shape \(array.shape), expected \(expect.shape)") }
            }
        }
        if native.size == 0 { r.failures.append("zero-element tensor") }
        guard r.failures.isEmpty else { return r }
        r.elementCount = native.size

        if case .exact = rule, !isFloating(native) {
            r.exact = KleinCompare.bytes(native) == KleinCompare.bytes(reference)
            if r.exact != true {
                let a = native.asType(.int64).asArray(Int64.self), b = reference.asType(.int64).asArray(Int64.self)
                let first = zip(a, b).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                r.failures.append("values differ" + (first.map { " (first at flat index \($0): native \(a[$0]) reference \(b[$0]))" } ?? ""))
            }
            r.passed = r.failures.isEmpty
            return r
        }

        let n = KleinCompare.floats(native), f = KleinCompare.floats(reference)
        let badNative = n.filter { !$0.isFinite }.count, badReference = f.filter { !$0.isFinite }.count
        r.nonFiniteCount = badNative + badReference
        guard r.nonFiniteCount == 0 else {
            r.failures.append("\(r.nonFiniteCount) non-finite values (native \(badNative), reference \(badReference))")
            return r
        }
        switch rule {
        case .exact:
            r.exact = KleinCompare.bytes(native) == KleinCompare.bytes(reference)
            if r.exact != true {
                let first = zip(n, f).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                r.failures.append("values differ" + (first.map { " (first at flat index \($0): native \(n[$0]) reference \(f[$0]))" } ?? ""))
            }
            r.metrics = ParityMetrics.compare(native: n, reference: f, limits: ParityLimits(relativeL2: 0, normalizedMaxAbs: nil))
        case .limit(let limit):
            let metrics = ParityMetrics.compare(native: n, reference: f, limits: ParityLimits(relativeL2: limit.relativeL2, normalizedMaxAbs: limit.normalizedMaxAbs))
            r.metrics = metrics
            if !metrics.passed {
                r.failures.append("over budget: relativeL2 \(metrics.relativeL2.map { "\($0)" } ?? "n/a") (limit \(limit.relativeL2)), maxAbsDiff \(metrics.maxAbsDiff)"
                    + (metrics.zeroReferenceFallback ? ", zero-reference fallback: absolute L2 \(metrics.absoluteL2)" : ""))
            }
        case .diagnostic:
            r.metrics = ParityMetrics.compare(native: n, reference: f, limits: ParityLimits(relativeL2: .infinity, normalizedMaxAbs: nil))
        }
        r.passed = r.failures.isEmpty
        return r
    }

    static func diagnostic(_ name: String, native: MLXArray?, reference: MLXArray?, expect: KleinTransformerTensors.Expect? = nil) -> KleinDiagnostic {
        let r = compareTensor(name, native: native, reference: reference, rule: .diagnostic, expect: expect)
        return KleinDiagnostic(name: name, metrics: r.metrics, failures: r.failures)
    }

    // MARK: Rules

    static let embeddingKeys: Set<String> = Set(
        ["temb_raw", "temb", "x_embed", "ctx_embed", "rope_img_cos", "rope_img_sin", "rope_txt_cos", "rope_txt_sin"]
            + (0..<6).flatMap { ["mod_img.\($0)", "mod_txt.\($0)"] } + (0..<3).map { "mod_single.\($0)" })

    /// How each tensor of an independent (free-running) primary run is held. Block and final intermediates of a free run accumulate
    /// upstream differences, so they are diagnostics; the isolated block checks decide those.
    static func primaryRule(_ key: String) -> Rule {
        switch key {
        case "text_ids", "latents", "latent_ids", "sigmas", "timesteps", "timestep_scaled", "chain.0": return .exact
        case "prompt_embeds": return .limit(KleinTransformerBudgets.promptEmbeds)
        case "noise_compiled": return .limit(KleinTransformerBudgets.fullPrediction)
        default:
            if key.hasPrefix("chain_noise.") { return .exact }
            if key.hasPrefix("chain.") { return .limit(KleinTransformerBudgets.noiseChain) }
            if embeddingKeys.contains(key) { return .limit(KleinTransformerBudgets.embedding) }
            return .diagnostic
        }
    }

    // MARK: Schedule sanity (each side, independent of the other)

    /// sigmas end at 0 and decrease strictly from 1; timesteps are the float32 `sigmas[:-1] * 1000`.
    static func scheduleFailures(side: String, label: String, sigmas: MLXArray?, timesteps: MLXArray?) -> [String] {
        guard let sigmas, let timesteps, sigmas.size > 1, timesteps.size == sigmas.size - 1 else { return ["\(side) \(label): sigmas/timesteps missing or inconsistent in length"] }
        let s = KleinCompare.floats(sigmas), t = KleinCompare.floats(timesteps)
        var failures: [String] = []
        if s.contains(where: { !$0.isFinite }) || t.contains(where: { !$0.isFinite }) { return ["\(side) \(label): non-finite schedule"] }
        if s.first != 1 { failures.append("\(side) \(label): first sigma \(s.first ?? .nan), expected 1") }
        if s.last != 0 { failures.append("\(side) \(label): last sigma \(s.last ?? .nan), expected 0") }
        if !zip(s, s.dropFirst()).allSatisfy({ $0.0 > $0.1 }) { failures.append("\(side) \(label): sigmas are not strictly decreasing") }
        if zip(s, t).contains(where: { $0.0 * 1000 != $0.1 }) { failures.append("\(side) \(label): timesteps are not sigmas * 1000 in float32") }
        return failures
    }

    static func knownScheduleFailures(side: String, label: String, sigmas: MLXArray?, timesteps: MLXArray?) -> [String] {
        guard let sigmas, let timesteps else { return ["\(side) \(label): schedule missing"] }
        var failures: [String] = []
        if KleinCompare.floats(sigmas) != KleinTransformerCases.knownSigmas1024Steps4 { failures.append("\(side) \(label): sigmas differ from the production values measured for 1024x1024 / 4 steps") }
        if KleinCompare.floats(timesteps) != KleinTransformerCases.knownTimesteps1024Steps4 { failures.append("\(side) \(label): timesteps differ from the production values measured for 1024x1024 / 4 steps") }
        return failures
    }

    // MARK: Whole comparison

    struct Pins: Decodable {
        let casesSHA256: String, casesDigest: String, budgetsSHA256: String
        enum CodingKeys: String, CodingKey { case casesSHA256 = "cases_sha256", casesDigest = "cases_digest", budgetsSHA256 = "budgets_sha256" }
    }

    /// Compares a reference run, an independent native run and the native isolated run. Unreadable inputs throw; everything else is
    /// recorded in the report.
    public static func run(
        referenceDirectory: URL, nativeDirectory: URL, isolatedDirectory: URL, casesJSON: URL, budgetsJSON: URL, frozenJSON: URL,
        log: (String) -> Void = { _ in }
    ) throws -> KleinTransformerReport {
        let casesData = try Data(contentsOf: casesJSON)
        let casesSHA = KleinHashing.sha256(of: casesData)
        let declaredSpecs = try KleinTransformerCases.parse(casesData)
        // Always enumerate the compiled corpus, even when the supplied file has been shortened or altered.
        let specs = KleinTransformerCases.frozen
        let expectedIDs = specs.map(\.id)
        let budgetsSHA = try KleinHashing.sha256(of: budgetsJSON)
        let frozenData = try Data(contentsOf: frozenJSON)
        let frozen = try JSONDecoder().decode(Pins.self, from: frozenData)
        let manifestURLs = [referenceDirectory, nativeDirectory, isolatedDirectory].map { $0.appendingPathComponent("manifest.json") }
        let reference = try KleinJSON.read(KleinTransformerManifest.self, from: manifestURLs[0])
        let native = try KleinJSON.read(KleinTransformerManifest.self, from: manifestURLs[1])
        let isolated = try KleinJSON.read(KleinTransformerManifest.self, from: manifestURLs[2])
        let manifestSHAs = try manifestURLs.map { try KleinHashing.sha256(of: $0) }
        var checks: [KleinCheck] = []

        // Frozen inputs: the cases and budgets are the ones recorded before any real run, on every side.
        var frozenFailures = KleinTransformerCases.differences(declaredSpecs, from: specs)
        frozenFailures += KleinTransformerCases.structuralFailures(declaredSpecs)
        if casesSHA != frozen.casesSHA256 { frozenFailures.append("cases file SHA-256 \(casesSHA) is not the frozen \(frozen.casesSHA256)") }
        if KleinTransformerCases.digest(of: declaredSpecs) != frozen.casesDigest || KleinTransformerCases.compiledDigest != frozen.casesDigest {
            frozenFailures.append("cases digest differs from the frozen \(frozen.casesDigest)")
        }
        if budgetsSHA != frozen.budgetsSHA256 { frozenFailures.append("budgets file SHA-256 \(budgetsSHA) is not the frozen \(frozen.budgetsSHA256)") }
        frozenFailures += try KleinTransformerBudgets.check(budgetsJSON: budgetsJSON)
        for (label, side, manifest) in [("reference", "reference", reference), ("native", "native", native), ("isolated", "isolated", isolated)] {
            if manifest.schema != 1 { frozenFailures.append("\(label) manifest schema \(manifest.schema), expected 1") }
            if manifest.side != side { frozenFailures.append("\(label) manifest declares side \(manifest.side)") }
            if manifest.casesSHA256 != casesSHA { frozenFailures.append("\(label) ran a different cases file (\(manifest.casesSHA256))") }
            if manifest.casesDigest != frozen.casesDigest { frozenFailures.append("\(label) ran cases with digest \(manifest.casesDigest)") }
            if manifest.budgetsSHA256 != budgetsSHA { frozenFailures.append("\(label) ran with a different budgets file (\(manifest.budgetsSHA256))") }
        }
        checks.append(KleinCompare.check("frozen-inputs", frozenFailures))

        // Runtime provenance.
        var runtime = KleinCompare.referenceVersionFailures(reference.runtime)
        for key in ["python", "python_executable"] where (reference.runtime[key] ?? "").isEmpty { runtime.append("reference runtime did not record \(key)") }
        if reference.runtime["is_m1_or_m2"] != "false" { runtime.append("reference reports is_m1_or_m2=\(reference.runtime["is_m1_or_m2"] ?? "nil"); production compiles predict only off M1/M2, and that branch is not validated here") }
        if reference.runtime["production_predict_compiled"] != "true" { runtime.append("reference did not run the production compiled `_predict`") }
        let gotModules = reference.runtimeModules ?? [:]
        for (module, pin) in KleinTransformerPins.runtimeModules.sorted(by: { $0.key < $1.key }) where gotModules[module] != pin {
            runtime.append("\(module): runtime .pyc \(gotModules[module] ?? "not reported"), audited \(pin)")
        }
        for module in Set(gotModules.keys).subtracting(KleinTransformerPins.runtimeModules.keys).sorted() { runtime.append("unexpected module \(module)") }
        for (label, manifest) in [("native", native), ("isolated", isolated)] {
            if manifest.runtime["self_reported_mlx_swift_revision"] != KleinPins.mlxSwiftRevision {
                runtime.append("\(label) self-reported mlx-swift revision \(manifest.runtime["self_reported_mlx_swift_revision"] ?? "nil"), pinned \(KleinPins.mlxSwiftRevision)")
            }
            if !(manifest.runtime["self_reported_build_settings"] ?? "").contains("MTL_FAST_MATH=NO") {
                runtime.append("\(label) self-reported build settings \(manifest.runtime["self_reported_build_settings"] ?? "nil") do not record MTL_FAST_MATH=NO")
            }
            if !KleinCompare.isSHA256Hex(manifest.runtime["executable_sha256"]) { runtime.append("\(label) runtime did not record a well-formed executable_sha256") }
        }
        if native.runtime["executable_sha256"] != isolated.runtime["executable_sha256"] { runtime.append("native and isolated runs used different binaries") }
        if isolated.runtime["reference_manifest_sha256"] != manifestSHAs[0] { runtime.append("the isolated run did not read this reference run (manifest SHA-256 mismatch)") }
        checks.append(KleinCompare.check("runtime-provenance", runtime))

        var artifacts: [String] = []
        for (label, manifest) in [("reference", reference), ("native", native), ("isolated", isolated)] {
            if manifest.modelFilesBefore != KleinTransformerPins.modelFiles { artifacts.append("\(label): model files before the run differ from the pins") }
            if manifest.modelFilesAfter != KleinTransformerPins.modelFiles { artifacts.append("\(label): model files after the run differ from the pins") }
        }
        checks.append(KleinCompare.check("model-artifacts-unchanged", artifacts))

        // Loaded weight bytes: text (905) and transformer (387), MFLUX applier vs native loader, against the architecture.
        var weightFailures: [String] = []
        func compareWeights(_ label: String, _ layout: [String: (dtype: String, shape: [Int])], _ sides: [(String, [String: KleinTensorDigest])]) {
            for (side, digests) in sides where Set(digests.keys) != Set(layout.keys) {
                weightFailures.append("\(label) \(side): loaded tensor names differ from the \(layout.count) stored names: missing \(Set(layout.keys).subtracting(digests.keys).sorted().prefix(5)), unexpected \(Set(digests.keys).subtracting(layout.keys).sorted().prefix(5))")
            }
            for (name, want) in layout.sorted(by: { $0.key < $1.key }) {
                let found = sides.map { $0.1[name] }
                guard found.allSatisfy({ $0 != nil }) else { continue }
                if found.dropFirst().contains(where: { $0! != found[0]! }) { weightFailures.append("\(label) \(name): digests differ between \(sides.map(\.0)) (\(found.map { $0!.sha256.prefix(12) }))") }
                let d = found[0]!
                if d.shape != want.shape || d.dtype != KleinTextWeights.mlxName(of: want.dtype) { weightFailures.append("\(label) \(name): loaded \(d.dtype) \(d.shape), stored \(want.dtype) \(want.shape)") }
                let elementBytes = want.dtype == "U32" || want.dtype == "F32" ? 4 : 2
                if d.bytes != want.shape.reduce(1, *) * elementBytes || !KleinCompare.isSHA256Hex(d.sha256) {
                    weightFailures.append("\(label) \(name): invalid loaded byte count or SHA-256")
                }
            }
        }
        compareWeights("text", TextEncoderLayout.expectedTensors(), [("reference", reference.textWeights), ("native", native.textWeights)])
        compareWeights("transformer", KleinTransformerLayout.expectedTensors(), [("reference", reference.transformerWeights), ("native", native.transformerWeights), ("isolated", isolated.transformerWeights)])
        if !isolated.textWeights.isEmpty { weightFailures.append("the isolated run unexpectedly reports text weights") }
        checks.append(KleinCompare.check("loaded-weight-bytes", KleinCompare.capped(weightFailures)))

        var quant: [String] = []
        func compareQuant(_ label: String, expected: Set<String>, _ sides: [(String, [KleinQuantizedModule])]) {
            let sorted = sides.map { ($0.0, $0.1.sorted { $0.name < $1.name }) }
            for (side, modules) in sorted {
                if Set(modules.map(\.name)) != expected || modules.count != expected.count { quant.append("\(label) \(side): quantized modules are not the architecture's \(expected.count) matrices (\(modules.count) reported)") }
                if modules.contains(where: { $0.bits != 4 || $0.groupSize != 64 || $0.mode != "affine" }) { quant.append("\(label) \(side) reports a module that is not 4-bit/64/affine") }
            }
            if Set(sorted.map { $0.1 }.map { $0.map { "\($0.name)/\($0.bits)/\($0.groupSize)/\($0.mode)" } }).count > 1 { quant.append("\(label) quantized modules differ between \(sides.map(\.0))") }
        }
        compareQuant("text", expected: Set(KleinTextWeights.quantizedMatrices().map(\.name)), [("reference", reference.textQuantizedModules), ("native", native.textQuantizedModules)])
        compareQuant("transformer", expected: Set(KleinTransformerLayout.matrices().map(\.name)),
                     [("reference", reference.transformerQuantizedModules), ("native", native.transformerQuantizedModules), ("isolated", isolated.transformerQuantizedModules)])
        checks.append(KleinCompare.check("quantization", quant))

        // Case list and per-case parameters.
        var listing: [String] = []
        for (label, manifest) in [("reference", reference), ("native", native), ("isolated", isolated)] {
            listing += KleinCompare.caseListFailures(label: label, ids: manifest.cases.map(\.id), expected: expectedIDs)
            for spec in specs {
                guard let c = manifest.cases.first(where: { $0.id == spec.id }) else { continue }
                let grid = try? KleinLatent.grid(height: spec.height, width: spec.width)
                if Array(c.prompt.utf8) != Array(spec.prompt.utf8) || c.category != spec.category || c.width != spec.width || c.height != spec.height
                    || c.seed != Int(spec.seed) || c.steps != spec.steps || c.timestepIndex != spec.timestepIndex
                    || c.imageSeqLen != spec.imageSeqLen || c.gridHeight != grid?.height || c.gridWidth != grid?.width {
                    listing.append("\(label) case \(spec.id) ran other parameters than the frozen table")
                }
            }
        }
        for spec in specs {
            guard let r = reference.cases.first(where: { $0.id == spec.id }), let n = native.cases.first(where: { $0.id == spec.id }) else { continue }
            if r.renderedText != n.renderedText || r.numValid != n.numValid || r.untruncatedCount != n.untruncatedCount {
                listing.append("\(spec.id): tokenization disagrees (valid \(r.numValid)/\(n.numValid), untruncated \(r.untruncatedCount)/\(n.untruncatedCount))")
            }
            if spec.category == "long", r.untruncatedCount <= KleinTokenizer.maxLength { listing.append("\(spec.id): the long case does not exceed the 512-token limit (\(r.untruncatedCount))") }
        }
        checks.append(KleinCompare.check("case-list", listing))
        var complete = listing.isEmpty && declaredSpecs.map(\.id) == specs.map(\.id)

        // Controls: seeded latents, grid IDs, pack/unpack and the schedule for six sizes and eight step counts.
        var controlResults: [KleinTensorResult] = []
        var controlFailures: [String] = []
        let controlRefURL = referenceDirectory.appendingPathComponent("controls.safetensors"), controlNativeURL = nativeDirectory.appendingPathComponent("controls.safetensors")
        var referenceFiles: [String: String] = [:], nativeFiles: [String: String] = [:], isolatedFiles: [String: String] = [:]
        if FileManager.default.fileExists(atPath: controlRefURL.path), FileManager.default.fileExists(atPath: controlNativeURL.path) {
            let ref = try loadArrays(url: controlRefURL), nat = try loadArrays(url: controlNativeURL)
            referenceFiles["controls"] = try KleinHashing.sha256(of: controlRefURL)
            nativeFiles["controls"] = try KleinHashing.sha256(of: controlNativeURL)
            let required = KleinTransformerControls.requiredKeys()
            for (label, arrays, manifest) in [("reference", ref, reference), ("native", nat, native)] {
                if Set(arrays.keys) != Set(required) {
                    controlFailures.append("\(label) controls hold \(arrays.count) tensors; required \(required.count) (missing \(Set(required).subtracting(arrays.keys).sorted().prefix(3)), unexpected \(Set(arrays.keys).subtracting(required).sorted().prefix(3)))")
                    complete = false
                }
                if Set(manifest.controlTensors.keys) != Set(arrays.keys) {
                    controlFailures.append("\(label) controls recorded \(manifest.controlTensors.count) tensors but saved \(arrays.count)")
                    complete = false
                }
                for (name, info) in manifest.controlTensors {
                    if let a = arrays[name], KleinDType.name(a.dtype) != info.dtype || a.shape != info.shape {
                        controlFailures.append("\(label) control \(name): manifest dtype/shape differs from the saved tensor")
                    }
                }
            }
            for key in required {
                let layout = KleinTransformerControls.expectedLayout(key: key)
                let expect = layout.map { KleinTransformerTensors.Expect(dtype: $0.dtype, shape: $0.shape) }
                let r = compareTensor(key, native: nat[key], reference: ref[key], rule: .exact, expect: expect)
                controlResults.append(r)
                if !r.passed { controlFailures.append("\(key): \(r.failures.joined(separator: "; "))") }
            }
            for size in KleinTransformerCases.controlSizes {
                let k = KleinTransformerControls.key(size.width, size.height)
                for n in KleinTransformerCases.controlSteps {
                    for (side, arrays) in [("reference", ref), ("native", nat)] {
                        controlFailures += scheduleFailures(side: side, label: "\(k).n\(n)", sigmas: arrays["sigmas.\(k).n\(n)"], timesteps: arrays["timesteps.\(k).n\(n)"])
                    }
                }
            }
            for (side, arrays) in [("reference", ref), ("native", nat)] {
                controlFailures += knownScheduleFailures(side: side, label: "1024x1024.n4", sigmas: arrays["sigmas.1024x1024.n4"], timesteps: arrays["timesteps.1024x1024.n4"])
            }
        } else {
            controlFailures.append("controls.safetensors missing (reference \(FileManager.default.fileExists(atPath: controlRefURL.path)), native \(FileManager.default.fileExists(atPath: controlNativeURL.path)))")
            complete = false
        }
        checks.append(KleinCompare.check("controls", KleinCompare.capped(controlFailures)))

        // Per case.
        var results: [KleinTransformerCaseResult] = []
        for spec in specs {
            guard reference.cases.contains(where: { $0.id == spec.id }), native.cases.contains(where: { $0.id == spec.id }), isolated.cases.contains(where: { $0.id == spec.id }) else { continue }
            let result = try compareCase(
                spec: spec, reference: reference, native: native, isolated: isolated, referenceDirectory: referenceDirectory,
                nativeDirectory: nativeDirectory, isolatedDirectory: isolatedDirectory, referenceFiles: &referenceFiles, nativeFiles: &nativeFiles, isolatedFiles: &isolatedFiles)
            results.append(result)
            log("\(spec.id): \(result.passed ? "PASS" : "FAIL") \(result.failures.prefix(2))")
        }
        let incomplete = results.contains {
            $0.failures.contains { $0.hasPrefix("tensor file missing") || $0.contains("saved tensors differ from the required set") }
                || $0.tensors.contains { $0.failures.contains { $0.hasPrefix("missing") } }
                || $0.diagnostics.contains { $0.failures.contains { $0.hasPrefix("missing") } }
        }
        let verdict = KleinCompare.status(complete: complete && !incomplete, checks: checks, cases: results.map { KleinTransformerCompare.asSliceOneResult($0) }, expectedCount: expectedIDs.count)
        return KleinTransformerReport(
            status: verdict, passed: verdict == "COMPLETE_PASS", expectedCaseIDs: expectedIDs, casesSHA256: casesSHA, casesDigest: KleinTransformerCases.digest(of: declaredSpecs),
            budgetsSHA256: budgetsSHA, budgets: KleinTransformerBudgets.asReport, frozenInputsSHA256: KleinHashing.sha256(of: frozenData),
            referenceManifestSHA256: manifestSHAs[0], nativeManifestSHA256: manifestSHAs[1], isolatedManifestSHA256: manifestSHAs[2],
            referenceTensorFilesSHA256: referenceFiles, nativeTensorFilesSHA256: nativeFiles, isolatedTensorFilesSHA256: isolatedFiles,
            referenceRuntime: reference.runtime, nativeRuntime: native.runtime, isolatedRuntime: isolated.runtime,
            referenceRuntimeModules: reference.runtimeModules ?? [:], checks: checks, controls: controlResults, cases: results)
    }

    /// `KleinCompare.status` works on slice 1's case result type; only ids and verdicts matter to it.
    static func asSliceOneResult(_ r: KleinTransformerCaseResult) -> KleinCaseResult {
        KleinCaseResult(id: r.id, category: r.category, validTokens: 0, untruncatedTokens: 0, referenceEncoderOutcome: "ok", nativeEncoderOutcome: "ok", tensors: [], failures: r.failures, passed: r.passed)
    }

    static func compareCase(
        spec: KleinTransformerCase, reference: KleinTransformerManifest, native: KleinTransformerManifest, isolated: KleinTransformerManifest,
        referenceDirectory: URL, nativeDirectory: URL, isolatedDirectory: URL,
        referenceFiles: inout [String: String], nativeFiles: inout [String: String], isolatedFiles: inout [String: String]
    ) throws -> KleinTransformerCaseResult {
        var failures: [String] = []
        var result = KleinTransformerCaseResult(id: spec.id, category: spec.category, imageSeqLen: spec.imageSeqLen, tensors: [], diagnostics: [], failures: [], passed: false)
        let refURL = referenceDirectory.appendingPathComponent("\(spec.id).safetensors"), natURL = nativeDirectory.appendingPathComponent("\(spec.id).safetensors")
        let isoURL = isolatedDirectory.appendingPathComponent("\(spec.id).safetensors")
        let present = [refURL, natURL, isoURL].map { FileManager.default.fileExists(atPath: $0.path) }
        guard present.allSatisfy({ $0 }) else {
            result.failures = ["tensor file missing (reference \(present[0]), native \(present[1]), isolated \(present[2]))"]
            return result
        }
        let ref = try loadArrays(url: refURL), nat = try loadArrays(url: natURL), iso = try loadArrays(url: isoURL)
        referenceFiles[spec.id] = try KleinHashing.sha256(of: refURL)
        nativeFiles[spec.id] = try KleinHashing.sha256(of: natURL)
        isolatedFiles[spec.id] = try KleinHashing.sha256(of: isoURL)

        // The saved tensors must be exactly the required set and what each producer recorded.
        let primaryKeys = Set(KleinTransformerTensors.primaryKeys(spec)), isolatedKeys = Set(KleinTransformerTensors.isolatedKeys(spec))
        for (label, manifest, arrays, required) in [("reference", reference, ref, primaryKeys), ("native", native, nat, primaryKeys), ("isolated", isolated, iso, isolatedKeys)] {
            let saved = Set(arrays.keys)
            if saved != required { failures.append("\(label) saved tensors differ from the required set: missing \(required.subtracting(saved).sorted().prefix(5)), unexpected \(saved.subtracting(required).sorted().prefix(5))") }
            guard let entry = manifest.cases.first(where: { $0.id == spec.id }) else { continue }
            if Set(entry.tensors.keys) != saved { failures.append("\(label) recorded \(entry.tensors.count) tensors but saved \(saved.count)") }
            for (name, info) in entry.tensors {
                if let a = arrays[name], KleinDType.name(a.dtype) != info.dtype || a.shape != info.shape { failures.append("\(label) \(name): file holds \(KleinDType.name(a.dtype)) \(a.shape), manifest says \(info.dtype) \(info.shape)") }
            }
        }

        var tensors: [KleinTensorResult] = []
        var diagnostics: [KleinDiagnostic] = []
        let layout = KleinTransformerTensors.primaryLayout(spec)
        for key in KleinTransformerTensors.primaryKeys(spec) {
            switch primaryRule(key) {
            case .diagnostic: diagnostics.append(diagnostic("free-run \(key): native eager vs MFLUX eager", native: nat[key], reference: ref[key], expect: layout[key]))
            case let rule: tensors.append(compareTensor(key, native: nat[key], reference: ref[key], rule: rule, expect: layout[key]))
            }
        }
        // Compiled and eager cross-comparisons of the full prediction (the production compiled path decides above).
        diagnostics.append(diagnostic("noise: native eager vs MFLUX compiled", native: nat["noise_eager"], reference: ref["noise_compiled"], expect: layout["noise_compiled"]))
        diagnostics.append(diagnostic("noise: native compiled vs MFLUX eager", native: nat["noise_compiled"], reference: ref["noise_eager"], expect: layout["noise_eager"]))
        diagnostics.append(diagnostic("noise: MFLUX eager vs MFLUX compiled (reference drift)", native: ref["noise_eager"], reference: ref["noise_compiled"], expect: layout["noise_compiled"]))
        diagnostics.append(diagnostic("noise: native eager vs native compiled", native: nat["noise_eager"], reference: nat["noise_compiled"], expect: layout["noise_compiled"]))

        // Isolated components from the reference's captured inputs.
        let isoLayout = KleinTransformerTensors.isolatedLayout(spec)
        for key in KleinTransformerTensors.isolatedComponentKeys {
            tensors.append(compareTensor("iso.\(key)", native: iso["iso.\(key)"], reference: ref[key], rule: .limit(KleinTransformerBudgets.embedding), expect: isoLayout["iso.\(key)"]))
        }
        for key in KleinTransformerTensors.isolatedBlockKeys {
            tensors.append(compareTensor("iso.\(key)", native: iso["iso.\(key)"], reference: ref[key], rule: .limit(KleinTransformerBudgets.singleBlock), expect: isoLayout["iso.\(key)"]))
        }
        for t in 0..<spec.steps {
            tensors.append(compareTensor("iso.step.\(t)", native: iso["iso.step.\(t)"], reference: ref["chain.\(t + 1)"], rule: .exact, expect: isoLayout["iso.step.\(t)"]))
        }
        diagnostics.append(diagnostic("isolated full forward from reference inputs: native eager vs MFLUX eager", native: iso["iso.noise_eager"], reference: ref["noise_eager"], expect: isoLayout["iso.noise_eager"]))
        diagnostics.append(diagnostic("isolated full forward from reference inputs: native compiled vs MFLUX compiled", native: iso["iso.noise_compiled"], reference: ref["noise_compiled"], expect: isoLayout["iso.noise_compiled"]))
        diagnostics.append(diagnostic("isolated full forward from reference inputs: native eager vs MFLUX compiled", native: iso["iso.noise_eager"], reference: ref["noise_compiled"], expect: isoLayout["iso.noise_eager"]))

        // Each side's own schedule is checked independently; the production values are pinned for 1024x1024 / 4 steps.
        for (side, arrays) in [("reference", ref), ("native", nat)] {
            failures += scheduleFailures(side: side, label: spec.id, sigmas: arrays["sigmas"], timesteps: arrays["timesteps"])
            if spec.width == 1024, spec.height == 1024, spec.steps == 4 { failures += knownScheduleFailures(side: side, label: spec.id, sigmas: arrays["sigmas"], timesteps: arrays["timesteps"]) }
        }
        failures += KleinCompare.textIDsFailures(side: "reference", ids: ref["text_ids"]) + KleinCompare.textIDsFailures(side: "native", ids: nat["text_ids"])
        for t in tensors where !t.passed { failures.append("\(t.name): \(t.failures.joined(separator: "; "))") }
        for d in diagnostics where !d.failures.isEmpty { failures.append("diagnostic \(d.name): \(d.failures.joined(separator: "; "))") }
        result.tensors = tensors
        result.diagnostics = diagnostics
        result.failures = failures
        result.passed = failures.isEmpty
        return result
    }
}
