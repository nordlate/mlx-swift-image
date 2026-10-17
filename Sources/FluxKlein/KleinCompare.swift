import Foundation
import MLX
import ZImage

public struct KleinTensorResult: Codable, Sendable {
    public var name: String
    public var dtype: String?
    public var shape: [Int]?
    /// Set for exactly-compared tensors (token IDs, masks, text IDs).
    public var exact: Bool?
    /// Set for numerically compared tensors.
    public var metrics: ComparisonMetrics?
    public var nonFiniteCount: Int
    public var failures: [String]
    public var passed: Bool
    /// Elements compared numerically; 0 means only presence, dtype and shape were verified (the empty prompt's zero-length tensors).
    public var elementCount: Int = 0
}

public struct KleinCaseResult: Codable, Sendable {
    public var id: String
    public var category: String
    public var validTokens: Int
    public var untruncatedTokens: Int
    public var referenceEncoderOutcome: String
    public var nativeEncoderOutcome: String
    public var tensors: [KleinTensorResult]
    public var failures: [String]
    public var passed: Bool
}

public struct KleinCheck: Codable, Sendable {
    public var name: String
    public var failures: [String]
    public var passed: Bool
}

public struct KleinReport: Codable, Sendable {
    /// `COMPLETE_PASS`, `COMPLETE_FAIL` (everything was compared and something differs) or `INCOMPLETE` (a case, tensor or file
    /// the comparison requires is missing, duplicated or unreadable). Only `COMPLETE_PASS` is a pass.
    public var status: String
    public var passed: Bool
    public var expectedCaseIDs: [String]
    public var casesSHA256: String
    public var budgetsSHA256: String
    public var budgets: [String: [String: Double]]
    public var referenceManifestSHA256: String
    public var nativeManifestSHA256: String
    public var referenceTensorFilesSHA256: [String: String]
    public var nativeTensorFilesSHA256: [String: String]
    public var referenceRuntime: [String: String]
    public var nativeRuntime: [String: String]
    public var referenceRuntimeModules: [String: String]
    public var checks: [KleinCheck]
    public var cases: [KleinCaseResult]
}

public enum KleinCompare {
    // MARK: Tensors

    // mlx-swift's byte accessors build an unsafe buffer pointer that is invalid for zero-element arrays, so every conversion is
    // guarded: a zero-element array has no values and converts to nothing, after (not instead of) the dtype/shape checks.
    static func floats(_ array: MLXArray) -> [Float] { array.size == 0 ? [] : array.asType(.float32).asArray(Float.self) }

    static func bytes(_ array: MLXArray) -> Data { array.size == 0 ? Data() : array.asData(access: .copy).data }

    /// Compares one tensor numerically. Missing tensors, dtype or shape disagreement (between the sides or with the architecture)
    /// and any non-finite value on either side fail closed; metrics are computed only when both sides are well formed.
    public static func floatTensor(
        name: String, native: MLXArray?, reference: MLXArray?, limit: KleinBudgets.Limit, expected: (dtype: String, shape: [Int])?
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
        if let expected {
            for (side, array) in [("native", native), ("reference", reference)] where KleinDType.name(array.dtype) != expected.dtype || array.shape != expected.shape {
                r.failures.append("\(side) is \(KleinDType.name(array.dtype)) \(array.shape), expected \(expected.dtype) \(expected.shape)")
            }
        }
        guard r.failures.isEmpty else { return r }
        r.elementCount = native.size
        if r.elementCount == 0 {
            // Same dtype, same (zero-element) shape, as the architecture expects: nothing numerical to compare, and none is claimed.
            r.passed = true
            return r
        }
        let n = floats(native), f = floats(reference)
        r.nonFiniteCount = n.filter { !$0.isFinite }.count + f.filter { !$0.isFinite }.count
        guard r.nonFiniteCount == 0 else {
            r.failures.append("\(r.nonFiniteCount) non-finite values (native \(n.filter { !$0.isFinite }.count), reference \(f.filter { !$0.isFinite }.count))")
            return r
        }
        let metrics = ParityMetrics.compare(native: n, reference: f, limits: limit.parityLimits)
        r.metrics = metrics
        if !metrics.passed {
            r.failures.append("over budget: relativeL2 \(metrics.relativeL2.map { "\($0)" } ?? "n/a") (limit \(limit.relativeL2)), normalizedMaxAbs \(metrics.normalizedMaxAbs.map { "\($0)" } ?? "n/a") (limit \(limit.normalizedMaxAbs)), maxAbsDiff \(metrics.maxAbsDiff)")
        }
        r.passed = r.failures.isEmpty
        return r
    }

    /// Exact comparison: same dtype, shape and bytes.
    public static func exactTensor(
        name: String, native: MLXArray?, reference: MLXArray?, expected: (dtype: String, shape: [Int])?
    ) -> KleinTensorResult {
        var r = KleinTensorResult(name: name, dtype: nil, shape: nil, exact: false, metrics: nil, nonFiniteCount: 0, failures: [], passed: false)
        guard let native, let reference else {
            r.failures.append("missing: native \(native == nil ? "absent" : "present"), reference \(reference == nil ? "absent" : "present")")
            return r
        }
        r.dtype = KleinDType.name(native.dtype)
        r.shape = native.shape
        if native.dtype != reference.dtype { r.failures.append("dtype: native \(native.dtype) reference \(reference.dtype)") }
        if native.shape != reference.shape { r.failures.append("shape: native \(native.shape) reference \(reference.shape)") }
        if let expected {
            for (side, array) in [("native", native), ("reference", reference)] where KleinDType.name(array.dtype) != expected.dtype || array.shape != expected.shape {
                r.failures.append("\(side) is \(KleinDType.name(array.dtype)) \(array.shape), expected \(expected.dtype) \(expected.shape)")
            }
        }
        if r.failures.isEmpty {
            r.exact = bytes(native) == bytes(reference)
            if r.exact != true {
                let a = native.size == 0 ? [] : native.asType(.int64).asArray(Int64.self)
                let b = reference.size == 0 ? [] : reference.asType(.int64).asArray(Int64.self)
                let first = zip(a, b).enumerated().first { $0.element.0 != $0.element.1 }?.offset
                r.failures.append("values differ" + (first.map { " (first at flat index \($0): native \(a[$0]) reference \(b[$0]))" } ?? ""))
            }
        }
        r.passed = r.failures.isEmpty
        return r
    }

    /// `prompt_embeds` must be exactly the three selected states side by side, layer-major per token. Checked on each side
    /// independently, so a wrong assembly layout cannot hide behind the cross-side comparison of the states.
    public static func assemblyFailures(side: String, arrays: [String: MLXArray]) -> [String] {
        guard let embeds = arrays["prompt_embeds"] else { return ["\(side): prompt_embeds missing"] }
        var failures: [String] = []
        for (k, key) in ["hs9", "hs18", "hs27"].enumerated() {
            guard let state = arrays[key] else { failures.append("\(side): \(key) missing"); continue }
            let hidden = state.dim(-1)
            guard embeds.ndim == 3, state.ndim == 3, embeds.dim(-1) == 3 * hidden, embeds.dim(0) == state.dim(0), embeds.dim(1) == state.dim(1) else {
                failures.append("\(side): prompt_embeds \(embeds.shape) is not three \(key) \(state.shape) wide")
                continue
            }
            let slice = embeds[0..., 0..., (k * hidden) ..< ((k + 1) * hidden)]
            if embeds.dtype != state.dtype || bytes(slice) != bytes(state) { failures.append("\(side): prompt_embeds slice \(k) is not \(key)") }
        }
        return failures
    }

    // MARK: Case list and verdicts

    /// The manifests must list exactly the expected case IDs, in order, once each.
    public static func caseListFailures(label: String, ids: [String], expected: [String]) -> [String] {
        var failures: [String] = []
        let duplicates = Dictionary(grouping: ids, by: { $0 }).filter { $0.value.count > 1 }.keys.sorted()
        if !duplicates.isEmpty { failures.append("\(label): duplicate case ids \(duplicates)") }
        let missing = Set(expected).subtracting(ids).sorted(), extra = Set(ids).subtracting(expected).sorted()
        if !missing.isEmpty { failures.append("\(label): missing cases \(missing)") }
        if !extra.isEmpty { failures.append("\(label): unexpected cases \(extra)") }
        if missing.isEmpty, extra.isEmpty, duplicates.isEmpty, ids != expected { failures.append("\(label): cases are not in cases.json order") }
        return failures
    }

    public static func coverageFailures(categories: Set<String>, untruncatedCounts: [Int]) -> [String] {
        var failures: [String] = []
        let missing = KleinCases.requiredCategories.subtracting(categories).sorted()
        if !missing.isEmpty { failures.append("categories not covered: \(missing)") }
        let counts = Set(untruncatedCounts)
        let absent = KleinCases.requiredBoundaryCounts.subtracting(counts).sorted()
        if !absent.isEmpty { failures.append("no case renders to \(absent) tokens (limit boundary 512)") }
        return failures
    }

    /// `COMPLETE_PASS` only if the case set is complete and nothing failed anywhere; incomplete runs are never a pass.
    public static func status(complete: Bool, checks: [KleinCheck], cases: [KleinCaseResult], expectedCount: Int) -> String {
        guard complete, cases.count == expectedCount, Set(cases.map(\.id)).count == expectedCount else { return "INCOMPLETE" }
        return checks.allSatisfy(\.passed) && cases.allSatisfy(\.passed) ? "COMPLETE_PASS" : "COMPLETE_FAIL"
    }

    /// text_ids rows are `(0, 0, 0, position)` over the padded length (the arange of the sequence, never of the valid count); a
    /// zero-length sequence has no rows. Checked on each side independently of the cross-side comparison.
    public static func textIDsFailures(side: String, ids: MLXArray?) -> [String] {
        guard let ids else { return ["\(side): text_ids missing"] }
        guard ids.ndim == 3, ids.dim(0) == 1, ids.dim(2) == 4, ids.dtype == .int32 else {
            return ["\(side): text_ids is \(ids.dtype) \(ids.shape), expected int32 [1, S, 4]"]
        }
        guard ids.size > 0 else { return [] }  // zero rows (the empty prompt): shape and dtype were validated above
        let flat = ids.asArray(Int32.self)
        for position in 0..<ids.dim(1) {
            let base = position * 4
            let expectedRow: [Int32] = [0, 0, 0, Int32(position)]
            if Array(flat[base..<(base + 4)]) != expectedRow { return ["\(side): text_ids row \(position) is not (0, 0, 0, position)"] }
        }
        return []
    }

    /// A case whose tensor file or tensor is missing on either side was not compared at all.
    static func isIncomplete(_ result: KleinCaseResult) -> Bool {
        result.failures.contains { $0.hasPrefix("tensor file missing") } || result.tensors.contains { $0.failures.contains { $0.hasPrefix("missing") } }
    }

    static func check(_ name: String, _ failures: [String]) -> KleinCheck {
        KleinCheck(name: name, failures: failures, passed: failures.isEmpty)
    }

    static func capped(_ failures: [String], limit: Int = 12) -> [String] {
        failures.count <= limit ? failures : Array(failures.prefix(limit)) + ["... and \(failures.count - limit) more"]
    }

    // MARK: Pins

    static func isSHA256Hex(_ value: String?) -> Bool {
        guard let value, value.count == 64 else { return false }
        return value.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }

    /// The corpus must be the frozen one, whatever both sides ran.
    static func frozenCorpusFailures(casesSHA256: String, caseCount: Int) -> [String] {
        var failures: [String] = []
        if casesSHA256 != KleinPins.frozenCasesSHA256 { failures.append("cases file SHA-256 \(casesSHA256) is not the frozen \(KleinPins.frozenCasesSHA256)") }
        if caseCount != KleinPins.frozenCaseCount { failures.append("cases file has \(caseCount) cases, frozen count is \(KleinPins.frozenCaseCount)") }
        return failures
    }

    /// The reference must be the pinned production MFLUX and MLX versions, exactly.
    static func referenceVersionFailures(_ runtime: [String: String]) -> [String] {
        var failures: [String] = []
        if runtime["mflux_version"] != KleinPins.referenceMfluxVersion { failures.append("reference mflux version \(runtime["mflux_version"] ?? "nil"), pinned \(KleinPins.referenceMfluxVersion)") }
        if runtime["mlx_version"] != KleinPins.referenceMLXVersion { failures.append("reference mlx version \(runtime["mlx_version"] ?? "nil"), pinned \(KleinPins.referenceMLXVersion)") }
        return failures
    }

    // MARK: Whole comparison

    /// Compares a reference run and a native run. Unreadable inputs throw; everything else is recorded in the report.
    public static func run(
        referenceDirectory: URL, nativeDirectory: URL, casesJSON: URL, budgetsJSON: URL, log: (String) -> Void = { _ in }
    ) throws -> KleinReport {
        let (specs, casesSHA) = try KleinCases.load(url: casesJSON)
        let expectedIDs = specs.map(\.id)
        let referenceManifestURL = referenceDirectory.appendingPathComponent("manifest.json")
        let nativeManifestURL = nativeDirectory.appendingPathComponent("manifest.json")
        let reference = try KleinJSON.read(KleinManifest.self, from: referenceManifestURL)
        let native = try KleinJSON.read(KleinManifest.self, from: nativeManifestURL)
        let budgetsData = try Data(contentsOf: budgetsJSON)
        var checks: [KleinCheck] = []

        // Provenance and identity.
        var sides: [String] = []
        if reference.side != "reference" { sides.append("reference manifest declares side \(reference.side)") }
        if native.side != "native" { sides.append("native manifest declares side \(native.side)") }
        if reference.casesSHA256 != casesSHA { sides.append("reference ran a different cases file (\(reference.casesSHA256))") }
        if native.casesSHA256 != casesSHA { sides.append("native ran a different cases file (\(native.casesSHA256))") }
        sides += frozenCorpusFailures(casesSHA256: casesSHA, caseCount: specs.count)
        let budgetFailures = try KleinBudgets.check(budgetsJSON: budgetsJSON)
        checks.append(check("frozen-inputs", sides + budgetFailures))

        var runtime = referenceVersionFailures(reference.runtime)
        for key in ["python", "python_executable"] where (reference.runtime[key] ?? "").isEmpty {
            runtime.append("reference runtime did not record \(key)")
        }
        if reference.runtimeModules != KleinPins.runtimeModules {
            let got = reference.runtimeModules ?? [:]
            for (module, pin) in KleinPins.runtimeModules.sorted(by: { $0.key < $1.key }) where got[module] != pin {
                runtime.append("\(module): runtime .pyc \(got[module] ?? "not reported"), audited \(pin)")
            }
            for module in Set(got.keys).subtracting(KleinPins.runtimeModules.keys).sorted() { runtime.append("unexpected module \(module)") }
        }
        // The native manifest's revision and build settings are SELF-REPORTED by the binary: a consistency check, not a build
        // attestation. The binary's SHA-256 is recorded for binding to the build log and dependency verification outside this
        // program; only its presence and shape are checked here.
        if native.runtime["self_reported_mlx_swift_revision"] != KleinPins.mlxSwiftRevision {
            runtime.append("native self-reported mlx-swift revision \(native.runtime["self_reported_mlx_swift_revision"] ?? "nil"), pinned \(KleinPins.mlxSwiftRevision)")
        }
        if !(native.runtime["self_reported_build_settings"] ?? "").contains("MTL_FAST_MATH=NO") {
            runtime.append("native self-reported build settings \(native.runtime["self_reported_build_settings"] ?? "nil") do not record MTL_FAST_MATH=NO")
        }
        if !isSHA256Hex(native.runtime["executable_sha256"]) { runtime.append("native runtime did not record a well-formed executable_sha256") }
        checks.append(check("runtime-provenance", runtime))

        var artifacts: [String] = []
        for (label, manifest) in [("reference", reference), ("native", native)] {
            if manifest.modelFilesBefore != KleinPins.modelFiles { artifacts.append("\(label): model files before the run differ from the pins") }
            if manifest.modelFilesAfter != KleinPins.modelFiles { artifacts.append("\(label): model files after the run differ from the pins") }
        }
        checks.append(check("model-artifacts-unchanged", artifacts))

        let layout = TextEncoderLayout.expectedTensors()
        var weightFailures: [String] = []
        for (label, manifest) in [("reference", reference), ("native", native)] where Set(manifest.weights.keys) != Set(layout.keys) {
            weightFailures.append("\(label) loaded tensor names differ from the 905 stored names: missing \(Set(layout.keys).subtracting(manifest.weights.keys).sorted().prefix(5)), unexpected \(Set(manifest.weights.keys).subtracting(layout.keys).sorted().prefix(5))")
        }
        for (name, want) in layout.sorted(by: { $0.key < $1.key }) {
            guard let r = reference.weights[name], let n = native.weights[name] else { continue }
            if r != n { weightFailures.append("\(name): reference \(r.dtype) \(r.shape) \(r.sha256.prefix(12)), native \(n.dtype) \(n.shape) \(n.sha256.prefix(12))") }
            if n.shape != want.shape || n.dtype != KleinTextWeights.mlxName(of: want.dtype) { weightFailures.append("\(name): loaded \(n.dtype) \(n.shape), stored \(want.dtype) \(want.shape)") }
        }
        checks.append(check("loaded-weight-bytes", capped(weightFailures)))

        var quant: [String] = []
        let sortedRef = reference.quantizedModules.sorted { $0.name < $1.name }, sortedNative = native.quantizedModules.sorted { $0.name < $1.name }
        if sortedRef != sortedNative { quant.append("quantized modules differ between MFLUX and native") }
        let expectedModules = Set(KleinTextWeights.quantizedMatrices().map { $0.name })
        if Set(sortedRef.map(\.name)) != expectedModules || sortedRef.count != expectedModules.count {
            quant.append("reference quantized modules are not the architecture's \(expectedModules.count) matrices (\(sortedRef.count) reported)")
        }
        if sortedRef.contains(where: { $0.bits != 4 || $0.groupSize != 64 || $0.mode != "affine" }) { quant.append("reference reports a module that is not 4-bit/64/affine") }
        checks.append(check("quantization", quant))

        // Case list, coverage and agreement.
        var listing = caseListFailures(label: "reference", ids: reference.cases.map(\.id), expected: expectedIDs)
        listing += caseListFailures(label: "native", ids: native.cases.map(\.id), expected: expectedIDs)
        listing += coverageFailures(categories: Set(reference.cases.map(\.category)), untruncatedCounts: reference.cases.map(\.untruncatedCount))
        for spec in specs {
            for (label, manifest) in [("reference", reference), ("native", native)] {
                if let c = manifest.cases.first(where: { $0.id == spec.id }), c.prompt != spec.prompt || c.category != spec.category {
                    listing.append("\(label) case \(spec.id) ran a different prompt or category than cases.json")
                }
            }
        }
        checks.append(check("case-list", listing))
        let complete = listing.isEmpty

        // Per case.
        var results: [KleinCaseResult] = []
        var referenceFiles: [String: String] = [:], nativeFiles: [String: String] = [:]
        for spec in specs {
            guard let rc = reference.cases.first(where: { $0.id == spec.id }), let nc = native.cases.first(where: { $0.id == spec.id }) else { continue }
            let result = try compareCase(
                spec: spec, reference: rc, native: nc, referenceDirectory: referenceDirectory, nativeDirectory: nativeDirectory,
                referenceFiles: &referenceFiles, nativeFiles: &nativeFiles)
            results.append(result)
            log("\(spec.id): \(result.passed ? "PASS" : "FAIL") \(result.failures.prefix(2))")
        }

        let verdict = status(complete: complete && !results.contains(where: isIncomplete), checks: checks, cases: results, expectedCount: expectedIDs.count)
        let referenceManifestSHA = try KleinHashing.sha256(of: referenceManifestURL)
        let nativeManifestSHA = try KleinHashing.sha256(of: nativeManifestURL)
        return KleinReport(
            status: verdict, passed: verdict == "COMPLETE_PASS", expectedCaseIDs: expectedIDs, casesSHA256: casesSHA,
            budgetsSHA256: KleinHashing.sha256(of: budgetsData), budgets: KleinBudgets.asReport,
            referenceManifestSHA256: referenceManifestSHA, nativeManifestSHA256: nativeManifestSHA,
            referenceTensorFilesSHA256: referenceFiles, nativeTensorFilesSHA256: nativeFiles, referenceRuntime: reference.runtime,
            nativeRuntime: native.runtime, referenceRuntimeModules: reference.runtimeModules ?? [:], checks: checks, cases: results)
    }

    static func compareCase(
        spec: KleinCaseSpec, reference rc: KleinManifestCase, native nc: KleinManifestCase, referenceDirectory: URL, nativeDirectory: URL,
        referenceFiles: inout [String: String], nativeFiles: inout [String: String]
    ) throws -> KleinCaseResult {
        var failures: [String] = []
        func agree<T: Equatable>(_ what: String, _ a: T, _ b: T) { if a != b { failures.append("\(what): reference \(a), native \(b)") } }
        agree("rendered chat text", rc.renderedText, nc.renderedText)
        agree("valid token count", rc.numValid, nc.numValid)
        agree("untruncated token count", rc.untruncatedCount, nc.untruncatedCount)
        agree("input_ids dtype", rc.idsDtype, nc.idsDtype)
        agree("input_ids shape", rc.idsShape, nc.idsShape)
        agree("attention_mask dtype", rc.maskDtype, nc.maskDtype)
        agree("attention_mask shape", rc.maskShape, nc.maskShape)

        let refURL = referenceDirectory.appendingPathComponent("\(spec.id).safetensors")
        let nativeURL = nativeDirectory.appendingPathComponent("\(spec.id).safetensors")
        guard FileManager.default.fileExists(atPath: refURL.path), FileManager.default.fileExists(atPath: nativeURL.path) else {
            failures.append("tensor file missing (reference \(FileManager.default.fileExists(atPath: refURL.path)), native \(FileManager.default.fileExists(atPath: nativeURL.path)))")
            return KleinCaseResult(
                id: spec.id, category: spec.category, validTokens: nc.numValid, untruncatedTokens: nc.untruncatedCount,
                referenceEncoderOutcome: rc.encoderOutcome, nativeEncoderOutcome: nc.encoderOutcome, tensors: [], failures: failures, passed: false)
        }
        let ref = try loadArrays(url: refURL), nat = try loadArrays(url: nativeURL)
        referenceFiles[spec.id] = try KleinHashing.sha256(of: refURL)
        nativeFiles[spec.id] = try KleinHashing.sha256(of: nativeURL)
        // A producer's recorded dtype/shape must be what it actually saved.
        for (label, manifestCase, arrays) in [("reference", rc, ref), ("native", nc, nat)] {
            let recorded = Set(manifestCase.tensors.keys), saved = Set(arrays.keys)
            if recorded != saved { failures.append("\(label) saved tensors \(saved.sorted()) but recorded \(recorded.sorted())") }
            for (name, info) in manifestCase.tensors {
                if let a = arrays[name], KleinDType.name(a.dtype) != info.dtype || a.shape != info.shape {
                    failures.append("\(label) \(name): file holds \(KleinDType.name(a.dtype)) \(a.shape), manifest says \(info.dtype) \(info.shape)")
                }
            }
        }

        // Every case, empty or not, is held to the same set of tensors and outcome. The empty prompt is a `[1, 0]` int32 batch that
        // production MFLUX encodes into zero-length tensors; its layout is expected, never skipped. Zero-element tensors carry no values,
        // so for that case only the outcome, presence, dtypes and shapes are verified (the report records elementCount 0).
        var tensors: [KleinTensorResult] = []
        let layout = KleinBudgets.expectedLayout(empty: spec.prompt.isEmpty)
        if rc.encoderOutcome != "ok" || nc.encoderOutcome != "ok" {
            failures.append("encoder outcome: reference \(rc.encoderOutcome), native \(nc.encoderOutcome)")
        }
        for name in KleinBudgets.exactTensors {
            tensors.append(exactTensor(name: name, native: nat[name], reference: ref[name], expected: layout[name]))
        }
        for name in KleinBudgets.tensors.keys.sorted() {
            tensors.append(floatTensor(name: name, native: nat[name], reference: ref[name], limit: KleinBudgets.tensors[name]!, expected: layout[name]))
        }
        failures += assemblyFailures(side: "reference", arrays: ref) + assemblyFailures(side: "native", arrays: nat)
        if let mask = ref["attention_mask"], (mask.size == 0 ? 0 : mask.asType(.int64).sum().item(Int.self)) != rc.numValid {
            failures.append("reference num_valid \(rc.numValid) disagrees with its attention mask")
        }
        failures += textIDsFailures(side: "reference", ids: ref["text_ids"]) + textIDsFailures(side: "native", ids: nat["text_ids"])
        for t in tensors where !t.passed { failures.append("\(t.name): \(t.failures.joined(separator: "; "))") }
        return KleinCaseResult(
            id: spec.id, category: spec.category, validTokens: nc.numValid, untruncatedTokens: nc.untruncatedCount,
            referenceEncoderOutcome: rc.encoderOutcome, nativeEncoderOutcome: nc.encoderOutcome, tensors: tensors, failures: failures,
            passed: failures.isEmpty)
    }
}

extension KleinTextWeights {
    /// MLX Python's name for a safetensors dtype string of the layout.
    static func mlxName(of safetensorsDType: String) -> String {
        switch safetensorsDType {
        case "BF16": "mlx.core.bfloat16"
        case "U32": "mlx.core.uint32"
        case "F32": "mlx.core.float32"
        default: safetensorsDType
        }
    }
}
