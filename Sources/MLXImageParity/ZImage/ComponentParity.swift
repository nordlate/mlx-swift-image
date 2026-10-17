import ZImage
import Foundation
import MLX
import MLXImageCore

// Slice 2 parity machinery: one independently runnable check per component (seeded latents, scheduler, q4 transformer
// weights, patch/timestep embeddings, first block). Each reads only its own reference directory (written by its own
// exporter in reference/) and, where it needs them, the model weights. No check consumes another check's output, and no
// native component feeds another's input: every comparison starts from the reference's own isolated inputs.

public struct CheckResult: Codable, Sendable {
    public var name: String
    /// `exact` (bitwise), `relative-l2` or `pixels` (a predeclared limit decides) or `diagnostic` (reported, never decides).
    public var kind: String
    public var decisive: Bool
    public var passed: Bool
    public var metrics: ComparisonMetrics?
    /// Set on decoded-pixel rows (uint8 levels), which are judged by max and mean abs difference instead of relative L2.
    public var pixels: PixelMetrics?
    public var detail: String?
}

/// Decoded-pixel comparison in uint8 levels, judged by the predeclared max abs and mean abs limits.
public struct PixelMetrics: Codable, Equatable, Sendable {
    public var maxAbsDiff: Int
    public var meanAbsDiff: Double
    public var differingValues: Int
    public var valueCount: Int
    public var passed: Bool

    public static func compare(native: [UInt8], reference: [UInt8]) -> PixelMetrics {
        precondition(native.count == reference.count)
        var maxDiff = 0, differing = 0, sum = 0
        for (n, r) in zip(native, reference) {
            let d = abs(Int(n) - Int(r))
            if d != 0 { differing += 1 }
            maxDiff = max(maxDiff, d)
            sum += d
        }
        let mean = native.isEmpty ? 0 : Double(sum) / Double(native.count)
        return PixelMetrics(
            maxAbsDiff: maxDiff, meanAbsDiff: mean, differingValues: differing, valueCount: native.count,
            passed: Double(maxDiff) <= ParityTolerance.pixelMaxAbsLevels && mean <= ParityTolerance.pixelMeanAbsLevels)
    }
}

public struct ComponentCaseResult: Codable, Sendable {
    public var id: String
    public var checks: [CheckResult]
    /// True when at least one decisive check exists and every decisive check passed.
    public var passed: Bool

    init(id: String, checks: [CheckResult]) {
        self.id = id
        self.checks = checks
        let decisive = checks.filter(\.decisive)
        passed = !decisive.isEmpty && decisive.allSatisfy(\.passed)
    }
}

public struct ComponentReport: Codable, Sendable {
    public var component: String
    public var tolerances: [String: Double]
    public var mlxSwiftRevision: String
    public var referenceMLXPythonVersion: String?
    public var cases: [ComponentCaseResult]
    public var passed: Bool
}

/// The explicit case lists of `reference/component_cases.json`, shared with the Python exporters.
struct ComponentCases: Decodable {
    struct Latent: Decodable { let id: String; let seed: UInt64; let width: Int; let height: Int }
    struct Scheduler: Decodable { let id: String; let steps: Int; let width: Int; let height: Int }
    struct Chain: Decodable { let id: String; let seed: UInt64; let steps: Int; let width: Int; let height: Int }
    struct Timestep: Decodable { let id: String; let steps: Int; let width: Int; let height: Int; let step: Int }
    struct Scene: Decodable { let id: String; let seed: UInt64; let width: Int; let height: Int; let steps: Int; let step: Int; let prompt: String }
    let latentCases: [Latent]
    let schedulerCases: [Scheduler]
    let schedulerChains: [Chain]
    let timestepCases: [Timestep]
    let embeddingCases: [Scene]
    let blockCases: [Scene]
    let blockMiddleCases: [Scene]
    let blockLastCases: [Scene]
    let finalCases: [Scene]
    let outputCases: [Scene]

    enum CodingKeys: String, CodingKey {
        case latentCases = "latent_cases", schedulerCases = "scheduler_cases", schedulerChains = "scheduler_chains"
        case timestepCases = "timestep_cases", embeddingCases = "embedding_cases", blockCases = "block_cases"
        case blockMiddleCases = "block_middle_cases", blockLastCases = "block_last_cases"
        case finalCases = "final_cases", outputCases = "output_cases"
    }

    static func load(_ url: URL) throws -> ComponentCases { try JSONDecoder().decode(ComponentCases.self, from: Data(contentsOf: url)) }
}

/// What every reference manifest carries that the checks need.
struct ReferenceInfo: Decodable {
    struct IDOnly: Decodable { let id: String }
    let component: String
    let mlxPythonVersion: String?
    let cases: [IDOnly]?
    enum CodingKeys: String, CodingKey { case component, mlxPythonVersion = "mlx_python_version", cases }

    static func load(directory: URL, component: String) throws -> ReferenceInfo {
        let info = try JSONDecoder().decode(ReferenceInfo.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        guard info.component == component else {
            throw ComponentParityError("reference directory holds the '\(info.component)' reference, not '\(component)'")
        }
        return info
    }
}

public struct ComponentParityError: Error, CustomStringConvertible {
    public let description: String
    init(_ message: String) { description = message }
}

public enum ComponentCheck {
    /// Bitwise comparison: same shape, dtype and bytes.
    static func exact(_ name: String, native: MLXArray, reference: MLXArray) -> CheckResult {
        var result = CheckResult(name: name, kind: "exact", decisive: true, passed: false, metrics: nil, detail: nil)
        if let problem = layoutProblem(native: native, reference: reference) { result.detail = problem; return result }
        if bytesEqual(native, reference) { result.passed = true; return result }
        let (n, r) = (native.asType(.float32).asArray(Float.self), reference.asType(.float32).asArray(Float.self))
        let differing = zip(n, r).enumerated().filter { $0.element.0.bitPattern != $0.element.1.bitPattern }
        let first = differing.first.map { "first at \($0.offset): native \($0.element.0) reference \($0.element.1)" } ?? "bytes differ"
        result.detail = "\(differing.count) of \(n.count) elements differ; \(first)"
        return result
    }

    /// A predeclared relative-L2 limit decides (`limits`); shape or dtype differences fail. `decisive: false` records the
    /// same metrics as a diagnostic that cannot fail a case.
    static func tolerance(_ name: String, native: MLXArray, reference: MLXArray, limits: ParityLimits, decisive: Bool = true) -> CheckResult {
        var result = CheckResult(name: name, kind: decisive ? "relative-l2" : "diagnostic", decisive: decisive, passed: !decisive, metrics: nil, detail: nil)
        if let problem = layoutProblem(native: native, reference: reference) { result.detail = problem; return result }
        let metrics = ParityMetrics.compare(
            native: native.asType(.float32).asArray(Float.self), reference: reference.asType(.float32).asArray(Float.self), limits: limits)
        result.metrics = metrics
        if decisive { result.passed = metrics.passed }
        if bytesEqual(native, reference) { result.detail = "bit-identical" }
        return result
    }

    /// The four comparisons of one core that has an eager and a compiled reference. Only native eager vs MFLUX eager decides
    /// (the predeclared limit). `mx.compile` fuses element-wise ops, so MFLUX's compiled result may differ from its own eager
    /// one; that drift, and how the native eager and compiled results sit against both, are recorded, never failed.
    static func eagerAndCompiled(
        _ name: String, nativeEager: MLXArray, nativeCompiled: MLXArray, referenceEager: MLXArray, referenceCompiled: MLXArray,
        limits: ParityLimits
    ) -> [CheckResult] {
        [
            tolerance(name, native: nativeEager, reference: referenceEager, limits: limits),
            tolerance("\(name): native eager vs MFLUX compiled", native: nativeEager, reference: referenceCompiled, limits: limits, decisive: false),
            tolerance("\(name): MFLUX eager vs MFLUX compiled (reference drift)", native: referenceEager, reference: referenceCompiled, limits: limits, decisive: false),
            tolerance("\(name): native compiled vs MFLUX compiled", native: nativeCompiled, reference: referenceCompiled, limits: limits, decisive: false),
        ]
    }

    /// The comparisons of the prediction under the owner's decision for slice 4: native compiled against MFLUX's production
    /// compiled path decides (the predeclared limit); native eager vs MFLUX eager, native eager vs MFLUX compiled and MFLUX
    /// eager vs MFLUX compiled (the reference's own drift) are recorded as diagnostics and never fail a case.
    static func compiledDecisive(
        _ name: String, nativeEager: MLXArray, nativeCompiled: MLXArray, referenceEager: MLXArray, referenceCompiled: MLXArray,
        limits: ParityLimits
    ) -> [CheckResult] {
        [
            tolerance("\(name): native compiled vs MFLUX compiled (production path)", native: nativeCompiled, reference: referenceCompiled, limits: limits),
            tolerance("\(name): native eager vs MFLUX eager (retained)", native: nativeEager, reference: referenceEager, limits: limits, decisive: false),
            tolerance("\(name): native eager vs MFLUX compiled", native: nativeEager, reference: referenceCompiled, limits: limits, decisive: false),
            tolerance("\(name): MFLUX eager vs MFLUX compiled (reference drift)", native: referenceEager, reference: referenceCompiled, limits: limits, decisive: false),
        ]
    }

    /// Decoded pixels `[H, W, 3]` (uint8 levels): max abs and mean abs difference against the predeclared limits.
    static func pixels(_ name: String, native: [UInt8], reference: [UInt8], nativeShape: [Int], referenceShape: [Int], decisive: Bool = true) -> CheckResult {
        var result = CheckResult(name: name, kind: decisive ? "pixels" : "diagnostic", decisive: decisive, passed: !decisive, metrics: nil, detail: nil)
        guard nativeShape == referenceShape, native.count == reference.count else {
            result.passed = false
            result.detail = "native \(nativeShape), reference \(referenceShape)"
            return result
        }
        let metrics = PixelMetrics.compare(native: native, reference: reference)
        result.pixels = metrics
        if decisive { result.passed = metrics.passed }
        if metrics.differingValues == 0 { result.detail = "pixel-identical" }
        return result
    }

    static func layoutProblem(native: MLXArray, reference: MLXArray) -> String? {
        guard native.shape == reference.shape, native.dtype == reference.dtype else {
            return "native \(native.dtype) \(native.shape), reference \(reference.dtype) \(reference.shape)"
        }
        return nil
    }

    static func bytesEqual(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.asData(access: .copy).data == b.asData(access: .copy).data
    }

    /// A manifest must list exactly the declared case IDs, in order, and have at least one.
    static func requireCaseList(manifest: ReferenceInfo, declared: [String], what: String) throws {
        let listed = manifest.cases?.map(\.id) ?? []
        guard !declared.isEmpty, listed == declared else {
            throw ComponentParityError("\(what): reference case list (\(listed.count)) differs from component_cases.json (\(declared.count))")
        }
    }

    static func report(
        component: String, tolerances: [String: Double], mlxSwiftRevision: String, manifest: ReferenceInfo, cases: [ComponentCaseResult]
    ) -> ComponentReport {
        ComponentReport(
            component: component, tolerances: tolerances, mlxSwiftRevision: mlxSwiftRevision,
            referenceMLXPythonVersion: manifest.mlxPythonVersion, cases: cases, passed: !cases.isEmpty && cases.allSatisfy(\.passed))
    }

    static func logLine(_ c: ComponentCaseResult) -> String {
        let decisive = c.checks.filter(\.decisive)
        let worst = decisive.compactMap { $0.metrics?.relativeL2 }.max()
        let kinds = decisive.contains { $0.kind == "relative-l2" } ? "max relL2=\(worst.map { String(format: "%.3e", $0) } ?? "n/a")" : "bitwise"
        return "\(c.id): \(decisive.count) checks, \(decisive.filter { !$0.passed }.count) failed, \(kinds) \(c.passed ? "PASS" : "FAIL")"
    }
}
