import Foundation
import MLX
import ZImage

public struct KleinGenerationPixelResult: Codable, Sendable {
    public var name: String
    public var maxAbs: Int
    public var meanAbs: Double
    public var passed: Bool
}
public struct KleinGenerationCaseResult: Codable, Sendable {
    public var id: String
    public var tensorCounts: [String: Int]
    public var tensors: [KleinTensorResult]
    public var pixels: [KleinGenerationPixelResult]
    public var failures: [String]
    public var passed: Bool
}
public struct KleinGenerationReport: Codable, Sendable {
    public var status: String
    public var passed: Bool
    public var expectedCaseIDs: [String]
    public var actualCaseIDs: [String: [String]]
    public var freeze: [String: String]
    public var artifactSHA256: [String: String]
    public var runtime: [String: [String: String]]
    public var checks: [KleinCheck]
    public var cases: [KleinGenerationCaseResult]
}
public enum KleinGenerationCompare {
    /// Rule assignment is compiled in; input files cannot substitute a diagnostic for a deciding latent/pixel comparison.
    static func rule(_ name: String,synthetic: Bool,isolated: Bool) -> KleinTransformerCompare.Rule {
        if name == "packed" && !synthetic && !isolated { return .limit(.init(relativeL2: KleinGenerationBudgets.postStep,normalizedMaxAbs: nil)) }
        if name == "pixels" { return .diagnostic } // The separate integer pixel budget below decides.
        if name == "conversion" || name == "packed" || name == "initial" || name == "latent_ids" || name == "text_ids" || name == "sigmas" || name == "timesteps" { return .exact }
        if name == "prompt_embeds" { return .limit(.init(relativeL2: KleinGenerationBudgets.prompt,normalizedMaxAbs: KleinGenerationBudgets.promptMax)) }
        if name.hasPrefix("noise.") || (name == "unpacked" && !synthetic && !isolated) || (name == "decoded" && !synthetic && !isolated) { return .diagnostic }
        if name.hasPrefix("step.") || name == "after_loop" { return .limit(.init(relativeL2: KleinGenerationBudgets.postStep,normalizedMaxAbs: nil)) }
        if name == "decoded" { return .limit(.init(relativeL2: KleinGenerationBudgets.completeVAE,normalizedMaxAbs: nil)) }
        return .limit(.init(relativeL2: KleinGenerationBudgets.component,normalizedMaxAbs: nil))
    }
    /// Bind decode inputs to the captured actual final sampler state, without substituting a denoising computation.
    static func chainBindingFailures(_ arrays: [String: MLXArray],width: Int,height: Int,steps: Int) -> [String] {
        let shape = [1,(height/16)*(width/16),128]
        guard let last = arrays["step.\(steps)"], let final = arrays["after_loop"], let packed = arrays["packed"],
              last.shape == shape, final.shape == shape, packed.shape == [1,128,height/16,width/16],
              last.dtype == .bfloat16, final.dtype == .bfloat16, packed.dtype == .bfloat16 else { return ["final sampler/decode binding missing or invalid"] }
        var failures: [String] = []
        if KleinCompare.bytes(last) != KleinCompare.bytes(final) { failures.append("after_loop differs from actual final post-step latent") }
        let unpacked = final.reshaped(1,height/16,width/16,128).transposed(0,3,1,2)
        if KleinCompare.bytes(packed) != KleinCompare.bytes(unpacked) { failures.append("decode packed input differs from final sampler state") }
        return failures
    }
    static func weightFailures(_ manifests: [String: KleinGenerationManifest]) -> [String] {
        var layout: [String: (dtype: String,shape: [Int])] = [:]
        for (n,e) in TextEncoderLayout.expectedTensors() { layout["text_encoder."+n] = e }
        for (n,e) in KleinTransformerLayout.expectedTensors() { layout["transformer."+n] = e }
        for (n,e) in KleinVAELayout.expectedTensors() { layout["vae."+n] = (e.dtype,e.shape) }
        var failures: [String] = []
        for (side,m) in manifests {
            let expected = side == "isolated" ? layout.filter { $0.key.hasPrefix("vae.") } : layout
            if Set(m.weights.keys) != Set(expected.keys) { failures.append("\(side): loaded tensor names/count differ, expected \(expected.count), actual \(m.weights.count)") }
            for (n,e) in expected {
                guard let d = m.weights[n] else { continue }
                let bytes = e.shape.reduce(1,*) * (e.dtype == "U32" || e.dtype == "F32" ? 4 : 2)
                if d.dtype != KleinTextWeights.mlxName(of: e.dtype) || d.shape != e.shape || d.bytes != bytes || !KleinCompare.isSHA256Hex(d.sha256) {
                    failures.append("\(side): invalid loaded metadata \(n)")
                }
                if let reference = manifests["reference"]?.weights[n], d != reference { failures.append("\(side): loaded payload differs \(n)") }
            }
            let modules = KleinGenerationNative.quantized(isolated: side == "isolated").sorted { $0.name < $1.name }
            if m.quantizedModules.sorted(by: { $0.name < $1.name }) != modules { failures.append("\(side): quantized attributes/names/duplicates differ") }
        }
        return failures
    }
    public static func run(referenceDirectory: URL,nativeDirectory: URL,isolatedDirectory: URL,casesJSON: URL,budgetsJSON: URL,frozenJSON: URL,log: (String) -> Void = { _ in }) throws -> KleinGenerationReport {
        // CLI additionally selects CPU; library callers receive the same CPU-only comparison contract.
        try Device.withDefaultDevice(.cpu) {
            try compare(referenceDirectory: referenceDirectory,nativeDirectory: nativeDirectory,isolatedDirectory: isolatedDirectory,casesJSON: casesJSON,budgetsJSON: budgetsJSON,frozenJSON: frozenJSON,log: log)
        }
    }
    static func compare(referenceDirectory: URL,nativeDirectory: URL,isolatedDirectory: URL,casesJSON: URL,budgetsJSON: URL,frozenJSON: URL,log: (String) -> Void) throws -> KleinGenerationReport {
        var checks: [KleinCheck] = [], complete = true, hashes: [String: String] = [:], frozen: [String: String] = [:]
        func check(_ name: String,_ failures: [String]) { checks.append(KleinCompare.check(name,failures)) }
        var inputFailures: [String] = []
        do { try KleinGenerationCases.check(casesJSON) } catch { inputFailures.append("\(error)") }
        do { try KleinGenerationBudgets.check(budgetsJSON) } catch { inputFailures.append("\(error)") }
        do {
            frozen = try KleinJSON.read([String: String].self,from: frozenJSON)
            let expected = ["cases_sha256":try KleinHashing.sha256(of: casesJSON),"budgets_sha256":try KleinHashing.sha256(of: budgetsJSON),"cases_digest":KleinGenerationCases.digest]
            if frozen != expected { inputFailures.append("freeze does not bind current files and compiled corpus") }
            hashes["frozen"] = try KleinHashing.sha256(of: frozenJSON)
        } catch { inputFailures.append("freeze missing/unreadable: \(error)"); complete = false }
        check("frozen-inputs",inputFailures)
        let directories = ["reference":referenceDirectory,"native":nativeDirectory,"isolated":isolatedDirectory]
        var manifests: [String: KleinGenerationManifest] = [:]
        var provenance: [String] = []
        for (side,dir) in directories.sorted(by: { $0.key < $1.key }) {
            do {
                let url = dir.appendingPathComponent("manifest.json"), m = try KleinJSON.read(KleinGenerationManifest.self,from: url)
                hashes[side+"/manifest"] = try KleinHashing.sha256(of: url); manifests[side] = m
                if m.schema != 1 || m.side != side || m.freeze != frozen { provenance.append("\(side): schema/side/freeze differs") }
                if m.modelFilesBefore != KleinGenerationPins.modelFiles || m.modelFilesAfter != KleinGenerationPins.modelFiles { provenance.append("\(side): model files changed or differ from all 11 pins") }
                if m.cases.map(\.id) != KleinGenerationCases.allIDs { provenance.append("\(side): case IDs/order/duplicates differ"); complete = false }
                if side == "reference" {
                    provenance += KleinCompare.referenceVersionFailures(m.runtime)
                    if m.runtimeModules != KleinGenerationPins.runtimeModules { provenance.append("runtime module hashes differ from the 53 production direct.pyc pins") }
                    if m.runtime["is_m1_or_m2"] != "false" || m.runtime["production_predict_compiled"] != "true" || m.runtime["oracle"] != "Flux2Klein.generate_image" || m.runtime["mlx_core_commit"] != "UNKNOWN" {
                        provenance.append("reference oracle/compiled branch/core provenance differs")
                    }
                    for k in ["python","python_executable","mflux_dist_metadata_sha256","mflux_dist_location"] where (m.runtime[k] ?? "").isEmpty { provenance.append("reference missing \(k)") }
                } else {
                    if m.runtime["self_reported_mlx_swift_revision"] != KleinPins.mlxSwiftRevision || !KleinCompare.isSHA256Hex(m.runtime["executable_sha256"]) { provenance.append("\(side): revision/binary digest missing or differs") }
                    let settings = m.runtime["self_reported_build_settings"] ?? ""
                    for setting in ["MTL_FAST_MATH=NO","MACOSX_DEPLOYMENT_TARGET=26.2","ENABLE_CODE_COVERAGE=NO","CLANG_ENABLE_CODE_COVERAGE=NO","CLANG_COVERAGE_MAPPING=NO","GCC_GENERATE_TEST_COVERAGE_FILES=NO","GCC_INSTRUMENT_PROGRAM_FLOW_ARCS=NO"] where !settings.contains(setting) { provenance.append("\(side): self-reported setting missing \(setting)") }
                }
            } catch { provenance.append("\(side): manifest missing/unreadable: \(error)"); complete = false }
        }
        if manifests["native"]?.runtime["executable_sha256"] != manifests["isolated"]?.runtime["executable_sha256"] { provenance.append("native/isolated binaries differ") }
        if manifests["isolated"]?.runtime["reference_manifest_sha256"] != hashes["reference/manifest"] { provenance.append("isolated reference manifest binding differs") }
        check("runtime-model-corpus",provenance)
        check("loaded-1558-weights-and-quantization",weightFailures(manifests))
        let specs = KleinGenerationCases.frozen.map { ($0.id,$0.width,$0.height,$0.seed,$0.steps,$0.prompt,false) }
            + KleinGenerationCases.vae.map { ($0.id,$0.width,$0.height,$0.seed,0,"",true) }
        var results: [KleinGenerationCaseResult] = []
        for (id,w,h,seed,steps,prompt,synthetic) in specs {
            var failures: [String] = [], counts: [String: Int] = [:], arrays: [String: [String: MLXArray]] = [:], pngs: [String: [UInt8]] = [:], tensors: [KleinTensorResult] = [], pixelResults: [KleinGenerationPixelResult] = []
            for (side,dir) in directories.sorted(by: { $0.key < $1.key }) {
                let layout = KleinGenerationTensors.layout(width: w,height: h,steps: steps,synthetic: synthetic,isolated: side == "isolated")
                do {
                    let url = dir.appendingPathComponent(id+".safetensors"), a = try loadArrays(url: url)
                    arrays[side] = a; counts[side] = a.count; hashes[side+"/"+id+".safetensors"] = try KleinHashing.sha256(of: url)
                    if Set(a.keys) != Set(layout.keys) { failures.append("\(side): tensor set missing/extra"); complete = false }
                    if let entry = manifests[side]?.cases.first(where: { $0.id == id }) {
                    if entry.width != w || entry.height != h || entry.seed != seed || entry.steps != steps || Data(entry.prompt.utf8) != Data(prompt.utf8) { failures.append("\(side): case parameters differ") }
                    if entry.tensors != KleinTransformerNative.info(a) { failures.append("\(side): declared tensor metadata differs") }
                    if side != "isolated" && !synthetic {
                        failures += chainBindingFailures(a,width: w,height: h,steps: steps).map { "\(side): \($0)" }
                        if !(1...512).contains(entry.numValid) || entry.untruncatedCount < entry.numValid { failures.append("\(side): invalid token counts") }
                        if id == "g3-landscape1152x864-long", entry.numValid != 512 || entry.untruncatedCount != 1419 { failures.append("\(side): long prompt did not demonstrate 1419 to 512 truncation") }
                        if side == "native", let r = manifests["reference"]?.cases.first(where: { $0.id == id }),
                           (r.numValid != entry.numValid || r.untruncatedCount != entry.untruncatedCount) { failures.append("native/reference token counts differ") }
                    }
                    } else { failures.append("\(side): manifest case missing"); complete = false }
                    // Validate every artifact independently, including reference-only tensors and diagnostic tensors.
                    for (key,e) in layout {
                        let validation = KleinTransformerCompare.compareTensor("\(side).\(key)",native: a[key],reference: a[key],rule: .diagnostic,expect: e)
                        if !validation.passed { failures += validation.failures.map { "\(side).\(key): \($0)" } }
                    }
                } catch { failures.append("\(side): tensor artifact missing/unreadable: \(error)"); complete = false }
                do {
                    let url = dir.appendingPathComponent(id+".png"), info = try ZImageImage.pngInfo(at: url), image = try ZImageImage.readPNG(at: url)
                    hashes[side+"/"+id+".png"] = info.sha256
                    guard info.width == w, info.height == h, info.bitDepth == 8, info.colorType == 2, image.width == w, image.height == h else { throw KleinTransformerError.invalidInput("PNG must be actual-size 8-bit RGB without alpha") }
                    pngs[side] = image.bytes
                    guard let pixels = arrays[side]?["pixels"], pixels.shape == [h,w,3], pixels.dtype == .uint8 else { throw KleinTransformerError.invalidInput("PNG pixels missing or invalid") }
                    if image.bytes != pixels.asArray(UInt8.self) { failures.append("\(side): PNG raw RGB readback differs from own pixel array") }
                } catch { failures.append("\(side): PNG missing/invalid: \(error)"); complete = false }
            }
            let ref = arrays["reference"] ?? [:]
            for side in ["native","isolated"] {
                let layout = KleinGenerationTensors.layout(width: w,height: h,steps: steps,synthetic: synthetic,isolated: side == "isolated")
                for name in layout.keys.sorted() {
                    let referenceName = name == "conversion" ? "pixels" : name
                    tensors.append(KleinTransformerCompare.compareTensor(side+"."+name,native: arrays[side]?[name],reference: ref[referenceName],rule: rule(name,synthetic: synthetic,isolated: side == "isolated"),expect: layout[name]))
                }
                if let a = arrays[side]?["pixels"], let b = ref["pixels"], a.dtype == .uint8, b.dtype == .uint8, a.shape == [h,w,3], b.shape == a.shape {
                    let budget = KleinGenerationPixels.budget(native: a.asArray(UInt8.self),reference: b.asArray(UInt8.self))
                    pixelResults.append(.init(name: side+".pixels",maxAbs: budget.max,meanAbs: budget.mean,passed: budget.passed))
                    log("\(id) \(side) pixels: max=\(budget.max), mean=\(budget.mean)")
                    if !budget.passed { failures.append("\(side): pixels over 2/0.1 levels (max \(budget.max), mean \(budget.mean))") }
                } else { failures.append("\(side): pixel comparison missing/invalid"); complete = false }
                if let a = pngs[side], let b = pngs["reference"] {
                    let budget = KleinGenerationPixels.budget(native: a,reference: b)
                    pixelResults.append(.init(name: side+".png",maxAbs: budget.max,meanAbs: budget.mean,passed: budget.passed))
                    if !budget.passed { failures.append("\(side): decoded PNG pixels over 2/0.1 levels (max \(budget.max), mean \(budget.mean))") }
                } else { complete = false }
            }
            let passed = failures.isEmpty && tensors.allSatisfy(\.passed)
            results.append(KleinGenerationCaseResult(id: id,tensorCounts: counts,tensors: tensors,pixels: pixelResults,failures: failures,passed: passed))
            log("\(id): \(passed ? "PASS" : "FAIL")")
        }
        let passed = complete && checks.allSatisfy(\.passed) && results.count == 6 && results.allSatisfy(\.passed)
        return KleinGenerationReport(status: complete ? (passed ? "COMPLETE_PASS" : "COMPLETE_FAIL") : "INCOMPLETE",passed: passed,
            expectedCaseIDs: KleinGenerationCases.allIDs,actualCaseIDs: manifests.mapValues { $0.cases.map(\.id) },freeze: frozen,artifactSHA256: hashes,
            runtime: manifests.mapValues(\.runtime),checks: checks,cases: results)
    }
}
