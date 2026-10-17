import FluxKlein
import Foundation
import MLX
import MLXImageCore

public enum KleinStartingImageCompare {
    static func rule(_ name: String,isolated: Bool) -> KleinTransformerCompare.Rule {
        if ["source_rgb","prepared","encoder_input","bn_mean","bn_var","same_input_encoded","decoder_input","seeded_noise","latent_ids","text_ids","sigmas","timesteps","start","callback_indices","conversion"].contains(name) { return .exact }
        if name == "prompt_embeds" { return .limit(.init(relativeL2: 1e-3,normalizedMaxAbs: 1e-2)) }
        if name.hasPrefix("noise.") || name == "pixels" || (!isolated && (name == "decoded" || name == "unpacked")) { return .diagnostic }
        if name.hasPrefix("same_input.") || name.hasPrefix("encoder.") { return .limit(.init(relativeL2: 1e-3,normalizedMaxAbs: nil)) }
        if name == "encoded" || name == "clean" || name == "initial" || name == "matched" || name == "patched" || name == "normalized_latent" { return .limit(.init(relativeL2: 2e-3,normalizedMaxAbs: nil)) }
        if isolated && name == "packed" { return .exact }
        if isolated { return .limit(.init(relativeL2: name == "decoded" ? 2e-3 : 1e-3,normalizedMaxAbs: nil)) }
        return .limit(.init(relativeL2: 5e-3,normalizedMaxAbs: nil))
    }
    /// Exact intra-case bindings tie the actual encoder return to clean, blend, executed predict inputs and decode payload.
    static func chainFailures(_ a: [String: MLXArray],c: KleinStartingImageCase,isolated: Bool,reference: [String: MLXArray]? = nil) -> [String] {
        guard !a.isEmpty else { return ["missing source/encoder/clean chain tensors"] }
        var failures: [String] = []
        func equal(_ name: String,_ x: MLXArray?,_ y: MLXArray?) {
            guard let x, let y, x.shape == y.shape, x.dtype == y.dtype,
                  KleinCompare.floats(x).allSatisfy(\.isFinite), KleinCompare.floats(y).allSatisfy(\.isFinite) else { failures.append("\(name): missing/invalid chain input"); return }
            if KleinCompare.bytes(x) != KleinCompare.bytes(y) { failures.append("\(name): chain differs") }
        }
        let raw = KleinStartingImageCases.rgb(c)
        equal("source-fixture",a["source_rgb"],MLXArray(raw).reshaped(c.height,c.width,3))
        let image = KleinPreparedStartingImage(width: c.width,height: c.height,rgb: raw,fileSHA256: "")
        equal("source-normalization",a["prepared"],image.normalized())
        equal("encoder-input",a["encoder_input"],isolated ? reference?["encoder_input"] : a["prepared"])
        func preparation(_ input: MLXArray?,_ prefix: String) {
            guard let input, let mean = a["bn_mean"], let variance = a["bn_var"] else { failures.append("missing encoder/BN chain input"); return }
            do {
                let matched = try KleinStartingImageMath.spatial(input,height: c.height/8,width: c.width/8)
                equal(prefix+"matched",a[prefix+"matched"],matched)
                let patch = try KleinLatent.patchify(matched); equal(prefix+"patched",a[prefix+"patched"],patch)
                let bn = try KleinStartingImageMath.normalize(patch,mean: mean,variance: variance)
                // GPU/CPU sqrt may round differently. Check the mathematical BN binding at the fixed component gate.
                let e = KleinTransformerTensors.Expect(dtype: "mlx.core.bfloat16",shape: bn.shape)
                if !KleinTransformerCompare.compareTensor(prefix+"BN-chain",native: a[prefix+"normalized_latent"],reference: bn,rule: .limit(.init(relativeL2: 1e-3,normalizedMaxAbs: nil)),expect: e).passed { failures.append(prefix+"BN chain differs") }
                if let actualBN = a[prefix+"normalized_latent"] { equal(prefix+"pack",a[prefix+"clean"],try KleinLatent.pack(actualBN)) }
            } catch { failures.append("preparation chain: \(error)") }
        }
        preparation(a["encoded"],"")
        if c.tiny {
            if let quant = a["encoder.quant"], quant.shape == [1,64,c.height/8,c.width/8], quant.dtype == .bfloat16 {
                equal("quant-mean-return",a["encoded"],(quant[0...,0..<32,0...,0...] - 0.0) * 1.0)
            } else { failures.append("tiny encoder quant/mean binding missing") }
        }
        if !c.tiny {
            if let decoded = a["decoded"] {
                do { equal("actual-decoded-pixels",a["pixels"],try KleinGenerationPixels.convert(decoded,width: c.width,height: c.height)) }
                catch { failures.append("decoded pixel binding: \(error)") }
            } else { failures.append("decoded pixel binding missing") }
        }
        if isolated {
            equal("reference-encoded-input",a["same_input_encoded"],reference?["encoded"])
            preparation(a["same_input_encoded"],"same_input.")
            if !c.tiny {
                equal("reference-decode-input",a["decoder_input"],reference?["packed"])
                equal("actual-decode-input",a["packed"],a["decoder_input"])
            }
        } else if !c.tiny {
            do {
                let noise = try KleinLatent.preparePacked(seed: c.seed,height: c.height,width: c.width)
                let scheduler = try KleinScheduler(steps: c.steps,height: c.height,width: c.width)
                equal("own-seeded-noise",a["seeded_noise"],noise.latents); equal("own-grid",a["latent_ids"],noise.ids)
                equal("own-sigmas",a["sigmas"],scheduler.sigmas); equal("own-timesteps",a["timesteps"],scheduler.timesteps)
                equal("start",a["start"],MLXArray(Int32(c.start)))
                equal("callbacks",a["callback_indices"],MLXArray((c.start..<c.steps).map { Int32($0+1) }))
                if let clean = a["clean"] { equal("actual-initial-blend",a["initial"],try KleinStartingImageMath.blend(clean: clean,noise: noise.latents,sigma: scheduler.sigmas[c.start])) }
                for t in c.start..<c.steps { equal("actual-predict-input.\(t)",a["input.\(t)"],a[t == c.start ? "initial" : "step.\(t)"]) }
                equal("actual-final",a["after_loop"],a["step.\(c.steps)"])
                if let last = a["after_loop"] { equal("actual-decode-payload",a["packed"],try KleinLatent.unpack(last,height: c.height,width: c.width)) }
            } catch { failures.append("denoising chain: \(error)") }
        }
        return failures
    }
    public static func run(referenceDirectory: URL,nativeDirectory: URL,isolatedDirectory: URL,casesJSON: URL,budgetsJSON: URL,
                           fixtures: URL,frozenJSON: URL,log: (String) -> Void = { _ in }) throws -> KleinGenerationReport {
        try Device.withDefaultDevice(.cpu) {
            var checks: [KleinCheck] = [], complete = true, hashes: [String: String] = [:], frozen: [String: String] = [:]
            func check(_ n: String,_ f: [String]) { checks.append(KleinCompare.check(n,f)) }
            var inputFailures: [String] = [], fixturePins: [String: String] = [:]
            do {
                frozen = try KleinStartingImageFreeze.require(frozenJSON,cases: casesJSON,budgets: budgetsJSON,fixtures: fixtures)
                fixturePins = try KleinStartingImageFixtures.check(fixtures)
                hashes["frozen"] = try KleinHashing.sha256(of: frozenJSON)
            } catch { inputFailures.append("freeze/fixture input missing or changed: \(error)"); complete = false }
            check("frozen-inputs-and-source-fixtures",inputFailures)
            let directories = ["reference":referenceDirectory,"native":nativeDirectory,"isolated":isolatedDirectory]
            var manifests: [String: KleinGenerationManifest] = [:], provenance: [String] = []
            for (side,dir) in directories.sorted(by: { $0.key < $1.key }) {
                do {
                    let url = dir.appendingPathComponent("manifest.json"), wrapper = try KleinJSON.read(KleinStartingImageManifest.self,from: url), m = wrapper.payload
                    hashes[side+"/manifest"] = try KleinHashing.sha256(of: url); manifests[side] = m
                    if m.schema != 1 || m.side != side || m.freeze != frozen { provenance.append("\(side): schema/side/freeze differs") }
                    if m.modelFilesBefore != KleinGenerationPins.modelFiles || m.modelFilesAfter != KleinGenerationPins.modelFiles { provenance.append("\(side): 11-file model audit differs") }
                    if wrapper.fixturesBefore != fixturePins || wrapper.fixturesAfter != fixturePins || fixturePins.count != 6 { provenance.append("\(side): source PNG before/after differs") }
                    if m.cases.map(\.id) != KleinStartingImageCases.allIDs { provenance.append("\(side): missing/duplicate/extra cases"); complete = false }
                    if side == "reference" {
                        provenance += KleinCompare.referenceVersionFailures(m.runtime)
                        if m.runtimeModules != KleinStartingImagePins.runtimeModules { provenance.append("reference 56 direct.pyc module pins differ") }
                        if m.runtime["oracle"] != "Flux2Klein.generate_image(image_path,image_strength)" || m.runtime["production_predict_compiled"] != "true" || m.runtime["is_m1_or_m2"] != "false" || m.runtime["mlx_core_commit"] != "UNKNOWN" { provenance.append("reference oracle/branch/core differs") }
                        for k in ["python","python_executable","mflux_dist_metadata_sha256","mflux_dist_location"] where (m.runtime[k] ?? "").isEmpty { provenance.append("reference missing \(k)") }
                        if !(m.runtime["python"] ?? "").hasPrefix("3.12.") { provenance.append("reference Python must be 3.12") }
                    } else {
                        if m.runtime["self_reported_mlx_swift_revision"] != KleinPins.mlxSwiftRevision || !KleinCompare.isSHA256Hex(m.runtime["executable_sha256"]) { provenance.append("\(side): native revision/executable differs") }
                        for setting in ["MTL_FAST_MATH=NO","MACOSX_DEPLOYMENT_TARGET=26.2","ENABLE_CODE_COVERAGE=NO","CLANG_ENABLE_CODE_COVERAGE=NO","CLANG_COVERAGE_MAPPING=NO","GCC_GENERATE_TEST_COVERAGE_FILES=NO","GCC_INSTRUMENT_PROGRAM_FLOW_ARCS=NO"] where !(m.runtime["self_reported_build_settings"] ?? "").contains(setting) { provenance.append("\(side): missing setting \(setting)") }
                    }
                } catch { provenance.append("\(side): missing manifest: \(error)"); complete = false }
            }
            if manifests["isolated"]?.runtime["reference_manifest_sha256"] != hashes["reference/manifest"] { provenance.append("isolation reference binding differs") }
            if manifests["native"]?.runtime["executable_sha256"] != manifests["isolated"]?.runtime["executable_sha256"] { provenance.append("native binaries differ") }
            check("runtime-model-fixtures-corpus",provenance)
            check("all-1558-loaded-weights-and-quant-attributes",KleinGenerationCompare.weightFailures(manifests))
            var results: [KleinGenerationCaseResult] = []
            for c in KleinStartingImageCases.all {
                var failures: [String] = [], counts: [String: Int] = [:], arrays: [String: [String: MLXArray]] = [:]
                var tensors: [KleinTensorResult] = [], pixels: [KleinGenerationPixelResult] = []
                for (side,dir) in directories.sorted(by: { $0.key < $1.key }) {
                    let layout = KleinStartingImageTensors.layout(c,isolated: side == "isolated")
                    do {
                        let file = dir.appendingPathComponent(c.id+".safetensors"), a = try loadArrays(url: file)
                        arrays[side] = a; counts[side] = a.count; hashes[side+"/"+c.id] = try KleinHashing.sha256(of: file)
                        if Set(a.keys) != Set(layout.keys) { failures.append("\(side): static tensor set differs"); complete = false }
                        for (key,e) in layout {
                            let r = KleinTransformerCompare.compareTensor(side+"."+key,native: a[key],reference: a[key],rule: .diagnostic,expect: e)
                            if !r.passed { failures += r.failures.map { side+"."+key+": "+$0 } }
                        }
                        if let entry = manifests[side]?.cases.first(where: { $0.id == c.id }) {
                            if entry.width != c.width || entry.height != c.height || entry.seed != c.seed || entry.steps != c.steps || entry.prompt != c.prompt || entry.tensors != KleinTransformerNative.info(a) { failures.append("\(side): declared case/tensor metadata differs") }
                            if !c.tiny && side != "isolated" {
                                if !(1...512).contains(entry.numValid) || entry.untruncatedCount < entry.numValid { failures.append("\(side): invalid token summary") }
                                if side == "native", let r = manifests["reference"]?.cases.first(where: { $0.id == c.id }), (r.numValid != entry.numValid || r.untruncatedCount != entry.untruncatedCount) { failures.append("token summaries differ") }
                            }
                        } else { failures.append("\(side): missing manifest entry"); complete = false }
                        for (name,weight) in [("bn_mean","vae.bn.running_mean"),("bn_var","vae.bn.running_var")] {
                            if let tensor = a[name], let digest = manifests[side]?.weights[weight],
                               KleinHashing.sha256(of: Data(KleinCompare.bytes(tensor))) == digest.sha256 { } else { failures.append("\(side): captured BN statistics differ from loaded parameter bytes") }
                        }
                        if !c.tiny {
                            let file = dir.appendingPathComponent(c.id+".png"), info = try ImageOutput.pngInfo(at: file), png = try ImageOutput.readPNG(at: file)
                            hashes[side+"/"+c.id+".png"] = info.sha256
                            if info.width != c.width || info.height != c.height || info.bitDepth != 8 || info.colorType != 2 || a["pixels"]?.dtype != .uint8 || png.bytes != a["pixels"]?.asArray(UInt8.self) { failures.append("\(side): PNG readback differs from own pixel payload") }
                        }
                    } catch { failures.append("\(side): missing/invalid artifacts: \(error)"); complete = false }
                }
                let ref = arrays["reference"] ?? [:]
                for side in ["reference","native","isolated"] {
                    failures += chainFailures(arrays[side] ?? [:],c: c,isolated: side == "isolated",reference: ref).map { side+": "+$0 }
                }
                for side in ["native","isolated"] {
                    for (name,e) in KleinStartingImageTensors.layout(c,isolated: side == "isolated").sorted(by: { $0.key < $1.key }) {
                        let refName: String
                        if name.hasPrefix("same_input.") { refName = String(name.dropFirst("same_input.".count)) }
                        else if name == "same_input_encoded" { refName = "encoded" }
                        else if name == "decoder_input" { refName = "packed" }
                        else if name == "conversion" { refName = "pixels" }
                        else { refName = name }
                        tensors.append(KleinTransformerCompare.compareTensor(side+"."+name,native: arrays[side]?[name],reference: ref[refName],rule: rule(name,isolated: side == "isolated"),expect: e))
                    }
                    if !c.tiny {
                        if let a = arrays[side]?["pixels"], let b = ref["pixels"], a.dtype == .uint8, b.dtype == .uint8, a.shape == b.shape, a.shape == [c.height,c.width,3] {
                            let budget = KleinGenerationPixels.budget(native: a.asArray(UInt8.self),reference: b.asArray(UInt8.self))
                            pixels.append(.init(name: side,maxAbs: budget.max,meanAbs: budget.mean,passed: budget.passed))
                            if !budget.passed { failures.append("\(side): final pixels exceed max 2 AND mean 0.1") }
                        } else { failures.append("\(side): missing pixel comparison"); complete = false }
                    }
                }
                let passed = failures.isEmpty && tensors.allSatisfy(\.passed)
                results.append(.init(id: c.id,tensorCounts: counts,tensors: tensors,pixels: pixels,failures: failures,passed: passed)); log("\(c.id): \(passed ? "PASS" : "FAIL")")
            }
            let passed = complete && checks.allSatisfy(\.passed) && results.count == 6 && results.allSatisfy(\.passed)
            return .init(status: complete ? (passed ? "COMPLETE_PASS" : "COMPLETE_FAIL") : "INCOMPLETE",passed: passed,expectedCaseIDs: KleinStartingImageCases.allIDs,
                         actualCaseIDs: manifests.mapValues { $0.cases.map(\.id) },freeze: frozen,artifactSHA256: hashes,runtime: manifests.mapValues(\.runtime),checks: checks,cases: results)
        }
    }
}
