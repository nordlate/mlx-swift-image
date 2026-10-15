import Foundation
import MLX
import ZImageNative

public struct KleinStartingImageManifest: Codable, Sendable {
    public var payload: KleinGenerationManifest
    public var fixturesBefore: [String: String], fixturesAfter: [String: String]
}
public enum KleinStartingImageTensors {
    public static func layout(_ c: KleinStartingImageCase,isolated: Bool) -> [String: KleinTransformerTensors.Expect] {
        let h = c.height/16, w = c.width/16
        func e(_ shape: [Int],_ dtype: String = "mlx.core.bfloat16") -> KleinTransformerTensors.Expect { .init(dtype: dtype,shape: shape) }
        let fp = "mlx.core.float32", int = "mlx.core.int32"
        var a = ["bn_mean":e([128]),"bn_var":e([128]),"source_rgb":e([c.height,c.width,3],"mlx.core.uint8"),"prepared":e([1,3,c.height,c.width],fp),
                 "encoder_input":e([1,3,c.height,c.width],fp),"encoded":e([1,32,h*2,w*2]),"matched":e([1,32,h*2,w*2]),
                 "patched":e([1,128,h,w]),"normalized_latent":e([1,128,h,w]),"clean":e([1,h*w,128])]
        if c.tiny {
            a["encoder.conv_in"] = e([1,128,c.height,c.width],fp)
            for (i,channels) in [128,256,512,512].enumerated() {
                let divisor = 1 << min(i+1,3)
                a["encoder.down.\(i)"] = e([1,channels,c.height/divisor,c.width/divisor],fp)
            }
            a["encoder.mid"] = e([1,512,c.height/8,c.width/8],fp)
            a["encoder.conv_out"] = e([1,64,c.height/8,c.width/8]); a["encoder.quant"] = e([1,64,c.height/8,c.width/8])
        }
        if isolated {
            a["same_input_encoded"] = e([1,32,h*2,w*2])
            for name in ["matched","patched","normalized_latent","clean"] { a["same_input."+name] = a[name] }
            if !c.tiny {
                a["decoder_input"] = e([1,128,h,w],fp); a["packed"] = e([1,128,h,w],fp)
                a["unpacked"] = e([1,32,h*2,w*2],fp); a["decoded"] = e([1,3,c.height,c.width])
                a["pixels"] = e([c.height,c.width,3],"mlx.core.uint8"); a["conversion"] = a["pixels"]
            }
        } else if !c.tiny {
            a["prompt_embeds"] = e([1,512,7680]); a["text_ids"] = e([1,512,4],int)
            a["latent_ids"] = e([1,h*w,4],int); a["seeded_noise"] = e([1,h*w,128])
            a["sigmas"] = e([c.steps+1],fp); a["timesteps"] = e([c.steps],fp); a["start"] = e([],int)
            a["callback_indices"] = e([c.steps-c.start],int)
            for name in ["initial","after_loop"] { a[name] = e([1,h*w,128],fp) }
            a["packed"] = e([1,128,h,w],fp); a["unpacked"] = e([1,32,h*2,w*2],fp)
            a["decoded"] = e([1,3,c.height,c.width]); a["pixels"] = e([c.height,c.width,3],"mlx.core.uint8")
            for t in c.start..<c.steps { for name in ["input.\(t)","noise.\(t)","step.\(t+1)"] { a[name] = e([1,h*w,128],fp) } }
        }
        return a
    }
}
public enum KleinStartingImageNative {
    static func saveCase(_ c: KleinStartingImageCase,arrays: [String: MLXArray],out: URL,summary: (Int,Int) = (0,0)) throws -> KleinGenerationEntry {
        eval(Array(arrays.values))
        try save(arrays: arrays,url: out.appendingPathComponent(c.id+".safetensors"))
        if let pixels = arrays["pixels"] { try ZImageImage.encodePNG(pixels: pixels).write(to: out.appendingPathComponent(c.id+".png"),options: .withoutOverwriting) }
        return .init(id: c.id,width: c.width,height: c.height,seed: c.seed,steps: c.steps,prompt: c.prompt,numValid: summary.0,untruncatedCount: summary.1,tensors: KleinTransformerNative.info(arrays))
    }
    public static func run(modelDirectory: URL,casesJSON: URL,budgetsJSON: URL,fixtures: URL,frozenJSON: URL,
                           outputDirectory: URL,referenceDirectory: URL? = nil,log: (String) -> Void = { _ in }) throws {
        let frozen = try KleinStartingImageFreeze.require(frozenJSON,cases: casesJSON,budgets: budgetsJSON,fixtures: fixtures)
        try KleinStartingImagePaths.requireNew(outputDirectory,model: modelDirectory)
        let before = try KleinGenerationPins.verifyModelFiles(modelDirectory: modelDirectory), fixturesBefore = try KleinStartingImageFixtures.check(fixtures)
        let isolated = referenceDirectory != nil
        var reference: KleinStartingImageManifest?
        if let referenceDirectory {
            reference = try KleinJSON.read(KleinStartingImageManifest.self,from: referenceDirectory.appendingPathComponent("manifest.json"))
            guard reference?.payload.side == "reference", reference?.payload.freeze == frozen else { throw KleinTransformerError.invalidInput("isolated reference freeze/side differs") }
        }
        try FileManager.default.createDirectory(at: outputDirectory,withIntermediateDirectories: true)
        var after: [String: String]?, fixtureAfter: [String: String]?
        func audit() throws {
            if after == nil {
                var value: [String: String] = [:]
                for name in KleinGenerationPins.modelFiles.keys {
                    do { value[name] = try KleinHashing.sha256(of: modelDirectory.appendingPathComponent(name)) }
                    catch { value[name] = "AUDIT_ERROR: \(error)" }
                }
                try KleinJSON.write(value,to: outputDirectory.appendingPathComponent("model-after.json")); after = value
            }
            if fixtureAfter == nil {
                var value: [String: String] = [:]
                for c in KleinStartingImageCases.all {
                    do { value[c.id] = try KleinHashing.sha256(of: fixtures.deletingLastPathComponent().appendingPathComponent(c.id+".png")) }
                    catch { value[c.id] = "AUDIT_ERROR: \(error)" }
                }
                try KleinJSON.write(value,to: outputDirectory.appendingPathComponent("fixtures-after.json")); fixtureAfter = value
            }
            guard after == before, fixtureAfter == fixturesBefore,
                  try KleinHashing.sha256(of: fixtures) == frozen["fixtures_sha256"] else {
                throw KleinTransformerError.artifactChanged("model/source PNG/fixture manifest changed; actual after hashes retained")
            }
        }
        defer { do { try audit() } catch { log("AFTER AUDIT FAILED: \(error)") } }
        let generator = isolated ? nil : try KleinGenerator(modelDirectory: modelDirectory)
        let vae = try generator?.vae ?? KleinVAE(weights: KleinVAEWeights.load(modelDirectory: modelDirectory))
        var entries: [KleinGenerationEntry] = []
        for c in KleinStartingImageCases.all {
            var arrays: [String: MLXArray] = [:]
            func capture(_ n: String,_ a: MLXArray) throws {
                guard arrays[n] == nil else { throw KleinTransformerError.invalidInput("duplicate trace: \(n)") }
                eval(a); arrays[n] = a
            }
            try capture("bn_mean",vae.weights.tensor("bn.running_mean")); try capture("bn_var",vae.weights.tensor("bn.running_var"))
            let file = fixtures.deletingLastPathComponent().appendingPathComponent(c.id+".png")
            if !isolated && !c.tiny {
                _ = try generator!.generate(.init(prompt: c.prompt,width: c.width,height: c.height,seed: c.seed,steps: c.steps,image: file,strength: c.strength),trace: capture)
            } else {
                let image = try KleinPreparedStartingImage.load(file,width: c.width,height: c.height)
                try capture("source_rgb",MLXArray(image.rgb).reshaped(c.height,c.width,3)); try capture("prepared",image.normalized())
                let ref = try referenceDirectory.map { try loadArrays(url: $0.appendingPathComponent(c.id+".safetensors")) }
                if let ref {
                    let layout = KleinStartingImageTensors.layout(c,isolated: false)
                    guard Set(ref.keys) == Set(layout.keys) else { throw KleinTransformerError.invalidInput("incomplete reference tensors") }
                    for (n,e) in layout {
                        guard KleinTransformerCompare.compareTensor(n,native: ref[n],reference: ref[n],rule: .diagnostic,expect: e).passed else { throw KleinTransformerError.invalidInput("invalid reference tensor: \(n)") }
                    }
                }
                let input = ref?["encoder_input"] ?? arrays["prepared"]!
                try capture("encoder_input",input)
                let encoded = try vae.encodePrepared(input) { n,a in if c.tiny || n == "encoded" { try capture(n,a) } }
                _ = try KleinStartingImageMath.clean(encoded,vae: vae,width: c.width,height: c.height,trace: capture)
                if let ref {
                    try capture("same_input_encoded",ref["encoded"]!)
                    _ = try KleinStartingImageMath.clean(ref["encoded"]!,vae: vae,width: c.width,height: c.height) { n,a in try capture("same_input."+n,a) }
                    if !c.tiny {
                        let packed = ref["packed"]!; try capture("decoder_input",packed); try capture("packed",packed)
                        let decoded = try vae.decodeStartingImagePacked(packed) { n,a in if n == "unpacked" { try capture(n,a) } }
                        try capture("decoded",decoded)
                        try capture("pixels",KleinGenerationPixels.convert(decoded,width: c.width,height: c.height))
                        try capture("conversion",KleinGenerationPixels.convert(ref["decoded"]!,width: c.width,height: c.height))
                    }
                }
            }
            let summary = (!isolated && !c.tiny) ? generator!.tokenSummary(c.prompt) : (0,0)
            entries.append(try saveCase(c,arrays: arrays,out: outputDirectory,summary: summary))
            log("\(c.id): \(isolated ? "same-input isolation" : "independent prepared-PNG path") captured")
        }
        var weights = KleinGenerationNative.prefixed(vae.weights.digests(),"vae")
        if let generator {
            weights.merge(KleinGenerationNative.prefixed(generator.textDigests,"text_encoder")) { a,_ in a }
            weights.merge(KleinGenerationNative.prefixed(generator.transformerDigests,"transformer")) { a,_ in a }
        }
        var extra: [String: String] = [:]
        if let referenceDirectory { extra["reference_manifest_sha256"] = try KleinHashing.sha256(of: referenceDirectory.appendingPathComponent("manifest.json")) }
        try audit()
        let payload = KleinGenerationManifest(side: isolated ? "isolated" : "native",freeze: frozen,runtime: try KleinTransformerNative.runtimeInfo(extra: extra),runtimeModules: [:],
            modelFilesBefore: before,modelFilesAfter: after!,weights: weights,quantizedModules: KleinGenerationNative.quantized(isolated: isolated),cases: entries)
        try KleinJSON.write(KleinStartingImageManifest(payload: payload,fixturesBefore: fixturesBefore,fixturesAfter: fixtureAfter!),to: outputDirectory.appendingPathComponent("manifest.json"))
    }
}
