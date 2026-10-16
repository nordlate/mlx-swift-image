import Foundation
import MLX
import ZImageNative

public struct KleinGenerationEntry: Codable, Equatable, Sendable {
    public var id: String
    public var width: Int, height: Int
    public var seed: UInt64
    public var steps: Int
    public var prompt: String
    public var numValid: Int
    public var untruncatedCount: Int
    public var tensors: [String: KleinTensorInfo]
}
public struct KleinGenerationManifest: Codable, Sendable {
    public var schema: Int = 1
    public var side: String
    public var freeze: [String: String]
    public var runtime: [String: String]
    public var runtimeModules: [String: String]
    public var modelFilesBefore: [String: String], modelFilesAfter: [String: String]
    /// Names prefixed text_encoder./transformer./vae., including the validated but unused VAE encoder.
    public var weights: [String: KleinTensorDigest]
    public var quantizedModules: [KleinQuantizedModule]
    public var cases: [KleinGenerationEntry]
}
public enum KleinGenerationTensors {
    public static func layout(width: Int,height: Int,steps: Int,synthetic: Bool,isolated: Bool) -> [String: KleinTransformerTensors.Expect] {
        let h = height/16, w = width/16
        func e(_ s: [Int],_ d: String = "mlx.core.bfloat16") -> KleinTransformerTensors.Expect { .init(dtype: d,shape: s) }
        var out = ["packed":e([1,128,h,w]),"unpacked":e([1,32,h*2,w*2]),"decoded":e([1,3,height,width]),"pixels":e([height,width,3],"mlx.core.uint8")]
        if isolated { out["conversion"] = e([height,width,3],"mlx.core.uint8") }
        if synthetic {
            out["bn"] = e([1,128,h,w]); out["post_quant"] = e([1,32,h*2,w*2]); out["conv_in"] = e([1,512,h*2,w*2]); out["mid"] = e([1,512,h*2,w*2])
            for (i,c) in [512,512,256,128].enumerated() { let scale = 2 << min(i+1,3); out["up.\(i)"] = e([1,c,h*scale,w*scale]) }
        } else if !isolated {
            out["prompt_embeds"] = e([1,512,7680]); out["text_ids"] = e([1,512,4],"mlx.core.int32")
            out["latent_ids"] = e([1,h*w,4],"mlx.core.int32"); out["initial"] = e([1,h*w,128]); out["after_loop"] = e([1,h*w,128])
            out["sigmas"] = e([steps+1],"mlx.core.float32"); out["timesteps"] = e([steps],"mlx.core.float32")
            for t in 0..<steps { out["noise.\(t)"] = e([1,h*w,128]); out["step.\(t+1)"] = e([1,h*w,128]) }
        }
        return out
    }
}
public enum KleinGenerationNative {
    static func freeze(casesJSON: URL,budgetsJSON: URL) throws -> [String: String] {
        let data = Data(try KleinGenerationFreeze.record(casesJSON: casesJSON,budgetsJSON: budgetsJSON).utf8)
        return try JSONDecoder().decode([String: String].self,from: data)
    }
    static func prefixed(_ d: [String: KleinTensorDigest],_ prefix: String) -> [String: KleinTensorDigest] { Dictionary(uniqueKeysWithValues: d.map { (prefix + "." + $0.key,$0.value) }) }
    static func quantized(isolated: Bool) -> [KleinQuantizedModule] {
        let text = KleinTextWeights.quantizedMatrices().map { KleinQuantizedModule(name: $0.name,bits: 4,groupSize: 64,mode: "affine") }
        return (isolated ? [] : [("text_encoder",text),("transformer",KleinTransformerWeights.quantizedModules())]).flatMap { prefix,modules in
            modules.map { KleinQuantizedModule(name: prefix + "." + $0.name,bits: $0.bits,groupSize: $0.groupSize,mode: $0.mode) }
        } + KleinVAEWeights.quantizedModules().map { KleinQuantizedModule(name: "vae." + $0.name,bits: $0.bits,groupSize: $0.groupSize,mode: $0.mode) }
    }
    static func saveCase(_ id: String,width: Int,height: Int,seed: UInt64,steps: Int,prompt: String,arrays: [String: MLXArray],out: URL,numValid: Int = 0,untruncatedCount: Int = 0) throws -> KleinGenerationEntry {
        eval(Array(arrays.values))
        try save(arrays: arrays,url: out.appendingPathComponent(id + ".safetensors"))
        guard let pixels = arrays["pixels"] else { throw KleinTransformerError.invalidInput("missing pixels") }
        let png = out.appendingPathComponent(id + ".png")
        try KleinGenerationPaths.requireNew(png)
        try ZImageImage.encodePNG(pixels: pixels).write(to: png,options: .withoutOverwriting)
        return KleinGenerationEntry(id: id,width: width,height: height,seed: seed,steps: steps,prompt: prompt,numValid: numValid,untruncatedCount: untruncatedCount,tensors: KleinTransformerNative.info(arrays))
    }
    public static func run(modelDirectory: URL,casesJSON: URL,budgetsJSON: URL,outputDirectory: URL,log: (String) -> Void = { _ in }) throws {
        let frozen = try freeze(casesJSON: casesJSON,budgetsJSON: budgetsJSON)
        try KleinGenerationPaths.requireNew(outputDirectory,modelDirectory: modelDirectory)
        let before = try KleinGenerationPins.verifyModelFiles(modelDirectory: modelDirectory)
        try KleinGenerationPaths.create(outputDirectory,modelDirectory: modelDirectory)
        var afterAudit: [String: String]?
        func auditAfter() throws -> [String: String] {
            if let afterAudit { return afterAudit }
            let hashes = try KleinGenerationPins.verifyModelFiles(modelDirectory: modelDirectory)
            try KleinJSON.write(hashes,to: outputDirectory.appendingPathComponent("model-after.json"))
            afterAudit = hashes
            return hashes
        }
        defer {
            if afterAudit == nil {
                do { _ = try auditAfter() } catch { log("MODEL AFTER-AUDIT FAILED: \(error)") }
            }
        }
        let generator = try KleinGenerator(modelDirectory: modelDirectory)
        var entries: [KleinGenerationEntry] = []
        for c in KleinGenerationCases.frozen {
            var arrays: [String: MLXArray] = [:]
            _ = try generator.generate(.init(prompt: c.prompt,width: c.width,height: c.height,seed: c.seed,steps: c.steps),trace: { n,a in
                guard arrays[n] == nil else { throw KleinTransformerError.invalidInput("duplicate trace \(n)") }; eval(a); arrays[n] = a
            })
            let summary = generator.tokenSummary(c.prompt)
            entries.append(try saveCase(c.id,width: c.width,height: c.height,seed: c.seed,steps: c.steps,prompt: c.prompt,arrays: arrays,out: outputDirectory,numValid: summary.valid,untruncatedCount: summary.untruncated))
            log("\(c.id): full independent generation captured")
        }
        for c in KleinGenerationCases.vae {
            let packed = try KleinLatent.prepare(seed: c.seed,height: c.height,width: c.width).latents
            var arrays = ["packed":packed]
            let decoded = try generator.vae.decodePacked(packed) { n,a in eval(a); arrays[n] = a }
            arrays["pixels"] = try KleinGenerationPixels.convert(decoded,width: c.width,height: c.height)
            entries.append(try saveCase(c.id,width: c.width,height: c.height,seed: c.seed,steps: 0,prompt: "",arrays: arrays,out: outputDirectory))
            log("\(c.id): synthetic VAE captured")
        }
        var weights = prefixed(generator.textDigests(),"text_encoder")
        weights.merge(prefixed(generator.transformerDigests(),"transformer")) { a,_ in a }; weights.merge(prefixed(generator.vaeDigests(),"vae")) { a,_ in a }
        let manifest = KleinGenerationManifest(side: "native",freeze: frozen,runtime: try KleinTransformerNative.runtimeInfo(),runtimeModules: [:],
            modelFilesBefore: before,modelFilesAfter: try auditAfter(),weights: weights,
            quantizedModules: quantized(isolated: false),cases: entries)
        try KleinJSON.write(manifest,to: outputDirectory.appendingPathComponent("manifest.json"))
    }
    public static func runIsolated(modelDirectory: URL,referenceDirectory: URL,casesJSON: URL,budgetsJSON: URL,outputDirectory: URL,log: (String) -> Void = { _ in }) throws {
        let frozen = try freeze(casesJSON: casesJSON,budgetsJSON: budgetsJSON)
        let referenceURL = referenceDirectory.appendingPathComponent("manifest.json")
        let reference = try KleinJSON.read(KleinGenerationManifest.self,from: referenceURL)
        guard reference.side == "reference", reference.freeze == frozen else { throw KleinTransformerError.invalidInput("reference freeze differs") }
        try KleinGenerationPaths.requireNew(outputDirectory,modelDirectory: modelDirectory)
        let before = try KleinGenerationPins.verifyModelFiles(modelDirectory: modelDirectory)
        try KleinGenerationPaths.create(outputDirectory,modelDirectory: modelDirectory)
        var afterAudit: [String: String]?
        func auditAfter() throws -> [String: String] {
            if let afterAudit { return afterAudit }
            let hashes = try KleinGenerationPins.verifyModelFiles(modelDirectory: modelDirectory)
            try KleinJSON.write(hashes,to: outputDirectory.appendingPathComponent("model-after.json"))
            afterAudit = hashes
            return hashes
        }
        defer {
            if afterAudit == nil {
                do { _ = try auditAfter() } catch { log("MODEL AFTER-AUDIT FAILED: \(error)") }
            }
        }
        let weights = try KleinVAEWeights.load(modelDirectory: modelDirectory), vae = KleinVAE(weights: weights)
        let digests = weights.digests()
        var entries: [KleinGenerationEntry] = []
        let specs = KleinGenerationCases.frozen.map { ($0.id,$0.width,$0.height,$0.seed,$0.steps,$0.prompt,false) }
            + KleinGenerationCases.vae.map { ($0.id,$0.width,$0.height,$0.seed,0,"",true) }
        for (id,w,h,seed,steps,prompt,synthetic) in specs {
            let input = try loadArrays(url: referenceDirectory.appendingPathComponent(id + ".safetensors"))
            let expected = KleinGenerationTensors.layout(width: w,height: h,steps: steps,synthetic: synthetic,isolated: false)
            for key in ["packed","decoded"] {
                guard let a = input[key], a.shape == expected[key]?.shape, KleinDType.name(a.dtype) == expected[key]?.dtype,
                      KleinCompare.floats(a).allSatisfy(\.isFinite) else { throw KleinTransformerError.invalidInput("\(id): invalid reference \(key)") }
            }
            var arrays = ["packed":input["packed"]!]
            let decoded = try vae.decodePacked(input["packed"]!) { n,a in
                if synthetic || n == "unpacked" || n == "decoded" { eval(a); arrays[n] = a }
            }
            arrays["pixels"] = try KleinGenerationPixels.convert(decoded,width: w,height: h)
            arrays["conversion"] = try KleinGenerationPixels.convert(input["decoded"]!,width: w,height: h)
            entries.append(try saveCase(id,width: w,height: h,seed: seed,steps: steps,prompt: prompt,arrays: arrays,out: outputDirectory))
            log("\(id): isolated complete VAE and conversion captured")
        }
        let manifest = KleinGenerationManifest(side: "isolated",freeze: frozen,
            runtime: try KleinTransformerNative.runtimeInfo(extra: ["reference_manifest_sha256":try KleinHashing.sha256(of: referenceURL)]),runtimeModules: [:],
            modelFilesBefore: before,modelFilesAfter: try auditAfter(),weights: prefixed(digests,"vae"),
            quantizedModules: quantized(isolated: true),cases: entries)
        try KleinJSON.write(manifest,to: outputDirectory.appendingPathComponent("manifest.json"))
    }
}
