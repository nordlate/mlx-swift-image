import Foundation
import ImageIO
import MLX
import MLXImageCore

/// Only the already prepared, job-owned RGB PNG from StartingImagePreparer is accepted. No raw-photo preprocessing.
public struct KleinPreparedStartingImage {
    public let width: Int, height: Int
    public let rgb: [UInt8]
    public let fileSHA256: String
    public var rgbSHA256: String { KleinHashing.sha256(of: Data(rgb)) }
    public static func load(_ url: URL, width: Int, height: Int) throws -> Self {
        guard url.isFileURL else { throw KleinTransformerError.invalidInput("prepared input must be a local PNG file") }
        _ = try KleinLatent.grid(height: height,width: width)
        let data = try Data(contentsOf: url), info = try ImageOutput.pngInfo(at: url)
        // The bytes validated below (first in-memory read) must be the ones the pinned file digest names; the digest is checked
        // again after the ImageIO decode (a later read of the file), so a change between the reads cannot pass either one.
        guard KleinHashing.sha256(of: data) == info.sha256 else { throw KleinTransformerError.invalidInput("input changed during read") }
        guard data.count >= 33, Array(data[8..<12]) == [0,0,0,13], info.width == width, info.height == height,
              info.bitDepth == 8, info.colorType == 2, data[26] == 0, data[27] == 0, data[28] == 0 else {
            throw KleinTransformerError.invalidInput("prepared input must be exact-size, noninterlaced 8-bit RGB PNG")
        }
        // Reject animated PNG and truncated chunks. ImageIO supplies decoded raw RGB samples, without a drawing/color context.
        var offset = 8, ended = false
        while offset + 12 <= data.count {
            let count = data[offset..<offset+4].reduce(0) { $0 << 8 | Int($1) }
            guard count <= data.count - offset - 12 else { throw KleinTransformerError.invalidInput("truncated PNG chunk") }
            let kind = String(decoding: data[offset+4..<offset+8],as: UTF8.self)
            guard kind != "acTL", kind != "fcTL", kind != "fdAT" else { throw KleinTransformerError.invalidInput("one still PNG required") }
            if kind == "eXIf" { try validateEXIF(Array(data[offset+8..<offset+8+count])) }
            offset += count + 12
            if kind == "IEND" {
                guard count == 0 else { throw KleinTransformerError.invalidInput("invalid PNG IEND") }
                ended = true; break
            }
        }
        guard ended, offset == data.count,
              let source = CGImageSourceCreateWithURL(url as CFURL,nil), CGImageSourceGetCount(source) == 1 else {
            throw KleinTransformerError.invalidInput("invalid still PNG")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString: Any]
        let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        guard orientation == 1 else { throw KleinTransformerError.invalidInput("unprepared EXIF orientation") }
        let image = try ImageOutput.readPNG(at: url)
        guard image.width == width, image.height == height, image.bytes.count == width*height*3,
              try KleinHashing.sha256(of: url) == info.sha256 else { throw KleinTransformerError.invalidInput("input changed during read") }
        return Self(width: width,height: height,rgb: image.bytes,fileSHA256: info.sha256)
    }
    /// PNG eXIf is a TIFF stream, without a JPEG Exif preamble. Validate IFD0 orientation independently of ImageIO.
    static func validateEXIF(_ data: [UInt8]) throws {
        func invalid() -> KleinTransformerError { .invalidInput("unprepared or malformed PNG EXIF orientation") }
        guard data.count >= 8, Array(data[0..<2]) == [0x49,0x49] || Array(data[0..<2]) == [0x4d,0x4d] else { throw invalid() }
        let little = data[0] == 0x49
        func read(_ offset: Int,_ count: Int) throws -> Int {
            guard offset >= 0, offset <= data.count-count else { throw invalid() }
            let bytes = data[offset..<offset+count]
            return (little ? Array(bytes.reversed()) : Array(bytes)).reduce(0) { ($0 << 8) | Int($1) }
        }
        guard try read(2,2) == 42 else { throw invalid() }
        let ifd = try read(4,4), count = try read(ifd,2)
        guard ifd >= 8, count <= (data.count-ifd-2)/12 else { throw invalid() }
        var found = false
        for i in 0..<count {
            let entry = ifd+2+i*12
            if try read(entry,2) == 0x0112 {
                guard !found, try read(entry+2,2) == 3, try read(entry+4,4) == 1, try read(entry+8,2) == 1 else { throw invalid() }
                found = true
            }
        }
    }
    public func normalized() -> MLXArray {
        // numpy float32 division happens BEFORE mx.array; do not replace with a reciprocal multiply or BF16 conversion.
        let unit = MLXArray(rgb.map { Float($0) / Float(255) }).reshaped(1,height,width,3).transposed(0,3,1,2)
        return 2.0 * unit - 1.0
    }
}
public struct KleinStartingImageRequest: Sendable {
    public let prompt: String
    public let width: Int, height: Int, steps: Int
    public let seed: UInt64
    public let image: URL
    public let strength: Double, guidance: Double
    public let scheduler: String
    public let tiling: Bool
    public init(prompt: String,width: Int,height: Int,seed: UInt64,steps: Int,image: URL,strength: Double,
                guidance: Double = 1,scheduler: String = "flow_match_euler_discrete",tiling: Bool = false) {
        self.prompt = prompt; self.width = width; self.height = height; self.seed = seed; self.steps = steps
        self.image = image; self.strength = strength; self.guidance = guidance; self.scheduler = scheduler; self.tiling = tiling
    }
    public static func start(steps: Int,strength: Double) throws -> Int {
        guard (2...8).contains(steps), strength.isFinite, strength > 0, strength < 1 else {
            throw KleinTransformerError.invalidInput("img2img requires steps 2...8 and finite 0 < strength < 1")
        }
        let start = max(1,Int(Double(steps)*strength))
        guard start < steps else { throw KleinTransformerError.invalidInput("no remaining denoising step") }
        return start
    }
    /// Host change slider mapping: the exact expression a host app uses to map a strength slider to `imageStrength`
    /// (`Double(steps - change + 1) / Double(steps) - 0.5 / Double(steps)`), so every bit pattern the host sends is the one used here.
    public static func strength(steps: Int,change: Int) throws -> Double {
        guard (2...8).contains(steps), (1..<steps).contains(change) else { throw KleinTransformerError.invalidInput("invalid host change") }
        return Double(steps-change+1)/Double(steps)-0.5/Double(steps)
    }
    public func validate() throws -> KleinPreparedStartingImage {
        guard !prompt.isEmpty, guidance == 1, scheduler == "flow_match_euler_discrete", !tiling else {
            throw KleinTransformerError.invalidInput("nonempty prompt, guidance=1, untiled flow_match_euler_discrete required")
        }
        _ = try Self.start(steps: steps,strength: strength)
        return try KleinPreparedStartingImage.load(image,width: width,height: height)
    }
}
public enum KleinStartingImageMath {
    public static func spatial(_ encoded: MLXArray,height: Int,width: Int) throws -> MLXArray {
        var x = encoded
        if x.ndim == 5, x.dim(2) == 1 { x = x[0...,0...,0,0...,0...] }
        guard x.ndim == 4, x.dim(0) == 1, x.dim(1) == 32, x.dim(2) >= 2, x.dim(3) >= 2,
              height > 0, width > 0, height % 2 == 0, width % 2 == 0 else { throw KleinTransformerError.invalidInput("invalid encoded spatial shape") }
        let eh = x.dim(2) - x.dim(2)%2, ew = x.dim(3) - x.dim(3)%2
        x = x[0...,0...,0..<eh,0..<ew]
        if eh > height { let offset = (eh-height)/2; x = x[0...,0...,offset..<offset+height,0...] }
        if eh < height { let p = (height-eh)/2; x = padded(x,widths: [.init(0),.init(0),.init((p,height-eh-p)),.init(0)]) }
        if ew > width { let offset = (ew-width)/2; x = x[0...,0...,0...,offset..<offset+width] }
        if ew < width { let p = (width-ew)/2; x = padded(x,widths: [.init(0),.init(0),.init(0),.init((p,width-ew-p))]) }
        return x
    }
    public static func normalize(_ x: MLXArray,mean: MLXArray,variance: MLXArray) throws -> MLXArray {
        guard x.ndim == 4, x.dim(1) == 128, x.size > 0, x.dtype == .bfloat16,
              mean.shape == [128], variance.shape == [128], mean.dtype == .bfloat16, variance.dtype == .bfloat16 else {
            throw KleinTransformerError.invalidInput("BN requires BF16 patched encoded and statistics")
        }
        let m = mean.reshaped(1,128,1,1).asType(x.dtype)
        let std = MLX.sqrt(variance.reshaped(1,128,1,1)+KleinVAELayout.bnEps).asType(x.dtype)
        return (x-m)/std
    }
    public static func clean(_ encoded: MLXArray,vae: KleinVAE,width: Int,height: Int,
                             trace: ((String,MLXArray) throws -> Void)? = nil) throws -> MLXArray {
        let matched = try spatial(encoded,height: height/8,width: width/8); try trace?("matched",matched)
        let patched = try KleinLatent.patchify(matched); try trace?("patched",patched)
        let bn = try normalize(patched,mean: vae.weights.tensor("bn.running_mean"),variance: vae.weights.tensor("bn.running_var"))
        try trace?("normalized_latent",bn)
        let clean = try KleinLatent.pack(bn); try trace?("clean",clean)
        return clean
    }
    public static func blend(clean: MLXArray,noise: MLXArray,sigma: MLXArray) throws -> MLXArray {
        guard clean.shape == noise.shape, clean.size > 0, clean.dtype == .bfloat16, noise.dtype == .bfloat16,
              sigma.ndim == 0, sigma.dtype == .float32 else { throw KleinTransformerError.invalidInput("invalid img2img interpolation") }
        return (1 - sigma) * clean + sigma * noise
    }
}
extension KleinGenerator {
    /// onStep uses the production absolute 1-based schedule position, including skipped steps.
    public func generate(_ request: KleinStartingImageRequest,onStep: ((Int,MLXArray) throws -> Void)? = nil,
                         trace: ((String,MLXArray) throws -> Void)? = nil) throws -> KleinGenerationOutput {
        let image = try request.validate() // Fail before prompt/model computation.
        let start = try KleinStartingImageRequest.start(steps: request.steps,strength: request.strength)
        let source = MLXArray(image.rgb).reshaped(request.height,request.width,3), normalized = image.normalized()
        try trace?("source_rgb",source); try trace?("prepared",normalized); try trace?("encoder_input",normalized)
        let encoded = try vae.encodePrepared(normalized) { n,a in if n == "encoded" { try trace?(n,a) } }
        let clean = try KleinStartingImageMath.clean(encoded,vae: vae,width: request.width,height: request.height,trace: trace)
        let prompt = try encode(request.prompt)
        let noise = try KleinLatent.preparePacked(seed: request.seed,height: request.height,width: request.width)
        let scheduler = try KleinScheduler(steps: request.steps,height: request.height,width: request.width)
        var latents = try KleinStartingImageMath.blend(clean: clean,noise: noise.latents,sigma: scheduler.sigmas[start])
        for (n,a) in ["prompt_embeds":prompt.promptEmbeds,"text_ids":prompt.textIDs,"latent_ids":noise.ids,"seeded_noise":noise.latents,
                      "sigmas":scheduler.sigmas,"timesteps":scheduler.timesteps,"initial":latents,"start":MLXArray(Int32(start))] { eval(a); try trace?(n,a) }
        var callbacks: [Int32] = []
        for t in start..<request.steps {
            try trace?("input.\(t)",latents)
            let prediction = predict(latents,noise.ids,prompt.promptEmbeds,prompt.textIDs,scheduler.timesteps[t])
            latents = try scheduler.step(noise: prediction,timestep: t,latents: latents); eval(latents)
            try trace?("noise.\(t)",prediction); try trace?("step.\(t+1)",latents)
            callbacks.append(Int32(t+1)); try onStep?(t+1,latents)
        }
        try trace?("callback_indices",MLXArray(callbacks)); try trace?("after_loop",latents)
        let packed = try KleinLatent.unpack(latents,height: request.height,width: request.width); try trace?("packed",packed)
        let decoded = try vae.decodeStartingImagePacked(packed) { n,a in if n == "unpacked" { try trace?(n,a) } }
        let pixels = try KleinGenerationPixels.convert(decoded,width: request.width,height: request.height)
        eval(decoded,pixels); try trace?("decoded",decoded); try trace?("pixels",pixels)
        guard try KleinHashing.sha256(of: request.image) == image.fileSHA256 else { throw KleinTransformerError.artifactChanged("source PNG changed") }
        return KleinGenerationOutput(latents: latents,packed: packed,decoded: decoded,pixels: pixels)
    }
}
