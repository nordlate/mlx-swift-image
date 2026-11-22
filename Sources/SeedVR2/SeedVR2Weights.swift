import Foundation
import MLX
import MLXImageCore

/// The loaded SeedVR2 3B weights, keyed by the names stored in the fp16 files. A quantized linear layer `name` holds
/// `name.weight` (packed uint32), `name.scales` and `name.biases` (fp16), as MLX's `QuantizedLinear` does, plus its `name.bias` if
/// the layer has one; every other tensor is the stored fp16 tensor, unchanged (VAE conv weights transposed to MLX's
/// `[out, kt, kh, kw, in]`, as MFLUX's `transpose_conv3d_weight` does).
public struct SeedVR2Weights {
    public let transformer: [String: MLXArray]
    public let vae: [String: MLXArray]

    public init(transformer: [String: MLXArray], vae: [String: MLXArray]) {
        self.transformer = transformer
        self.vae = vae
    }

    /// The transformer and VAE linear layers that are stored quantized after load.
    public static let transformerQuantized = SeedVR2Layout.transformerQuantizedWeights()
    public static let vaeQuantized = SeedVR2Layout.vaeQuantizedWeights()

    /// Validates both files (size and header: names, dtypes, shapes) before reading any tensor data.
    public static func validate(modelDirectory: URL) throws {
        for (file, expected) in [(SeedVR2Pins.transformer, SeedVR2Layout.transformerTensors()), (SeedVR2Pins.vae, SeedVR2Layout.vaeTensors())] {
            let url = modelDirectory.appendingPathComponent(file.name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw SeedVR2Error.invalidWeights("\(file.name) is missing from \(modelDirectory.path)")
            }
            let size = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int
            guard size == file.bytes else {
                throw SeedVR2Error.invalidWeights("\(file.name) is \(size ?? -1) bytes, expected \(file.bytes) (revision \(SeedVR2Pins.revision))")
            }
            try SeedVR2Layout.validate(header: try SafetensorsHeader.read(from: url), expected: expected, file: file.name)
        }
    }

    /// Validates, then loads both files and quantizes the linear layers to 4 bits (affine, group size 64) with MLX's `quantize`, the
    /// kernel MFLUX's `nn.quantize` runs. Tensors are read, quantized and evaluated one at a time and the fp16 source is released
    /// before the next is read, so the peak during loading is the quantized model plus one fp16 matrix (with `Memory.cacheLimit = 0`;
    /// otherwise MLX keeps freed buffers in its cache).
    public static func load(modelDirectory: URL) throws -> SeedVR2Weights {
        try validate(modelDirectory: modelDirectory)
        let transformer = try loadQuantizing(
            url: modelDirectory.appendingPathComponent(SeedVR2Layout.transformerFile), quantized: transformerQuantized, transposeConvs: false)
        let vae = try loadQuantizing(url: modelDirectory.appendingPathComponent(SeedVR2Layout.vaeFile), quantized: vaeQuantized, transposeConvs: true)
        return SeedVR2Weights(transformer: transformer, vae: vae)
    }

    static func loadQuantizing(url: URL, quantized: Set<String>, transposeConvs: Bool) throws -> [String: MLXArray] {
        var source = try loadArrays(url: url)
        var loaded: [String: MLXArray] = [:]
        for name in source.keys.sorted() {
            guard let array = source.removeValue(forKey: name) else { continue }
            if quantized.contains(name) {
                let prefix = String(name.dropLast(".weight".count))
                let (wq, scales, biases) = MLX.quantized(array, groupSize: SeedVR2Layout.groupSize, bits: SeedVR2Layout.bits, mode: .affine)
                guard let biases else { throw SeedVR2Error.invalidWeights("\(name): affine quantization returned no biases") }
                eval(wq, scales, biases)
                loaded[prefix + ".weight"] = wq
                loaded[prefix + ".scales"] = scales
                loaded[prefix + ".biases"] = biases
            } else {
                let stored = transposeConvs && array.ndim == 5 ? array.transposed(0, 2, 3, 4, 1) : array
                eval(stored)
                loaded[name] = stored
            }
        }
        return loaded
    }
}

extension Dictionary where Key == String, Value == MLXArray {
    func tensor(_ name: String) -> MLXArray {
        guard let t = self[name] else { preconditionFailure("SeedVR2 tensor \(name) missing; the loader validates the full layout") }
        return t
    }
}
