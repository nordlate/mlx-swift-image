import Foundation
import MLX

/// Port of mflux's `Flux2LatentCreator` (`flux2/latent_creator/flux2_latent_creator.py`), op for op, for text-to-image: seeded packed
/// latents, grid position IDs, and the pack / unpack reshapes. The seeded draw is `mx.random.normal` float32 under
/// `mx.random.key(seed)`, then cast to bfloat16 (`ModelConfig.precision`).
public enum KleinLatent {
    public static let latentChannels = 32
    public static let patchedChannels = 128  // latent_channels * 2 * 2
    public static let vaeScaleFactor = 8
    /// `Config` rounds sizes down to a multiple of 16; the native port refuses anything else instead of rounding silently.
    public static let sizeMultiple = 16
    public static let sizeRange = 16...8192

    /// The latent grid `(height, width)` in tokens for an image of `height` x `width` pixels: `(H / 16, W / 16)`.
    public static func grid(height: Int, width: Int) throws -> (height: Int, width: Int) {
        for (label, value) in [("height", height), ("width", width)] {
            guard sizeRange.contains(value), value % sizeMultiple == 0 else {
                throw KleinTransformerError.invalidInput("\(label) \(value) must be a multiple of \(sizeMultiple) within \(sizeRange)")
            }
        }
        return (height / sizeMultiple, width / sizeMultiple)
    }

    /// `Flux2LatentCreator.patchify_latents`: `[B, C, H, W]` to `[B, 4C, H/2, W/2]` (a 5D input with a singleton frame axis is squeezed).
    public static func patchify(_ latents: MLXArray) throws -> MLXArray {
        var x = latents
        if x.ndim == 5, x.dim(2) == 1 { x = x[0..., 0..., 0, 0..., 0...] }
        guard x.ndim == 4, x.dim(2) % 2 == 0, x.dim(3) % 2 == 0 else {
            throw KleinTransformerError.invalidInput("patchify expects [B, C, even H, even W], got \(latents.shape)")
        }
        let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(b, c, h / 2, 2, w / 2, 2).transposed(0, 1, 3, 5, 2, 4).reshaped(b, c * 4, h / 2, w / 2)
    }

    /// `pack_latents`: `[B, C, H, W]` to `[B, H*W, C]`.
    public static func pack(_ latents: MLXArray) throws -> MLXArray {
        guard latents.ndim == 4 else { throw KleinTransformerError.invalidInput("pack expects [B, C, H, W], got \(latents.shape)") }
        let (b, c, h, w) = (latents.dim(0), latents.dim(1), latents.dim(2), latents.dim(3))
        return latents.reshaped(b, c, h * w).transposed(0, 2, 1)
    }

    /// `unpack_latents`: packed `[B, N, C]` to `[B, C, H/16, W/16]` for an image of `height` x `width` pixels. The sequence length must
    /// match the grid; a 4D input is returned unchanged, as in mflux.
    public static func unpack(_ latents: MLXArray, height: Int, width: Int) throws -> MLXArray {
        let g = try grid(height: height, width: width)
        if latents.ndim == 4 { return latents }
        guard latents.ndim == 3 else { throw KleinTransformerError.invalidInput("unpack expects packed [B, N, C], got \(latents.shape)") }
        let (b, n, c) = (latents.dim(0), latents.dim(1), latents.dim(2))
        guard n == g.height * g.width else {
            throw KleinTransformerError.invalidInput("packed latent seq_len \(n) does not match \(g.height)x\(g.width) for \(width)x\(height)")
        }
        return latents.reshaped(b, g.height, g.width, c).transposed(0, 3, 1, 2)
    }

    /// `prepare_grid_ids(latents, t_coord)`: int32 `[B, H*W, 4]` rows `(t, h, w, 0)` in row-major grid order.
    public static func gridIDs(batch: Int, height: Int, width: Int, tCoord: Int = 0) throws -> MLXArray {
        guard batch >= 1, (1...(sizeRange.upperBound / sizeMultiple)).contains(height),
              (1...(sizeRange.upperBound / sizeMultiple)).contains(width), Int32(exactly: tCoord) != nil else {
            throw KleinTransformerError.invalidInput("invalid grid dimensions, batch or int32 time coordinate")
        }
        let hIDs = MLXArray(0 ..< Int32(height)), wIDs = MLXArray(0 ..< Int32(width))
        let hGrid = broadcast(expandedDimensions(hIDs, axis: 1), to: [height, width])
        let wGrid = broadcast(expandedDimensions(wIDs, axis: 0), to: [height, width])
        let flatH = hGrid.reshaped(-1), flatW = wGrid.reshaped(-1)
        let t = MLXArray.full(flatH.shape, values: MLXArray(Int32(tCoord)), dtype: .int32)
        let layer = MLXArray.zeros(flatH.shape, dtype: .int32)
        let coords = expandedDimensions(stacked([t, flatH, flatW, layer], axis: 1), axis: 0)
        return broadcast(coords, to: [batch, coords.dim(1), coords.dim(2)])
    }

    /// `prepare_latents`: the unpacked seeded noise `[B, 128, H/16, W/16]` in bfloat16, the grid IDs and the grid size.
    public static func prepare(seed: UInt64, height: Int, width: Int, batch: Int = 1) throws -> (latents: MLXArray, ids: MLXArray, gridHeight: Int, gridWidth: Int) {
        guard batch >= 1 else { throw KleinTransformerError.invalidInput("batch must be at least 1") }
        let g = try grid(height: height, width: width)
        let noise = MLXRandom.normal([batch, patchedChannels, g.height, g.width], key: MLXRandom.key(seed)).asType(.bfloat16)
        return (noise, try gridIDs(batch: batch, height: g.height, width: g.width, tCoord: 0), g.height, g.width)
    }

    /// `prepare_packed_latents`: packed `[B, N, 128]` bfloat16 noise, `[B, N, 4]` int32 IDs and the grid size.
    public static func preparePacked(seed: UInt64, height: Int, width: Int, batch: Int = 1) throws -> (latents: MLXArray, ids: MLXArray, gridHeight: Int, gridWidth: Int) {
        let p = try prepare(seed: seed, height: height, width: width, batch: batch)
        return (try pack(p.latents), p.ids, p.gridHeight, p.gridWidth)
    }
}
