import Foundation
import MLX

/// Port of MFLUX 0.20's `VAETiler` with SeedVR2's `TilingConfig()` (encode: 512-pixel tiles, 64-pixel overlap; decode: 64-latent
/// tiles, 8-latent overlap). Each tile is encoded or decoded and evaluated on its own, then blended on the CPU in float32 with
/// raised-cosine ramps over the overlaps, as MFLUX does in numpy. This keeps the VAE's peak at one tile, whatever the output size.
public enum SeedVR2VAETiler {
    public static let encodeTileSize = 512
    public static let encodeTileOverlap = 64
    public static let decodeTileSize = 512
    /// `vae_decode_overlap` (latents) times the spatial scale: 64 pixels.
    public static let decodeTileOverlap = 8 * SeedVR2Layout.vaeSpatialScale

    /// `VAEUtil.encode` with tiling on: `[1, 3, H, W]` to the latent `[1, 16, 1, ceil(H/8), ceil(W/8)]` (float32).
    public static func encode(_ image: MLXArray, vae: SeedVR2VAE) -> MLXArray {
        let (height, width) = (image.dim(2), image.dim(3))
        if height <= encodeTileSize && width <= encodeTileSize {
            return vae.encode(image.expandedDimensions(axis: 2))
        }
        let scale = SeedVR2Layout.vaeSpatialScale
        let latentTile = max(1, encodeTileSize / scale)
        let latentOverlap = max(0, min(encodeTileOverlap / scale, latentTile - 1))
        let stride = max(1, latentTile - latentOverlap)
        let (latentHeight, latentWidth) = ((height + scale - 1) / scale, (width + scale - 1) / scale)
        let ramp = cosRamp(latentOverlap)
        var blend = Blend(channels: SeedVR2Layout.latentChannels, height: latentHeight, width: latentWidth)
        for y in Swift.stride(from: 0, to: latentHeight, by: stride) {
            let yEnd = min(y + latentTile, latentHeight)
            for x in Swift.stride(from: 0, to: latentWidth, by: stride) {
                let xEnd = min(x + latentTile, latentWidth)
                // Too-small sliver tiles at the end are skipped (the previous tile already covers them).
                if (y > 0 && yEnd - y <= latentOverlap) || (x > 0 && xEnd - x <= latentOverlap) { continue }
                let tile = image[0..., 0..., (y * scale) ..< min(yEnd * scale, height), (x * scale) ..< min(xEnd * scale, width)]
                var encoded = vae.encode(tile.expandedDimensions(axis: 2))
                if encoded.dim(2) == 1 { encoded = encoded[0..., 0..., 0] }
                let effH = min(yEnd - y, encoded.dim(2), latentHeight - y), effW = min(xEnd - x, encoded.dim(3), latentWidth - x)
                blend.add(
                    tile: encoded, at: (y, x), size: (effH, effW), overlap: (latentOverlap, latentOverlap), ramp: ramp,
                    lastRow: yEnd >= latentHeight, lastColumn: xEnd >= latentWidth)
            }
        }
        return blend.result().reshaped(1, SeedVR2Layout.latentChannels, 1, latentHeight, latentWidth)
    }

    /// `VAEUtil.decode` with tiling on: `[1, 16, 1, h, w]` to `[1, 3, 8h, 8w]` (float32).
    public static func decode(_ latent: MLXArray, vae: SeedVR2VAE) -> MLXArray {
        let latent = latent.ndim == 4 ? latent.expandedDimensions(axis: 2) : latent
        let scale = SeedVR2Layout.vaeSpatialScale
        let (latentHeight, latentWidth) = (latent.dim(3), latent.dim(4))
        let (height, width) = (latentHeight * scale, latentWidth * scale)
        let latentTile = max(1, decodeTileSize / scale)
        if latentHeight <= latentTile && latentWidth <= latentTile {
            let decoded = vae.decode(latent)
            return decoded.dim(2) == 1 ? decoded[0..., 0..., 0] : decoded
        }
        let latentOverlap = max(0, min(decodeTileOverlap / scale, latentTile - 1))
        let stride = max(1, latentTile - latentOverlap)
        let ramp = cosRamp(decodeTileOverlap)
        var blend = Blend(channels: 3, height: height, width: width)
        for y in Swift.stride(from: 0, to: latentHeight, by: stride) {
            let yEnd = min(y + latentTile, latentHeight)
            for x in Swift.stride(from: 0, to: latentWidth, by: stride) {
                let xEnd = min(x + latentTile, latentWidth)
                if (y > 0 && yEnd - y <= latentOverlap) || (x > 0 && xEnd - x <= latentOverlap) { continue }
                var decoded = vae.decode(latent[0..., 0..., 0..., y ..< yEnd, x ..< xEnd])
                if decoded.dim(2) == 1 { decoded = decoded[0..., 0..., 0] }
                let (outY, outX) = (y * scale, x * scale)
                let effH = min((yEnd - y) * scale, decoded.dim(2), height - outY)
                let effW = min((xEnd - x) * scale, decoded.dim(3), width - outX)
                blend.add(
                    tile: decoded, at: (outY, outX), size: (effH, effW), overlap: (decodeTileOverlap, decodeTileOverlap), ramp: ramp,
                    lastRow: yEnd >= latentHeight, lastColumn: xEnd >= latentWidth)
            }
        }
        return blend.result().reshaped(1, 3, height, width)
    }

    /// The weighted sum and weight total of the tiles, in float32, channels first.
    struct Blend {
        let channels: Int, height: Int, width: Int
        var sum: [Float]
        var count: [Float]

        init(channels: Int, height: Int, width: Int) {
            (self.channels, self.height, self.width) = (channels, height, width)
            sum = [Float](repeating: 0, count: channels * height * width)
            count = [Float](repeating: 0, count: height * width)
        }

        /// Evaluates the tile (`[1, C, h, w]`), crops it to `size` and adds it with weights that ramp up over the overlap where a tile
        /// lies before it and down where one follows (`ov = min(overlap, eff - 1)`), as `VAETiler` does.
        mutating func add(
            tile: MLXArray, at origin: (Int, Int), size: (Int, Int), overlap: (Int, Int), ramp: [Float], lastRow: Bool, lastColumn: Bool
        ) {
            let values = tile.asType(.float32)
            eval(values)
            let data = values.asArray(Float.self)
            let (tileH, tileW) = (values.dim(2), values.dim(3))
            func weights(_ count: Int, _ overlap: Int, first: Bool, last: Bool) -> [Float] {
                var w = [Float](repeating: 1, count: count)
                let ov = max(0, min(overlap, count - 1))
                if ov > 0 {
                    if !first { for i in 0 ..< ov { w[i] = ramp[i] } }
                    if !last { for i in 0 ..< ov { w[count - ov + i] = 1 - ramp[i] } }
                }
                return w
            }
            let wh = weights(size.0, overlap.0, first: origin.0 == 0, last: lastRow)
            let ww = weights(size.1, overlap.1, first: origin.1 == 0, last: lastColumn)
            let (channels, plane, width) = (channels, height * width, width)
            data.withUnsafeBufferPointer { d in
                sum.withUnsafeMutableBufferPointer { s in
                    count.withUnsafeMutableBufferPointer { n in
                        for y in 0 ..< size.0 {
                            for x in 0 ..< size.1 {
                                let w = wh[y] * ww[x]
                                let target = (origin.0 + y) * width + origin.1 + x
                                for c in 0 ..< channels { s[c * plane + target] += d[(c * tileH + y) * tileW + x] * w }
                                n[target] += w
                            }
                        }
                    }
                }
            }
        }

        /// `out / clip(count, 1e-6)`, as `[C, H, W]` float32.
        func result() -> MLXArray {
            var out = [Float](repeating: 0, count: sum.count)
            let plane = height * width
            sum.withUnsafeBufferPointer { s in
                count.withUnsafeBufferPointer { n in
                    out.withUnsafeMutableBufferPointer { o in
                        for c in 0 ..< channels {
                            for i in 0 ..< plane { o[c * plane + i] = s[c * plane + i] / max(n[i], Float(1e-6)) }
                        }
                    }
                }
            }
            return MLXArray(out, [channels, height, width])
        }
    }

    /// `VAETiler._cos_ramp(n)`: `0.5 - 0.5 * cos(linspace(0, 1, n) * pi)` in numpy float32. numpy's float32 cosine is not the C
    /// library's (5 of the 64 decode values differ in the last bit), so the two ramps the tiling uses are pinned as numpy computes them.
    static func cosRamp(_ n: Int) -> [Float] {
        if let pinned = pinnedRamps[n] { return pinned.map(Float.init(bitPattern:)) }
        guard n > 0 else { return [] }
        return (0 ..< n).map { i in
            let t = i == n - 1 ? Float(1) : Float(Double(i) * (1.0 / Double(n - 1)))
            return Float(0.5) - Float(0.5) * Foundation.cos(t * Float.pi)
        }
    }

    static let pinnedRamps: [Int: [UInt32]] = [
        8: [0x0000_0000, 0x3D4A_D0D8, 0x3E40_C5F4, 0x3EC7_08E0, 0x3F1C_7B92, 0x3F4F_CE85, 0x3F73_52F3, 0x3F80_0000],
        64: [
            0x0000_0000, 0x3A22_EE00, 0x3B22_D500, 0x3BB6_FF00, 0x3C22_6D60, 0x3C7D_51A0, 0x3CB5_F950, 0x3CF7_04E0,
            0x3D20_D118, 0x3D4A_D0D8, 0x3D79_66F8, 0x3D96_3AEC, 0x3DB1_EE78, 0x3DCF_BC80, 0x3DEF_920C, 0x3E08_AD6E,
            0x3E1A_80B6, 0x3E2D_3786, 0x3E40_C5F4, 0x3E55_1F8E, 0x3E6A_3764, 0x3E80_0000, 0x3E8B_35C8, 0x3E96_B5E4,
            0x3EA2_7904, 0x3EAE_77AA, 0x3EBA_AA34, 0x3EC7_08E0, 0x3ED3_8BCC, 0x3EE0_2B04, 0x3EEC_DE7F, 0x3EF9_9E26,
            0x3F03_30EF, 0x3F09_90C2, 0x3F0F_EA80, 0x3F16_3A1C, 0x3F1C_7B92, 0x3F22_AAE8, 0x3F28_C42D, 0x3F2E_C380,
            0x3F34_A510, 0x3F3A_651E, 0x3F40_0000, 0x3F45_722A, 0x3F4A_B81E, 0x3F4F_CE85, 0x3F54_B220, 0x3F59_5FD4,
            0x3F5D_D4A5, 0x3F62_0DC0, 0x3F66_0870, 0x3F69_C231, 0x3F6D_38A3, 0x3F70_6990, 0x3F73_52F3, 0x3F75_F2EF,
            0x3F78_47DA, 0x3F7A_5036, 0x3F7C_0ABA, 0x3F7D_764B, 0x3F7E_9202, 0x3F7F_5D2B, 0x3F7F_D744, 0x3F80_0000,
        ],
    ]
}
