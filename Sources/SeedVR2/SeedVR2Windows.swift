import Foundation
import MLX

/// Port of MFLUX's `WindowPartitioner` for one batch row: the windows of a `(t, h, w)` patch grid, the gather that orders the tokens
/// window by window (row-major inside each window), its inverse, and the window shapes. Computed on the CPU from integers only.
public struct SeedVR2Windows: Equatable {
    public struct Window: Equatable, Sendable {
        public let t: Range<Int>, h: Range<Int>, w: Range<Int>
        public var shape: [Int] { [t.count, h.count, w.count] }
        public var count: Int { t.count * h.count * w.count }
    }

    public let grid: [Int]
    public let windows: [Window]
    /// Token index (row-major in the grid) of each partitioned position.
    public let forward: [Int32]
    /// The inverse permutation (`argsort(forward)`).
    public let reverse: [Int32]

    public init(t: Int, h: Int, w: Int, shift: Bool) {
        grid = [t, h, w]
        windows = Self.makeWindows(t: t, h: h, w: w, shift: shift)
        var forward: [Int32] = []
        forward.reserveCapacity(t * h * w)
        for window in windows {
            for it in window.t {
                for ih in window.h {
                    for iw in window.w { forward.append(Int32((it * h + ih) * w + iw)) }
                }
            }
        }
        var reverse = [Int32](repeating: 0, count: forward.count)
        for (position, token) in forward.enumerated() { reverse[Int(token)] = Int32(position) }
        self.forward = forward
        self.reverse = reverse
    }

    /// `WindowPartitioner._make_windows(size, num_windows=(4, 3, 3), shift)`: the grid is rescaled to about 45 x 80 patches, split into
    /// 3 x 3 windows (and time into 4), and shifted by half a window on odd blocks. Python's `round` (half to even), `math.ceil` and
    /// `int` (toward zero) are reproduced on doubles.
    static func makeWindows(t: Int, h: Int, w: Int, shift: Bool) -> [Window] {
        let (numT, numH, numW) = (SeedVR2Layout.window.t, SeedVR2Layout.window.h, SeedVR2Layout.window.w)
        let scale = (Double(45 * 80) / Double(h * w)).squareRoot()
        let resizedH = (Double(h) * scale).rounded(.toNearestOrEven), resizedW = (Double(w) * scale).rounded(.toNearestOrEven)
        let wh = Int((resizedH / Double(numH)).rounded(.up))
        let ww = Int((resizedW / Double(numW)).rounded(.up))
        let wt = Int((Double(min(t, 30)) / Double(numT)).rounded(.up))
        var (st, sh, sw) = (0.0, 0.0, 0.0)
        var (nt, nh, nw): (Int, Int, Int)
        if shift {
            st = wt < t ? 0.5 : 0
            sh = wh < h ? 0.5 : 0
            sw = ww < w ? 0.5 : 0
            nt = st > 0 ? Int(((Double(t) - st) / Double(wt)).rounded(.up)) + 1 : 1
            nh = sh > 0 ? Int(((Double(h) - sh) / Double(wh)).rounded(.up)) + 1 : 1
            nw = sw > 0 ? Int(((Double(w) - sw) / Double(ww)).rounded(.up)) + 1 : 1
        } else {
            nt = Int((Double(t) / Double(wt)).rounded(.up))
            nh = Int((Double(h) / Double(wh)).rounded(.up))
            nw = Int((Double(w) / Double(ww)).rounded(.up))
        }
        func range(_ i: Int, _ s: Double, _ size: Int, _ limit: Int) -> Range<Int>? {
            let start = max(Int((Double(i) - s) * Double(size)), 0)
            let end = min(Int((Double(i) - s + 1) * Double(size)), limit)
            return end > start ? start ..< end : nil
        }
        var windows: [Window] = []
        for iw in 0 ..< nw {
            guard let wr = range(iw, sw, ww, w) else { continue }
            for ih in 0 ..< nh {
                guard let hr = range(ih, sh, wh, h) else { continue }
                for it in 0 ..< nt {
                    guard let tr = range(it, st, wt, t) else { continue }
                    windows.append(Window(t: tr, h: hr, w: wr))
                }
            }
        }
        return windows
    }
}

/// Port of MFLUX's `RoPEModule` with `freqs_for="lang"` and text RoPE (the 3B configuration), for one batch row.
enum SeedVR2RoPE {
    /// `_get_axial_freqs(freqs, *dims)`: per axis `outer(arange(n), freqs)` with each frequency repeated twice, broadcast over the
    /// grid and concatenated over the axes (`[..., 2 * len(freqs) * len(dims)]`).
    static func axialFrequencies(_ freqs: MLXArray, _ dims: [Int]) -> MLXArray {
        let perAxis = freqs.dim(0) * 2
        let target = dims + [perAxis]
        let f32 = freqs.asType(.float32)
        var all: [MLXArray] = []
        for (axis, size) in dims.enumerated() {
            let position = MLXArray((0 ..< size).map { Float($0) })
            var axisFreqs = repeated(outer(position, f32), count: 2, axis: -1)
            var shape = [Int](repeating: 1, count: dims.count) + [perAxis]
            shape[axis] = size
            axisFreqs = axisFreqs.reshaped(shape)
            all.append(broadcast(axisFreqs, to: target))
        }
        return concatenated(all, axis: -1)
    }

    /// `_rotate_half`: pairs `(x1, x2)` become `(-x2, x1)`.
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let pairs = x.reshaped(Array(shape.dropLast()) + [-1, 2])
        let x1 = pairs[.ellipsis, 0], x2 = pairs[.ellipsis, 1]
        return stacked([-x2, x1], axis: -1).reshaped(shape)
    }

    /// `_apply_rotary_emb(freqs, t)`: rotate the first `freqs.shape[-1]` channels in float32, cast back, keep the rest.
    static func apply(_ freqs: MLXArray, _ t: MLXArray) -> MLXArray {
        let rotated = freqs.dim(-1)
        let middle = t[.ellipsis, 0 ..< rotated], right = t[.ellipsis, rotated...]
        let dtype = middle.dtype
        let f = freqs.asType(.float32), m = middle.asType(.float32)
        let transformed = ((m * cos(f)) + (rotateHalf(m) * sin(f))).asType(dtype)
        if right.dim(-1) > 0 { return concatenated([transformed, right], axis: -1) }
        return transformed
    }

    /// `_apply_mm_rope_3d` (lang): the frequency tables for every window's video tokens (temporal positions after the text) and for
    /// the text tokens, which are the same in every window.
    static func tables(freqs: MLXArray, windows: [SeedVR2Windows.Window], textLength: Int) -> (video: MLXArray, text: MLXArray) {
        let maxTemporal = windows.map { $0.t.count + textLength }.max()!
        let maxHeight = windows.map(\.h.count).max()!, maxWidth = windows.map(\.w.count).max()!
        let full = axialFrequencies(freqs, [min(maxTemporal + 16, 1024), min(maxHeight + 4, 128), min(maxWidth + 4, 128)])
        let text1D = axialFrequencies(freqs, [min(textLength + 16, 1024)])
        let video = concatenated(
            windows.map { full[textLength ..< textLength + $0.t.count, 0 ..< $0.h.count, 0 ..< $0.w.count].reshaped(-1, full.dim(-1)) }, axis: 0)
        let text = tiled(text1D[0 ..< textLength], repetitions: [1, 3])
        return (video, text)
    }
}
