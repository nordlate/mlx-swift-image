import FluxKlein
import Foundation
import MLX

/// Per-tensor digests of the loaded weights and the quantized-module lists that the parity manifests record. Hashing every tensor of a
/// 4.6 GB snapshot is slow, so none of this runs on the inference path.

public extension KleinTextWeights {
    /// Per-tensor name, dtype, shape, byte count and SHA-256 of the loaded bytes (what the reference exporter reports for MFLUX's
    /// applied parameters).
    func digests() -> [String: KleinTensorDigest] {
        KleinDigestAudit.record(tensors.count)
        var result: [String: KleinTensorDigest] = [:]
        for (name, array) in tensors {
            eval(array)
            let data = array.asData(access: .copy).data  // row-major bytes as stored, what the Python reference hashes
            result[name] = KleinTensorDigest(
                dtype: KleinDType.name(array.dtype), shape: array.shape, bytes: data.count, sha256: KleinHashing.sha256(of: data))
        }
        return result
    }
}

public extension KleinTransformerWeights {
    /// Per-tensor dtype, shape, byte count and SHA-256 of the loaded bytes (what the reference exporter reports for MFLUX's applied
    /// transformer parameters).
    func digests() -> [String: KleinTensorDigest] {
        KleinDigestAudit.record(tensors.count)
        var result: [String: KleinTensorDigest] = [:]
        for (name, array) in tensors {
            eval(array)
            let data = KleinCompare.bytes(array)
            result[name] = KleinTensorDigest(dtype: KleinDType.name(array.dtype), shape: array.shape, bytes: data.count, sha256: KleinHashing.sha256(of: data))
        }
        return result
    }

    /// The module the applier quantizes at the stored precision: all 109 matrices, 4-bit, group 64, affine.
    static func quantizedModules() -> [KleinQuantizedModule] {
        KleinTransformerLayout.matrices().map {
            KleinQuantizedModule(name: $0.name, bits: KleinTransformerLayout.bits, groupSize: KleinTransformerLayout.groupSize, mode: "affine")
        }
    }
}

public extension KleinVAEWeights {
    func digests() -> [String: KleinTensorDigest] {
        KleinDigestAudit.record(tensors.count)
        var result: [String: KleinTensorDigest] = [:]
        for (name, array) in tensors {
            eval(array)
            let data = KleinCompare.bytes(array)
            result[name] = KleinTensorDigest(dtype: KleinDType.name(array.dtype), shape: array.shape, bytes: data.count, sha256: KleinHashing.sha256(of: data))
        }
        return result
    }

    static func quantizedModules() -> [KleinQuantizedModule] {
        KleinVAELayout.matrices().map {
            KleinQuantizedModule(name: $0.name, bits: KleinVAELayout.bits, groupSize: KleinVAELayout.groupSize, mode: "affine")
        }
    }
}

/// Counts the tensors hashed by `digests()` inside one `measure` scope (a test seam: the production load path must hash nothing). The count is task-local, so digests computed by
/// other tests running in parallel, or outside the scope, never reach it.
public enum KleinDigestAudit {
    public final class Scope: @unchecked Sendable {
        private let lock = NSLock()
        private var total = 0
        func add(_ count: Int) { lock.lock(); total += count; lock.unlock() }
        public var tensorCount: Int { lock.lock(); defer { lock.unlock() }; return total }
    }
    @TaskLocal static var scope: Scope?
    static func record(_ count: Int) { scope?.add(count) }
    /// Runs `body` and returns its result with the number of tensors `digests()` hashed inside it (on this task only).
    public static func measure<T>(_ body: () throws -> T) rethrows -> (result: T, tensorCount: Int) {
        let scope = Scope()
        let result = try $scope.withValue(scope, operation: body)
        return (result, scope.tensorCount)
    }
}

public struct KleinTensorDigest: Codable, Equatable, Sendable {
    public var dtype: String
    public var shape: [Int]
    public var bytes: Int
    public var sha256: String
}

public enum KleinDType {
    /// MLX Python's `str(dtype)`.
    public static func name(_ dtype: DType) -> String {
        switch dtype {
        case .bfloat16: "mlx.core.bfloat16"
        case .float32: "mlx.core.float32"
        case .uint32: "mlx.core.uint32"
        case .int64: "mlx.core.int64"
        case .int32: "mlx.core.int32"
        default: "mlx.core.\(dtype)"
        }
    }
}
