// swift-tools-version: 6.0
// MLXSwiftImage: native MLX image generation for Apple silicon (Z-Image Turbo q4, FLUX.2 Klein 4B q4). Unofficial; not affiliated with ml-explore.
// Built for the `zimage-worker` product, a helper process that apps embed as their local-generation runtime (see README.md).
import PackageDescription

// Pinned by commit, not by version string. A consumer that builds the worker for release should also check the checked-out gitlinks of
// mlx-swift's submodules (MLX core, mlx-c) against the values it expects.
let mlxSwiftRevision = "19601207e9a0de51e03ee6ec0c3c5f3784275075"  // release 0.32.3

let package = Package(
    name: "MLXSwiftImage",
    // The helper and its Metal target are validated against 26.2; a host app may target an older macOS, since it only launches the helper.
    platforms: [.macOS("26.2")],
    products: [
        .library(name: "MLXImageCore", targets: ["MLXImageCore"]),
        .library(name: "ZImage", targets: ["ZImage"]),
        .library(name: "FluxKlein", targets: ["FluxKlein"]),
        .library(name: "ImageWorker", targets: ["ImageWorker"]),
        .executable(name: "zimage-worker", targets: ["zimage-worker"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", revision: mlxSwiftRevision),
    ],
    targets: [
        // What Z-Image and Klein share: the Qwen3 q4 text-encoder layout and its identical ops, the Qwen tokenizer, safetensors headers,
        // parity metrics and PNG/JPEG output. Numerics that differ between the two models stay in their own targets.
        .target(
            name: "MLXImageCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .target(
            name: "ZImage",
            dependencies: [
                "MLXImageCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        // Klein does not depend on ZImage: tokenizer, safetensors/layout loader, metrics and image output come from MLXImageCore.
        .target(
            name: "FluxKlein",
            dependencies: [
                "MLXImageCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        // The worker (contract v2). Sources/ImageWorker/LocalWorkerContract.swift is the host/worker contract; a host app that compiles
        // its own copy of that file must keep it identical. It serves Z-Image Turbo q4 and FLUX.2 Klein 4B q4.
        .target(name: "ImageWorker", dependencies: ["MLXImageCore", "ZImage", "FluxKlein", .product(name: "MLX", package: "mlx-swift")]),
        .executableTarget(name: "zimage-worker", dependencies: ["ImageWorker"]),
        .testTarget(name: "MLXImageCoreTests", dependencies: ["MLXImageCore"]),
        .testTarget(name: "FluxKleinTests", dependencies: ["FluxKlein", "MLXImageCore"]),
        .testTarget(name: "ZImageTests", dependencies: ["ZImage", "MLXImageCore"]),
        .testTarget(name: "ImageWorkerTests", dependencies: ["ImageWorker"]),
    ],
    swiftLanguageModes: [.v5]
)
