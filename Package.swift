// swift-tools-version: 6.0
// MLXSwiftImage: native MLX image generation for Apple silicon (Z-Image Turbo and Base q4, FLUX.2 Klein 4B and 9B q4, SeedVR2 3B upscaling). Unofficial; not affiliated with ml-explore.
import PackageDescription

// Pinned by commit, not by version string. A consumer that builds for release should also check the checked-out gitlinks of
// mlx-swift's submodules (MLX core, mlx-c) against the values it expects.
let mlxSwiftRevision = "19601207e9a0de51e03ee6ec0c3c5f3784275075"  // release 0.32.3

let package = Package(
    name: "MLXSwiftImage",
    // The Metal target is validated against 26.2.
    platforms: [.macOS("26.2")],
    products: [
        .library(name: "MLXImageCore", targets: ["MLXImageCore"]),
        .library(name: "ZImage", targets: ["ZImage"]),
        .library(name: "FluxKlein", targets: ["FluxKlein"]),
        .library(name: "SeedVR2", targets: ["SeedVR2"]),
        .library(name: "MLXImageParity", targets: ["MLXImageParity"]),
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
        // SeedVR2 3B super-resolution: depends only on MLXImageCore. The fixed positive text embedding (ByteDance's pos_emb, Apache 2.0)
        // ships as a resource, so no text encoder is needed.
        .target(
            name: "SeedVR2",
            dependencies: [
                "MLXImageCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            resources: [.copy("Resources/pos_emb.safetensors"), .copy("Resources/pos_emb.LICENSE-Apache-2.0.txt")]
        ),
        // The parity and validation harness for the models: frozen cases and budgets, reference comparison, native exports, pins and
        // manifests. Depends on the model modules; they never depend on it.
        .target(
            name: "MLXImageParity",
            dependencies: [
                "MLXImageCore", "ZImage", "FluxKlein", "SeedVR2",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        .testTarget(name: "MLXImageCoreTests", dependencies: ["MLXImageCore"]),
        .testTarget(name: "FluxKleinTests", dependencies: ["FluxKlein", "MLXImageCore"]),
        .testTarget(name: "ZImageTests", dependencies: ["ZImage", "MLXImageCore"]),
        .testTarget(name: "SeedVR2Tests", dependencies: ["SeedVR2", "MLXImageCore"]),
        .testTarget(name: "MLXImageParityTests", dependencies: ["MLXImageParity", "ZImage", "FluxKlein", "SeedVR2", "MLXImageCore"]),
    ],
    swiftLanguageModes: [.v5]
)
