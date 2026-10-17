// swift-tools-version: 6.0
// MLXSwiftImage: native MLX image generation for Apple silicon (Z-Image Turbo q4, FLUX.2 Klein 4B q4). Unofficial; not affiliated with ml-explore.
// Taltio embeds the `zimage-worker` product as its production runtime: LocalWorker/build-native.sh builds it from here (pinned by
// LocalWorker/native.lock); the Xcode project does not reference the package.
import PackageDescription

// Pinned by commit, not by version string. LocalWorker/build-native.sh checks the checked-out gitlinks of mlx-swift's submodules
// (MLX core, mlx-c) against the values in LocalWorker/native.lock.
let mlxSwiftRevision = "19601207e9a0de51e03ee6ec0c3c5f3784275075"  // release 0.32.3

let package = Package(
    name: "MLXSwiftImage",
    // The helper and its Metal target are validated against 26.2; the Taltio app stays at 15.3.
    platforms: [.macOS("26.2")],
    products: [
        .library(name: "ZImage", targets: ["ZImage"]),
        .library(name: "FluxKlein", targets: ["FluxKlein"]),
        .library(name: "ImageWorker", targets: ["ImageWorker"]),
        .executable(name: "zimage-worker", targets: ["zimage-worker"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", revision: mlxSwiftRevision),
    ],
    targets: [
        .target(
            name: "ZImage",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        // Klein text encoding reuses ZImage's tokenizer, safetensors/layout loader and metrics; nothing there changes behavior.
        .target(
            name: "FluxKlein",
            dependencies: [
                "ZImage",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ]
        ),
        // The Taltio worker (contract v2). Sources/ImageWorker/LocalWorkerContract.swift is a symlink to the app's own file, so
        // host and helper compile the same contract. It serves Z-Image Turbo q4 and FLUX.2 Klein 4B q4.
        .target(name: "ImageWorker", dependencies: ["ZImage", "FluxKlein", .product(name: "MLX", package: "mlx-swift")]),
        .executableTarget(name: "zimage-worker", dependencies: ["ImageWorker"]),
        .testTarget(name: "FluxKleinTests", dependencies: ["FluxKlein", "ZImage"]),
        .testTarget(name: "ZImageTests", dependencies: ["ZImage"]),
        .testTarget(name: "ImageWorkerTests", dependencies: ["ImageWorker"]),
    ],
    swiftLanguageModes: [.v5]
)
