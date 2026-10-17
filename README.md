# mlx-swift-image

Native Swift inference for image generation on Apple silicon, built on [MLX Swift](https://github.com/ml-explore/mlx-swift):

- **FLUX.2 Klein 4B** (q4): text to image, and image to image from a prepared starting image
- **Z-Image Turbo** (q4): text to image

> **Unofficial.** This project is not affiliated with, endorsed by or maintained by Apple or the ml-explore/MLX project, nor by the authors of the
> models it runs (Z-Image: Alibaba Tongyi-MAI; FLUX.2: Black Forest Labs).

The Swift code is a
port of the reference Python/MFLUX pipelines, checked against frozen parity cases and golden pixels.

## Models

Model weights are **not included** and are not downloaded by this package. Each model has its own licence and terms: check the model cards of the
snapshots you use before running or redistributing them. The loaders read pre-quantized (q4) snapshots in the `mflux-save` layout and have no
quantizer of their own.

## Requirements

- Apple silicon Mac
- macOS 26.2 or later (the Metal target is validated against 26.2)
- A Swift toolchain that supports `swift-tools-version: 6.0` (developed with Xcode 27); the package builds in Swift 5 language mode
- mlx-swift 0.32.3, pinned by commit in `Package.swift` and `Package.resolved`

## Layout

| Target | What |
|---|---|
| `MLXImageCore` | What both models share: the Qwen3 q4 text-encoder layout and ops, the Qwen tokenizer, safetensors header checks, PNG/JPEG output |
| `ZImage` | Z-Image Turbo: text encoder, transformer, scheduler, VAE, weight loaders, generator |
| `FluxKlein` | FLUX.2 Klein 4B: text encoder, transformer, VAE (with encoder for starting images), scheduler, weight loaders, generator |
| `MLXImageParity` | Validation harness for the two models: frozen cases, comparisons against reference exports, tolerances and budgets, parity metrics, pins and manifests. Not needed for inference |

`ZImage` and `FluxKlein` depend on `MLXImageCore` and never on each other, and neither depends on `MLXImageParity`. The two models keep separate
text encoders, schedulers, VAEs and transformers on purpose: they differ numerically (dtype flow, norm formulation and eps, rotation order,
attention precision, compiled or eager steps) and the golden pixels pin each. Only code that is semantically identical lives in `MLXImageCore`.

Tests live in `Tests/`, one test target per module (`MLXImageCoreTests`, `ZImageTests`, `FluxKleinTests`, `MLXImageParityTests`), with frozen
parity inputs in `Tests/Fixtures`.

## Using the library

Add the package at a pinned revision and depend on the library products you need (`MLXImageCore`, `ZImage`, `FluxKlein`, and `MLXImageParity` only if you run the parity harness):

```swift
.package(url: "https://github.com/nordlate/mlx-swift-image", revision: "<commit>")
```

```swift
import FluxKlein
import ZImage

// Z-Image Turbo (steps 4...12, width and height multiples of 16)
let zimage = try ZImageGenerator(modelDirectory: URL(fileURLWithPath: "/path/to/z-image-turbo-q4"))
let image = try zimage.generate(GenerationRequest(prompt: "a lighthouse at dusk", seed: 1, width: 512, height: 512, steps: 8))

// FLUX.2 Klein 4B (steps 1...8); KleinGenerationRequest also takes an optional starting image
let klein = try KleinGenerator(modelDirectory: URL(fileURLWithPath: "/path/to/flux2-klein-4b-q4"))
```

See `Sources/ZImage/Pipeline.swift` and `Sources/FluxKlein/KleinGeneration.swift` for the full request and output types. MLX needs a Metal
library at run time; the stock mlx-swift build provides it.

## Tests

```bash
swift build
swift test
```

Tests that need model files or a reference export skip themselves with a message naming the missing variable.

| Variable | Used for |
|---|---|
| `ZIMAGE_MODEL_DIR` (alias `ZIMAGE_Q4_MODEL_DIR`) | A Z-Image Turbo q4 snapshot (read only) |
| `KLEIN_MODEL_DIR` (alias `KLEIN_Q4_MODEL_DIR`) | A FLUX.2 Klein 4B q4 snapshot (read only) |
| `ZIMAGE_REFERENCE_DIR` | A Python/MFLUX reference export (`manifest.json` and per-case safetensors) for the Z-Image tokenizer parity test |

Without any of them the model-dependent tests are skipped and the rest run.

## Licence

MIT, see `LICENSE`. Model weights are separate and covered by their own licences.
