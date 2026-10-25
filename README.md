# mlx-swift-image

Native Swift inference for image generation on Apple silicon, built on [MLX Swift](https://github.com/ml-explore/mlx-swift):

- **FLUX.2 Klein 4B** (q4): text to image, and image to image from a prepared starting image
- **Z-Image Turbo** (q4): text to image
- **Z-Image Base** (q4): text to image with classifier-free guidance and a negative prompt

> **Unofficial.** This project is not affiliated with, endorsed by or maintained by Apple or the ml-explore/MLX project, nor by the authors of the
> models it runs (Z-Image: Alibaba Tongyi-MAI; FLUX.2: Black Forest Labs).

The Swift code is a
port of the reference Python/MFLUX pipelines, checked against frozen parity cases and golden pixels. Pixel-identical means identical to MFLUX on those cases with the MLX build used when they were exported; against other MLX builds, a small `rms_norm` difference between the Python wheel and the Swift package (see below) can move pixels by a few levels on average (mean 0.6–2.8 levels per channel on the Z-Image Turbo cases), with isolated pixels differing by up to about 150 levels.

## Models

| Model | Product | Modes | Steps | Upstream model card (weights licence) | Parity |
|---|---|---|---|---|---|
| FLUX.2 Klein 4B, q4 | `FluxKlein` | text to image, image to image | 1...8 | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | pixel-identical to MFLUX on the golden cases with the MLX build used when they were exported; not re-run against MFLUX since; 0.2.0 does not change Klein code, and a native-versus-native check against 0.1.0 gave identical output |
| Z-Image Turbo, q4 | `ZImage` | text to image | 4...12 | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | pixel-identical to MFLUX on the golden cases when they were exported; not reproduced with a later local re-export (see below) |
| Z-Image Base, q4 | `ZImage` | text to image, guidance and negative prompt | 1...50 (28...50 recommended) | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Apache 2.0; snapshot [mflux-community/z-image-base-mflux-q4](https://huggingface.co/mflux-community/z-image-base-mflux-q4)) | sigmas bit-identical and guided steps within tolerance against MFLUX 0.20; pixels within MFLUX's own eager-versus-compiled drift (see below) |

Model weights are **not included** and are not downloaded by this package. Each model has its own licence and terms: check the model cards of the
snapshots you use before running or redistributing them. The loaders read pre-quantized (q4) snapshots in the `mflux-save` layout and have no
quantizer of their own.

> **No safeguards.** These pipelines run the models directly. They do not include the content filters, safety checkers or other safeguards
> that model makers or hosted services may provide. If you ship them in an app, you are responsible for any safeguards your use needs.

## Requirements

- Apple silicon Mac
- macOS 26.2 or later (the Metal target is validated against 26.2)
- A Swift toolchain that supports `swift-tools-version: 6.0` (developed with Xcode 27); the package builds in Swift 5 language mode
- mlx-swift 0.32.3, pinned by commit in `Package.swift` and `Package.resolved`

## Layout

| Target | What |
|---|---|
| `MLXImageCore` | What both models share: the Qwen3 q4 text-encoder layout and ops, the Qwen tokenizer, safetensors header checks, PNG/JPEG output |
| `ZImage` | Z-Image Turbo and Base: text encoder, transformer, schedulers, VAE, weight loaders, generator |
| `FluxKlein` | FLUX.2 Klein 4B: text encoder, transformer, VAE (with encoder for starting images), scheduler, weight loaders, generator |
| `MLXImageParity` | Validation harness for the models: frozen cases, comparisons against reference exports, tolerances and budgets, parity metrics, pins and manifests. Not needed for inference |

`ZImage` and `FluxKlein` depend on `MLXImageCore` and never on each other, and neither depends on `MLXImageParity`. The two models keep separate
text encoders, schedulers, VAEs and transformers on purpose: they differ numerically (dtype flow, norm formulation and eps, rotation order,
attention precision, compiled or eager steps) and the golden pixels pin each. Only code that is semantically identical lives in `MLXImageCore`.

Tests live in `Tests/`, one test target per module (`MLXImageCoreTests`, `ZImageTests`, `FluxKleinTests`, `MLXImageParityTests`), with frozen
parity inputs in `Tests/Fixtures`.

## Using the library

Add the package at a pinned version (or revision) and depend on the library products you need (`MLXImageCore`, `ZImage`, `FluxKlein`, and `MLXImageParity` only if you run the parity harness):

```swift
.package(url: "https://github.com/nordlate/mlx-swift-image", exact: "0.2.0")
```

```swift
import FluxKlein
import ZImage

// Z-Image Turbo (steps 4...12, width and height multiples of 16)
let zimage = try ZImageGenerator(modelDirectory: URL(fileURLWithPath: "/path/to/z-image-turbo-q4"))
let image = try zimage.generate(GenerationRequest(prompt: "a lighthouse at dusk", seed: 1, width: 512, height: 512, steps: 8))

// Z-Image Base (steps 1...50, 28...50 recommended): guidance above 1.0 runs a negative-prompt pass per step
let base = try ZImageGenerator(modelDirectory: URL(fileURLWithPath: "/path/to/z-image-base-q4"), variant: .base)
let guided = try base.generate(GenerationRequest(
    prompt: "a lighthouse at dusk", seed: 1, width: 1024, height: 1024, steps: 28, guidance: 4.0, negativePrompt: "blurry, low quality"))

// FLUX.2 Klein 4B (steps 1...8); image to image from a prepared starting image uses KleinStartingImageRequest
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
| `ZIMAGE_BASE_MODEL_DIR` | A Z-Image Base q4 snapshot (read only) |
| `ZIMAGE_BASE_REFERENCE_DIR` | A Python/MFLUX 0.20 Z-Image Base reference export (`manifest.json`, per-case safetensors) for the guided step and pixel parity tests |
| `ZIMAGE_REFERENCE_DIR` | A Python/MFLUX reference export (`manifest.json` and per-case safetensors) for the Z-Image tokenizer parity test |

Without any of them the model-dependent tests are skipped and the rest run.

### Z-Image Base parity

Base follows MFLUX 0.20 (`ModelConfig.z_image()`): `FlowMatchEulerDiscreteScheduler` with MFLUX's empirical `mu` (not the diffusers `shift` of
6.0), and guidance combined as `noise + guidance * (noise - negativeNoise)`, run only above guidance 1.0 (a blank negative prompt is a single
space). Sigmas are compared bit for bit (from the reference export; nothing generated is committed). Guided steps are compared one at a time from MFLUX's own state
within the harness's limits. The unguided step-0 prediction differs from MFLUX's by 6e-4 to 1.1e-3 relative L2, compiled and eager alike (MFLUX's and the native compiled step-0 predictions equal
their own eager ones), even though every input (latents, sigma, timestep embedding, caption embedding, rotary tables) is bit-identical. The first differing tensor is
`mx.fast.rms_norm` on float32 activations, about 6e-8 apart between the Python MLX wheel and the Swift package (same MLX 0.32.2 core), compounded over the blocks. It is not
specific to Base: Turbo's forward differs by 6e-4 on the same inputs, and on three Turbo jobs exported with the local MFLUX 0.20 / MLX 0.32.2 Python wheel the native pixels differ by mean 0.6 to 2.8 levels (max 89 to 147), the same as before this change,
so Turbo does not meet the 2 / 0.1 level limits against that export. The original Turbo golden references are not in the repository, so whether they came from a different MLX build is unverified. Guided pixels differ further: the guidance weights amplify the difference, and MFLUX's own eager and compiled
runs of the same job differ by mean 2 to 14 levels. The pixel test therefore requires the native mean difference to stay within twice that measured MFLUX drift.

## Versioning

Semantic versioning, with a release tag per version; changes are listed per product in `CHANGELOG.md`. Before 1.0.0 a minor version may change
the public API. Pinning a tag or a commit is recommended.

## Contributing

Maintained for the needs of the app that uses it (Taltio); contributions are welcome. See `CONTRIBUTING.md`.

## Licence

MIT, see `LICENSE`. The ports derive from MFLUX (MIT); attributions are in `NOTICE`. Model weights are separate and covered by their own licences.
