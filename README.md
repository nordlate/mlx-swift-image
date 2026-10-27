# mlx-swift-image

Native Swift inference for image generation on Apple silicon, built on [MLX Swift](https://github.com/ml-explore/mlx-swift):

- **FLUX.2 Klein 4B** (q4): text to image, and image to image from a prepared starting image
- **FLUX.2 Klein 9B** (q4): the same modes as 4B, through the same code path, and a converter that makes its q4 snapshot from the official
  bf16 download, byte-identical to `mflux-save`
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
| FLUX.2 Klein 9B, q4 | `FluxKlein` (`KleinVariant.klein9B`) | text to image, image to image | 1...8 | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) (FLUX Non-Commercial License; gated; snapshot made locally with `mflux-save` 0.20 `--quantize 4`) | prompt embeddings, ids, sigmas and seeded noise bit-identical to MFLUX 0.20; steps, starting-image latents and pixels within MFLUX's own eager-versus-compiled (or CPU-versus-GPU) drift (see below) |
| Z-Image Turbo, q4 | `ZImage` | text to image | 4...12 | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | pixel-identical to MFLUX on the golden cases when they were exported; not reproduced with a later local re-export (see below) |
| Z-Image Base, q4 | `ZImage` | text to image, guidance and negative prompt | 1...50 (28...50 recommended) | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Apache 2.0; snapshot [mflux-community/z-image-base-mflux-q4](https://huggingface.co/mflux-community/z-image-base-mflux-q4)) | sigmas bit-identical and guided steps within tolerance against MFLUX 0.20; pixels within MFLUX's own eager-versus-compiled drift (see below) |

Model weights are **not included** and are not downloaded by this package. Each model has its own licence and terms: check the model cards of the
snapshots you use before running or redistributing them. The loaders read pre-quantized (q4) snapshots in the `mflux-save` layout. For FLUX.2
Klein 9B, `KleinConverter` makes that snapshot from the official bf16 download (see [Converting FLUX.2 Klein 9B](#converting-flux2-klein-9b));
the other models have no quantizer here. FLUX.2 Klein 9B is under the FLUX Non-Commercial License and is gated on Hugging Face: you need to
accept its terms, download it and convert it yourself; nothing in this repository contains or fetches its weights.

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
| `FluxKlein` | FLUX.2 Klein 4B and 9B (`KleinVariant`): text encoder, transformer, VAE (with encoder for starting images), scheduler, weight loaders, generator, and the 9B converter (`KleinConverter`). One code path; the variant sets the transformer and text-encoder sizes (`KleinTransformerLayout.klein4B`/`.klein9B`, `TextEncoderLayout.qwen3_4B`/`.qwen3_8B`) and the pinned snapshot layout |
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

// FLUX.2 Klein 9B: the same requests; the snapshot is validated as 9B before any tensor is read, and a 4B snapshot is refused
let klein9B = try KleinGenerator(modelDirectory: URL(fileURLWithPath: "/path/to/flux2-klein-9b-q4"), variant: .klein9B)
```

See `Sources/ZImage/Pipeline.swift` and `Sources/FluxKlein/KleinGeneration.swift` for the full request and output types. MLX needs a Metal
library at run time; the stock mlx-swift build provides it.

## Converting FLUX.2 Klein 9B

`KleinConverter` turns the official bf16 diffusers release ([black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B),
about 35 GB) into the q4 snapshot `KleinGenerator(modelDirectory:variant: .klein9B)` loads. The output is byte for byte what
`mflux-save --base-model flux2-klein-9b --quantize 4` (MFLUX 0.20) writes from the same files: tensor names, dtypes, shapes, data order,
shard split, metadata (`mflux_version` 0.20.0) and the tokenizer files the loader pins. No Python is needed.

```swift
import FluxKlein

let source = URL(fileURLWithPath: "/path/to/FLUX.2-klein-9B")        // the Hugging Face download, read only
let output = URL(fileURLWithPath: "/path/to/flux2-klein-9b-q4")      // must not exist yet

let snapshot = try KleinSourceSnapshot.inspect(directory: source)    // structure, configs, small-file hashes; no weights read
let needed = try KleinConverter.requiredBytes(for: snapshot.variant) // 9,541,191,231 bytes for 9B
let result = try await KleinConverter.convert(sourceDirectory: source, outputDirectory: output) { progress in
    print(progress.fractionCompleted)
}
let klein9B = try KleinGenerator(modelDirectory: result.outputDirectory, variant: .klein9B)
```

- **Validation** (`KleinSourceSnapshot.inspect`): `model_index.json` must name `Flux2KleinPipeline` with the Klein components and
  `is_distilled: true`; the transformer config selects the variant and both configs must agree with it; every required file must be present
  with the official size and every small file with its SHA-256; the component folders may hold only files of the release; the weight files'
  headers must hold exactly the expected bf16 tensors. Errors are `KleinConversionError` cases: `notKlein`, `unsupportedVariant` (another size,
  a base model, or Klein 4B, which has no conversion pins yet), `missingFile`, `unexpectedFile`, `sizeMismatch`, `hashMismatch`, `invalidFile`.
  The pins (`KleinOfficialRelease.klein9B`) are the Hugging Face LFS SHA-256s of revision `92196c8e`.
- **Conversion**: every linear layer and the token embedding are quantized (affine 4-bit, group size 64) on the CPU by `KleinQ4Quantizer`,
  which reproduces MLX's Metal `affine_quantize` kernel exactly. MLX's own CPU quantizer is not used: it starts the group maximum at minus
  infinity instead of 0 and rounds half to even instead of away from zero, so it gives different bytes for some groups. No GPU kernel runs.
  The weight files are memory-mapped and streamed tensor by tensor; each is hashed in the same pass, so the 35 GB are read once.
  A mismatching hash fails the conversion.
- **Output**: written into a hidden staging folder next to the output, synced, checked with the loaders' header validation and then renamed
  into place. Cancellation (`isCancelled`, or cancelling the task of the async `convert`) and any failure remove the staging folder. The
  source folder is only read. `requiredBytes(for:)` is the exact size of the output, for a disk-space check.
- **Data order**: MLX writes tensor data in the iteration order of a C++ `std::unordered_map`. The converter gets that order by inserting
  the same names, in mflux's parameter order, into the same map through mlx-c and saving placeholder scalars, then writes the real data at
  those offsets.

Measured on an M5 Max (64 GB), release build: 18.4 s for the 9B release (bound by reading and hashing the 35 GB; hashing alone takes 13 s),
peak `phys_footprint` 0.1 GiB (the source pages are file-backed and released after each tensor). MFLUX's own conversion peaks at 19.2 GB.

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
| `KLEIN9B_MODEL_DIR` | A FLUX.2 Klein 9B q4 snapshot written by `mflux-save` 0.20 (read only) |
| `KLEIN9B_BF16_DIR` | The official FLUX.2 Klein 9B bf16 download (read only). With `KLEIN9B_MODEL_DIR`, the converter test converts it, compares every file and tensor with the MFLUX snapshot and generates one image from each; the output goes under `KLEIN_CONVERSION_OUTPUT_DIR` (default: the temporary folder) and is removed unless `KLEIN_CONVERSION_KEEP=1` |
| `KLEIN9B_REFERENCE_DIR`, `KLEIN_REFERENCE_DIR` | Python/MFLUX 0.20 Klein 9B and 4B generation exports (`manifest.json`, per-case safetensors, input PNGs) for the cases in `Tests/Fixtures/reference/klein_generation_cases.json`; `KLEIN_PARITY_REPORT` optionally names a directory for the JSON reports |
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

### FLUX.2 Klein 9B

9B is the 4B graph with larger sizes: 8 double and 24 single blocks, 32 heads (width 4096), joint attention width 12288, and a Qwen3 8B text
encoder (hidden 4096, MLP 12288; the same 36 layers, heads and selected layers 9, 18, 27). The `mflux-save` 0.20 q4 snapshot has the same
format as 4B's (affine 4-bit, group size 64, bf16 scales and biases and norms, f32 `inv_freq`), split into two text-encoder shards (905
tensors), three transformer shards (341 + 136 + 62 = 539 tensors) and one VAE shard, with `mflux_version` 0.20.0. The VAE and tokenizer
files are identical to 4B's. About 9.5 GB on disk.

Against five MFLUX 0.20 jobs (three text to image at 1024², 672x896 and 512²; two image to image at 768² and 896x672) the prompt embeddings,
text and latent ids, sigmas, timesteps, seeded noise and starting-image RGB are bit-identical. Each step is compared from MFLUX's own state
and must stay within twice MFLUX's own eager-versus-compiled difference for that step (the largest, 1.5e-2 relative L2 against MFLUX's
1.6e-2), and the starting-image latents within twice MFLUX's own CPU-versus-GPU encoder difference. Klein in bf16 is that sensitive: MFLUX's
compiled and eager step-0 predictions already differ by about 2e-2. Native pixels differ from MFLUX's by mean 0.8 to 2.4 levels, against
0.7 to 2.7 between MFLUX's own eager and compiled runs; the test requires the native mean to stay within twice that drift. Klein 4B shows the
same against an MFLUX 0.20 export made in the same environment (mean 0.9 to 2.4 levels).

Measured on an M5 Max (64 GB), release build, warm runs, 4 steps: loading takes 0.3 to 0.6 s (the transformer is materialized, the text
encoder is read on first use); 1024² takes 6.9 s (Klein 4B: 3.5 s) and 512² 2.1 s (4B: 1.0 s); the text encoder takes 0.14 s. MLX's
peak allocation is 16.8 GiB at 1024², 12.9 GiB at 768² and 10.6 GiB at 512² (4B: 12.7 and 6.5 GiB). The weights account for 8.9 GiB;
denoising adds under 1 GiB at 1024²; the rest of the 1024² peak is the untiled VAE decode. With MLX's buffer cache left uncapped the
process footprint reaches 31.6 GiB at 1024²; with `Memory.cacheLimit = 0` it peaks at 12.5 GiB (the weights are memory-mapped), at about
10 % more time (7.6 s). A Mac with 16 GB is not enough: the GPU working-set limit there is about 10.7 GiB, below even the 512² peak. 32 GB
is the practical minimum for 1024²; 24 GB may run smaller sizes but was not tested.

## Versioning

Semantic versioning, with a release tag per version; changes are listed per product in `CHANGELOG.md`. Before 1.0.0 a minor version may change
the public API. Pinning a tag or a commit is recommended.

## Contributing

Maintained for the needs of the app that uses it (Taltio); contributions are welcome. See `CONTRIBUTING.md`.

## Licence

MIT, see `LICENSE`. The ports derive from MFLUX (MIT); attributions are in `NOTICE`. Model weights are separate and covered by their own licences.
