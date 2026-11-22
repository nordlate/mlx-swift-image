# mlx-swift-image

Native Swift inference for image generation on Apple silicon, built on [MLX Swift](https://github.com/ml-explore/mlx-swift):

- **FLUX.2 Klein 4B** (q4): text to image, and image to image from a prepared starting image
- **FLUX.2 Klein 9B** (q4): the same modes as 4B, through the same code path, and a converter that makes its q4 snapshot from the official
  bf16 download, byte-identical to `mflux-save`
- **Z-Image Turbo** (q4): text to image
- **Z-Image Base** (q4): text to image with classifier-free guidance and a negative prompt
- **SeedVR2 3B** (quantized to 4 bits at load): image super-resolution (upscaling) in one diffusion step

> **Unofficial.** This project is not affiliated with, endorsed by or maintained by Apple or the ml-explore/MLX project, nor by the authors of the
> models it runs (Z-Image: Alibaba Tongyi-MAI; FLUX.2: Black Forest Labs; SeedVR2: ByteDance Seed).

The Swift code is a
port of the reference Python/MFLUX pipelines, checked against frozen parity cases and golden pixels. Pixel-identical means identical to MFLUX on those cases with the MLX build used when they were exported; against other MLX builds, a small `rms_norm` difference between the Python wheel and the Swift package (see below) can move pixels by a few levels on average (mean 0.6–2.8 levels per channel on the Z-Image Turbo cases), with isolated pixels differing by up to about 150 levels.

## Models

| Model | Product | Modes | Steps | Upstream model card (weights licence) | Parity |
|---|---|---|---|---|---|
| FLUX.2 Klein 4B, q4 | `FluxKlein` | text to image, image to image | 1...8 | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | pixel-identical to MFLUX on the golden cases with the MLX build used when they were exported; not re-run against MFLUX since; 0.2.0 does not change Klein code, and a native-versus-native check against 0.1.0 gave identical output |
| FLUX.2 Klein 9B, q4 | `FluxKlein` (`KleinVariant.klein9B`) | text to image, image to image | 1...8 | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) (FLUX Non-Commercial License; gated; snapshot made locally with `mflux-save` 0.20 `--quantize 4`) | prompt embeddings, ids, sigmas and seeded noise bit-identical to MFLUX 0.20; steps, starting-image latents and pixels within MFLUX's own eager-versus-compiled (or CPU-versus-GPU) drift (see below) |
| Z-Image Turbo, q4 | `ZImage` | text to image | 4...12 | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | pixel-identical to MFLUX on the golden cases when they were exported; not reproduced with a later local re-export (see below) |
| Z-Image Base, q4 | `ZImage` | text to image, guidance and negative prompt | 1...50 (28...50 recommended) | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) (Apache 2.0; snapshot [mflux-community/z-image-base-mflux-q4](https://huggingface.co/mflux-community/z-image-base-mflux-q4)) | sigmas bit-identical and guided steps within tolerance against MFLUX 0.20; pixels within MFLUX's own eager-versus-compiled drift (see below) |
| SeedVR2 3B, 4-bit at load | `SeedVR2` | upscaling (image to image), `softness` 0...1 | 1 | [ByteDance-Seed/SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) (Apache 2.0; fp16 safetensors from [numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI), Apache 2.0) | preprocessing, seeded noise and the scheduler step bit-identical to MFLUX 0.20; VAE encode and decode and the transformer step within the harness's limits; pixels within MFLUX's own eager-versus-compiled drift (see below) |

Model weights are **not included** and are not downloaded by this package. Each model has its own licence and terms: check the model cards of the
snapshots you use before running or redistributing them. The Z-Image and Klein loaders read pre-quantized (q4) snapshots in the `mflux-save`
layout. For FLUX.2 Klein 9B, `KleinConverter` makes that snapshot from the official bf16 download (see [Converting FLUX.2 Klein 9B](#converting-flux2-klein-9b));
the other text-to-image models have no quantizer here. SeedVR2 reads the fp16 files and quantizes them to 4 bits when it loads them, as MFLUX
does; its only bundled data is SeedVR2's fixed positive text embedding (ByteDance's Apache 2.0 `pos_emb`, with its licence, see `NOTICE`). FLUX.2 Klein 9B is under the FLUX Non-Commercial License and is gated on Hugging Face: you need to
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
| `MLXImageCore` | What the models share: the Qwen3 q4 text-encoder layout and ops, the Qwen tokenizer, safetensors header checks, PNG/JPEG output |
| `ZImage` | Z-Image Turbo and Base: text encoder, transformer, schedulers, VAE, weight loaders, generator |
| `FluxKlein` | FLUX.2 Klein 4B and 9B (`KleinVariant`): text encoder, transformer, VAE (with encoder for starting images), scheduler, weight loaders, generator, and the 9B converter (`KleinConverter`). One code path; the variant sets the transformer and text-encoder sizes (`KleinTransformerLayout.klein4B`/`.klein9B`, `TextEncoderLayout.qwen3_4B`/`.qwen3_8B`) and the pinned snapshot layout |
| `SeedVR2` | SeedVR2 3B upscaling (`SeedVR2Upscaler`): weight validation and 4-bit quantization at load, the NaDiT transformer with windowed attention and 3D RoPE, the causal 3D VAE with MFLUX's tiling, Pillow-exact bicubic preprocessing, the seeded noise, MFLUX's colour correction, and the bundled positive text embedding |
| `MLXImageParity` | Validation harness for the models: frozen cases, comparisons against reference exports, tolerances and budgets, parity metrics, pins and manifests. Not needed for inference |

`ZImage`, `FluxKlein` and `SeedVR2` depend on `MLXImageCore` and never on each other, and none depends on `MLXImageParity`. The two models keep separate
text encoders, schedulers, VAEs and transformers on purpose: they differ numerically (dtype flow, norm formulation and eps, rotation order,
attention precision, compiled or eager steps) and the golden pixels pin each. Only code that is semantically identical lives in `MLXImageCore`.

Tests live in `Tests/`, one test target per module (`MLXImageCoreTests`, `ZImageTests`, `FluxKleinTests`, `SeedVR2Tests`, `MLXImageParityTests`), with frozen
parity inputs in `Tests/Fixtures`.

## Using the library

Add the package at a pinned version (or revision) and depend on the library products you need (`MLXImageCore`, `ZImage`, `FluxKlein`, `SeedVR2`, and `MLXImageParity` only if you run the parity harness):

```swift
.package(url: "https://github.com/nordlate/mlx-swift-image", exact: "0.3.0")
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

```swift
import MLX
import MLXImageCore
import SeedVR2

// SeedVR2 3B: the folder holds seedvr2_ema_3b_fp16.safetensors and ema_vae_fp16.safetensors (numz/SeedVR2_comfyUI @ 09ced710).
Memory.cacheLimit = 0                                   // keeps the peak at the live arrays (see "SeedVR2 3B" below)
let upscaler = try SeedVR2Upscaler(modelDirectory: URL(fileURLWithPath: "/path/to/seedvr2-comfyui"))
let input = try ImageOutput.readPNG(at: URL(fileURLWithPath: "/path/to/input.png"))   // or ImageOutput.RGBImage(width:height:bytes:)
let upscaled = try upscaler.upscale(SeedVR2Request(image: input, resolution: .scale(2), softness: 0.5, seed: 42))
try upscaled.writePNG(to: URL(fileURLWithPath: "/path/to/output.png"))
```

See `Sources/ZImage/Pipeline.swift`, `Sources/FluxKlein/KleinGeneration.swift` and `Sources/SeedVR2/SeedVR2Upscaler.swift` for the full
request and output types. MLX needs a Metal
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
| `SEEDVR2_MODEL_DIR` | The folder with `seedvr2_ema_3b_fp16.safetensors` and `ema_vae_fp16.safetensors` from `numz/SeedVR2_comfyUI` (read only; the model test also checks their SHA-256) |
| `SEEDVR2_REFERENCE_DIR` | A Python/MFLUX 0.20 SeedVR2 export (`manifest.json`, the input PNGs, per-case safetensors) for the cases in `Tests/Fixtures/reference/seedvr2_cases.json`; `SEEDVR2_PARITY_REPORT` optionally names a directory for the JSON report |
| `SEEDVR2_MEASURE` | `<input png>:<shortest side>:<softness>`: the opt-in time and peak-memory measurement of one upscale (`SeedVR2MeasurementTests`, one job per process; `SEEDVR2_MEASURE_OUTPUT` optionally names the output PNG; `SEEDVR2_MEASURE_PHASES=1` also records the peak after each stage, from a separate untimed run) |

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

### SeedVR2 3B

SeedVR2 follows MFLUX 0.20's `SeedVR2(quantize=4)`: `softness` (0...1) first downscales the input by `1 + 7 * softness` with Pillow's
bicubic filter and scales it back, the input is resized to the output size and padded with black to multiples of 16, the VAE encodes it
in 512-pixel tiles (64-pixel overlap), one NaDiT step denoises seeded noise conditioned on that latent and on SeedVR2's fixed positive
text embedding, the VAE decodes in 64-latent tiles (8-latent overlap), and MFLUX's wavelet and CIELAB colour correction is applied
against the input. The transformer's linear layers are quantized to 4 bits (group size 64) when the fp16 file is loaded; no converter
or pre-quantized snapshot is needed.

Against an MFLUX 0.20 export of the same weights (nine preprocessing cases, three noise cases, five complete upscales: 512² to 1024²
and 1024² to 2048², each at softness 0 and 0.5, and a padded 400x300 crop to 1080x1440), preprocessing, the output size and the seeded
noise are bit-identical, and the scheduler step from MFLUX's prediction is exact. The transformer step from MFLUX's own noise and
condition differs by 5.7e-4 to 8.5e-4 relative L2 (MFLUX's own compiled-versus-eager difference is about 8e-4), the tiled VAE encode and
decode of MFLUX's tensors by at most 8e-4 (bit-identical at 2048²), and native pixels differ from MFLUX's by mean 0.05 to 0.12 levels
(max 3 to 7), the same as MFLUX's own compiled-versus-eager runs; the test requires the native mean to stay within twice that drift.

Measured on an M5 Max (64 GB), release build, one upscale per process at softness 0.5, `Memory.cacheLimit = 0` in both, peak
`phys_footprint` (MFLUX 0.20 in brackets): loading takes 0.8 s and peaks at 2.6 GB, which is the peak during quantization at load
(MFLUX loads lazily and quantizes inside its first upscale); 512² to 1024² takes 11.2 s at 11.1 GB (12.3 s, 11.46 GB); 1024² to 2048²
44.2 s at 11.1 GB (48.5 s, 11.56 GB); 1024² to 4096² 178.7 s at 11.6 GB (194.9 s, 13.22 GB). The quantized weights take about 2.5 GB;
most of the remaining peak is the transformer's activations (at 4096² the video MLP runs in token chunks to keep them down) and the
VAE's tile activations, not the weights.

## Versioning

Semantic versioning, with a release tag per version; changes are listed per product in `CHANGELOG.md`. Before 1.0.0 a minor version may change
the public API. Pinning a tag or a commit is recommended.

## Contributing

Maintained for the needs of the app that uses it (Taltio); contributions are welcome. See `CONTRIBUTING.md`.

## Licence

MIT, see `LICENSE`. The ports derive from MFLUX (MIT); attributions are in `NOTICE`. Model weights are separate and covered by their own licences.
