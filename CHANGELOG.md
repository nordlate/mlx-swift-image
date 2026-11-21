# Changelog

The package uses semantic versioning. Before 1.0.0, a minor version may change the public API. Entries are grouped by product.

## 0.3.0

- **SeedVR2** (new product): SeedVR2 3B super-resolution, ported from MFLUX 0.20. `SeedVR2Upscaler(modelDirectory:)` loads the fp16
  files of `numz/SeedVR2_comfyUI` (validated against `SeedVR2Pins` and `SeedVR2Layout` before any tensor is read) and quantizes the
  linear layers to 4 bits at load, as MFLUX's `quantize=4` does; `upscale(_:)` takes a `SeedVR2Request` (8-bit RGB input, a
  `SeedVR2Resolution` of `.shortestSide(_:)` or `.scale(_:)`, `softness` 0...1, a seed) and returns a `SeedVR2Output` with the pixels.
  One transformer step, tiled VAE encode and decode, and MFLUX's colour correction. SeedVR2's fixed positive text embedding is bundled
  (ByteDance's Apache 2.0 `pos_emb`, with its licence).
- **MLXImageCore**: `ImageOutput.RGBImage` has a public initializer.
- **MLXImageParity**: SeedVR2 parity against an MFLUX 0.20 export (`seedVR2Preprocess`, `seedVR2Noise`, `seedVR2Upscales`) and its
  frozen cases (`seedvr2_cases.json`).
- **FluxKlein**: FLUX.2 Klein 9B q4 (`KleinVariant.klein9B`, `KleinGenerator(modelDirectory:variant:)`): text to image and image to image
  through the same code path as 4B, from a snapshot written by `mflux-save` 0.20. The loaders take a `variant:` (default `.klein4B`) and
  refuse a snapshot of the other size before reading tensors. `KleinTransformerLayout` is now a struct with the presets `.klein4B` and
  `.klein9B`; its sizes and `expectedTensors()` are instance members. `KleinPromptEncoder.embeddingWidth` takes the text-encoder layout.
  `KleinPins.storedMfluxVersion` moved to `KleinVariant.storedMfluxVersion`. Klein 4B output is unchanged (native-versus-native identical).
- **MLXImageCore**: `TextEncoderLayout` is now a struct with the presets `.qwen3_4B` and `.qwen3_8B`; `hiddenSize`, `intermediateSize`,
  `layerMatrices`, `quantizedMatrices` and `expectedTensors()` are instance members. Z-Image output is unchanged.
- **FluxKlein**: `KleinConverter` converts the official FLUX.2 Klein 9B bf16 release into the q4 snapshot the loaders read, byte-identical
  to `mflux-save` 0.20: `KleinSourceSnapshot.inspect(directory:verifyingWeightHashes:isCancelled:)` validates a folder and finds the variant
  (`KleinConversionError` for anything else), `KleinConverter.convert(sourceDirectory:outputDirectory:progress:isCancelled:)` (and an
  async form that stops when its task is cancelled) converts through a staging folder, and `KleinConverter.requiredBytes(for:)` gives the
  output size. Quantization runs on the CPU (`KleinQ4Quantizer`, the arithmetic of MLX's Metal kernel); no GPU kernel is used. The official
  file pins are `KleinOfficialRelease.klein9B`. Klein 4B is recognized but not converted (no pins).
- **MLXImageParity**: `KleinVariantParity`, generation parity for either Klein variant against an MFLUX 0.20 export, with drift-relative
  limits, and its frozen cases (`klein_generation_cases.json`).

## 0.2.0

- **ZImage**: Z-Image Base q4 (`ZImageVariant.base`, `ZImageGenerator(modelDirectory:variant:)`): text to image with classifier-free guidance and a
  negative prompt (`GenerationRequest.guidance`, `negativePrompt`; steps 1...50), following MFLUX 0.20's scheduler and guidance. Turbo's API
  and output are unchanged; Turbo rejects guidance and a negative prompt; Base rejects a non-blank negative prompt with guidance of 1.0 or below, which would be ignored.
- **MLXImageParity**: Z-Image Base checks (scheduler sigmas, guided steps, pixels) and their frozen cases.

## 0.1.0

First public release.

- **MLXImageCore**: the Qwen3 q4 text-encoder layout and ops, the Qwen tokenizer, safetensors header checks and PNG/JPEG output, shared by both models.
- **ZImage**: Z-Image Turbo q4, text to image (steps 4...12).
- **FluxKlein**: FLUX.2 Klein 4B q4, text to image and image to image from a prepared starting image (steps 1...8).
- **MLXImageParity**: the parity harness for both models (frozen cases, reference comparisons, tolerances, metrics).
- Built on mlx-swift 0.32.3, pinned by commit.
