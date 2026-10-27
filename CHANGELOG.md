# Changelog

The package uses semantic versioning. Before 1.0.0, a minor version may change the public API. Entries are grouped by product.

## Unreleased

- **FluxKlein**: FLUX.2 Klein 9B q4 (`KleinVariant.klein9B`, `KleinGenerator(modelDirectory:variant:)`): text to image and image to image
  through the same code path as 4B, from a snapshot written by `mflux-save` 0.20. The loaders take a `variant:` (default `.klein4B`) and
  refuse a snapshot of the other size before reading tensors. `KleinTransformerLayout` is now a struct with the presets `.klein4B` and
  `.klein9B`; its sizes and `expectedTensors()` are instance members. `KleinPromptEncoder.embeddingWidth` takes the text-encoder layout.
  `KleinPins.storedMfluxVersion` moved to `KleinVariant.storedMfluxVersion`. Klein 4B output is unchanged (native-versus-native identical).
- **MLXImageCore**: `TextEncoderLayout` is now a struct with the presets `.qwen3_4B` and `.qwen3_8B`; `hiddenSize`, `intermediateSize`,
  `layerMatrices`, `quantizedMatrices` and `expectedTensors()` are instance members. Z-Image output is unchanged.
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
