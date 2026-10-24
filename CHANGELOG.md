# Changelog

The package uses semantic versioning. Before 1.0.0, a minor version may change the public API. Entries are grouped by product.

## 0.2.0

- **ZImage**: Z-Image Base q4 (`ZImageVariant.base`, `ZImageGenerator(modelDirectory:variant:)`): text to image with classifier-free guidance and a
  negative prompt (`GenerationRequest.guidance`, `negativePrompt`; steps 1...50), following MFLUX 0.20's scheduler and guidance. Turbo's API,
  output and golden pixels are unchanged; Turbo rejects guidance and a negative prompt.
- **MLXImageParity**: Z-Image Base checks (scheduler sigmas, guided steps, pixels) and their frozen cases.

## 0.1.0

First public release.

- **MLXImageCore**: the Qwen3 q4 text-encoder layout and ops, the Qwen tokenizer, safetensors header checks and PNG/JPEG output, shared by both models.
- **ZImage**: Z-Image Turbo q4, text to image (steps 4...12).
- **FluxKlein**: FLUX.2 Klein 4B q4, text to image and image to image from a prepared starting image (steps 1...8).
- **MLXImageParity**: the parity harness for both models (frozen cases, reference comparisons, tolerances, metrics).
- Built on mlx-swift 0.32.3, pinned by commit.
