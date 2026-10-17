# Changelog

The package uses semantic versioning. Before 1.0.0, a minor version may change the public API. Entries are grouped by product.

## 0.1.0

First public release.

- **MLXImageCore**: the Qwen3 q4 text-encoder layout and ops, the Qwen tokenizer, safetensors header checks and PNG/JPEG output, shared by both models.
- **ZImage**: Z-Image Turbo q4, text to image (steps 4...12).
- **FluxKlein**: FLUX.2 Klein 4B q4, text to image and image to image from a prepared starting image (steps 1...8).
- **MLXImageParity**: the parity harness for both models (frozen cases, reference comparisons, tolerances, metrics).
- Built on mlx-swift 0.32.3, pinned by commit.
