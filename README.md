# mlx-swift-image

Native Swift inference for image generation on Apple silicon, built on [MLX Swift](https://github.com/ml-explore/mlx-swift):

- **FLUX.2 Klein 4B** (q4): text to image, and image to image from a prepared starting image
- **Z-Image Turbo** (q4): text to image

The package also contains `zimage-worker`, a small helper process that runs one of these models for a host app over a JSON-lines protocol on
stdin/stdout.

> **Unofficial.** This project is not affiliated with, endorsed by or maintained by Apple or the ml-explore/MLX project, nor by the authors of the
> models it runs (Z-Image: Alibaba Tongyi-MAI; FLUX.2: Black Forest Labs).

It started inside the Taltio app, which embeds the worker for local image generation, and was extracted into its own package. The Swift code is a
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
| `MLXImageCore` | What both models share: the Qwen3 q4 text-encoder layout and ops, the Qwen tokenizer, safetensors header checks, parity metrics, PNG/JPEG output |
| `ZImage` | Z-Image Turbo: text encoder, transformer, scheduler, VAE, weight loaders, parity helpers |
| `FluxKlein` | FLUX.2 Klein 4B: text encoder, transformer, VAE (with encoder for starting images), scheduler, pins, parity harnesses |
| `ImageWorker` | The worker implementation used by the `zimage-worker` executable: request validation, engines for both models, event stream, atomic file output. Its contract types are internal, not a consumable API |
| `zimage-worker` | The worker executable (a thin `main.swift` over `ImageWorker`) |

`ZImage` and `FluxKlein` depend on `MLXImageCore` and never on each other. The two models keep separate text encoders, schedulers, VAEs and
transformers on purpose: they differ numerically (dtype flow, norm formulation and eps, rotation order, attention precision, compiled or eager
steps) and the golden pixels pin each. Only code that is semantically identical lives in `MLXImageCore`.

Other directories: `Tests/` (with frozen parity inputs in `Tests/Fixtures`), `Tools/` and `Patches/` (see
[Reduced Metal library](#reduced-metal-library-no-jit)).

## Using the library

Add the package at a pinned revision and depend on the library products you need (`MLXImageCore`, `ZImage`, `FluxKlein`):

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
library at run time: either the stock JIT build, or a colocated `mlx.metallib` when you build with the patch described below.

## The worker

`zimage-worker` serves Z-Image Turbo q4 and FLUX.2 Klein 4B q4. The host starts it, writes one JSON request line to stdin (optionally followed by
`{"type":"cancel"}`), and reads JSON event lines from stdout; stderr is free-form logging. Each job ends with exactly one terminal event
(`done`, `cancelled` or `error`).

- Request fields: `version` (2), `jobID`, `family` (`flux2-klein-4b` or `z-image-turbo`), `modelDirectory`, `quantize` (4), `prompt`, `seed`,
  `width`, `height`, `steps`, `outputPath`, `cacheLimitMB`, and optionally `textChunks`, `sourceImagePath` with `imageStrength` (Klein only),
  `reuse` and `preview`.
- Events: `started`, `loaded`, `progress`, `preview`, `done`, `cancelled`, `error`, each carrying `version` and `jobID`.
- With `reuse`, the worker stays alive after a job and accepts `{"type":"next","request":{…}}` lines; end of stdin releases the model and exits.
- The PNG at `outputPath` is complete when `done` is sent: the worker writes `<outputPath>.partial` and renames it.

The contract is defined in `Sources/ImageWorker/LocalWorkerContract.swift` (types, versioning rules and the event decoder); validation rules
and messages are in `Sources/ImageWorker/RequestValidation.swift`. A host that compiles its own copy of the contract file must keep it identical.

Worker environment variables: `TALTIO_WORKER_IDLE_MS` (idle exit for reusable workers, default 60000), `TALTIO_WORKER_PARENT_GRACE_MS` (default
5000), `TALTIO_WORKER_MLX_METRICS=1` (write MLX memory figures to stderr), and `TALTIO_WORKER_FAKE_ENGINE=1` with the `TALTIO_WORKER_FAKE_*`
variables for a timed stand-in engine used by the lifecycle tests. The `TALTIO_` prefix is kept because it is part of the worker contract.

## Reduced Metal library (no JIT)

Stock MLX compiles many Metal kernels from source at run time, which costs seconds on a cold start. `Tools/` and `Patches/` produce a worker that
instead loads a small precompiled `mlx.metallib` containing only the kernels these models use. The scripts fail closed: any unexpected anchor,
hash or flag stops the run. They are CPU-only apart from the build commands they replay, and they never write to the checkout you give them.

1. **No-JIT source.** `Tools/prepare_source.py --variant precompiled --stock-checkout <mlx-swift checkout> --experiment-dir . --out <out>` verifies
   the checkout is the pinned mlx-swift commit and unmodified, clones it, and applies `Patches/package-nojit.patch` (a one-line edit of the copy's
   `Package.swift` so SwiftPM compiles `nojit_kernels.cpp`, looking kernels up in a precompiled library, instead of `jit_kernels.cpp`). The patch it
   generates must equal the committed one byte for byte. `--experiment-dir` is the package to build against the copy (this repository's root);
   `Tools/prepare_reduced.py` and `Tools/prepare_trace_source.py` take the same arguments.
2. **Full library.** Build the CMake target `mlx-metallib` of the MLX core in the copy with `-DMLX_METAL_JIT=OFF` (Release, arm64) and keep the verbose
   build log. `Tools/metallib_record.py` checks the cache, the compile flags (`-fno-fast-math`, deployment target) and the resulting library.
3. **Kernel allowlist.** `Tools/prepare_reduced.py --flavor trace` builds a discovery worker whose hook (`Patches/device-names-trace.patch`, active
   only when `TALTIO_MLX_TRACE=<absolute prefix>` is set) records the distinct kernel names MLX asks the library for. Running your generation
   workloads with it gives traces; `Tools/allowlist.py derive` takes their union (with a declaration file listing the jobs) and `verify` checks it.
4. **Reduction.** `Tools/reduce_library.py build` replays the full build's recorded `metal` commands, removing the explicit kernel instantiations that
   are not in the allowlist (`Tools/kernel_filter.py`, with an independent proof that nothing else changed) and relinks. `verify` compares the
   reflected function names (`Tools/metallib_names.swift`) with the allowlist and the full library. `Tools/macho_check.py` offers Mach-O and
   build-log evidence checks.

Use the resulting `mlx.metallib` next to the worker binary built from the patched package. The declaration and the frozen allowlist depend on the
workload you want covered and are not part of this repository.

## Tests

```bash
swift build
swift test
python3 -I -B -m unittest discover -s Tools -p 'test_*.py'
```

Tests that need model files, a reference export or an mlx-swift checkout skip themselves with a message naming the missing variable.

| Variable | Used for |
|---|---|
| `ZIMAGE_MODEL_DIR` (alias `ZIMAGE_Q4_MODEL_DIR`) | A Z-Image Turbo q4 snapshot (read only) |
| `KLEIN_MODEL_DIR` (alias `KLEIN_Q4_MODEL_DIR`) | A FLUX.2 Klein 4B q4 snapshot (read only) |
| `ZIMAGE_REFERENCE_DIR` | A Python/MFLUX reference export (`manifest.json` and per-case safetensors) for the Z-Image tokenizer parity test |
| `MLX_SWIFT_CHECKOUT` | A pinned, unmodified mlx-swift checkout, for the `Tools` tests that read the real `Package.swift` and `device.cpp` |

Without any of them the model-dependent tests are skipped and the rest run.

## Licence

MIT, see `LICENSE`. Model weights are separate and covered by their own licences.
