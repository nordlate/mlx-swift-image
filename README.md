# MLXSwiftImage

Native image generation on Apple silicon with [MLX Swift](https://github.com/ml-explore/mlx-swift): Z-Image Turbo q4 and FLUX.2 Klein 4B q4
(text to image, and image to image for Klein), plus `zimage-worker`, the sandbox-friendly helper process Taltio embeds for local generation.

Unofficial. This package is not affiliated with, endorsed by or maintained by ml-explore or Apple.

It lives inside the Taltio repository for now and is meant to be extracted unchanged as the public `mlx-swift-image` repository
(nordlate/Taltio#254).

## Layout

| Path | What |
|---|---|
| `Sources/ZImage` | Z-Image Turbo: tokenizer, text encoder, transformer, scheduler, VAE, safetensors loader, PNG output, parity helpers |
| `Sources/FluxKlein` | FLUX.2 Klein 4B: text encoder, transformer, VAE (with encoder for starting images), scheduler, pins and parity harnesses |
| `Sources/ImageWorker` | The worker: request validation, engines for both families, event stream, atomic files. `LocalWorkerContract.swift` is a symlink to Taltio's own contract file, so host and worker compile one definition |
| `Sources/zimage-worker` | The executable (the name is kept: Taltio's build, lock and release check refer to it) |
| `Tests/` | `ZImageTests`, `FluxKleinTests`, `ImageWorkerTests`; `Tests/Fixtures` holds the frozen parity cases the tests read |
| `Tools/` | Build tooling and its tests: the no-JIT source preparation and the Metal library reduction (`prepare_*.py`, `reduce_library.py`, `kernel_filter.py`, `allowlist.py`, `metallib_*`, `macho_check.py`) |
| `Patches/` | `package-nojit.patch`: the one-line mlx-swift `Package.swift` edit that compiles the precompiled kernels, so MLX loads a colocated `mlx.metallib` |
| `Package.resolved` | The pinned dependency graph (mlx-swift 0.32.3 by commit) |

Requires macOS 26.2 and Apple silicon (the Metal target is validated against 26.2). Models are not part of the package.

## How Taltio builds it

The Xcode project does not reference this package. `LocalWorker/build-native.sh` resolves the pinned mlx-swift, applies `Patches/package-nojit.patch`
to a scratch copy, builds the `zimage-worker` product (Release, `MTL_FAST_MATH=NO`), builds the full precompiled Metal library with CMake and
reduces it with `Tools/reduce_library.py` to the frozen allowlist in `LocalWorker/native-inputs/`. `LocalWorker/native.lock` pins the inputs, including
the hash of this package's worker source tree; `LocalWorker/embed-runtime.sh` embeds and signs the result, and `LocalWorker/release-check.sh` verifies
the signed app with golden pixels. See `LocalWorker/README.md`.

## Tests

```bash
swift test            # in this directory; tests that need an installed model skip themselves
python3 -I -B Tools/test_prepare_source.py   # likewise for the other Tools/test_*.py
```

Measurement, lifecycle and parity command-line tools that use this package stay in `Experiments/native-mlx-zimage`.
