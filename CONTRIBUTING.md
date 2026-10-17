# Contributing

This library is maintained for the needs of the app that uses it (Taltio). Contributions are welcome, but there is no promised roadmap,
support window or response time.

## Before you open a pull request

- Open an issue first for anything beyond a small fix, so the scope can be agreed before you spend time on it.
- Keep each model's numerics unchanged unless the change is the point of the pull request. The ports match MFLUX's output pixel for
  pixel; a change that alters the output of an existing model needs a reason, and new parity evidence.
- Put code in `MLXImageCore` only when it is semantically identical for every model that uses it (see "Layout" in the README).
- Run `swift build` and `swift test`. If you change a model, also run its model-dependent tests with the variables listed in the README.
- Never add model weights, reference exports or other large binaries to the repository.

## New model families

A new family is a larger commitment: a port, parity cases against a reference implementation, and ongoing maintenance on each mlx-swift
update. Please open an issue to discuss it before starting.

## Licence

By contributing you agree that your contribution is licensed under the MIT License of this repository (see `LICENSE`).
