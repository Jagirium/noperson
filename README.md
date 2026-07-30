# noperson

GPU-first face swapping in Rust. The current base release runs the photo and
live webcam pipeline with ONNX Runtime, CUDA, and optional TensorRT execution
on NVIDIA hardware.

## Status

- Linux/NVIDIA is the validated runtime target.
- Model inference and image processing stay on the GPU across the live path.
- Windows is kept at compile compatibility in this release; runtime completion
  follows later.
- Background removal is intentionally outside the current build.
- DFM models are user-supplied local files and are not distributed by this
  project.

## Requirements

- NVIDIA GPU with a current driver
- CUDA 12.8-compatible runtime
- Rust toolchain with edition 2024 support
- Linux webcam support through V4L2; `v4l2loopback` for virtual-camera output

## Models

Download the first-party model bundle into `models/`:

```bash
mkdir -p models
gh release download models-v0.1.0 \
  --repo Jagirium/noperson \
  --pattern '*.onnx' \
  --pattern 'emap.bin' \
  --dir models
```

Every runtime asset is pinned by filename and SHA-256 in
`src/models/registry.rs`. The model release also includes `SHA256SUMS` and a
machine-readable selection manifest. DFM files stay outside the release.

## Build and run

```bash
cargo run --locked --release
```

Useful verification commands:

```bash
cargo test --locked
cargo test --locked --test swap -- --ignored --test-threads=1
cargo check --locked --target x86_64-pc-windows-msvc
```

## License

The source code is licensed under AGPL-3.0. The model release manifest records
source and release hashes; model assets retain their own upstream terms.
