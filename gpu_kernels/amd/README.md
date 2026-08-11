# AMD HIP kernels

Each `.hip.cpp` module mirrors the logical module and exported `extern "C"`
entry points under `../nvidia/`. `compat.hpp` contains the subgroup operations
that must work on both native wave32 and wave64 hardware; HIP kernels must not
embed CUDA's fixed 32-lane ballot or reduction assumptions.

Run the offline AMD codegen check on any x86_64 Docker host:

```bash
./scripts/kernels/check-amd-codegen.sh --check
```

The wrapper uses the pinned official
`rocm/dev-ubuntu-24.04:7.2.4` image without GPU passthrough. It compiles every
module with `hipcc --genco`, extracts the raw AMDGPU ELF code object, verifies
target and wavefront metadata, and writes a BLAKE3 inventory under
`.cache/amd-codegen/rocm-7.2.4/`.

The initial compile matrix is `gfx90a`/wave64, `gfx942`/wave64, and
`gfx1100`/wave32. These checks prove frontend and AMDGPU code generation, not
runtime correctness or performance on AMD hardware.
