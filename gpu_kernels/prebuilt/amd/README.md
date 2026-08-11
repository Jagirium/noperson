# AMD kernel artifacts

ROCm code objects are stored as
`rocm-<major>.<minor>/<gfx-target>-wave<32|64>/*.hsaco` with a BLAKE3 manifest
beside each target inventory. The loader must match the exact `gfx` target and
subgroup width before loading an artifact; it must never fall back between
wave32 and wave64.
