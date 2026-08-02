#!/usr/bin/env bash
set -Eeuo pipefail

die() {
    printf 'windows-runtime-pack: %s\n' "$*" >&2
    exit 1
}

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die 'run from a git worktree'
libs=${1:-$repo_root/dist/runtime-windows-tree}
output=${2:-$repo_root/dist/runtime-windows}
test -d "$libs/base" || die "Windows runtime base is missing: $libs/base"
test -d "$libs/trt/base" || die "Windows TensorRT base is missing: $libs/trt/base"
test ! -e "$output" || die "output already exists: $output"
command -v tar >/dev/null || die 'tar is required'
command -v zstd >/dev/null || die 'zstd is required'

parent=$(dirname -- "$output")
mkdir -p "$parent"
stage=$(mktemp -d "$parent/.runtime-windows-pack.XXXXXXXX")
cleanup() { rm -rf -- "$stage"; }
trap cleanup EXIT INT TERM
epoch=${SOURCE_DATE_EPOCH:-$(git show -s --format=%ct HEAD)}

pack() {
    local filename=$1
    shift
    tar --sort=name --mtime="@$epoch" --owner=0 --group=0 --numeric-owner \
        --mode='u+rwX,go+rX,go-w' \
        -C "$libs" -cf - "$@" \
        | zstd -q -T2 -3 -o "$stage/$filename"
}

pack noperson-runtime-base-windows-x86_64-v1.tar.zst \
    base RUNTIME-MANIFEST BLAKE3SUMS
pack noperson-runtime-trt-base-windows-x86_64-v1.tar.zst \
    trt/base/onnxruntime_providers_tensorrt.dll \
    trt/base/nvinfer_10.dll \
    trt/base/nvonnxparser_10.dll
pack noperson-runtime-trt-universal-windows-x86_64-v1.tar.zst \
    trt/base/nvinfer_builder_resource_10.dll \
    trt/sm75 trt/sm80 trt/sm86 trt/sm89 trt/sm90 trt/sm100 trt/sm120 trt/ptx

hash_file() {
    local runtime_file=$1
    if command -v b3sum >/dev/null; then
        b3sum --no-names "$runtime_file"
        return
    fi
    local python=${BLAKE3_PYTHON:-python3}
    "$python" -c 'import blake3, sys
h = blake3.blake3()
with open(sys.argv[1], "rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
        h.update(chunk)
print(h.hexdigest())' "$runtime_file" \
        || die 'install b3sum or provide BLAKE3_PYTHON with the blake3 module'
}

while IFS= read -r -d '' archive; do
    printf '%s %s %s\n' \
        "$(hash_file "$archive")" \
        "$(stat -Lc '%s' "$archive")" \
        "$(basename -- "$archive")"
done < <(find "$stage" -type f -name '*.tar.zst' -print0 | sort -z) \
    >"$stage/MANIFEST_BLAKE3.txt"

mv -- "$stage" "$output"
trap - EXIT INT TERM
cat "$output/MANIFEST_BLAKE3.txt"
