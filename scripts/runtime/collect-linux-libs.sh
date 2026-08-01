#!/usr/bin/env bash
set -Eeuo pipefail

die() {
    printf 'runtime-libs: %s\n' "$*" >&2
    exit 1
}

command -v readelf >/dev/null || die 'readelf is required'
command -v find >/dev/null || die 'find is required'

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die 'run from a git worktree'
output=${1:-$repo_root/libs}
case "$output" in
    /|"$repo_root"|"$repo_root/") die 'refusing unsafe output directory' ;;
esac
test ! -e "$output" || die "output already exists: $output"

cuda_root=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda-12.8}}
cuda_lib=$(readlink -f "$cuda_root/lib64")
ort_lib=${ORT_LIB_DIR:-$repo_root/target/release}
trt_lib=${TRT_LIB_DIR:-/usr/lib/x86_64-linux-gnu}
cudnn_lib=${CUDNN_LIB_DIR:-/usr/lib/x86_64-linux-gnu}

test -d "$cuda_lib" || die "CUDA library directory is missing: $cuda_lib"
test -d "$ort_lib" || die "ORT library directory is missing: $ort_lib"
test -d "$trt_lib" || die "TensorRT library directory is missing: $trt_lib"
test -d "$cudnn_lib" || die "cuDNN library directory is missing: $cudnn_lib"

parent=$(dirname -- "$output")
mkdir -p "$parent"
stage=$(mktemp -d "$parent/.runtime-libs.XXXXXXXX")
cleanup() { rm -rf -- "$stage"; }
trap cleanup EXIT INT TERM

mkdir -p "$stage/base" "$stage/trt/base"

copy_runtime() {
    local source=$1 destination=$2 real real_name soname source_name
    test -e "$source" || die "required library is missing: $source"
    real=$(readlink -f "$source")
    test -f "$real" || die "library target is not a file: $source"
    real_name=$(basename -- "$real")
    source_name=$(basename -- "$source")
    install -m 0755 "$real" "$destination/$real_name"

    soname=$(readelf -d "$real" | awk '/\(SONAME\)/ { gsub(/[\[\]]/, "", $5); print $5; exit }')
    if test -n "$soname" && test "$soname" != "$real_name"; then
        ln -s "$real_name" "$destination/$soname"
    fi
    if test "$source_name" != "$real_name" && test "$source_name" != "$soname"; then
        ln -s "$real_name" "$destination/$source_name"
    fi
}

# ORT providers selected by the application.
copy_runtime "$ort_lib/libonnxruntime_providers_shared.so" "$stage/base"
copy_runtime "$ort_lib/libonnxruntime_providers_cuda.so" "$stage/base"
copy_runtime "$ort_lib/libonnxruntime_providers_tensorrt.so" "$stage/trt/base"

# CUDA 12.8 libraries required by ORT CUDA and cuDNN. The device driver is external.
for library in \
    libcudart.so.12 \
    libcublas.so.12 \
    libcublasLt.so.12 \
    libcurand.so.10 \
    libcufft.so.11 \
    libnvrtc.so.12 \
    libnvrtc-builtins.so.12.8 \
    libnppc.so.12 \
    libnppig.so.12 \
    libnppif.so.12 \
    libnppim.so.12
do
    copy_runtime "$cuda_lib/$library" "$stage/base"
done

# cuDNN uses dlopen for its component libraries, so the whole runtime family is explicit.
for library in \
    libcudnn.so.9 \
    libcudnn_adv.so.9 \
    libcudnn_cnn.so.9 \
    libcudnn_engines_precompiled.so.9 \
    libcudnn_engines_runtime_compiled.so.9 \
    libcudnn_graph.so.9 \
    libcudnn_heuristic.so.9 \
    libcudnn_ops.so.9
do
    copy_runtime "$cudnn_lib/$library" "$stage/base"
done

# TensorRT common builder/parser. Plugins are omitted until a production graph needs one.
copy_runtime "$trt_lib/libnvinfer.so.10" "$stage/trt/base"
copy_runtime "$trt_lib/libnvonnxparser.so.10" "$stage/trt/base"

# TensorRT 10.16 loads one internal builder resource selected for the target GPU.
for architecture in sm75 sm80 sm86 sm89 sm90 sm100 sm120 ptx; do
    shard="$stage/trt/$architecture"
    mkdir -p "$shard"
    resource=$(find "$trt_lib" -maxdepth 1 -type f \
        -name "libnvinfer_builder_resource_${architecture}.so.*" -print -quit)
    test -n "$resource" || die "TensorRT builder resource is missing: $architecture"
    copy_runtime "$resource" "$shard"
done

is_external_system_library() {
    case "$1" in
        ld-linux-x86-64.so.2|libc.so.6|libdl.so.2|libgcc_s.so.1|libm.so.6|\
        libpthread.so.0|librt.so.1|libstdc++.so.6|libz.so.1) return 0 ;;
        *) return 1 ;;
    esac
}

available=$(find "$stage/base" "$stage/trt/base" -maxdepth 1 \
    \( -type f -o -type l \) -printf '%f\n' | sort -u)
while IFS= read -r elf; do
    while IFS= read -r needed; do
        test -n "$needed" || continue
        is_external_system_library "$needed" && continue
        printf '%s\n' "$available" | awk -v wanted="$needed" '$0 == wanted { found=1 } END { exit !found }' \
            || die "unresolved packaged dependency: $(basename -- "$elf") -> $needed"
    done < <(readelf -d "$elf" | awk '/\(NEEDED\)/ { gsub(/[\[\]]/, "", $5); print $5 }')
done < <(find "$stage/base" "$stage/trt/base" -maxdepth 1 -type f -print | sort)

hash_file() {
    local file=$1
    if command -v b3sum >/dev/null; then
        b3sum "$file" | awk '{print $1}'
        return
    fi
    python=${BLAKE3_PYTHON:-python3}
    "$python" -c 'import blake3, sys
h = blake3.blake3()
with open(sys.argv[1], "rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
        h.update(chunk)
print(h.hexdigest())' "$file" \
        || die 'install b3sum or provide BLAKE3_PYTHON with the blake3 module'
}

{
    printf 'format=1\n'
    printf 'cuda=12.8\n'
    printf 'cudnn=%s\n' "$(readlink -f "$cudnn_lib/libcudnn.so.9" | sed 's/.*libcudnn\.so\.//')"
    printf 'tensorrt=%s\n' "$(readlink -f "$trt_lib/libnvinfer.so.10" | sed 's/.*libnvinfer\.so\.//')"
    printf 'shards=sm75,sm80,sm86,sm89,sm90,sm100,sm120,ptx\n'
} >"$stage/RUNTIME-MANIFEST"

manifest="$stage/BLAKE3SUMS"
while IFS= read -r -d '' file; do
    relative=${file#"$stage/"}
    printf '%s  %s\n' "$(hash_file "$file")" "$relative"
done < <(find "$stage" -type f ! -name BLAKE3SUMS -print0 | sort -z) >"$manifest"

mv -- "$stage" "$output"
trap - EXIT INT TERM
printf 'runtime-libs: %s\n' "$output"
du -sh "$output/base" "$output/trt/base" "$output"/trt/sm* "$output/trt/ptx"
