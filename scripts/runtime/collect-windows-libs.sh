#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

die() {
    printf 'windows-runtime-libs: %s\n' "$*" >&2
    exit 1
}

command -v objdump >/dev/null || die 'objdump is required for PE dependency verification'
command -v rg >/dev/null || die 'ripgrep is required'

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die 'run from a git worktree'
output=${1:-$repo_root/dist/runtime-windows-tree}
cuda_root=${WINDOWS_CUDA_ROOT:-$repo_root/win/CUDA_128}
ort_hash=8a54165e2dfc85e9f6afbdaf154e7c1c74582e6269a2d0ec93b11e1459309555
cache_root=${XDG_CACHE_HOME:-$HOME/.cache}
ort_root=${WINDOWS_ORT_ROOT:-$cache_root/ort.pyke.io/dfbin/x86_64-pc-windows-msvc/$ort_hash}

case "$output" in
    /|"$repo_root"|"$repo_root/") die 'refusing unsafe output directory' ;;
esac
test ! -e "$output" || die "output already exists: $output"
test -d "$cuda_root/bin" || die "Windows CUDA bin directory is missing: $cuda_root/bin"
test -d "$cuda_root/lib" || die "Windows TensorRT directory is missing: $cuda_root/lib"
test -d "$ort_root" || die "ORT CUDA 12 Windows distribution is missing: $ort_root"

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

rg -q '"version"\s*:\s*"12\.8\.57"' "$cuda_root/version.json" \
    || die 'CUDA runtime is not 12.8.57'
rg -q '^#define CUDNN_MAJOR 9$' "$cuda_root/include/cudnn_version.h" \
    || die 'cuDNN major version is not 9'
rg -q '^#define CUDNN_MINOR 11$' "$cuda_root/include/cudnn_version.h" \
    || die 'cuDNN minor version is not 11'
rg -q '^#define CUDNN_PATCHLEVEL 0$' "$cuda_root/include/cudnn_version.h" \
    || die 'cuDNN patch version is not 0'
rg -q '^#define TRT_MAJOR_ENTERPRISE 10\r?$' "$cuda_root/include/NvInferVersion.h" \
    || die 'TensorRT major version is not 10'
rg -q '^#define TRT_MINOR_ENTERPRISE 13\r?$' "$cuda_root/include/NvInferVersion.h" \
    || die 'TensorRT minor version is not 13'
rg -q '^#define TRT_PATCH_ENTERPRISE 0\r?$' "$cuda_root/include/NvInferVersion.h" \
    || die 'TensorRT patch version is not 0'
rg -q '^#define TRT_BUILD_ENTERPRISE 35\r?$' "$cuda_root/include/NvInferVersion.h" \
    || die 'TensorRT build version is not 35'
declare -A pinned_runtime_hashes=(
    ["$cuda_root/bin/cudnn64_9.dll"]=e635c9af06c64e599a781098466e91b51e19fd0f25f9ac12a23ab511aee3dacf
    ["$cuda_root/bin/cudnn_adv64_9.dll"]=0c2ff93897203d0115b88a010a76d268ed89ff2a2f628fbed30662310394c122
    ["$cuda_root/bin/cudnn_cnn64_9.dll"]=2197a3aa79c23179ad5203e6b594a50d5c00e3afcd22a688727a38bd03a8a06e
    ["$cuda_root/bin/cudnn_engines_precompiled64_9.dll"]=0c93a034083746bc0a2ca4e1fca8f9ba014b22ba2ff4f523f0a82fb3058e6f90
    ["$cuda_root/bin/cudnn_engines_runtime_compiled64_9.dll"]=5dfd44d256f3c87d7f173d3d5fc7e648a7476545d0e592dfe8b38c0f0fbd6f35
    ["$cuda_root/bin/cudnn_graph64_9.dll"]=d4716bdcb38a7c86e5da1d3bbfdd77ce759bfb43ef86fb2454a35aa9b3c9f170
    ["$cuda_root/bin/cudnn_heuristic64_9.dll"]=9af12af62cb9eddc8abc7566aa75ac1762bc03b5497de2e6149807e1bccaad75
    ["$cuda_root/bin/cudnn_ops64_9.dll"]=0fa44c9406bf2da0430df3c223d11bec36467e5e801a9eb59be28afc004bbb41
    ["$cuda_root/lib/nvinfer_10.dll"]=d56bc3423265bc1f8499edb6a6fe19f300ac1861bf0cecf767fbaf060c007318
    ["$cuda_root/lib/nvonnxparser_10.dll"]=df0860579a695aea3a6bd0b4213acbd51dc85dff41d715379c15520849268932
    ["$cuda_root/lib/nvinfer_builder_resource_10.dll"]=4475cee39d6119a17cdd13450f4e9b4370ebb293dc09b713cd608c3112c812bf
)
for runtime_file in "${!pinned_runtime_hashes[@]}"; do
    test -f "$runtime_file" || die "pinned Windows runtime is missing: $runtime_file"
    test "$(hash_file "$runtime_file")" = "${pinned_runtime_hashes[$runtime_file]}" \
        || die "Windows runtime does not match the pinned build: $(basename -- "$runtime_file")"
done

parent=$(dirname -- "$output")
mkdir -p "$parent"
stage=$(mktemp -d "$parent/.runtime-windows.XXXXXXXX")
cleanup() { rm -rf -- "$stage"; }
trap cleanup EXIT INT TERM
mkdir -p "$stage/base" "$stage/trt/base"

copy_runtime() {
    local source=$1 destination=$2
    test -f "$source" || die "required Windows library is missing: $source"
    install -m 0755 "$source" "$destination/$(basename -- "$source")"
}

for library in \
    onnxruntime_providers_shared.dll \
    onnxruntime_providers_cuda.dll
do
    copy_runtime "$ort_root/$library" "$stage/base"
done

for library in \
    cudart64_12.dll \
    cublas64_12.dll \
    cublasLt64_12.dll \
    cufft64_11.dll \
    nvrtc64_120_0.dll \
    nvrtc-builtins64_128.dll \
    cudnn64_9.dll \
    cudnn_adv64_9.dll \
    cudnn_cnn64_9.dll \
    cudnn_engines_precompiled64_9.dll \
    cudnn_engines_runtime_compiled64_9.dll \
    cudnn_graph64_9.dll \
    cudnn_heuristic64_9.dll \
    cudnn_ops64_9.dll \
    nppc64_12.dll \
    nppig64_12.dll \
    nppif64_12.dll \
    nppim64_12.dll
do
    copy_runtime "$cuda_root/bin/$library" "$stage/base"
done

copy_runtime "$ort_root/onnxruntime_providers_tensorrt.dll" "$stage/trt/base"
for library in nvinfer_10.dll nvonnxparser_10.dll nvinfer_builder_resource_10.dll; do
    copy_runtime "$cuda_root/lib/$library" "$stage/trt/base"
done

# TensorRT 10.13 for Windows ships one universal builder resource. Marker
# directories preserve today's platform-neutral RuntimeLayout without copying
# the 1.7 GiB DLL once per compute capability.
for architecture in sm75 sm80 sm86 sm89 sm90 sm100 sm120 ptx; do
    mkdir -p "$stage/trt/$architecture"
    printf 'nvinfer_builder_resource_10.dll\n' >"$stage/trt/$architecture/WINDOWS-UNIVERSAL-TRT"
done

is_external_windows_library() {
    local name=${1,,}
    case "$name" in
        api-ms-win-*.dll|ext-ms-win-*.dll|kernel32.dll|advapi32.dll|user32.dll|\
        shell32.dll|ole32.dll|dbghelp.dll|ntdll.dll|ucrtbase.dll|ws2_32.dll|\
        msvcp140.dll|vcruntime140.dll|vcruntime140_1.dll|nvcuda.dll) return 0 ;;
        *) return 1 ;;
    esac
}

available=$(find "$stage/base" "$stage/trt/base" -maxdepth 1 -type f -iname '*.dll' \
    -printf '%f\n' | tr '[:upper:]' '[:lower:]' | sort -u)
while IFS= read -r pe; do
    imports=$(objdump -p "$pe") \
        || die "could not inspect PE dependencies: $(basename -- "$pe")"
    dependencies=$(awk '/DLL Name:/ { print $3 }' <<<"$imports") \
        || die "could not parse PE dependencies: $(basename -- "$pe")"
    while IFS= read -r needed; do
        test -n "$needed" || continue
        is_external_windows_library "$needed" && continue
        needed=${needed,,}
        printf '%s\n' "$available" \
            | awk -v wanted="$needed" '$0 == wanted { found=1 } END { exit !found }' \
            || die "unresolved packaged dependency: $(basename -- "$pe") -> $needed"
    done <<<"$dependencies"
done < <(find "$stage/base" "$stage/trt/base" -maxdepth 1 -type f -iname '*.dll' -print | sort)

{
    printf 'format=1\n'
    printf 'platform=windows-x86_64\n'
    printf 'generation=cuda12.8-cudnn9.11-trt10.13-v1\n'
    printf 'ort=1.24.2\n'
    printf 'cuda=12.8.57\n'
    printf 'cudnn=9.11.0.98\n'
    printf 'tensorrt=10.13.0.35\n'
    printf 'trt_resource=universal\n'
    printf 'external=nvcuda.dll,MSVCP140.dll,VCRUNTIME140.dll,VCRUNTIME140_1.dll\n'
} >"$stage/RUNTIME-MANIFEST"

while IFS= read -r -d '' runtime_file; do
    relative=${runtime_file#"$stage/"}
    printf '%s  %s\n' "$(hash_file "$runtime_file")" "$relative"
done < <(find "$stage" -type f ! -name BLAKE3SUMS -print0 | sort -z) >"$stage/BLAKE3SUMS"

mv -- "$stage" "$output"
trap - EXIT INT TERM
printf 'windows-runtime-libs: %s\n' "$output"
du -sh "$output/base" "$output/trt/base" "$output"/trt/sm* "$output/trt/ptx"
