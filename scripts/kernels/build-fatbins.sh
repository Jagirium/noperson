#!/usr/bin/env bash
set -Eeuo pipefail

readonly CUDA_RELEASE=12.8
readonly CUDA_TARGETS=(75 80 86 89 90 100 120)

die() {
    printf 'fatbins: %s\n' "$*" >&2
    exit 1
}

usage() {
    printf 'usage: %s --write|--check\n' "${0##*/}" >&2
    exit 2
}

test "$#" -eq 1 || usage
case "$1" in
    --write|--check) mode=$1 ;;
    *) usage ;;
esac

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd -P)
kernels_dir="$repo_root/gpu_kernels"
output_dir="$kernels_dir/prebuilt/cuda-$CUDA_RELEASE"
temporary_parent="$repo_root/.cache/release/kernels"
temporary="$temporary_parent/.fatbins.$$"

mkdir -p -- "$temporary/artifacts"
cleanup() {
    rm -r -- "$temporary"
}
trap cleanup EXIT INT TERM

cuda_root=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda-$CUDA_RELEASE}}
nvcc=${NVCC:-$cuda_root/bin/nvcc}
if ! test -x "$nvcc"; then
    nvcc=$(command -v nvcc 2>/dev/null || true)
fi
test -n "$nvcc" && test -x "$nvcc" || die "CUDA $CUDA_RELEASE nvcc was not found"

cuobjdump=${CUOBJDUMP:-$(dirname -- "$nvcc")/cuobjdump}
if ! test -x "$cuobjdump"; then
    cuobjdump=$(command -v cuobjdump 2>/dev/null || true)
fi
test -n "$cuobjdump" && test -x "$cuobjdump" || die 'cuobjdump was not found'

nvcc_version=$({ "$nvcc" --version; } 2>&1) || die 'nvcc --version failed'
case "$nvcc_version" in
    *"release $CUDA_RELEASE,"*) ;;
    *) die "CUDA Toolkit release $CUDA_RELEASE is required" ;;
esac
nvcc_build=$(printf '%s\n' "$nvcc_version" | sed -n 's/.*V\([0-9][0-9.]*\).*/\1/p' | tail -1)
test -n "$nvcc_build" || die 'could not resolve the complete nvcc version'

hash_file() {
    local path=$1 output python b3sum_bin
    b3sum_bin=${B3SUM:-$(command -v b3sum 2>/dev/null || true)}
    if test -n "$b3sum_bin" && test -x "$b3sum_bin"; then
        output=$("$b3sum_bin" -- "$path") || die "BLAKE3 failed: $path"
        printf '%s\n' "${output%% *}"
        return
    fi
    python=${BLAKE3_PYTHON:-python3}
    output=$("$python" -c \
        'import blake3, pathlib, sys; print(blake3.blake3(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
        "$path") || die 'install b3sum or provide BLAKE3_PYTHON with the blake3 module'
    printf '%s\n' "$output"
}

mapfile -d '' sources < <(
    find "$kernels_dir" -maxdepth 1 -type f -name '*.cu' -print0 | sort -z
)
test "${#sources[@]}" -gt 0 || die 'CUDA source inventory is empty'

gencode=(
    -gencode arch=compute_75,code=sm_75
    -gencode arch=compute_80,code=sm_80
    -gencode arch=compute_86,code=sm_86
    -gencode arch=compute_89,code=sm_89
    -gencode arch=compute_90,code=sm_90
    -gencode arch=compute_100,code=sm_100
    -gencode arch=compute_120,code=sm_120
    -gencode arch=compute_75,code=compute_75
)

for source in "${sources[@]}"; do
    stem=${source##*/}
    stem=${stem%.cu}
    relative_source=${source#"$repo_root/"}
    artifact="$temporary/artifacts/$stem.fatbin"
    printf 'fatbins: building %s\n' "$relative_source"
    (
        cd "$repo_root"
        "$nvcc" --fatbin -O3 "${gencode[@]}" -o "$artifact" "$relative_source"
    )

    elf_listing=$("$cuobjdump" --list-elf "$artifact") \
        || die "could not inspect ELF payloads: $stem.fatbin"
    for architecture in "${CUDA_TARGETS[@]}"; do
        case "$elf_listing" in
            *"sm_$architecture"*) ;;
            *) die "$stem.fatbin is missing sm_$architecture SASS" ;;
        esac
    done
    ptx_listing=$("$cuobjdump" --list-ptx "$artifact") \
        || die "could not inspect PTX payloads: $stem.fatbin"
    case "$ptx_listing" in
        *".sm_75.ptx"*) ;;
        *) die "$stem.fatbin is missing the compute_75 PTX fallback" ;;
    esac
done

manifest="$temporary/MANIFEST_BLAKE3.txt"
{
    printf '# noperson CUDA kernel artifacts v1\n'
    printf 'cuda_release=%s\n' "$CUDA_RELEASE"
    printf 'nvcc_version=%s\n' "$nvcc_build"
    printf 'sass=sm75,sm80,sm86,sm89,sm90,sm100,sm120\n'
    printf 'ptx=compute_75\n'
    printf '# BLAKE3 inventory\n'
} > "$manifest"

inventory="$temporary/inventory"
for source in "${sources[@]}"; do
    stem=${source##*/}
    stem=${stem%.cu}
    printf '%s\n' "${source#"$repo_root/"}"
    printf 'gpu_kernels/prebuilt/cuda-%s/%s.fatbin\n' "$CUDA_RELEASE" "$stem"
done | sort > "$inventory"

while IFS= read -r relative; do
    case "$relative" in
        gpu_kernels/prebuilt/*)
            stem=${relative##*/}
            actual="$temporary/artifacts/$stem"
            ;;
        *) actual="$repo_root/$relative" ;;
    esac
    digest=$(hash_file "$actual")
    printf '%s  %s\n' "$digest" "$relative" >> "$manifest"
done < "$inventory"

if test "$mode" = --check; then
    test -f "$output_dir/MANIFEST_BLAKE3.txt" || die 'tracked fatbin manifest is missing'
    for source in "${sources[@]}"; do
        stem=${source##*/}
        stem=${stem%.cu}
        cmp -s -- "$temporary/artifacts/$stem.fatbin" "$output_dir/$stem.fatbin" \
            || die "tracked fatbin is stale: $stem.fatbin"
    done
    cmp -s -- "$manifest" "$output_dir/MANIFEST_BLAKE3.txt" \
        || die 'tracked fatbin manifest is stale'
    printf 'fatbins: verified %s reproducible CUDA artifacts\n' "${#sources[@]}"
    exit 0
fi

mkdir -p -- "$output_dir"
for source in "${sources[@]}"; do
    stem=${source##*/}
    stem=${stem%.cu}
    partial="$output_dir/.$stem.fatbin.partial.$$"
    install -m 0644 -- "$temporary/artifacts/$stem.fatbin" "$partial"
    mv -f -- "$partial" "$output_dir/$stem.fatbin"
done
manifest_partial="$output_dir/.MANIFEST_BLAKE3.txt.partial.$$"
install -m 0644 -- "$manifest" "$manifest_partial"
mv -f -- "$manifest_partial" "$output_dir/MANIFEST_BLAKE3.txt"
printf 'fatbins: published %s verified CUDA artifacts\n' "${#sources[@]}"
