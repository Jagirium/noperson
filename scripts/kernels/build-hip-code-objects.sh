#!/usr/bin/env bash
set -Eeuo pipefail

readonly AMD_TARGETS=('gfx90a:64' 'gfx942:64' 'gfx1100:32')

die() {
    printf 'hip-codegen: %s\n' "$*" >&2
    exit 1
}

usage() {
    printf 'usage: %s --output PATH\n' "${0##*/}" >&2
    exit 2
}

if test "$#" -ne 2 || test "$1" != --output; then
    usage
fi
output_root=$2
test -d "$output_root" || die "output directory does not exist: $output_root"

repo_root=$(pwd -P)
sources_dir="$repo_root/gpu_kernels/amd"
hipcc=/opt/rocm/bin/hipcc
test -x "$hipcc" || die '/opt/rocm/bin/hipcc was not found'

readelf=/opt/rocm/llvm/bin/llvm-readelf
if ! test -x "$readelf"; then
    readelf=$(command -v readelf 2>/dev/null || true)
fi
if test -z "$readelf" || ! test -x "$readelf"; then
    die 'llvm-readelf/readelf was not found'
fi
bundler=/opt/rocm/llvm/bin/clang-offload-bundler
test -x "$bundler" || die 'clang-offload-bundler was not found'

mapfile -d '' sources < <(
    find "$sources_dir" -maxdepth 1 -type f -name '*.hip.cpp' -print0 | sort -z
)
test "${#sources[@]}" -gt 0 || die 'HIP source inventory is empty'

temporary="$output_root/.partial.$$"
mkdir -p -- "$temporary/.symbols"
cleanup() {
    rm -rf -- "$temporary"
}
trap cleanup EXIT INT TERM

target_spec=''
for target_spec in "${AMD_TARGETS[@]}"; do
    target=${target_spec%%:*}
    wave=${target_spec##*:}
    target_dir="$temporary/$target-wave$wave"
    mkdir -p -- "$target_dir"
    printf 'hip-codegen: target=%s wave=%s modules=%s\n' "$target" "$wave" "${#sources[@]}"

    for source in "${sources[@]}"; do
        stem=${source##*/}
        stem=${stem%.hip.cpp}
        artifact="$target_dir/$stem.hsaco"
        bundle="$target_dir/.$stem.offload-bundle"
        "$hipcc" \
            -x hip \
            -std=c++17 \
            -O3 \
            --genco \
            --offload-arch="$target" \
            -I"$sources_dir" \
            "$source" \
            -o "$bundle"
        "$bundler" \
            --unbundle \
            --type=o \
            --targets="hipv4-amdgcn-amd-amdhsa--$target" \
            --input="$bundle" \
            --output="$artifact"
        unlink -- "$bundle"
        test -s "$artifact" || die "empty code object: $target/$stem.hsaco"
        elf_header=$("$readelf" -h "$artifact") \
            || die "could not inspect code object: $target/$stem.hsaco"
        case "$elf_header" in
            *AMDGPU*|*"AMD GPU"*) ;;
            *) die "not an AMDGPU ELF code object: $target/$stem.hsaco" ;;
        esac
        metadata=$("$readelf" --notes "$artifact") \
            || die "could not inspect AMDGPU metadata: $target/$stem.hsaco"
        case "$metadata" in
            *"amdhsa.target:   amdgcn-amd-amdhsa--$target"*) ;;
            *) die "wrong AMDGPU target metadata: $target/$stem.hsaco" ;;
        esac
        case "$metadata" in
            *".wavefront_size: $wave"*) ;;
            *) die "wrong wavefront metadata: $target/$stem.hsaco (expected $wave)" ;;
        esac
        symbols=$("$readelf" --dyn-syms "$artifact" \
            | awk '$4 == "OBJECT" && $NF ~ /[.]kd$/ { print $NF }' \
            | sort) \
            || die "could not inspect kernel symbols: $target/$stem.hsaco"
        case "$symbols" in
            *.kd*) ;;
            *) die "code object has no kernel descriptors: $target/$stem.hsaco" ;;
        esac
        symbol_inventory="$temporary/.symbols/$stem.txt"
        if test -f "$symbol_inventory"; then
            test "$symbols" = "$(<"$symbol_inventory")" \
                || die "kernel symbol mismatch across targets: $stem"
        else
            printf '%s\n' "$symbols" > "$symbol_inventory"
        fi
    done
done

for target_spec in "${AMD_TARGETS[@]}"; do
    target=${target_spec%%:*}
    wave=${target_spec##*:}
    final_dir="$output_root/$target-wave$wave"
    mkdir -p -- "$final_dir"
    find "$temporary/$target-wave$wave" -maxdepth 1 -type f -name '*.hsaco' -print0 \
        | while IFS= read -r -d '' artifact; do
            install -m 0644 -- "$artifact" "$final_dir/.${artifact##*/}.partial.$$"
            mv -f -- "$final_dir/.${artifact##*/}.partial.$$" "$final_dir/${artifact##*/}"
        done
done

printf 'hip-codegen: published %s modules for %s targets\n' \
    "${#sources[@]}" "${#AMD_TARGETS[@]}"
