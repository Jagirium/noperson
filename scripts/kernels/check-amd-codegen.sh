#!/usr/bin/env bash
set -Eeuo pipefail

readonly ROCM_IMAGE='rocm/dev-ubuntu-24.04:7.2.4'
readonly ROCM_RELEASE='7.2.4'
readonly AMD_TARGETS=('gfx90a:64' 'gfx942:64' 'gfx1100:32')

die() {
    printf 'amd-codegen: %s\n' "$*" >&2
    exit 1
}

usage() {
    printf 'usage: %s --check|--print-plan\n' "${0##*/}" >&2
    exit 2
}

test "$#" -eq 1 || usage
mode=$1
case "$mode" in
    --check|--print-plan) ;;
    *) usage ;;
esac

print_plan() {
    printf 'image=%s\n' "$ROCM_IMAGE"
    printf 'rocm=%s\n' "$ROCM_RELEASE"
    local target
    for target in "${AMD_TARGETS[@]}"; do
        printf 'target=%s wave=%s\n' "${target%%:*}" "${target##*:}"
    done
}

if test "$mode" = --print-plan; then
    print_plan
    exit 0
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd -P)
output_dir="$repo_root/.cache/amd-codegen/rocm-$ROCM_RELEASE"

docker_bin=$(command -v docker 2>/dev/null || true)
test -n "$docker_bin" || die 'docker was not found'
"$docker_bin" image inspect "$ROCM_IMAGE" >/dev/null 2>&1 \
    || die "container image is missing; run: docker pull $ROCM_IMAGE"

mkdir -p -- "$output_dir"
printf 'amd-codegen: ROCm %s, 3 targets, no GPU passthrough\n' "$ROCM_RELEASE"
"$docker_bin" run --rm \
    --network=none \
    --security-opt=no-new-privileges \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$repo_root,dst=/work,readonly" \
    --mount "type=bind,src=$output_dir,dst=/output" \
    --workdir /work \
    "$ROCM_IMAGE" \
    bash scripts/kernels/build-hip-code-objects.sh --output /output

b3sum_bin=${B3SUM:-$(command -v b3sum 2>/dev/null || true)}
if test -z "$b3sum_bin" && test -x "$repo_root/.cache/release/tools/bin/b3sum"; then
    b3sum_bin="$repo_root/.cache/release/tools/bin/b3sum"
fi
if test -z "$b3sum_bin" || ! test -x "$b3sum_bin"; then
    die 'b3sum is required to inventory generated code objects'
fi
manifest_partial="$output_dir/.MANIFEST_BLAKE3.txt.partial.$$"
{
    printf '# noperson AMD HIP codegen cache v1\n'
    print_plan
    find "$output_dir" -mindepth 2 -maxdepth 2 -type f -name '*.hsaco' -print0 \
        | sort -z \
        | xargs -0 -r "$b3sum_bin" \
        | sed "s|$output_dir/||"
} > "$manifest_partial"
mv -f -- "$manifest_partial" "$output_dir/MANIFEST_BLAKE3.txt"
printf 'amd-codegen: verified artifacts in %s\n' "$output_dir"
