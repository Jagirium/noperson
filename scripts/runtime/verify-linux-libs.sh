#!/usr/bin/env bash
set -Eeuo pipefail

die() {
    printf 'runtime-verify: %s\n' "$*" >&2
    exit 1
}

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die 'run from a git worktree'
root=${1:-$repo_root/libs}
root=$(readlink -f "$root")
test -d "$root/base" || die "base runtime is missing: $root/base"
test -d "$root/trt/base" || die "TensorRT base is missing: $root/trt/base"
test -f "$root/BLAKE3SUMS" || die 'BLAKE3SUMS is missing'
test -f "$root/RUNTIME-MANIFEST" || die 'RUNTIME-MANIFEST is missing'
command -v ldd >/dev/null || die 'ldd is required'

broken=$(find "$root" -xtype l -print -quit)
test -z "$broken" || die "broken symlink: $broken"

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

while read -r expected relative; do
    test -n "$expected" || continue
    file="$root/$relative"
    test -f "$file" || die "manifest entry is missing: $relative"
    actual=$(hash_file "$file") || die "BLAKE3 hashing failed: $relative"
    test "$actual" = "$expected" || die "BLAKE3 mismatch: $relative"
done <"$root/BLAKE3SUMS"

manifest_files=$(wc -l <"$root/BLAKE3SUMS")
actual_files=$(find "$root" -path "$root/launch" -prune -o \
    -type f ! -name BLAKE3SUMS -print | wc -l)
test "$manifest_files" -eq "$actual_files" \
    || die "manifest coverage mismatch: hashes=$manifest_files files=$actual_files"

library_path="$root/base:$root/trt/base"
while IFS= read -r elf; do
    dependencies=$(LD_LIBRARY_PATH="$library_path" ldd "$elf" 2>&1) \
        || die "loader rejected: ${elf#"$root/"}"
    case "$dependencies" in
        *'not found'*) die "unresolved loader dependency: ${elf#"$root/"}" ;;
    esac
done < <(find "$root" -type f -name '*.so*' -print | sort)

cuda_provider="$root/base/libonnxruntime_providers_cuda.so"
trt_provider="$root/trt/base/libonnxruntime_providers_tensorrt.so"
for provider in "$cuda_provider" "$trt_provider"; do
    test -f "$provider" || die "ORT provider is missing: $provider"
    dependencies=$(LD_LIBRARY_PATH="$library_path" ldd "$provider")
    while IFS= read -r needed; do
        case "$dependencies" in
            *"$needed => $root/"*) ;;
            *) die "provider escaped the private runtime: $(basename -- "$provider") -> $needed" ;;
        esac
    done < <(readelf -d "$provider" \
        | awk '/\(NEEDED\)/ { gsub(/[\[\]]/, "", $5); print $5 }' \
        | awk '/^(libcublas|libcurand|libcufft|libcudart|libcudnn|libnvinfer|libnvonnxparser)/')
done

printf 'runtime-verify: hashes=%s files=%s status=ok\n' "$manifest_files" "$actual_files"
