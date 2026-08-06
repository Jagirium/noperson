#!/usr/bin/env bash
set -Eeuo pipefail

die() {
    printf 'runtime-pack: %s\n' "$*" >&2
    exit 1
}

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die 'run from a git worktree'
libs=${1:-$repo_root/libs}
output=${2:-$repo_root/dist/runtime}
test -d "$libs/base" || die "runtime base is missing: $libs/base"
test -d "$libs/trt/base" || die "TensorRT base is missing: $libs/trt/base"
test ! -e "$output" || die "output already exists: $output"
command -v tar >/dev/null || die 'tar is required'
command -v zstd >/dev/null || die 'zstd is required'

parent=$(dirname -- "$output")
mkdir -p "$parent"
stage=$(mktemp -d "$parent/.runtime-pack.XXXXXXXX")
cleanup() { rm -rf -- "$stage"; }
trap cleanup EXIT INT TERM

epoch=${SOURCE_DATE_EPOCH:-$(git show -s --format=%ct HEAD)}
pack() {
    local filename=$1
    shift
    tar --sort=name --mtime="@$epoch" --owner=0 --group=0 --numeric-owner \
        -C "$libs" -cf - "$@" \
        | zstd -q -T2 -3 -o "$stage/$filename"
}

pack noperson-runtime-base-linux-x86_64-v1.tar.zst base RUNTIME-MANIFEST BLAKE3SUMS
pack noperson-runtime-trt-base-linux-x86_64-v1.tar.zst trt/base
for shard in sm75 sm80 sm86 sm89 sm90 sm100 sm120 ptx; do
    pack "noperson-runtime-trt-$shard-linux-x86_64-v1.tar.zst" "trt/$shard"
done

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

while IFS= read -r -d '' archive; do
    hash=$(hash_file "$archive") || die "BLAKE3 hashing failed: $(basename -- "$archive")"
    printf '%s %s %s\n' \
        "$hash" \
        "$(stat -Lc '%s' "$archive")" \
        "$(basename -- "$archive")"
done < <(find "$stage" -type f -name '*.tar.zst' -print0 | sort -z) \
    >"$stage/MANIFEST_BLAKE3.txt"

mv -- "$stage" "$output"
trap - EXIT INT TERM
cat "$output/MANIFEST_BLAKE3.txt"
