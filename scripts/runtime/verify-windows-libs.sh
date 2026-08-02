#!/usr/bin/env bash
set -Eeuo pipefail

die() {
    printf 'windows-runtime-verify: %s\n' "$*" >&2
    exit 1
}

repo_root=$(git rev-parse --show-toplevel 2>/dev/null) || die 'run from a git worktree'
root=${1:-$repo_root/dist/runtime-windows-tree}
test -d "$root/base" || die "runtime base is missing: $root/base"
test -d "$root/trt/base" || die "TensorRT base is missing: $root/trt/base"
root=$(realpath -- "$root") || die 'could not normalize runtime root'
test -f "$root/BLAKE3SUMS" || die 'BLAKE3SUMS is missing'
command -v objdump >/dev/null || die 'objdump is required'

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

declare -A inventory_paths=()
while read -r expected relative extra; do
    test -n "$expected" && test -n "$relative" && test -z "${extra:-}" \
        || die 'invalid BLAKE3SUMS entry'
    [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "invalid BLAKE3 digest: $relative"
    case "$relative" in
        /*|../*|*/../*|*/..) die "unsafe BLAKE3SUMS path: $relative" ;;
    esac
    test -z "${inventory_paths[$relative]+present}" \
        || die "duplicate BLAKE3SUMS path: $relative"
    inventory_paths["$relative"]=1
    test -f "$root/$relative" || die "manifest file is missing: $relative"
    actual=$(hash_file "$root/$relative")
    test "$actual" = "$expected" || die "BLAKE3 mismatch: $relative"
done <"$root/BLAKE3SUMS"

manifest_files=$(wc -l <"$root/BLAKE3SUMS")
actual_files=$(find "$root" -type f ! -name BLAKE3SUMS -print | wc -l)
test "$manifest_files" -eq "$actual_files" \
    || die "unexpected file count: manifest=$manifest_files actual=$actual_files"
while IFS= read -r -d '' runtime_file; do
    relative=${runtime_file#"$root/"}
    test -n "${inventory_paths[$relative]+present}" \
        || die "file is not tracked by BLAKE3SUMS: $relative"
done < <(find "$root" -type f ! -name BLAKE3SUMS -print0)

test -f "$root/trt/base/nvinfer_builder_resource_10.dll" \
    || die 'nvinfer_builder_resource_10.dll is missing'
for architecture in sm75 sm80 sm86 sm89 sm90 sm100 sm120 ptx; do
    marker="$root/trt/$architecture/WINDOWS-UNIVERSAL-TRT"
    test -f "$marker" || die "universal TensorRT marker is missing: $architecture"
    test "$(sed -n '1p' "$marker")" = nvinfer_builder_resource_10.dll \
        || die "invalid universal TensorRT marker: $architecture"
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

available=$(find "$root/base" "$root/trt/base" -maxdepth 1 -type f -iname '*.dll' \
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
done < <(find "$root/base" "$root/trt/base" -maxdepth 1 -type f -iname '*.dll' -print | sort)

printf 'windows-runtime-verify: %s files verified\n' "$actual_files"
