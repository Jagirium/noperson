#!/usr/bin/env bash
set -Eeuo pipefail

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "$script_root/.." && pwd -P)
# shellcheck source=scripts/release/bootstrap.sh
source "$script_root/release/bootstrap.sh"

variant=
dev_mode=false
dev_args=()

die() {
    printf 'release: %s\n' "$*" >&2
    exit 1
}

resolve_variant() {
    local choice= argument
    for argument in "$@"; do
        case "$argument" in
            --dev) dev_mode=true; dev_args+=(--dev) ;;
            --docker|docker|1)
                test -z "$choice" || die 'choose exactly one release variant'
                choice=docker
                ;;
            --native|native|2)
                test -z "$choice" || die 'choose exactly one release variant'
                choice=native
                ;;
            --windows|windows|3)
                test -z "$choice" || die 'choose exactly one release variant'
                choice=windows
                ;;
            *) die "unknown release argument: $argument" ;;
        esac
    done
    if test -z "$choice"; then
        printf '%s\n' \
            'Choose release build:' \
            '  1) Linux GPU + Docker' \
            '  2) Linux GPU native' \
            '  3) Windows GPU native'
        read -r -p '> ' choice
        case "$choice" in
            1) choice=docker ;;
            2) choice=native ;;
            3) choice=windows ;;
            *) die "unknown release variant: $choice" ;;
        esac
    fi
    variant=$choice
}

validate_host() {
    command -v git >/dev/null 2>&1 || die 'git is required'
    command -v curl >/dev/null 2>&1 || die 'curl is required'
    command -v tar >/dev/null 2>&1 || die 'tar is required'
    command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required'
    git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
        || die 'release entry point must live in a git worktree'
    test "$(uname -m)" = x86_64 || die 'only x86_64 release builds are supported'
    case "$variant" in
        native|docker)
            cuda_root=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda-12.8}}
            nvcc_bin=${NVCC:-$cuda_root/bin/nvcc}
            if ! test -x "$nvcc_bin"; then
                nvcc_bin=$(command -v nvcc 2>/dev/null || true)
            fi
            test -n "$nvcc_bin" && test -x "$nvcc_bin" \
                || die 'CUDA Toolkit 12.8 nvcc is required'
            nvcc_version=$("$nvcc_bin" --version 2>&1) || die 'nvcc --version failed'
            case "$nvcc_version" in *'release 12.8,'*) ;; *) die 'CUDA Toolkit release 12.8 is required' ;; esac
            export NVCC="$nvcc_bin"
            ;;
        windows)
            case "$(uname -s)" in
                MINGW*|MSYS*|CYGWIN*) ;;
                *) die 'Windows GPU native must run from a Windows shell' ;;
            esac
            ;;
    esac
}

prepare_toolchain() {
    if test "$variant" = windows; then
        return
    fi
    bootstrap_prepare_toolchain
}

prepare_dependencies() {
    if test "$variant" = windows; then
        return
    fi
    bootstrap_prepare_dependencies "$variant"
}

verify_kernel_artifacts() {
    if test "$variant" = windows; then
        return
    fi
    B3SUM="$NOPERSON_RELEASE_TOOL_ROOT/bin/b3sum" \
        "$script_root/kernels/build-fatbins.sh" --check
}

run_platform_builder() {
    case "$variant" in
        docker)
            "$script_root/release/linux.sh" --orchestrated --docker "${dev_args[@]}"
            ;;
        native)
            "$script_root/release/linux.sh" --orchestrated --native "${dev_args[@]}"
            ;;
        windows)
            cmd.exe /c "${script_root//\//\\}\\release\\win.bat" -Orchestrated "${dev_args[@]}"
            ;;
    esac
}

run_stage() {
    local name=$1 log
    log="$NOPERSON_RELEASE_LOG_ROOT/$name.log"
    printf 'release: [%s]\n' "$name"
    if "$name" > >(tee "$log") 2>&1; then
        return
    else
        status=$?
    fi
    printf 'release: stage %s failed; log: %s\n' "$name" "$log" >&2
    return "$status"
}

resolve_variant "$@"
bootstrap_init "$repo_root"
cd "$repo_root"
run_stage validate_host
run_stage prepare_toolchain
run_stage prepare_dependencies
run_stage verify_kernel_artifacts
run_stage run_platform_builder
