#!/usr/bin/env bash

# Shared, pinned release bootstrap. This file is sourced by scripts/release.sh
# and the internal Linux packager; it intentionally performs no work on load.

RUST_TOOLCHAIN=1.97.1
RUSTUP_VERSION=1.29.0
RUSTUP_INIT_SHA256=4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10
B3SUM_VERSION=1.8.6
FFMPEG_VERSION=8.1.2
FFMPEG_SHA256=464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c
NV_CODEC_HEADERS_VERSION=n13.0.19.0
NV_CODEC_HEADERS_SHA256=86d15d1a7c0ac73a0eafdfc57bebfeba7da8264595bf531cf4d8db1c22940116
FFMPEG_RUNTIME_CACHE_VERSION=1
ZSTD_VERSION=1.5.7
ZSTD_SHA256=eb33e51f49a15e023950cd7825ca74a4a2b43db8354825ac24fc1b7ee09e6fa3
PATCHELF_VERSION=0.18.0
PATCHELF_SHA256=ce84f2447fb7a8679e58bc54a20dc2b01b37b5802e12c57eece772a6f14bf3f0

bootstrap_die() {
    printf 'release bootstrap: %s\n' "$*" >&2
    exit 1
}

bootstrap_init() {
    local repo_root=$1
    export NOPERSON_RELEASE_CACHE="$repo_root/.cache/release"
    export NOPERSON_RELEASE_DOWNLOAD_ROOT="$NOPERSON_RELEASE_CACHE/downloads"
    export NOPERSON_RELEASE_RUSTUP_HOME="$NOPERSON_RELEASE_CACHE/toolchains/rustup"
    export NOPERSON_RELEASE_CARGO_HOME="$NOPERSON_RELEASE_CACHE/toolchains/cargo"
    export NOPERSON_RELEASE_TOOL_ROOT="$NOPERSON_RELEASE_CACHE/tools"
    if test -f "$NOPERSON_RELEASE_CACHE/linux-x86_64/ffmpeg-runtime-v1/.complete"; then
        # Preserve the already-warm cache created by pre-orchestrator dev builds.
        export NOPERSON_RELEASE_DEPENDENCY_ROOT="$NOPERSON_RELEASE_CACHE/linux-x86_64"
    else
        export NOPERSON_RELEASE_DEPENDENCY_ROOT="$NOPERSON_RELEASE_CACHE/dependencies/linux-x86_64"
    fi
    export NOPERSON_RELEASE_TARGET_DIR="$NOPERSON_RELEASE_CACHE/cargo-target"
    export NOPERSON_RELEASE_LOG_ROOT="$NOPERSON_RELEASE_CACHE/logs"
    export NOPERSON_RELEASE_TEMP_ROOT="$NOPERSON_RELEASE_CACHE/tmp"
    export RUSTUP_HOME="$NOPERSON_RELEASE_RUSTUP_HOME"
    export CARGO_HOME="$NOPERSON_RELEASE_CARGO_HOME"
    export TMPDIR="$NOPERSON_RELEASE_TEMP_ROOT"
    mkdir -p -- \
        "$NOPERSON_RELEASE_DOWNLOAD_ROOT" \
        "$NOPERSON_RELEASE_RUSTUP_HOME" \
        "$NOPERSON_RELEASE_CARGO_HOME" \
        "$NOPERSON_RELEASE_TOOL_ROOT/bin" \
        "$NOPERSON_RELEASE_DEPENDENCY_ROOT" \
        "$NOPERSON_RELEASE_TARGET_DIR" \
        "$NOPERSON_RELEASE_LOG_ROOT" \
        "$NOPERSON_RELEASE_TEMP_ROOT"
    export PATH="$NOPERSON_RELEASE_TOOL_ROOT/bin:$NOPERSON_RELEASE_CARGO_HOME/bin:$PATH"
}

download_verified() {
    local url=$1 output=$2 expected_sha256=$3 partial="${2}.part"
    if test -f "$output" \
        && printf '%s  %s\n' "$expected_sha256" "$output" | sha256sum -c --status; then
        return
    fi
    command -v curl >/dev/null 2>&1 || bootstrap_die 'curl is required for bootstrap downloads'
    if ! curl -fL --retry 3 --continue-at - "$url" -o "$partial"; then
        curl -fL --retry 3 "$url" -o "$partial"
    fi
    printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum -c --status \
        || bootstrap_die "SHA-256 mismatch: ${output##*/}"
    mv -f -- "$partial" "$output"
}

bootstrap_prepare_toolchain() {
    local rustup_init completion
    completion="$NOPERSON_RELEASE_RUSTUP_HOME/.rust-$RUST_TOOLCHAIN.complete"
    if test -x "$NOPERSON_RELEASE_CARGO_HOME/bin/rustup" \
        && "$NOPERSON_RELEASE_CARGO_HOME/bin/rustup" run "$RUST_TOOLCHAIN" rustc --version >/dev/null 2>&1 \
        && test -f "$completion"; then
        printf 'release: using cached Rust toolchain %s\n' "$RUST_TOOLCHAIN"
        return
    fi
    test "$(uname -m)" = x86_64 || bootstrap_die 'release bootstrap supports x86_64 only'
    rustup_init="$NOPERSON_RELEASE_TOOL_ROOT/rustup-init-$RUSTUP_VERSION"
    download_verified \
        "https://static.rust-lang.org/rustup/archive/$RUSTUP_VERSION/x86_64-unknown-linux-gnu/rustup-init" \
        "$rustup_init" "$RUSTUP_INIT_SHA256"
    chmod 0755 "$rustup_init"
    if test -x "$NOPERSON_RELEASE_CARGO_HOME/bin/rustup"; then
        "$NOPERSON_RELEASE_CARGO_HOME/bin/rustup" toolchain install "$RUST_TOOLCHAIN" --profile minimal
    else
        "$rustup_init" -y --no-modify-path --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
    fi
    "$NOPERSON_RELEASE_CARGO_HOME/bin/rustup" run "$RUST_TOOLCHAIN" rustc --version > "$completion"
}

bootstrap_prepare_b3sum() {
    local destination="$NOPERSON_RELEASE_TOOL_ROOT/bin/b3sum"
    local completion="$NOPERSON_RELEASE_TOOL_ROOT/.b3sum-$B3SUM_VERSION.complete"
    if test -x "$destination" && test -f "$completion"; then
        printf 'release: using cached b3sum %s\n' "$B3SUM_VERSION"
        export B3SUM="$destination"
        return
    fi
    "$NOPERSON_RELEASE_CARGO_HOME/bin/cargo" "+$RUST_TOOLCHAIN" install \
        b3sum --version "=$B3SUM_VERSION" --locked \
        --root "$NOPERSON_RELEASE_TOOL_ROOT/b3sum-$B3SUM_VERSION"
    install -m 0755 \
        "$NOPERSON_RELEASE_TOOL_ROOT/b3sum-$B3SUM_VERSION/bin/b3sum" "$destination"
    "$destination" --version > "$completion"
    export B3SUM="$destination"
}

bootstrap_prepare_patchelf() {
    local destination="$NOPERSON_RELEASE_TOOL_ROOT/bin/patchelf"
    local completion="$NOPERSON_RELEASE_TOOL_ROOT/.patchelf-$PATCHELF_VERSION.complete"
    local archive="$NOPERSON_RELEASE_DOWNLOAD_ROOT/patchelf-$PATCHELF_VERSION-x86_64.tar.gz"
    local unpacked="$NOPERSON_RELEASE_TOOL_ROOT/patchelf-$PATCHELF_VERSION"
    if test -x "$destination" && test -f "$completion"; then
        printf 'release: using cached patchelf %s\n' "$PATCHELF_VERSION"
        return
    fi
    download_verified \
        "https://github.com/NixOS/patchelf/releases/download/$PATCHELF_VERSION/patchelf-$PATCHELF_VERSION-x86_64.tar.gz" \
        "$archive" "$PATCHELF_SHA256"
    mkdir -p -- "$unpacked"
    tar -xf "$archive" -C "$unpacked" --strip-components=1
    install -m 0755 "$unpacked/bin/patchelf" "$destination"
    "$destination" --version > "$completion"
}

bootstrap_prepare_zstd() {
    local destination="$NOPERSON_RELEASE_TOOL_ROOT/bin/zstd"
    local completion="$NOPERSON_RELEASE_TOOL_ROOT/.zstd-$ZSTD_VERSION.complete"
    local archive="$NOPERSON_RELEASE_DOWNLOAD_ROOT/zstd-$ZSTD_VERSION.tar.gz"
    local source="$NOPERSON_RELEASE_DEPENDENCY_ROOT/zstd-$ZSTD_VERSION"
    if test -x "$destination" && test -f "$completion"; then
        printf 'release: using cached zstd %s\n' "$ZSTD_VERSION"
        return
    fi
    download_verified \
        "https://github.com/facebook/zstd/releases/download/v$ZSTD_VERSION/zstd-$ZSTD_VERSION.tar.gz" \
        "$archive" "$ZSTD_SHA256"
    if ! test -f "$source/Makefile"; then
        tar -xf "$archive" -C "$NOPERSON_RELEASE_DEPENDENCY_ROOT"
    fi
    make -C "$source/programs" -s -j"$(nproc)" zstd
    install -m 0755 "$source/programs/zstd" "$destination"
    "$destination" --version > "$completion"
}

bootstrap_prepare_build_tools() {
    bootstrap_prepare_b3sum
    bootstrap_prepare_patchelf
    bootstrap_prepare_zstd
}

relocate_ffmpeg_pkg_config() {
    local package_config
    for package_config in "$NOPERSON_RELEASE_FFMPEG_PREFIX"/lib/pkgconfig/libav*.pc; do
        sed -i "s|^prefix=.*|prefix=$NOPERSON_RELEASE_FFMPEG_PREFIX|" "$package_config"
        sed -i "s|^libdir=.*|libdir=${NOPERSON_RELEASE_FFMPEG_PREFIX}/lib|" "$package_config"
        sed -i "s|^includedir=.*|includedir=${NOPERSON_RELEASE_FFMPEG_PREFIX}/include|" "$package_config"
    done
}

build_native_video_dependencies() {
    local root=$1
    local canonical_prefix="/ffmpeg-runtime-v${FFMPEG_RUNTIME_CACHE_VERSION}"
    local ffmpeg_archive="$root/ffmpeg-${FFMPEG_VERSION}.tar.xz"
    local headers_archive="$root/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz"
    local ffmpeg_pc pc_prefix
    local ffmpeg_build_log="$NOPERSON_RELEASE_LOG_ROOT/ffmpeg-build.log"
    mkdir -p -- "$root"
    download_verified \
        "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
        "$ffmpeg_archive" "$FFMPEG_SHA256"
    download_verified \
        "https://github.com/FFmpeg/nv-codec-headers/archive/refs/tags/${NV_CODEC_HEADERS_VERSION}.tar.gz" \
        "$headers_archive" "$NV_CODEC_HEADERS_SHA256"
    export NOPERSON_RELEASE_FFMPEG_SOURCE="$root/ffmpeg-${FFMPEG_VERSION}"
    export NOPERSON_RELEASE_FFMPEG_PREFIX="$root/ffmpeg-runtime-v${FFMPEG_RUNTIME_CACHE_VERSION}"
    export NOPERSON_RELEASE_NV_CODEC_HEADERS="$root/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}/include"
    if ! test -f "$NOPERSON_RELEASE_FFMPEG_SOURCE/configure"; then
        tar -xf "$ffmpeg_archive" -C "$root"
    fi
    if ! test -f "$NOPERSON_RELEASE_NV_CODEC_HEADERS/ffnvcodec/nvEncodeAPI.h"; then
        tar -xf "$headers_archive" -C "$root"
    fi
    ffmpeg_pc="$NOPERSON_RELEASE_FFMPEG_PREFIX/lib/pkgconfig/libavformat.pc"
    if test -f "$NOPERSON_RELEASE_FFMPEG_PREFIX/.complete" && test -f "$ffmpeg_pc"; then
        relocate_ffmpeg_pkg_config
        pc_prefix=$(sed -n 's/^prefix=//p' "$ffmpeg_pc" | head -1)
        if test "$pc_prefix" = "$NOPERSON_RELEASE_FFMPEG_PREFIX" \
            && test -f "$NOPERSON_RELEASE_FFMPEG_PREFIX/lib/libavformat.so" \
            && test -f "$NOPERSON_RELEASE_FFMPEG_PREFIX/lib/libavcodec.so" \
            && test -f "$NOPERSON_RELEASE_FFMPEG_PREFIX/lib/libavutil.so"; then
            export NOPERSON_FFMPEG_CACHE_KEY=$(
                sha256sum "$NOPERSON_RELEASE_FFMPEG_PREFIX"/lib/pkgconfig/libav*.pc \
                    | sha256sum | awk '{print $1}'
            )
            printf 'release: using cached minimal FFmpeg runtime\n'
            return
        fi
        printf 'release: rebuilding relocated minimal FFmpeg runtime\n'
    fi
    (
        cd "$NOPERSON_RELEASE_FFMPEG_SOURCE"
        if ! ./configure \
            --prefix="$canonical_prefix" \
            --disable-static --enable-shared --disable-gpl --disable-nonfree \
            --disable-programs --disable-doc --disable-debug --disable-x86asm --disable-network \
            --disable-avdevice --disable-avfilter --disable-swscale --disable-swresample \
            --disable-encoders --disable-decoders --disable-hwaccels \
            --disable-filters --disable-devices >"$ffmpeg_build_log" 2>&1; then
            printf 'release: FFmpeg configure failed; log: %s\n' "$ffmpeg_build_log" >&2
            exit 1
        fi
        printf 'release: building minimal FFmpeg runtime\n'
        if ! make -s -j"$(nproc)" >>"$ffmpeg_build_log" 2>&1; then
            printf 'release: FFmpeg build failed; log: %s\n' "$ffmpeg_build_log" >&2
            exit 1
        fi
        if ! make -s install DESTDIR="$root" >>"$ffmpeg_build_log" 2>&1; then
            printf 'release: FFmpeg install failed; log: %s\n' "$ffmpeg_build_log" >&2
            exit 1
        fi
        relocate_ffmpeg_pkg_config
        touch "$NOPERSON_RELEASE_FFMPEG_PREFIX/.complete"
    )
    export NOPERSON_FFMPEG_CACHE_KEY=$(
        sha256sum "$NOPERSON_RELEASE_FFMPEG_PREFIX"/lib/pkgconfig/libav*.pc \
            | sha256sum | awk '{print $1}'
    )
}

bootstrap_prepare_dependencies() {
    local variant=$1
    bootstrap_prepare_build_tools
    if test "$variant" = native; then
        build_native_video_dependencies "$NOPERSON_RELEASE_DEPENDENCY_ROOT"
    fi
}
