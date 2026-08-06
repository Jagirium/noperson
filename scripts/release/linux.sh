#!/usr/bin/env bash
set -Eeuo pipefail

RUST_TOOLCHAIN=1.97.1
APT_SNAPSHOT=20250701T000000Z
FFMPEG_VERSION=8.1.2
FFMPEG_SHA256=464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c
NV_CODEC_HEADERS_VERSION=n13.0.19.0
NV_CODEC_HEADERS_SHA256=86d15d1a7c0ac73a0eafdfc57bebfeba7da8264595bf531cf4d8db1c22940116
FFMPEG_RUNTIME_CACHE_VERSION=1

die() {
    printf 'release: %s\n' "$*" >&2
    exit 1
}

command -v git >/dev/null || die 'git is required'
command -v tar >/dev/null || die 'tar is required'
command -v readelf >/dev/null || die 'readelf is required'
command -v zstd >/dev/null || die 'zstd is required'
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die 'run from a git worktree'

verify_ort_cuda12() {
    local release_dir=$1
    local provider needed
    provider=$(find -L "$release_dir" -maxdepth 1 -type f \
        -name 'libonnxruntime_providers_cuda.so' -print -quit)
    test -n "$provider" || die 'CUDAExecutionProvider library is missing'
    needed=$(readelf -d "$provider")
    case "$needed" in *'libcublasLt.so.12'*) ;; *) die 'ORT provider does not target cuBLAS 12' ;; esac
    case "$needed" in *'libcudart.so.12'*) ;; *) die 'ORT provider does not target CUDA runtime 12' ;; esac
    case "$needed" in *'.so.13'*) die 'CUDA 13 dependency leaked into CUDA 12 release' ;; esac
}

download_verified() {
    local url=$1
    local output=$2
    local expected_sha256=$3
    local partial="${output}.part"
    if test -f "$output" \
        && printf '%s  %s\n' "$expected_sha256" "$output" | sha256sum -c --status; then
        return
    fi
    if ! curl -fL --retry 3 --continue-at - "$url" -o "$partial"; then
        curl -fL --retry 3 "$url" -o "$partial"
    fi
    printf '%s  %s\n' "$expected_sha256" "$partial" | sha256sum -c -
    mv -f -- "$partial" "$output"
}

build_native_video_dependencies() {
    local root=$1
    local ffmpeg_archive="$root/ffmpeg-${FFMPEG_VERSION}.tar.xz"
    local headers_archive="$root/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz"
    mkdir -p "$root"
    download_verified \
        "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
        "$ffmpeg_archive" "$FFMPEG_SHA256"
    download_verified \
        "https://github.com/FFmpeg/nv-codec-headers/archive/refs/tags/${NV_CODEC_HEADERS_VERSION}.tar.gz" \
        "$headers_archive" "$NV_CODEC_HEADERS_SHA256"
    ffmpeg_source="$root/ffmpeg-${FFMPEG_VERSION}"
    ffmpeg_prefix="$root/ffmpeg-runtime-v${FFMPEG_RUNTIME_CACHE_VERSION}"
    nv_codec_headers="$root/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}/include"
    if ! test -f "$ffmpeg_source/configure"; then
        tar -xf "$ffmpeg_archive" -C "$root"
    fi
    if ! test -f "$nv_codec_headers/ffnvcodec/nvEncodeAPI.h"; then
        tar -xf "$headers_archive" -C "$root"
    fi
    if test -f "$ffmpeg_prefix/.complete" \
        && test -f "$ffmpeg_prefix/lib/pkgconfig/libavformat.pc"; then
        printf 'release: using cached minimal FFmpeg runtime\n'
        return
    fi
    (
        cd "$ffmpeg_source"
        ./configure \
            --prefix="$ffmpeg_prefix" \
            --disable-static \
            --enable-shared \
            --disable-gpl \
            --disable-nonfree \
            --disable-programs \
            --disable-doc \
            --disable-debug \
            --disable-x86asm \
            --disable-network \
            --disable-avdevice \
            --disable-avfilter \
            --disable-swscale \
            --disable-swresample \
            --disable-encoders \
            --disable-decoders \
            --disable-hwaccels \
            --disable-filters \
            --disable-devices
        printf 'release: building minimal FFmpeg runtime\n'
        make -s -j"$(nproc)"
        make -s install
        touch "$ffmpeg_prefix/.complete"
    )
}

stage_native_video_dependencies() {
    local stage=$1
    mkdir -p "$stage/lib"
    cp -a "$ffmpeg_prefix/lib/libavformat.so"* "$stage/lib/"
    cp -a "$ffmpeg_prefix/lib/libavcodec.so"* "$stage/lib/"
    cp -a "$ffmpeg_prefix/lib/libavutil.so"* "$stage/lib/"
    # Preserve loader-relative ELF paths literally.
    # shellcheck disable=SC2016
    find "$stage/lib" -type f -name 'libav*.so*' -exec \
        patchelf --force-rpath --set-rpath '$ORIGIN' {} +
    # shellcheck disable=SC2016
    patchelf --force-rpath --set-rpath '$ORIGIN/lib' "$stage/noperson"
    install -m 0644 "$ffmpeg_source/LICENSE.md" "$stage/FFMPEG-LICENSE.md"
    {
        printf 'source=https://ffmpeg.org/releases/ffmpeg-%s.tar.xz\n' "$FFMPEG_VERSION"
        printf 'sha256=%s\n' "$FFMPEG_SHA256"
        printf 'configuration=%s\n' "$(sed -n 's/^FFMPEG_CONFIGURATION=//p' "$ffmpeg_source/ffbuild/config.mak")"
    } >"$stage/FFMPEG-SOURCE-OFFER"
}

verify_native_video_bundle() {
    local stage=$1
    local closure
    # shellcheck disable=SC2016
    test "$(patchelf --print-rpath "$stage/noperson")" = '$ORIGIN/lib' \
        || die 'binary FFmpeg RPATH is not loader-relative'
    closure=$(ldd "$stage/noperson")
    case "$closure" in *'not found'*) die 'bundled FFmpeg loader closure is incomplete' ;; esac
    for library in libavformat libavcodec libavutil; do
        case "$closure" in *"$stage/lib/${library}.so"*) ;;
            *) die "binary did not resolve bundled ${library}" ;;
        esac
    done
}

mode=
dev_mode=false
for argument in "$@"; do
    case "$argument" in
        --docker|--native)
            test -z "$mode" || test "$mode" = "$argument" \
                || die 'choose exactly one build mode: --docker or --native'
            mode=$argument
            ;;
        --dev)
            dev_mode=true
            ;;
        *) die 'usage: scripts/release/linux.sh [--docker|--native] [--dev]' ;;
    esac
done
mode=${mode:---docker}

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
if test "$dev_mode" != true; then
    git diff-index --quiet HEAD -- || die 'tracked files are dirty; commit the release inputs first'
    test -z "$(git status --porcelain --untracked-files=normal)" \
        || die 'untracked release inputs exist'
fi

machine=$(uname -m)
case "$machine" in
    x86_64)
        oci_arch=amd64
        artifact_arch=x86_64
        cuda_image=docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43
        ;;
    *) die "unsupported Linux architecture: $machine" ;;
esac
version=$(awk -F ' *= *' '$1 == "version" { gsub(/"/, "", $2); print $2; exit }' Cargo.toml)
test -n "$version" || die 'package version is missing'
commit=$(git rev-parse HEAD)
SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-$(git show -s --format=%ct HEAD)}
export SOURCE_DATE_EPOCH

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/noperson-release.XXXXXXXX")
archive_tmp=
checksum_tmp=
cleanup() {
    rm -rf -- "$work_dir"
    test -z "$archive_tmp" || rm -f -- "$archive_tmp"
    test -z "$checksum_tmp" || rm -f -- "$checksum_tmp"
}
trap cleanup EXIT INT TERM
source_dir="$work_dir/source"
mkdir -p "$source_dir" "$repo_root/dist"
git archive --format=tar HEAD | tar -xf - -C "$source_dir"

host_uid=$(id -u)
host_gid=$(id -g)
artifact="noperson-v${version}-linux-${artifact_arch}"

if test "$mode" = --native; then
    command -v rustup >/dev/null || die 'rustup is required for native build'
    command -v cargo >/dev/null || die 'cargo is required for native build'
    command -v curl >/dev/null || die 'curl is required for native build'
    command -v patchelf >/dev/null || die 'patchelf is required for native build'
    cuda_root=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}
    nvcc="$cuda_root/bin/nvcc"
    test -x "$nvcc" || die "nvcc is missing: $nvcc"
    nvcc_version=$("$nvcc" --version)
    case "$nvcc_version" in
        *'release 12.8'*) ;;
        *) die 'CUDA Toolkit release 12.8 is required' ;;
    esac
    if test "$dev_mode" = true \
        && rustup run "$RUST_TOOLCHAIN" rustc --version >/dev/null 2>&1; then
        printf 'release: using cached Rust toolchain %s\n' "$RUST_TOOLCHAIN"
    else
        rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal
    fi
    export CUDA_HOME="$cuda_root"
    export ORT_CUDA_VERSION=12
    export CARGO_INCREMENTAL=0
    export CARGO_PROFILE_RELEASE_LTO=true
    export NOPERSON_CUDA_ARCH=compute_75
    export CARGO_BUILD_JOBS=2
    dependency_root=$work_dir
    if test "$dev_mode" = true; then
        dependency_root="$repo_root/.cache/release/linux-${artifact_arch}"
        release_target_dir="$dependency_root/cargo-target"
    else
        release_target_dir="$work_dir/cargo-target"
    fi
    build_native_video_dependencies "$dependency_root"
    export PKG_CONFIG_PATH="$ffmpeg_prefix/lib/pkgconfig"
    export NOPERSON_NV_CODEC_HEADERS="$nv_codec_headers"
    export NOPERSON_REQUIRE_NV_CODEC_HEADERS=1
    if test "$dev_mode" != true; then
        export RUSTFLAGS="--remap-path-prefix=${repo_root}=. -C link-arg=-Wl,--build-id=none"
    fi
    CARGO_TARGET_DIR="$release_target_dir" \
        cargo "+$RUST_TOOLCHAIN" build --locked --release
    verify_ort_cuda12 "$release_target_dir/release"

    stage="$work_dir/$artifact"
    mkdir -p "$stage"
    install -m 0755 "$release_target_dir/release/noperson" "$stage/noperson"
    stage_native_video_dependencies "$stage"
    verify_native_video_bundle "$stage"
    install -m 0644 LICENSE README.md "$stage/"
    {
        printf 'commit=%s\n' "$commit"
        printf 'source_date_epoch=%s\n' "$SOURCE_DATE_EPOCH"
        printf 'rustc=%s\n' "$(rustc "+$RUST_TOOLCHAIN" --version)"
        printf 'cargo=%s\n' "$(cargo "+$RUST_TOOLCHAIN" --version)"
        printf 'nvcc=%s\n' "$("$nvcc" --version | tail -1)"
        printf 'cargo_lock_sha256=%s\n' "$(sha256sum Cargo.lock | awk '{print $1}')"
    } >"$stage/BUILD-MANIFEST"
    find "$stage" -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +
    archive="$repo_root/dist/${artifact}.tar.zst"
    checksum="${archive}.sha256"
    archive_tmp=$(mktemp "$repo_root/dist/.${artifact}.tar.zst.XXXXXXXX")
    checksum_tmp=$(mktemp "$repo_root/dist/.${artifact}.tar.zst.sha256.XXXXXXXX")
    tar --sort=name --mtime="@${SOURCE_DATE_EPOCH}" --owner=0 --group=0 \
        --numeric-owner -C "$work_dir" -cf - "$artifact" \
        | zstd -q -T0 -19 >"$archive_tmp"
    printf '%s  %s\n' "$(sha256sum "$archive_tmp" | awk '{print $1}')" \
        "${artifact}.tar.zst" >"$checksum_tmp"
    chmod 0644 "$archive_tmp" "$checksum_tmp"
    mv -f -- "$archive_tmp" "$archive"
    archive_tmp=
    mv -f -- "$checksum_tmp" "$checksum"
    checksum_tmp=
    printf 'release: %s\n' "$archive"
    printf 'release: %s\n' "$checksum"
    exit 0
fi

runtime=${CONTAINER_RUNTIME:-}
if test -z "$runtime"; then
    if command -v podman >/dev/null; then
        runtime=podman
    elif command -v docker >/dev/null; then
        runtime=docker
    else
        die 'Docker or Podman is required for --docker'
    fi
fi
command -v "$runtime" >/dev/null || die "container runtime not found: $runtime"

output_uid=$host_uid
output_gid=$host_gid
if test "$runtime" = podman; then
    rootless=$(podman info --format '{{.Host.Security.Rootless}}' 2>/dev/null || true)
    if test "$rootless" = true; then
        output_uid=0
        output_gid=0
    fi
fi

"$runtime" run --rm --interactive --platform "linux/${oci_arch}" \
    -e "RUST_TOOLCHAIN=$RUST_TOOLCHAIN" \
    -e "APT_SNAPSHOT=$APT_SNAPSHOT" \
    -e "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" \
    -e "RELEASE_COMMIT=$commit" \
    -e "RELEASE_ARTIFACT=$artifact" \
    -e "OUTPUT_UID=$output_uid" \
    -e "OUTPUT_GID=$output_gid" \
    -e "FFMPEG_VERSION=$FFMPEG_VERSION" \
    -e "FFMPEG_SHA256=$FFMPEG_SHA256" \
    -e "NV_CODEC_HEADERS_VERSION=$NV_CODEC_HEADERS_VERSION" \
    -e "NV_CODEC_HEADERS_SHA256=$NV_CODEC_HEADERS_SHA256" \
    -v "$source_dir:/input:ro" \
    -v "$repo_root/dist:/dist" \
    "$cuda_image" bash -Eeuo pipefail -s <<'BUILD'
export DEBIAN_FRONTEND=noninteractive
cat >/etc/apt/sources.list.d/ubuntu.sources <<EOF
Types: deb
URIs: https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}/
Suites: noble noble-updates noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
Check-Valid-Until: no
EOF
rm -f /etc/apt/sources.list.d/ubuntu.sources.curtin.orig /etc/apt/sources.list
apt-get -o Acquire::Check-Valid-Until=false update
apt-get -o Acquire::Check-Valid-Until=false install -y --no-install-recommends \
    build-essential ca-certificates curl git libclang-dev libgtk-3-dev libjpeg-dev \
    libssl-dev libudev-dev libv4l-dev libwayland-dev libx11-dev libxkbcommon-dev \
    libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev nasm patchelf pkg-config xz-utils zstd
rm -rf /var/lib/apt/lists/*

export CARGO_HOME=/opt/rust/cargo
export RUSTUP_HOME=/opt/rust/rustup
curl --proto '=https' --tlsv1.2 -fsS https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
export PATH="$CARGO_HOME/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export ORT_CUDA_VERSION=12
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_LTO=true
export NOPERSON_CUDA_ARCH=compute_75
export CARGO_BUILD_JOBS=2
export RUSTFLAGS="--remap-path-prefix=/build/source=. -C link-arg=-Wl,--build-id=none"

mkdir -p /build/source /build/stage /build/dependencies
cp -a /input/. /build/source/
curl -fL --retry 3 "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
    -o "/build/dependencies/ffmpeg-${FFMPEG_VERSION}.tar.xz"
printf '%s  %s\n' "$FFMPEG_SHA256" "/build/dependencies/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
    | sha256sum -c -
tar -xf "/build/dependencies/ffmpeg-${FFMPEG_VERSION}.tar.xz" -C /build/dependencies
curl -fL --retry 3 \
    "https://github.com/FFmpeg/nv-codec-headers/archive/refs/tags/${NV_CODEC_HEADERS_VERSION}.tar.gz" \
    -o "/build/dependencies/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz"
printf '%s  %s\n' "$NV_CODEC_HEADERS_SHA256" \
    "/build/dependencies/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz" | sha256sum -c -
tar -xf "/build/dependencies/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz" \
    -C /build/dependencies
ffmpeg_source="/build/dependencies/ffmpeg-${FFMPEG_VERSION}"
ffmpeg_prefix=/build/dependencies/ffmpeg-runtime
cd "$ffmpeg_source"
./configure \
    --prefix="$ffmpeg_prefix" \
    --disable-static --enable-shared --disable-gpl --disable-nonfree \
    --disable-programs --disable-doc --disable-debug --disable-x86asm --disable-network \
    --disable-avdevice --disable-avfilter --disable-swscale --disable-swresample \
    --disable-encoders --disable-decoders --disable-hwaccels \
    --disable-filters --disable-devices
printf 'release: building minimal FFmpeg runtime\n'
make -s -j"$(nproc)"
make -s install

cd /build/source
export PKG_CONFIG_PATH="$ffmpeg_prefix/lib/pkgconfig"
export NOPERSON_NV_CODEC_HEADERS="/build/dependencies/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}/include"
export NOPERSON_REQUIRE_NV_CODEC_HEADERS=1
cargo build --locked --release

provider=$(find -L target/release -maxdepth 1 -type f \
    -name 'libonnxruntime_providers_cuda.so' -print -quit)
test -n "$provider" || { echo 'release: CUDAExecutionProvider library is missing' >&2; exit 1; }
needed=$(readelf -d "$provider")
case "$needed" in *'libcublasLt.so.12'*) ;; *) echo 'release: ORT provider does not target cuBLAS 12' >&2; exit 1 ;; esac
case "$needed" in *'libcudart.so.12'*) ;; *) echo 'release: ORT provider does not target CUDA runtime 12' >&2; exit 1 ;; esac
case "$needed" in *'.so.13'*) echo 'release: CUDA 13 dependency leaked into CUDA 12 release' >&2; exit 1 ;; esac

stage="/build/stage/${RELEASE_ARTIFACT}"
mkdir -p "$stage/lib"
install -m 0755 target/release/noperson "$stage/noperson"
cp -a "$ffmpeg_prefix/lib/libavformat.so"* "$stage/lib/"
cp -a "$ffmpeg_prefix/lib/libavcodec.so"* "$stage/lib/"
cp -a "$ffmpeg_prefix/lib/libavutil.so"* "$stage/lib/"
# Preserve loader-relative ELF paths literally.
# shellcheck disable=SC2016
find "$stage/lib" -type f -name 'libav*.so*' -exec \
    patchelf --force-rpath --set-rpath '$ORIGIN' {} +
# shellcheck disable=SC2016
patchelf --force-rpath --set-rpath '$ORIGIN/lib' "$stage/noperson"
install -m 0644 "$ffmpeg_source/LICENSE.md" "$stage/FFMPEG-LICENSE.md"
{
    printf 'source=https://ffmpeg.org/releases/ffmpeg-%s.tar.xz\n' "$FFMPEG_VERSION"
    printf 'sha256=%s\n' "$FFMPEG_SHA256"
    printf 'configuration=%s\n' "$(sed -n 's/^FFMPEG_CONFIGURATION=//p' "$ffmpeg_source/ffbuild/config.mak")"
} >"$stage/FFMPEG-SOURCE-OFFER"
test "$(patchelf --print-rpath "$stage/noperson")" = '$ORIGIN/lib' \
    || { echo 'release: binary FFmpeg RPATH is not loader-relative' >&2; exit 1; }
closure=$(ldd "$stage/noperson")
case "$closure" in *'not found'*) echo 'release: bundled FFmpeg loader closure is incomplete' >&2; exit 1 ;; esac
for library in libavformat libavcodec libavutil; do
    case "$closure" in *"$stage/lib/${library}.so"*) ;;
        *) echo "release: binary did not resolve bundled ${library}" >&2; exit 1 ;;
    esac
done
install -m 0644 LICENSE README.md "$stage/"
{
    printf 'commit=%s\n' "$RELEASE_COMMIT"
    printf 'source_date_epoch=%s\n' "$SOURCE_DATE_EPOCH"
    printf 'rustc=%s\n' "$(rustc --version)"
    printf 'cargo=%s\n' "$(cargo --version)"
    printf 'nvcc=%s\n' "$(nvcc --version | tail -1)"
    printf 'cargo_lock_sha256=%s\n' "$(sha256sum Cargo.lock | awk '{print $1}')"
} >"$stage/BUILD-MANIFEST"

find "$stage" -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +
cd /build/stage
archive="/dist/${RELEASE_ARTIFACT}.tar.zst"
checksum="${archive}.sha256"
archive_tmp=$(mktemp "/dist/.${RELEASE_ARTIFACT}.tar.zst.XXXXXXXX")
checksum_tmp=$(mktemp "/dist/.${RELEASE_ARTIFACT}.tar.zst.sha256.XXXXXXXX")
tar --sort=name --mtime="@${SOURCE_DATE_EPOCH}" --owner=0 --group=0 \
    --numeric-owner -cf - "$RELEASE_ARTIFACT" \
    | zstd -q -T0 -19 >"$archive_tmp"
printf '%s  %s\n' "$(sha256sum "$archive_tmp" | awk '{print $1}')" \
    "${RELEASE_ARTIFACT}.tar.zst" >"$checksum_tmp"
chmod 0644 "$archive_tmp" "$checksum_tmp"
mv -f -- "$archive_tmp" "$archive"
mv -f -- "$checksum_tmp" "$checksum"
chown "$OUTPUT_UID:$OUTPUT_GID" "$archive" "$checksum"
BUILD

test -f "$repo_root/dist/${artifact}.tar.zst" \
    || die 'container did not export archive'
test -f "$repo_root/dist/${artifact}.tar.zst.sha256" \
    || die 'container did not export checksum'
printf 'release: %s\n' "$repo_root/dist/${artifact}.tar.zst"
printf 'release: %s\n' "$repo_root/dist/${artifact}.tar.zst.sha256"
