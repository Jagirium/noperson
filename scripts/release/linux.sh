#!/usr/bin/env bash
set -Eeuo pipefail

RUST_TOOLCHAIN=1.97.1
APT_SNAPSHOT=20250701T000000Z

die() {
    printf 'release: %s\n' "$*" >&2
    exit 1
}

command -v git >/dev/null || die 'git is required'
command -v tar >/dev/null || die 'tar is required'
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die 'run from a git worktree'

mode=${1:---docker}
case "$mode" in
    --docker|--native) ;;
    *) die 'usage: scripts/release/linux.sh [--docker|--native]' ;;
esac

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"
git diff-index --quiet HEAD -- || die 'tracked files are dirty; commit the release inputs first'
test -z "$(git status --porcelain --untracked-files=normal)" || die 'untracked release inputs exist'

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
cleanup() { rm -rf -- "$work_dir"; }
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
    cuda_root=${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}
    nvcc="$cuda_root/bin/nvcc"
    test -x "$nvcc" || die "nvcc is missing: $nvcc"
    case "$("$nvcc" --version | tail -1)" in
        *'release 12.8'*) ;;
        *) die 'CUDA Toolkit release 12.8 is required' ;;
    esac
    rustup toolchain install "$RUST_TOOLCHAIN" --profile minimal
    export CUDA_HOME="$cuda_root"
    export CARGO_INCREMENTAL=0
    export NOPERSON_CUDA_ARCH=sm_86
    export RUSTFLAGS="--remap-path-prefix=${repo_root}=. -C link-arg=-Wl,--build-id=none"
    cargo "+$RUST_TOOLCHAIN" build --locked --release

    stage="$work_dir/$artifact"
    mkdir -p "$stage/lib"
    install -m 0755 target/release/noperson "$stage/noperson"
    install -m 0644 LICENSE README.md "$stage/"
    find target/release -type f -name 'libonnxruntime.so*' -exec install -m 0755 {} "$stage/lib/" \;
    # These expressions belong in the generated launcher.
    # shellcheck disable=SC2016
    printf '%s\n' '#!/usr/bin/env bash' 'set -Eeuo pipefail' \
        'root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)' \
        'export LD_LIBRARY_PATH="$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' \
        'exec "$root/noperson" "$@"' >"$stage/run.sh"
    chmod 0755 "$stage/run.sh"
    {
        printf 'commit=%s\n' "$commit"
        printf 'source_date_epoch=%s\n' "$SOURCE_DATE_EPOCH"
        printf 'rustc=%s\n' "$(rustc "+$RUST_TOOLCHAIN" --version)"
        printf 'cargo=%s\n' "$(cargo "+$RUST_TOOLCHAIN" --version)"
        printf 'nvcc=%s\n' "$("$nvcc" --version | tail -1)"
        printf 'cargo_lock_sha256=%s\n' "$(sha256sum Cargo.lock | awk '{print $1}')"
    } >"$stage/BUILD-MANIFEST"
    find "$stage" -exec touch -h -d "@${SOURCE_DATE_EPOCH}" {} +
    tar --sort=name --mtime="@${SOURCE_DATE_EPOCH}" --owner=0 --group=0 \
        --numeric-owner -C "$work_dir" -cf - "$artifact" \
        | gzip -n -9 >"$repo_root/dist/${artifact}.tar.gz"
    cd "$repo_root/dist"
    sha256sum "${artifact}.tar.gz" >"${artifact}.tar.gz.sha256"
    printf 'release: %s\n' "$repo_root/dist/${artifact}.tar.gz"
    printf 'release: %s\n' "$repo_root/dist/${artifact}.tar.gz.sha256"
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

"$runtime" run --rm --platform "linux/${oci_arch}" \
    -e "RUST_TOOLCHAIN=$RUST_TOOLCHAIN" \
    -e "APT_SNAPSHOT=$APT_SNAPSHOT" \
    -e "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" \
    -e "RELEASE_COMMIT=$commit" \
    -e "RELEASE_ARTIFACT=$artifact" \
    -e "HOST_UID=$host_uid" \
    -e "HOST_GID=$host_gid" \
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
    libudev-dev libv4l-dev libwayland-dev libx11-dev libxkbcommon-dev \
    libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev pkg-config xz-utils
rm -rf /var/lib/apt/lists/*

export CARGO_HOME=/opt/rust/cargo
export RUSTUP_HOME=/opt/rust/rustup
curl --proto '=https' --tlsv1.2 -fsS https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain "$RUST_TOOLCHAIN"
export PATH="$CARGO_HOME/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export CARGO_INCREMENTAL=0
export RUSTFLAGS="--remap-path-prefix=/build/source=. -C link-arg=-Wl,--build-id=none"

mkdir -p /build/source /build/stage
cp -a /input/. /build/source/
cd /build/source
cargo build --locked --release

stage="/build/stage/${RELEASE_ARTIFACT}"
mkdir -p "$stage/lib"
install -m 0755 target/release/noperson "$stage/noperson"
install -m 0644 LICENSE README.md "$stage/"
find target/release -type f -name 'libonnxruntime.so*' -exec install -m 0755 {} "$stage/lib/" \;
cat >"$stage/run.sh" <<'RUNNER'
#!/usr/bin/env bash
set -Eeuo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export LD_LIBRARY_PATH="$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$root/noperson" "$@"
RUNNER
chmod 0755 "$stage/run.sh"
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
tar --sort=name --mtime="@${SOURCE_DATE_EPOCH}" --owner=0 --group=0 \
    --numeric-owner -cf - "$RELEASE_ARTIFACT" \
    | gzip -n -9 >"/dist/${RELEASE_ARTIFACT}.tar.gz"
cd /dist
sha256sum "${RELEASE_ARTIFACT}.tar.gz" >"${RELEASE_ARTIFACT}.tar.gz.sha256"
chown "$HOST_UID:$HOST_GID" "${RELEASE_ARTIFACT}.tar.gz" "${RELEASE_ARTIFACT}.tar.gz.sha256"
BUILD

printf 'release: %s\n' "$repo_root/dist/${artifact}.tar.gz"
printf 'release: %s\n' "$repo_root/dist/${artifact}.tar.gz.sha256"
