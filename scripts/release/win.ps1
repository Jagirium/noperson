[CmdletBinding()]
param(
    [switch]$Orchestrated,
    [switch]$Dev
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RustToolchain = '1.97.1'
$OrtVersion = '1.24.2'
$OrtUrl = 'https://huggingface.co/Jagirium/noperson-runtime/resolve/main/windows/ort/onnxruntime-1.24.2-x86_64-pc-windows-msvc-cu12.tar.lzma2'
$OrtSize = 81934635
$OrtBlake3 = 'c507789d21f3988502925b900249efdba1909c7ac4f5459e9efbd1b9a343009c'
$OrtSha256 = '8a54165e2dfc85e9f6afbdaf154e7c1c74582e6269a2d0ec93b11e1459309555'

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'release.ps1 must run natively on Windows'
}
if (-not $Orchestrated -and $env:NOPERSON_INTERNAL_RELEASE_TEST -ne '1') {
    throw 'Internal packager; use scripts\release.bat --windows (passes -Orchestrated)'
}
foreach ($Command in @('git', 'rustup', 'cargo', 'curl.exe')) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Required command is missing: $Command"
    }
}
if (-not $env:CUDA_PATH) { throw 'CUDA_PATH is not set' }
$Nvcc = Join-Path $env:CUDA_PATH 'bin\nvcc.exe'
if (-not (Test-Path -LiteralPath $Nvcc)) { throw "nvcc.exe is missing: $Nvcc" }
$NvccVersion = (& $Nvcc --version | Out-String)
if ($NvccVersion -notmatch 'release 12\.8') { throw 'CUDA Toolkit release 12.8 is required' }

$RepoRoot = (& git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Run from a git worktree' }
Set-Location -LiteralPath $RepoRoot
if (-not $Dev) {
    & git diff-index --quiet HEAD --
    if ($LASTEXITCODE -ne 0) { throw 'Tracked files are dirty; commit release inputs first' }
    if ((& git status --porcelain --untracked-files=normal).Count -ne 0) {
        throw 'Untracked release inputs exist'
    }
}

$ReleaseCache = Join-Path $RepoRoot '.cache\release'
$env:NOPERSON_RELEASE_RUSTUP_HOME = Join-Path $ReleaseCache 'toolchains\rustup'
$env:NOPERSON_RELEASE_CARGO_HOME = Join-Path $ReleaseCache 'toolchains\cargo'
$env:NOPERSON_RELEASE_DEPENDENCY_ROOT = Join-Path $ReleaseCache 'dependencies\windows-x86_64'
$env:NOPERSON_RELEASE_TARGET_DIR = Join-Path $ReleaseCache 'cargo-target\windows-x86_64'
$env:NOPERSON_RELEASE_DOWNLOAD_ROOT = Join-Path $ReleaseCache 'downloads'
$env:NOPERSON_RELEASE_TOOL_ROOT = Join-Path $ReleaseCache 'tools'
$env:ORT_CACHE_DIR = Join-Path $ReleaseCache 'ort.pyke.io'
$env:RUSTUP_HOME = $env:NOPERSON_RELEASE_RUSTUP_HOME
$env:CARGO_HOME = $env:NOPERSON_RELEASE_CARGO_HOME
foreach ($Directory in @(
    $env:NOPERSON_RELEASE_RUSTUP_HOME,
    $env:NOPERSON_RELEASE_CARGO_HOME,
    $env:NOPERSON_RELEASE_DEPENDENCY_ROOT,
    $env:NOPERSON_RELEASE_TARGET_DIR,
    $env:NOPERSON_RELEASE_DOWNLOAD_ROOT,
    $env:NOPERSON_RELEASE_TOOL_ROOT,
    $env:ORT_CACHE_DIR
)) {
    New-Item -ItemType Directory -Force -Path $Directory | Out-Null
}
$env:Path = (Join-Path $env:NOPERSON_RELEASE_CARGO_HOME 'bin') + ';' + $env:Path

function Test-OrtArchive {
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -ne $OrtSize) { return $false }
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    return $Actual -eq $OrtSha256
}

function Get-OrtExtractor {
    $OrtExtractorManifest = Join-Path $RepoRoot 'scripts\release\ort-extract\Cargo.toml'
    $OrtExtractorTarget = Join-Path $env:NOPERSON_RELEASE_TOOL_ROOT 'ort-extract-target'
    & cargo "+$RustToolchain" build --locked --release --manifest-path $OrtExtractorManifest --target-dir $OrtExtractorTarget
    if ($LASTEXITCODE -ne 0) {
        throw "ORT extractor build phase failed: $OrtExtractorManifest"
    }
    $OrtExtractor = Join-Path $OrtExtractorTarget 'release\noperson-ort-extract.exe'
    if (-not (Test-Path -LiteralPath $OrtExtractor -PathType Leaf)) {
        throw "ORT extractor activation failed: $OrtExtractor"
    }
    return $OrtExtractor
}

function Get-PublishedOrt {
    $OrtArchive = Join-Path $env:NOPERSON_RELEASE_DOWNLOAD_ROOT ([IO.Path]::GetFileName($OrtUrl))
    $OrtPart = "$OrtArchive.part"
    $OrtFinal = Join-Path $env:ORT_CACHE_DIR "dfbin\x86_64-pc-windows-msvc\$OrtSha256"
    $OrtRoot = Split-Path -Parent $OrtFinal
    $OrtComplete = Join-Path $OrtFinal '.complete'
    $OrtCompletionIdentity = "$OrtSha256/$OrtBlake3/$OrtSize"
    $CachedLibrary = Join-Path $OrtFinal 'onnxruntime.lib'
    if ((Test-Path -LiteralPath $CachedLibrary -PathType Leaf) -and
        (Test-Path -LiteralPath $OrtComplete -PathType Leaf) -and
        (Get-Content -LiteralPath $OrtComplete -Raw).Trim() -eq $OrtCompletionIdentity) {
        Write-Host "release: using cached ONNX Runtime $OrtVersion"
        return
    }

    New-Item -ItemType Directory -Force -Path $OrtRoot | Out-Null
    if (-not (Test-OrtArchive -Path $OrtArchive)) {
        if (Test-Path -LiteralPath $OrtArchive) {
            Remove-Item -LiteralPath $OrtArchive -Force
        }
        if ((Test-Path -LiteralPath $OrtPart) -and
            (Get-Item -LiteralPath $OrtPart).Length -gt $OrtSize) {
            Remove-Item -LiteralPath $OrtPart -Force
        }
        & curl.exe --fail --location --retry 3 --continue-at - --output $OrtPart $OrtUrl
        if ($LASTEXITCODE -ne 0 -or -not (Test-OrtArchive -Path $OrtPart)) {
            Remove-Item -LiteralPath $OrtPart -Force -ErrorAction SilentlyContinue
            & curl.exe --fail --location --retry 3 --output $OrtPart $OrtUrl
        }
        if ($LASTEXITCODE -ne 0) {
            throw "ORT download phase failed: $OrtUrl"
        }
        if ((Get-Item -LiteralPath $OrtPart).Length -ne $OrtSize) {
            throw "ORT size validation failed: $OrtPart"
        }
        if (-not (Test-OrtArchive -Path $OrtPart)) {
            throw "ORT SHA-256 validation failed: $OrtPart"
        }
        Move-Item -LiteralPath $OrtPart -Destination $OrtArchive -Force
    }

    $OrtStaging = Join-Path $OrtRoot ".$([IO.Path]::GetFileName($OrtFinal)).staging-$PID-$([Guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Force -Path $OrtStaging | Out-Null
    try {
        $OrtExtractor = Get-OrtExtractor
        & $OrtExtractor $OrtArchive $OrtStaging
        if ($LASTEXITCODE -ne 0) {
            throw "ORT extraction phase failed: $OrtArchive"
        }
        $StagedLibrary = Join-Path $OrtStaging 'onnxruntime.lib'
        if (-not (Test-Path -LiteralPath $StagedLibrary -PathType Leaf)) {
            throw "ORT extraction validation failed; onnxruntime.lib is missing: $OrtArchive"
        }
        $OrtCompletionIdentity | Set-Content -LiteralPath (Join-Path $OrtStaging '.complete') -Encoding ascii
        if (Test-Path -LiteralPath $OrtFinal) {
            Remove-Item -LiteralPath $OrtFinal -Recurse -Force
        }
        Move-Item -LiteralPath $OrtStaging -Destination $OrtFinal
    }
    finally {
        if (Test-Path -LiteralPath $OrtStaging) {
            Remove-Item -LiteralPath $OrtStaging -Recurse -Force
        }
    }
    $InstalledLibrary = Join-Path $OrtFinal 'onnxruntime.lib'
    if (-not (Test-Path -LiteralPath $InstalledLibrary -PathType Leaf)) {
        throw "ORT activation validation failed: $OrtFinal"
    }
}

& rustup toolchain install $RustToolchain --profile minimal
if ($LASTEXITCODE -ne 0) { throw 'Pinned Rust toolchain installation failed' }
$VersionLine = Select-String -LiteralPath 'Cargo.toml' -Pattern '^version\s*=\s*"([^"]+)"' |
    Select-Object -First 1
if (-not $VersionLine) { throw 'Package version is missing' }
$Version = $VersionLine.Matches[0].Groups[1].Value
$Commit = (& git rev-parse HEAD).Trim()
$env:SOURCE_DATE_EPOCH = (& git show -s --format=%ct HEAD).Trim()
$env:ORT_CUDA_VERSION = '12'
Get-PublishedOrt
$env:CARGO_INCREMENTAL = '0'
$env:CARGO_TARGET_DIR = $env:NOPERSON_RELEASE_TARGET_DIR
$env:RUSTFLAGS = '--remap-path-prefix=' + $RepoRoot + '=. -C link-arg=/Brepro'

& cargo "+$RustToolchain" build --locked --release
if ($LASTEXITCODE -ne 0) { throw 'Native Windows release build failed' }

$ArtifactName = "noperson-v$Version-windows-x86_64"
$Dist = Join-Path $RepoRoot 'dist'
$Stage = Join-Path $Dist $ArtifactName
$Archive = Join-Path $Dist "$ArtifactName.zip"
$Checksum = "$Archive.sha256"
if (Test-Path -LiteralPath $Stage) { Remove-Item -LiteralPath $Stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
$Binary = Join-Path $env:NOPERSON_RELEASE_TARGET_DIR 'release\noperson.exe'
Copy-Item -LiteralPath @($Binary, 'LICENSE', 'README.md') -Destination $Stage
if (Get-ChildItem -LiteralPath $Stage -Recurse -File | Where-Object {
    $_.Name -eq 'onnxruntime.lib' -or $_.Name -eq ([IO.Path]::GetFileName($OrtUrl))
}) {
    throw 'release archive contains forbidden ORT build input'
}
$BuildManifest = @(
    "commit=$Commit"
    "source_date_epoch=$($env:SOURCE_DATE_EPOCH)"
    "rustc=$(& rustc "+$RustToolchain" --version)"
    "cargo=$(& cargo "+$RustToolchain" --version)"
    "nvcc=$(& $Nvcc --version | Select-Object -Last 1)"
    "cargo_lock_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath 'Cargo.lock').Hash.ToLowerInvariant())"
)
$BuildManifestPath = Join-Path $Stage 'BUILD-MANIFEST'
$Utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllLines($BuildManifestPath, [string[]]$BuildManifest, $Utf8NoBom)

$Epoch = [DateTimeOffset]::FromUnixTimeSeconds([Int64]$env:SOURCE_DATE_EPOCH).UtcDateTime
Get-ChildItem -LiteralPath $Stage -Recurse -Force | ForEach-Object { $_.LastWriteTimeUtc = $Epoch }
if (Test-Path -LiteralPath $Archive) { Remove-Item -LiteralPath $Archive -Force }
Compress-Archive -LiteralPath $Stage -DestinationPath $Archive -CompressionLevel Optimal
$Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
"$Hash  $([IO.Path]::GetFileName($Archive))" | Set-Content -LiteralPath $Checksum -Encoding ascii
Write-Host "release: $Archive"
Write-Host "release: $Checksum"
