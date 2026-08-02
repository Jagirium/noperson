[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$RustToolchain = '1.97.1'

if (-not $IsWindows) { throw 'release.ps1 must run natively on Windows' }
foreach ($Command in @('git', 'rustup', 'cargo')) {
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
& git diff-index --quiet HEAD --
if ($LASTEXITCODE -ne 0) { throw 'Tracked files are dirty; commit release inputs first' }
if ((& git status --porcelain --untracked-files=normal).Count -ne 0) {
    throw 'Untracked release inputs exist'
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
$env:CARGO_INCREMENTAL = '0'
$env:NOPERSON_CUDA_ARCH = 'compute_75'
$env:RUSTFLAGS = '--remap-path-prefix=' + $RepoRoot + '=. -C link-arg=/Brepro'

& cargo "+$RustToolchain" build --locked --release
if ($LASTEXITCODE -ne 0) { throw 'Native Windows release build failed' }

$ArtifactName = "noperson-v$Version-windows-x86_64"
$Dist = Join-Path $RepoRoot 'dist'
$Stage = Join-Path $Dist $ArtifactName
$Archive = Join-Path $Dist "$ArtifactName.zip"
$Checksum = "$Archive.sha256"
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
Copy-Item -LiteralPath 'target\release\noperson.exe', 'LICENSE', 'README.md' -Destination $Stage
@(
    "commit=$Commit"
    "source_date_epoch=$($env:SOURCE_DATE_EPOCH)"
    "rustc=$(& rustc "+$RustToolchain" --version)"
    "cargo=$(& cargo "+$RustToolchain" --version)"
    "nvcc=$(& $Nvcc --version | Select-Object -Last 1)"
    "cargo_lock_sha256=$((Get-FileHash -Algorithm SHA256 -LiteralPath 'Cargo.lock').Hash.ToLowerInvariant())"
) | Set-Content -LiteralPath (Join-Path $Stage 'BUILD-MANIFEST') -Encoding utf8NoBOM

$Epoch = [DateTimeOffset]::FromUnixTimeSeconds([Int64]$env:SOURCE_DATE_EPOCH).UtcDateTime
Get-ChildItem -LiteralPath $Stage -Recurse -Force | ForEach-Object { $_.LastWriteTimeUtc = $Epoch }
if (Test-Path -LiteralPath $Archive) { Remove-Item -LiteralPath $Archive -Force }
Compress-Archive -LiteralPath $Stage -DestinationPath $Archive -CompressionLevel Optimal
$Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
"$Hash  $([IO.Path]::GetFileName($Archive))" | Set-Content -LiteralPath $Checksum -Encoding ascii
Write-Host "release: $Archive"
Write-Host "release: $Checksum"
