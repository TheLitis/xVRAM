[CmdletBinding()]
param(
    [string]$DependencyRoot = "D:/xVRAM-dependencies/phase6b0",
    [string]$HfPath,
    [ValidateSet("14b", "32b", "both", "none")][string]$Models = "both",
    [switch]$VerifyOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$manifestPath = Join-Path $repositoryRoot "python/xvram/compat_audit_profile.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.profile_id -ne "llama-b10819-qwen25-q4km-win-cuda133-v1") {
    throw "Unexpected Phase 6b audit dependency profile"
}
$dependencyPath = [IO.Path]::GetFullPath($DependencyRoot).TrimEnd('\', '/')
$repositoryPrefix = $repositoryRoot.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
if ($dependencyPath -eq [IO.Path]::GetPathRoot($dependencyPath).TrimEnd('\', '/') -or
    $dependencyPath -eq $repositoryRoot -or
    $dependencyPath.StartsWith($repositoryPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "DependencyRoot must be a dedicated directory outside the repository, not a drive root"
}

function Assert-NoReparsePath {
    param([string]$Path)
    $candidate = [IO.Path]::GetFullPath($Path)
    while ($candidate) {
        if (Test-Path -LiteralPath $candidate) {
            $entry = Get-Item -LiteralPath $candidate -Force
            if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Refusing a dependency path containing a reparse point: $candidate"
            }
        }
        $parent = [IO.Path]::GetDirectoryName($candidate)
        if (-not $parent -or $parent -eq $candidate) { break }
        $candidate = $parent
    }
}

function Get-DependencyFile {
    param([string]$Directory, [string]$Name)
    if ([IO.Path]::GetFileName($Name) -cne $Name -or $Name -in @("", ".", "..") -or
        $Name.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0) {
        throw "Invalid file name in pinned dependency manifest: $Name"
    }
    $candidate = [IO.Path]::GetFullPath((Join-Path $Directory $Name))
    $allowedPrefix = $dependencyPath + [IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($allowedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Dependency file escaped the selected directory"
    }
    Assert-NoReparsePath $candidate
    return $candidate
}

function Assert-FileDigest {
    param([string]$Path, [string]$ExpectedHash, [long]$ExpectedBytes = -1)
    if ($ExpectedHash -cnotmatch '^[a-f0-9]{64}$') { throw "Invalid SHA-256 pin" }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing dependency: $Path" }
    Assert-NoReparsePath $Path
    $entry = Get-Item -LiteralPath $Path
    if ($ExpectedBytes -ge 0 -and $entry.Length -ne $ExpectedBytes) {
        throw "Existing file has the wrong size; preserved without overwrite: $Path"
    }
    $actualHash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -cne $ExpectedHash) {
        throw "Existing file has the wrong SHA-256; preserved without overwrite: $Path"
    }
}

function Ensure-Directory {
    param([string]$Path)
    Assert-NoReparsePath $Path
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        if ($VerifyOnly) { throw "Missing dependency directory: $Path" }
        $null = New-Item -ItemType Directory -Path $Path
    }
}

Assert-NoReparsePath $dependencyPath
$modelKeys = @(switch ($Models) {
    "both" { @("14b", "32b") }
    "none" { }
    default { @($Models) }
})
$binaryDirectory = Join-Path $dependencyPath "llama-b10819"

# Validate all pre-existing pinned bytes before any download. A corrupt final file is
# an operator-visible failure, never a reason to silently request --force-download.
foreach ($archive in $manifest.upstream.archives) {
    $archivePath = Get-DependencyFile $dependencyPath $archive.name
    if (Test-Path -LiteralPath $archivePath) {
        Assert-FileDigest $archivePath $archive.sha256 $archive.bytes
    }
}
foreach ($file in $manifest.upstream.files.PSObject.Properties) {
    $binaryPath = Get-DependencyFile $binaryDirectory $file.Name
    if (Test-Path -LiteralPath $binaryPath) { Assert-FileDigest $binaryPath $file.Value }
}
foreach ($modelKey in $modelKeys) {
    $model = $manifest.models.PSObject.Properties[$modelKey].Value
    foreach ($file in $model.files) {
        $modelPath = Get-DependencyFile (Join-Path $dependencyPath "qwen$modelKey") $file.name
        if (Test-Path -LiteralPath $modelPath) {
            Assert-FileDigest $modelPath $file.sha256 $file.bytes
        }
    }
}

if (-not $VerifyOnly -and @($modelKeys).Count -gt 0) {
    if (-not $HfPath) { throw "Supply -HfPath to an existing hf CLI; this script does not install tools" }
    $hfCommand = Get-Command $HfPath -CommandType Application -ErrorAction Stop
    $resolvedHfPath = $hfCommand.Source
}

Ensure-Directory $dependencyPath
Ensure-Directory $binaryDirectory
Add-Type -AssemblyName System.IO.Compression.FileSystem
foreach ($archive in $manifest.upstream.archives) {
    $archivePath = Get-DependencyFile $dependencyPath $archive.name
    if (-not (Test-Path -LiteralPath $archivePath)) {
        if ($VerifyOnly) { throw "Missing archive: $archivePath" }
        $expectedUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b10819/" + $archive.name
        if ($archive.url -cne $expectedUrl) { throw "Unexpected archive source URL" }
        $temporaryPath = Get-DependencyFile $dependencyPath ($archive.name + ".download-" + [Guid]::NewGuid().ToString("N"))
        Write-Host "Downloading pinned archive $($archive.name)"
        try {
            Invoke-WebRequest -Uri $archive.url -OutFile $temporaryPath
            Assert-FileDigest $temporaryPath $archive.sha256 $archive.bytes
            Move-Item -LiteralPath $temporaryPath -Destination $archivePath
        } catch {
            Write-Warning "Any partial archive remains at $temporaryPath for inspection; no existing archive was replaced"
            throw
        }
    }
    Assert-FileDigest $archivePath $archive.sha256 $archive.bytes
    $zip = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        foreach ($entry in $zip.Entries) {
            # This pinned distribution is flat. Reject path-bearing entries rather
            # than broadening the extraction target or trusting ZIP path traversal.
            $destination = Get-DependencyFile $binaryDirectory $entry.FullName
            $source = $entry.Open()
            $hasher = [Security.Cryptography.SHA256]::Create()
            try {
                $entryHash = [BitConverter]::ToString($hasher.ComputeHash($source)).Replace("-", "").ToLowerInvariant()
            } finally {
                $hasher.Dispose()
                $source.Dispose()
            }
            $pinnedFile = $manifest.upstream.files.PSObject.Properties[$entry.FullName]
            if ($null -ne $pinnedFile -and $entryHash -cne $pinnedFile.Value) {
                throw "Verified archive entry disagrees with its extracted-file pin: $($entry.FullName)"
            }
            if (Test-Path -LiteralPath $destination) {
                Assert-FileDigest $destination $entryHash $entry.Length
                continue
            }
            if ($VerifyOnly) { throw "Missing extracted archive file: $destination" }
            $temporaryPath = Get-DependencyFile $binaryDirectory ($entry.FullName + ".extract-" + [Guid]::NewGuid().ToString("N"))
            $source = $entry.Open()
            $output = [IO.File]::Open($temporaryPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try { $source.CopyTo($output) } finally { $output.Dispose(); $source.Dispose() }
            Assert-FileDigest $temporaryPath $entryHash $entry.Length
            Move-Item -LiteralPath $temporaryPath -Destination $destination
        }
    } finally { $zip.Dispose() }
}
foreach ($file in $manifest.upstream.files.PSObject.Properties) {
    Assert-FileDigest (Get-DependencyFile $binaryDirectory $file.Name) $file.Value
}

foreach ($modelKey in $modelKeys) {
    $model = $manifest.models.PSObject.Properties[$modelKey].Value
    $modelDirectory = Join-Path $dependencyPath "qwen$modelKey"
    Ensure-Directory $modelDirectory
    foreach ($file in $model.files) {
        $destination = Get-DependencyFile $modelDirectory $file.name
        if (-not (Test-Path -LiteralPath $destination)) {
            if ($VerifyOnly) { throw "Missing model shard: $destination" }
            Write-Host "Downloading $($model.repo) $($file.name) at $($model.revision)"
            # One exact include per call: no other quantization, tokenizer file,
            # alternative shard, or floating revision is downloaded.
            $arguments = @("download", $model.repo, "--revision", $model.revision,
                "--include", $file.name, "--local-dir", $modelDirectory, "--max-workers", "2")
            & $resolvedHfPath @arguments
            if ($LASTEXITCODE -ne 0) { throw "hf download failed with exit $LASTEXITCODE" }
        }
        Assert-FileDigest $destination $file.sha256 $file.bytes
    }
}

Write-Host "Verified pinned Phase 6b audit dependencies at $dependencyPath"
Write-Host "Models: $Models. Shards remain unmerged; the upstream loader accepts the first shard."
Write-Host "This prepares observation inputs only; no llama.cpp executable or GPU workload was launched."
