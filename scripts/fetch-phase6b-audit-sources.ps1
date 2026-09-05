param(
    [Parameter(Mandatory = $true)][string]$SourceRoot,
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
$profilePath = Join-Path $PSScriptRoot '../python/xvram/compat_audit_source_profile.json'
$profile = Get-Content -Raw -LiteralPath $profilePath | ConvertFrom-Json
$sourceRootPath = [IO.Path]::GetFullPath($SourceRoot)
$repositoryPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$separator = [IO.Path]::DirectorySeparatorChar
if ($sourceRootPath.Equals($repositoryPath, [StringComparison]::OrdinalIgnoreCase) -or
    $sourceRootPath.StartsWith($repositoryPath + $separator, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Source dependencies must be outside the xVRAM repository.'
}
if ($profile.upstream_commit -cne '6a1a922d269908a29cbd4b49c27e6a8e7fd10fae' -or
    $profile.source_base_url -cne 'https://raw.githubusercontent.com/ggml-org/llama.cpp/6a1a922d269908a29cbd4b49c27e6a8e7fd10fae/') {
    throw 'Unexpected source provenance.'
}

function Assert-PlainPath([string]$Path) {
    $cursor = $Path
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -Force -LiteralPath $cursor
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw 'Source dependencies must not traverse reparse points.'
            }
        }
        $parent = [IO.Path]::GetDirectoryName($cursor)
        if ($parent -eq $cursor) { break }
        $cursor = $parent
    }
}

foreach ($entry in $profile.files) {
    if ($entry.path -notmatch '^ggml/(src|include)/[A-Za-z0-9_./-]+$' -or $entry.path.Contains('..') -or
        $entry.sha256 -notmatch '^[0-9a-f]{64}$' -or $entry.size_bytes -le 0 -or $entry.size_bytes -gt 2097152) {
        throw 'Invalid source manifest entry.'
    }
    $target = [IO.Path]::GetFullPath((Join-Path $sourceRootPath $entry.path))
    if (-not $target.StartsWith($sourceRootPath + $separator, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Source path escapes dependency root.'
    }
    Assert-PlainPath $target
    if (Test-Path -LiteralPath $target) {
        $existing = Get-Item -LiteralPath $target
        if ($existing.Length -ne $entry.size_bytes -or
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.sha256) {
            throw "Existing source differs from pinned bytes: $($entry.path)"
        }
    } elseif ($VerifyOnly) {
        throw "Missing source: $($entry.path)"
    } else {
        $targetDirectory = [IO.Path]::GetDirectoryName($target)
        New-Item -ItemType Directory -Force -Path $targetDirectory | Out-Null
        Assert-PlainPath $targetDirectory
        $temporary = Join-Path $targetDirectory ('.xvram-source-' + [Guid]::NewGuid().ToString('N') + '.tmp')
        try {
            Invoke-WebRequest -Uri ($profile.source_base_url + $entry.path) -OutFile $temporary -TimeoutSec 60
            if ((Get-Item -LiteralPath $temporary).Length -ne $entry.size_bytes -or
                (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.sha256) {
                throw "Downloaded source differs from pinned bytes: $($entry.path)"
            }
            Move-Item -LiteralPath $temporary -Destination $target
        } finally {
            if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
        }
    }
    Write-Output "Verified $($entry.path)"
}
