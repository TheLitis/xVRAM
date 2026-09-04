# Shared hardware-gate provenance. Dirty worktrees are allowed and recorded explicitly;
# source bytes and executable bytes must remain unchanged throughout a gate.

function Get-XvramAcceptanceSource {
    param([Parameter(Mandatory = $true)][string]$RepositoryRoot)

    $root = (Resolve-Path -LiteralPath $RepositoryRoot).Path
    $revision = @(& git -C $root rev-parse HEAD)
    if ($LASTEXITCODE -ne 0 -or $revision.Count -ne 1 -or
        $revision[0] -notmatch '^[0-9a-f]{40,64}$') {
        throw "Hardware acceptance provenance: unable to resolve Git HEAD"
    }
    $status = @(& git -C $root -c core.quotepath=false status --porcelain=v1 `
        --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw "Hardware acceptance provenance: unable to inspect the working tree"
    }
    $files = @(& git -C $root -c core.quotepath=false ls-files --cached --others `
        --exclude-standard -- src include python tests schemas cmake scripts `
        CMakeLists.txt CMakePresets.json VERSION)
    if ($LASTEXITCODE -ne 0) {
        throw "Hardware acceptance provenance: unable to enumerate source inputs"
    }
    $records = @($files | Sort-Object -Unique | ForEach-Object {
        $relative = [string]$_
        $path = Join-Path $root $relative
        $hash = if (Test-Path -LiteralPath $path -PathType Leaf) {
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        } else {
            "missing"
        }
        "$relative`0$hash"
    })
    $bytes = [Text.Encoding]::UTF8.GetBytes(($records -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
    return [pscustomobject][ordered]@{
        git_commit = $revision[0]
        git_commit_short = $revision[0].Substring(0, 12)
        working_tree_dirty = $status.Count -ne 0
        git_status_porcelain = @($status)
        source_files = $records.Count
        source_sha256 = $digest
    }
}

function Get-XvramAcceptanceBinary {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $item = Get-Item -LiteralPath $resolved
    if ($item.PSIsContainer) {
        throw "Hardware acceptance provenance: binary is not a file: $resolved"
    }
    return [pscustomobject][ordered]@{
        path = $resolved
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Assert-XvramAcceptanceReportRevision {
    param(
        [Parameter(Mandatory = $true)][string]$Observed,
        [Parameter(Mandatory = $true)][object]$Source
    )

    # Native reports embed Git's --short=12 value; Python reports may contain full HEAD.
    if ($Observed -notmatch '^[0-9a-f]{12,64}$' -or
        -not $Source.git_commit.StartsWith($Observed, [StringComparison]::Ordinal)) {
        throw "Hardware acceptance provenance: report revision does not match the gate's Git HEAD"
    }
}

function Assert-XvramAcceptanceBinaryUnchanged {
    param([Parameter(Mandatory = $true)][object]$Expected)

    $observed = Get-XvramAcceptanceBinary -Path $Expected.path
    if ($observed.sha256 -ne $Expected.sha256 -or $observed.bytes -ne $Expected.bytes) {
        throw "Hardware acceptance provenance: binary changed during the gate: $($Expected.path)"
    }
}

function Assert-XvramAcceptanceSourceUnchanged {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot,
        [Parameter(Mandatory = $true)][object]$Expected
    )

    $observed = Get-XvramAcceptanceSource -RepositoryRoot $RepositoryRoot
    if ($observed.git_commit -ne $Expected.git_commit -or
        $observed.source_sha256 -ne $Expected.source_sha256) {
        throw "Hardware acceptance provenance: Git HEAD or source inputs changed during the gate"
    }
    return $observed
}

function Initialize-XvramAcceptanceManifest {
    param([Parameter(Mandatory = $true)][string]$OutputDirectory)

    $root = (Resolve-Path -LiteralPath $OutputDirectory).Path
    $path = [IO.Path]::GetFullPath((Join-Path $root "acceptance-manifest.json"))
    if (-not [string]::Equals([IO.Path]::GetDirectoryName($path), $root,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "Hardware acceptance provenance: manifest escaped the resolved output directory"
    }
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        Remove-Item -LiteralPath $path -Force
    }
    return $path
}
