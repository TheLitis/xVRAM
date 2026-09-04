param([Parameter(Mandatory = $true)][string]$Helper)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. (Resolve-Path -LiteralPath $Helper).Path

function Assert-Test {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

function Assert-Rejected {
    param([scriptblock]$Action)
    $rejected = $false
    try { & $Action | Out-Null } catch { $rejected = $true }
    Assert-Test $rejected "changed provenance was accepted"
}

$temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$fixture = Join-Path $temporaryRoot ("xvram-provenance-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $fixture | Out-Null
try {
    New-Item -ItemType Directory -Path (Join-Path $fixture "src") | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $fixture "docs") | Out-Null
    $sourcePath = Join-Path $fixture "src/input.cpp"
    $docsPath = Join-Path $fixture "docs/note.md"
    [IO.File]::WriteAllText($sourcePath, "int fixture = 1;`n")
    [IO.File]::WriteAllText($docsPath, "fixture`n")
    & git -C $fixture init --quiet
    Assert-Test ($LASTEXITCODE -eq 0) "fixture git init failed"
    & git -C $fixture add -- src docs
    Assert-Test ($LASTEXITCODE -eq 0) "fixture git add failed"
    & git -C $fixture -c user.name=Fixture -c user.email=fixture@example.invalid `
        -c commit.gpgsign=false commit --quiet -m fixture
    Assert-Test ($LASTEXITCODE -eq 0) "fixture git commit failed"

    $baseline = Get-XvramAcceptanceSource -RepositoryRoot $fixture
    Assert-Test (-not $baseline.working_tree_dirty) "fresh fixture was reported dirty"
    Assert-Test ($baseline.source_files -eq 1) "docs entered the source snapshot"
    Assert-XvramAcceptanceReportRevision -Observed $baseline.git_commit -Source $baseline
    Assert-XvramAcceptanceReportRevision -Observed $baseline.git_commit_short -Source $baseline
    Assert-Rejected { Assert-XvramAcceptanceReportRevision -Observed "unknown" -Source $baseline }
    Assert-Rejected {
        Assert-XvramAcceptanceReportRevision -Observed ("0" * 40) -Source $baseline
    }
    Assert-Rejected {
        Assert-XvramAcceptanceReportRevision -Observed $baseline.git_commit.Substring(0, 7) `
            -Source $baseline
    }

    [IO.File]::WriteAllText($docsPath, "changed docs`n")
    $docsChanged = Assert-XvramAcceptanceSourceUnchanged -RepositoryRoot $fixture -Expected $baseline
    Assert-Test $docsChanged.working_tree_dirty "docs changes were not recorded"
    Assert-Test ($docsChanged.git_status_porcelain.Count -eq 1) "porcelain evidence was lost"
    [IO.File]::WriteAllText($sourcePath, "int fixture = 2;`n")
    Assert-Rejected {
        Assert-XvramAcceptanceSourceUnchanged -RepositoryRoot $fixture -Expected $baseline
    }
    $dirty = Get-XvramAcceptanceSource -RepositoryRoot $fixture
    Assert-Test $dirty.working_tree_dirty "modified source was not recorded"
    Assert-XvramAcceptanceSourceUnchanged -RepositoryRoot $fixture -Expected $dirty | Out-Null
    [IO.File]::WriteAllText((Join-Path $fixture "src/untracked.cpp"), "int new_fixture;`n")
    Assert-Rejected {
        Assert-XvramAcceptanceSourceUnchanged -RepositoryRoot $fixture -Expected $dirty
    }

    $binaryPath = Join-Path $fixture "fixture.bin"
    [IO.File]::WriteAllBytes($binaryPath, [byte[]]@(1, 2, 3))
    $binary = Get-XvramAcceptanceBinary -Path $binaryPath
    Assert-XvramAcceptanceBinaryUnchanged -Expected $binary
    [IO.File]::WriteAllBytes($binaryPath, [byte[]]@(1, 2, 4))
    Assert-Rejected { Assert-XvramAcceptanceBinaryUnchanged -Expected $binary }

    $output = Join-Path $fixture "output"
    New-Item -ItemType Directory -Path $output | Out-Null
    $manifest = Join-Path $output "acceptance-manifest.json"
    $report = Join-Path $output "report.json"
    [IO.File]::WriteAllText($manifest, '{"result":"completed"}')
    [IO.File]::WriteAllText($report, '{}')
    $resolvedManifest = Initialize-XvramAcceptanceManifest -OutputDirectory $output
    Assert-Test ($resolvedManifest -eq [IO.Path]::GetFullPath($manifest)) "manifest path changed"
    Assert-Test (-not (Test-Path -LiteralPath $manifest)) "stale completed manifest survived"
    Assert-Test (Test-Path -LiteralPath $report -PathType Leaf) "unrelated artifact was removed"
    Write-Host "Acceptance provenance fixture passed."
} finally {
    $resolvedFixture = [IO.Path]::GetFullPath($fixture)
    if (-not [string]::Equals([IO.Path]::GetDirectoryName($resolvedFixture),
            $temporaryRoot.TrimEnd([IO.Path]::DirectorySeparatorChar),
            [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($resolvedFixture) -notlike "xvram-provenance-*") {
        throw "Refusing to remove a fixture outside the exact temporary directory"
    }
    Remove-Item -LiteralPath $resolvedFixture -Recurse -Force
}
