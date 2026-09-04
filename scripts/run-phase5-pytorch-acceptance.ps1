[CmdletBinding()]
param(
    [Parameter()]
    [string]$BuildDirectory = "build/phase5-pytorch",

    [Parameter()]
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",

    [Parameter()]
    [ValidateRange(0, 64)]
    [int]$Device = 0,

    [Parameter()]
    [string]$OutputDirectory = "artifacts/phase5-pytorch-rtx3070",

    [Parameter()]
    [ValidateRange(1, 86400)]
    [int]$TimeoutSeconds = 900,

    [Parameter()]
    [string]$Python = "python",

    [Parameter()]
    [string]$CMake = "cmake",

    [Parameter()]
    [string]$Ninja = "ninja",

    [Parameter()]
    [string]$RuntimeLibrary,

    [Parameter()]
    [string]$ExpectedDeviceNamePattern = "^NVIDIA GeForce RTX 3070$",

    [Parameter()]
    [uint64]$ExpectedTotalVramBytes = 8589410304,

    [Parameter()]
    [switch]$IncludeIdentifiers,

    [Parameter()]
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
. (Join-Path $PSScriptRoot "acceptance-provenance.ps1")
$script:Phase4bRunner = Join-Path $PSScriptRoot "run-phase4b-rtx3070-acceptance.ps1"
$script:Validator = Join-Path $PSScriptRoot "validate-phase5-pytorch-acceptance.py"
$script:ReportSchema = Join-Path $script:RepositoryRoot `
    "schemas/pytorch-inference-report-v2.schema.json"
$script:TraceSchema = Join-Path $script:RepositoryRoot `
    "schemas/pytorch-inference-trace-record-v2.schema.json"

function Assert-Condition {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if (-not $Condition) {
        throw "Phase 5 PyTorch acceptance failed: $Message"
    }
}

function Resolve-Executable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Requested,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (Test-Path -LiteralPath $Requested -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Requested).Path
    }
    $commands = @(Get-Command $Requested -CommandType Application `
        -ErrorAction SilentlyContinue)
    Assert-Condition ($commands.Count -gt 0) "unable to locate $Description"
    return $commands[0].Source
}

function Get-XvramTorchWorkers {
    $workers = @()
    try {
        $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    } catch {
        throw "Phase 5 PyTorch acceptance failed: unable to inspect workers: " +
            $_.Exception.Message
    }
    foreach ($process in $processes) {
        $commandLine = [string]$process.CommandLine
        if (-not [string]::IsNullOrWhiteSpace($commandLine) -and
            $commandLine -match "(?i)xvram[.]torch_bench" -and
            $commandLine -match "(?i)(?:^|\s)--worker(?:$|\s)") {
            $workers += $process
        }
    }
    return $workers
}

function Assert-NoResidualWorker {
    Start-Sleep -Milliseconds 500
    $workers = @(Get-XvramTorchWorkers)
    if ($workers.Count -ne 0) {
        $processIds = @($workers | ForEach-Object { $_.ProcessId })
        throw "Phase 5 PyTorch acceptance failed: residual worker process IDs: " +
            ($processIds -join ", ")
    }
}

function ConvertTo-Base64Json {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Value
    )

    $json = $Value | ConvertTo-Json -Depth 12 -Compress
    return [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($json))
}

$runs = @(
    [pscustomobject]@{
        Name = "adaptive-incompressible-clock-42-p2"; Layers = 42; Ratio = 2.040
        Policy = "clock"; Prefetch = 2; Sequence = 32; Model = "llama2-like"
        Compression = "adaptive"; StatePattern = "incompressible"
        ParameterBytes = [uint64]17524498432; ActivationBytes = [uint64]4948224
        LogicalBytes = [uint64]17529454848; Oversubscribed = $true
    },
    [pscustomobject]@{
        Name = "capacity-structured-clock-47-p2"; Layers = 47; Ratio = 2.276
        Policy = "clock"; Prefetch = 2; Sequence = 32; Model = "llama2-like"
        Compression = "capacity"; StatePattern = "structured"
        ParameterBytes = [uint64]19548332032; ActivationBytes = [uint64]4948224
        LogicalBytes = [uint64]19553288448; Oversubscribed = $true
    },
    [pscustomobject]@{
        Name = "capacity-structured-lru-47-p2"; Layers = 47; Ratio = 2.276
        Policy = "lru"; Prefetch = 2; Sequence = 32; Model = "llama2-like"
        Compression = "capacity"; StatePattern = "structured"
        ParameterBytes = [uint64]19548332032; ActivationBytes = [uint64]4948224
        LogicalBytes = [uint64]19553288448; Oversubscribed = $true
    },
    [pscustomobject]@{
        Name = "capacity-structured-clock-47-p0"; Layers = 47; Ratio = 2.276
        Policy = "clock"; Prefetch = 0; Sequence = 32; Model = "llama2-like"
        Compression = "capacity"; StatePattern = "structured"
        ParameterBytes = [uint64]19548332032; ActivationBytes = [uint64]4948224
        LogicalBytes = [uint64]19553288448; Oversubscribed = $true
    },
    [pscustomobject]@{
        Name = "capacity-structured-attention-smoke-seq128"; Layers = 2; Ratio = 0.100
        Policy = "clock"; Prefetch = 2; Sequence = 128; Model = "operator-smoke"
        Compression = "capacity"; StatePattern = "structured"
        ParameterBytes = [uint64]1333829632; ActivationBytes = [uint64]19792896
        LogicalBytes = [uint64]1353655296; Oversubscribed = $false
    }
)

foreach ($path in @($script:Phase4bRunner, $script:Validator, $script:ReportSchema,
        $script:TraceSchema)) {
    Assert-Condition (Test-Path -LiteralPath $path -PathType Leaf) `
        "required gate input is missing: $path"
}

if ($DryRun) {
    [ordered]@{
        acceptance = "xvram.phase5.pytorch.rtx3070"
        phase4b_regression = $true
        report_schema = $script:ReportSchema
        trace_schema = $script:TraceSchema
        runs = $runs
    } | ConvertTo-Json -Depth 10
    return
}

$resolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $script:RepositoryRoot $OutputDirectory))
}
New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
$manifestPath = Initialize-XvramAcceptanceManifest -OutputDirectory $resolvedOutput
$sourceProvenance = Get-XvramAcceptanceSource -RepositoryRoot $script:RepositoryRoot
$phase4bOutput = Join-Path $resolvedOutput "phase4b-regression"

Assert-Condition ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) `
    "the mandatory hardware gate must run on Windows"
Assert-Condition ([Environment]::Is64BitProcess) "the gate requires a 64-bit process"
$script:PythonPath = Resolve-Executable -Requested $Python -Description "Python"

$phase4bArguments = @{
    BuildDirectory = $BuildDirectory
    Configuration = $Configuration
    Device = $Device
    OutputDirectory = $phase4bOutput
    TimeoutSeconds = $TimeoutSeconds
    Python = $Python
    CMake = $CMake
    Ninja = $Ninja
    ExpectedDeviceNamePattern = $ExpectedDeviceNamePattern
    ExpectedTotalVramBytes = $ExpectedTotalVramBytes
}
if (-not [string]::IsNullOrWhiteSpace($RuntimeLibrary)) {
    $phase4bArguments.RuntimeLibrary = $RuntimeLibrary
}
if ($IncludeIdentifiers) {
    $phase4bArguments.IncludeIdentifiers = $true
}

Write-Host "Running the complete Phase 4b regression gate..."
& $script:Phase4bRunner @phase4bArguments
Assert-Condition ($LASTEXITCODE -eq 0) "Phase 4b regression gate failed"
Assert-NoResidualWorker

$phase4bManifestPath = Join-Path $phase4bOutput "acceptance-manifest.json"
Assert-Condition (Test-Path -LiteralPath $phase4bManifestPath -PathType Leaf) `
    "Phase 4b manifest is missing"
$phase4bManifest = Get-Content -LiteralPath $phase4bManifestPath -Raw |
    ConvertFrom-Json
Assert-Condition ($phase4bManifest.result -eq "completed") `
    "Phase 4b manifest is not complete"
Assert-Condition ($phase4bManifest.validation.status -eq "passed" -and `
    $phase4bManifest.source.git_commit -eq $sourceProvenance.git_commit -and `
    $phase4bManifest.source.source_sha256 -eq $sourceProvenance.source_sha256) `
    "Phase 4b regression source provenance differs from the Phase 5 gate"
$hardware = $phase4bManifest.device
$runtimePath = [System.IO.Path]::GetFullPath([string]$phase4bManifest.runtime_library)
Assert-Condition (Test-Path -LiteralPath $runtimePath -PathType Leaf) `
    "Phase 4b runtime library no longer exists"
$binaryProvenance = [ordered]@{
    runtime = Get-XvramAcceptanceBinary -Path $runtimePath
    nvcomp_gpu = Get-XvramAcceptanceBinary -Path (Join-Path `
        (Split-Path $runtimePath -Parent) "nvcomp64_5.dll")
    nvcomp_cpu = Get-XvramAcceptanceBinary -Path (Join-Path `
        (Split-Path $runtimePath -Parent) "nvcomp_cpu64_5.dll")
}
Assert-Condition ($binaryProvenance.runtime.sha256 -eq $phase4bManifest.runtime_binary.sha256) `
    "runtime changed after the Phase 4b regression gate"

$previousPythonPath = $env:PYTHONPATH
$previousRuntimeLibrary = $env:XVRAM_TORCH_RUNTIME_LIBRARY
$previousGitCommit = $env:XVRAM_GIT_COMMIT
$previousPath = $env:PATH
try {
    $env:PYTHONPATH = Join-Path $script:RepositoryRoot "python"
    $env:XVRAM_TORCH_RUNTIME_LIBRARY = $runtimePath
    $env:XVRAM_GIT_COMMIT = $sourceProvenance.git_commit
    $torchLibraryProbe = 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")'
    $torchLibraryOutput = @(& $script:PythonPath -c $torchLibraryProbe)
    Assert-Condition ($LASTEXITCODE -eq 0 -and $torchLibraryOutput.Count -ge 1) `
        "PyTorch library-directory discovery failed"
    $torchLibraryDirectory = $torchLibraryOutput[-1].Trim()
    Assert-Condition (Test-Path -LiteralPath $torchLibraryDirectory -PathType Container) `
        "PyTorch library directory is missing"
    $env:PATH = (Split-Path $runtimePath -Parent) + ";" + $torchLibraryDirectory +
        ";" + $env:PATH

    $identifierArguments = @()
    if ($IncludeIdentifiers) {
        $identifierArguments = @("--include-identifiers")
    }
    $hardwareArgument = ConvertTo-Base64Json -Value $hardware
    $validated = @{}
    $manifestReports = @()

    foreach ($run in $runs) {
        $reportPath = Join-Path $resolvedOutput ($run.Name + ".json")
        $tracePath = Join-Path $resolvedOutput ($run.Name + ".jsonl")
        foreach ($artifact in @($reportPath, $tracePath)) {
            if (Test-Path -LiteralPath $artifact -PathType Leaf) {
                Remove-Item -LiteralPath $artifact -Force
            }
        }
        Write-Host "Running $($run.Name)..."
        $ratioArgument = ([double]$run.Ratio).ToString(
            "0.000", [Globalization.CultureInfo]::InvariantCulture)
        $arguments = @(
            "-m", "xvram.torch_bench",
            "--device", "$Device",
            "--model", $run.Model,
            "--model-ratio", $ratioArgument,
            "--layers", "$($run.Layers)",
            "--batch", "1",
            "--sequence", "$($run.Sequence)",
            "--hidden", "4096",
            "--intermediate", "11008",
            "--heads", "32",
            "--dtype", "float16",
            "--policy", $run.Policy,
            "--cache-target", "auto",
            "--chunk-size", "64MiB",
            "--device-headroom", "512MiB",
            "--scratch-cap", "512MiB",
            "--compression", $run.Compression,
            "--compression-codec", "lz4",
            "--state-pattern", $run.StatePattern,
            "--host-store-cap", "auto",
            "--host-headroom", "auto",
            "--compression-scratch-cap", "256MiB",
            "--codec-slots", "2",
            "--codec-workers", "2",
            "--prefetch-distance", "$($run.Prefetch)",
            "--sdpa-backend", "math",
            "--seed", "0x585652414D503035",
            "--timeout-seconds", "$TimeoutSeconds",
            "--trace", $tracePath,
            "--json", $reportPath,
            "--compact-json",
            "--no-text"
        ) + $identifierArguments
        & $script:PythonPath @arguments
        $runExitCode = $LASTEXITCODE
        Assert-NoResidualWorker
        Assert-Condition ($runExitCode -eq 0) `
            "$($run.Name) exited with code $runExitCode"

        $expected = [ordered]@{
            name = $run.Name
            device = $Device
            model = $run.Model
            model_ratio = [double]::Parse(
                $ratioArgument, [Globalization.CultureInfo]::InvariantCulture)
            layers = [int]$run.Layers
            sequence = [int]$run.Sequence
            policy = $run.Policy
            prefetch_distance = [int]$run.Prefetch
            compression = $run.Compression
            state_pattern = $run.StatePattern
            parameter_bytes = $run.ParameterBytes
            activation_bytes = $run.ActivationBytes
            logical_bytes = $run.LogicalBytes
            oversubscribed = [bool]$run.Oversubscribed
            include_identifiers = [bool]$IncludeIdentifiers
        }
        $expectedArgument = ConvertTo-Base64Json -Value $expected
        & $script:PythonPath $script:Validator `
            --report $reportPath `
            --trace $tracePath `
            --report-schema $script:ReportSchema `
            --trace-schema $script:TraceSchema `
            --expected-base64 $expectedArgument `
            --hardware-base64 $hardwareArgument
        Assert-Condition ($LASTEXITCODE -eq 0) `
            "strict report/trace validation failed for $($run.Name)"

        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        Assert-XvramAcceptanceReportRevision -Observed $report.build.git_commit `
            -Source $sourceProvenance
        $validated[$run.Name] = $report
        $manifestReports += [ordered]@{
            name = $run.Name
            report = $reportPath
            trace = $tracePath
            parameter_bytes = $report.model.parameter_bytes
            logical_bytes = $report.model.logical_bytes
            output_digest = $report.verification.output_digest
            graph_hash = $report.graph.hash
            pcie_h2d_bytes = $report.compression.pcie_h2d_bytes
            logical_h2d_bytes = $report.compression.logical_h2d_bytes
            prefetch_submitted = $report.cache.prefetch_submitted
        }
    }

    $comparisonNames = @(
        "capacity-structured-clock-47-p2",
        "capacity-structured-lru-47-p2",
        "capacity-structured-clock-47-p0"
    )
    $digests = @($comparisonNames | ForEach-Object {
        $validated[$_].verification.output_digest
    } | Select-Object -Unique)
    Assert-Condition ($digests.Count -eq 1) `
        "structured CLOCK/LRU/prefetch output digests differ"
    $graphHashes = @($comparisonNames | ForEach-Object {
        $validated[$_].graph.hash
    } | Select-Object -Unique)
    Assert-Condition ($graphHashes.Count -eq 1) `
        "structured CLOCK/LRU/prefetch graph hashes differ"
    Assert-Condition `
        ($validated["capacity-structured-clock-47-p0"].cache.prefetch_submitted -eq 0) `
        "distance-zero comparison submitted prefetch"
    Assert-Condition `
        ($validated["capacity-structured-clock-47-p2"].cache.prefetch_submitted -gt 0) `
        "distance-two comparison observed no prefetch"
    Assert-NoResidualWorker
    $sourceAtCompletion = Assert-XvramAcceptanceSourceUnchanged `
        -RepositoryRoot $script:RepositoryRoot -Expected $sourceProvenance
    foreach ($binary in $binaryProvenance.Values) {
        Assert-XvramAcceptanceBinaryUnchanged -Expected $binary
    }

    $manifest = [ordered]@{
        acceptance = "xvram.phase5.pytorch.rtx3070"
        generated_at_utc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
        build_configuration = $Configuration
        runtime_library = $runtimePath
        source = $sourceProvenance
        source_at_completion = $sourceAtCompletion
        binaries = $binaryProvenance
        validation = [ordered]@{
            status = "passed"
            report_revision_matched = $true
            phase4b_source_and_runtime_matched = $true
            source_snapshot_unchanged = $true
            binary_hashes_unchanged = $true
            dirty_working_tree_allowed_and_recorded = $true
        }
        device = $hardware
        driver_model = "WDDM"
        phase4b_regression_manifest = $phase4bManifestPath
        reports = $manifestReports
        result = "completed"
    }
    $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath `
        -Encoding utf8
    Write-Host "Phase 5 PyTorch RTX 3070 acceptance completed: $manifestPath"
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:XVRAM_TORCH_RUNTIME_LIBRARY = $previousRuntimeLibrary
    $env:XVRAM_GIT_COMMIT = $previousGitCommit
    $env:PATH = $previousPath
}
