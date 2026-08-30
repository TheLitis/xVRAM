[CmdletBinding()]
param(
    [Parameter()]
    [string]$BuildDirectory = "build/vs",

    [Parameter()]
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",

    [Parameter()]
    [ValidateRange(0, 64)]
    [int]$Device = 0,

    [Parameter()]
    [string]$OutputDirectory = "artifacts/phase3-rtx3070",

    [Parameter()]
    [ValidateRange(1, 86400)]
    [int]$TimeoutSeconds = 900,

    [Parameter()]
    [string]$ProbeExecutable,

    [Parameter()]
    [string]$GemmBenchExecutable,

    [Parameter()]
    [string]$ExpectedDeviceNamePattern = "RTX 3070"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$script:SchemaPath = Join-Path $script:RepositoryRoot "schemas/gemm-bench-report-v1.schema.json"

function Assert-Condition {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if (-not $Condition) {
        throw "Phase 3 acceptance failed: $Message"
    }
}

function Get-RequiredProperty {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,

        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "Phase 3 acceptance failed: $Context is missing '$Name'"
    }
    return $property.Value
}

function Resolve-XvramExecutable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter()]
        [string]$Override
    )

    if (-not [string]::IsNullOrWhiteSpace($Override)) {
        return (Resolve-Path -LiteralPath $Override).Path
    }

    $buildRoot = if ([System.IO.Path]::IsPathRooted($BuildDirectory)) {
        [System.IO.Path]::GetFullPath($BuildDirectory)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $script:RepositoryRoot $BuildDirectory))
    }
    $candidates = @(
        (Join-Path (Join-Path $buildRoot $Configuration) "$Name.exe"),
        (Join-Path $buildRoot "$Name.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "Unable to locate $Name.exe. Checked: $($candidates -join ', ')"
}

function Assert-JsonSchema {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportPath
    )

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -eq $python) {
        $python = Get-Command python -ErrorAction SilentlyContinue
    }
    if ($null -eq $python) {
        throw "Python is required for strict JSON Schema validation"
    }

    $validator = @'
import json
import pathlib
import sys
import jsonschema

schema_path = pathlib.Path(sys.argv[1])
report_path = pathlib.Path(sys.argv[2])
with schema_path.open("r", encoding="utf-8") as source:
    schema = json.load(source)
with report_path.open("r", encoding="utf-8") as source:
    report = json.load(source)
jsonschema.Draft202012Validator.check_schema(schema)
jsonschema.Draft202012Validator(schema).validate(report)
'@

    & $python.Source -c $validator $script:SchemaPath $ReportPath
    if ($LASTEXITCODE -ne 0) {
        throw "Strict schema validation failed for $ReportPath"
    }
}

function Get-ResidualGemmBenchProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ExecutablePath
    )

    $expected = [System.IO.Path]::GetFullPath($ExecutablePath)
    $matches = @()
    foreach ($process in @(Get-Process -Name "xvram-gemm-bench" -ErrorAction SilentlyContinue)) {
        try {
            if (-not [string]::IsNullOrWhiteSpace($process.Path) -and
                [string]::Equals(
                    [System.IO.Path]::GetFullPath($process.Path),
                    $expected,
                    [System.StringComparison]::OrdinalIgnoreCase)) {
                $matches += $process
            }
        } catch {
            # An inaccessible unrelated process cannot be identified as this executable.
        }
    }
    return $matches
}

function Assert-NoResidualWorker {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ExecutablePath
    )

    Start-Sleep -Milliseconds 250
    $residual = @(Get-ResidualGemmBenchProcess -ExecutablePath $ExecutablePath)
    if ($residual.Count -ne 0) {
        $processIds = @($residual | ForEach-Object { $_.Id })
        throw "Phase 3 acceptance failed: residual xvram-gemm-bench process IDs: " +
            ($processIds -join ", ")
    }
}

function Assert-BooleanLedger {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Ledger,

        [Parameter(Mandatory = $true)]
        [string[]]$Fields,

        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    foreach ($field in $Fields) {
        $value = Get-RequiredProperty -Object $Ledger -Name $field -Context $Context
        Assert-Condition ($value -eq $true) "$Context.$field must be true"
    }
}

function Assert-CommonReportSemantics {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Report,

        [Parameter(Mandatory = $true)]
        [string]$RawJson,

        [Parameter(Mandatory = $true)]
        [bool]$ExpectOversubscribed,

        [Parameter(Mandatory = $true)]
        [bool]$RequireReuse
    )

    Assert-Condition ($Report.schema_version -eq 1) "schema_version must be 1"
    Assert-Condition ($Report.report_type -eq "xvram.gemm_bench") `
        "report_type must be xvram.gemm_bench"
    Assert-Condition ($Report.outcome.status -eq "completed") "outcome must be completed"
    Assert-Condition ($Report.outcome.exit_code -eq 0) "outcome.exit_code must be 0"
    Assert-Condition (@($Report.diagnostics).Count -eq 0) "diagnostics must be empty"

    $proofFields = @(
        "tiled_execution_verified",
        "stable_virtual_addresses_verified",
        "set_access_after_map_verified",
        "event_boundaries_verified",
        "no_physical_aliases_verified",
        "dirty_writeback_verified",
        "cache_target_respected",
        "workspace_bounded",
        "all_workloads_match_reference",
        "raw_virtual_addresses_omitted"
    )
    Assert-BooleanLedger -Ledger $Report.proof -Fields $proofFields -Context "proof"
    Assert-Condition ($Report.proof.logical_data_exceeds_vram -eq $ExpectOversubscribed) `
        "logical_data_exceeds_vram does not match the requested case"

    if ($RequireReuse) {
        Assert-Condition ($Report.proof.cache_smaller_than_logical -eq $true) `
            "proof.cache_smaller_than_logical must be true"
        Assert-Condition ($Report.proof.handles_reused -eq $true) `
            "proof.handles_reused must be true"
        Assert-Condition ($Report.cache.handle_reuse_count -gt 0) `
            "handle reuse was not observed"
    }

    $cleanupFields = @(
        "complete",
        "operations_drained",
        "events_drained",
        "events_destroyed",
        "streams_destroyed",
        "cublas_handles_destroyed",
        "mappings_removed",
        "physical_handles_released",
        "workspace_released",
        "virtual_reservations_released",
        "pinned_staging_released",
        "host_backing_released",
        "context_released",
        "worker_terminated"
    )
    Assert-BooleanLedger -Ledger $Report.cleanup -Fields $cleanupFields -Context "cleanup"

    Assert-Condition ($Report.cache.unsafe_remap_count -eq 0) `
        "cache.unsafe_remap_count must be zero"
    Assert-Condition ($Report.cache.unsafe_transition_count -eq 0) `
        "cache.unsafe_transition_count must be zero"
    Assert-Condition ($Report.cache.mapping_count -eq $Report.cache.set_access_count) `
        "every mapping must have SetAccess"
    Assert-Condition ($Report.cache.mapping_count -eq $Report.cache.unmap_count) `
        "every completed mapping must be unmapped"
    Assert-Condition ($Report.cache.unmap_count -eq $Report.cache.event_boundary_count) `
        "every unmap must have a completed event boundary"
    Assert-Condition ($Report.cache.physical_handle_create_count -eq
        $Report.cache.physical_handle_release_count) "physical handle lifecycle mismatch"
    Assert-Condition ($Report.cache.resident_bytes_peak -le
        $Report.cache.target_bytes_maximum) "resident peak exceeded cache target"
    Assert-Condition ($Report.cache.workspace_bytes_peak -le
        $Report.configuration.effective_workspace_cap_bytes) `
        "workspace peak exceeded the configured cap"
    Assert-Condition ($Report.cache.pinned_staging_bytes -eq
        ($Report.configuration.effective_chunk_bytes * $Report.configuration.staging_slots)) `
        "pinned staging pool did not honor the configured slot bound"
    Assert-Condition ($Report.proof.maximum_working_set_bytes -le
        $Report.cache.target_bytes_minimum) "working set exceeded the minimum live target"

    Assert-Condition ($Report.numerics.validation_mode -ne "none") `
        "numerical validation was not performed"
    Assert-Condition ($Report.numerics.all_results_match -eq $true) `
        "numerical results did not match the reference"
    Assert-Condition ($Report.numerics.mismatch_count -eq 0) `
        "numerical mismatch_count must be zero"
    Assert-Condition ($Report.numerics.elements_verified -gt 0) `
        "no output elements were verified"
    [decimal]$expectedVerifiedElements = 0
    foreach ($workload in @($Report.workloads)) {
        $matchingPlans = @($Report.plans | Where-Object { $_.plan_id -eq $workload.plan_id })
        Assert-Condition ($matchingPlans.Count -eq 1) `
            "workload $($workload.name) does not reference exactly one plan"
        $matchingPlan = $matchingPlans[0]
        $expectedVerifiedElements += [decimal]$matchingPlan.m * [decimal]$matchingPlan.n * `
            [decimal]$workload.passes_completed
    }
    Assert-Condition ([decimal]$Report.numerics.elements_verified -eq $expectedVerifiedElements) `
        "full validation count does not cover every output element and completed pass"
    Assert-Condition ($Report.numerics.expected_digest128 -eq
        $Report.numerics.output_digest128) "numerical digest mismatch"

    Assert-Condition (@($Report.plans).Count -gt 0) "report contains no GEMM plan"
    foreach ($plan in @($Report.plans)) {
        Assert-Condition ($plan.status -eq "completed") "plan $($plan.plan_id) did not complete"
        Assert-Condition ($plan.tile_count -gt 0) "plan $($plan.plan_id) has no tiles"
        Assert-Condition ($plan.maximum_working_set_bytes -le
            $Report.cache.target_bytes_minimum) "plan working set exceeded live target"
        Assert-Condition ($plan.workspace_bytes -le
            $Report.configuration.effective_workspace_cap_bytes) `
            "plan workspace exceeded the configured cap"
    }
    foreach ($workload in @($Report.workloads)) {
        Assert-Condition ($workload.status -eq "completed") `
            "workload $($workload.name) did not complete"
        Assert-Condition ($workload.tiles_retired -eq $workload.tiles_total) `
            "workload $($workload.name) did not retire every tile"
        Assert-Condition ($workload.unsafe_remap_count -eq 0) `
            "workload $($workload.name) reported an unsafe remap"
        Assert-Condition ($workload.unsafe_transition_count -eq 0) `
            "workload $($workload.name) reported an unsafe transition"
        Assert-Condition ($workload.elapsed_ms -gt 0) `
            "workload $($workload.name) has no elapsed-time observation"
        Assert-Condition ($workload.achieved_tflops -ge 0) `
            "workload $($workload.name) has no throughput observation"
    }

    Assert-Condition ($Report.telemetry.cublas_version -gt 0) `
        "cuBLAS version observation is missing"
    Assert-Condition ($Report.telemetry.cublas_library_source -in
        @("system", "app_local", "explicit")) "cuBLAS library source is missing"
    Assert-Condition (($Report.telemetry.algorithms_selected +
        $Report.telemetry.algorithm_cache_hits) -gt 0) `
        "no cuBLAS algorithm execution was observed"
    Assert-Condition ($Report.telemetry.total_elapsed_ms -gt 0) `
        "total elapsed time is missing"
    Assert-Condition ($Report.telemetry.plan_timing.sample_count -gt 0 -and
        $Report.telemetry.plan_timing.total_ms -ge 0) "plan timing is missing"
    Assert-Condition ($Report.telemetry.gemm_timing.sample_count -gt 0 -and
        $Report.telemetry.gemm_timing.total_ms -gt 0) "GEMM timing is missing"

    if ($ExpectOversubscribed) {
        Assert-Condition ($Report.proof.logical_allocation_bytes -gt
            $Report.proof.total_vram_bytes) "logical allocation is not larger than VRAM"
        Assert-Condition (($Report.cache.clean_evictions + $Report.cache.dirty_evictions) -gt 0) `
            "oversubscribed case performed no eviction"
        Assert-Condition ($Report.cache.writebacks_completed -gt 0) `
            "oversubscribed case performed no write-back"
    }

    Assert-Condition ($null -eq $Report.device.uuid) "UUID must be redacted by default"
    Assert-Condition ($null -eq $Report.device.luid) "LUID must be redacted by default"
    Assert-Condition ($null -eq $Report.device.pci_bus_id) `
        "PCI bus ID must be redacted by default"
    Assert-Condition (-not ($RawJson -match
        '"(?:raw_)?(?:cuda_)?virtual_address"|"logical_(?:base|address)"|"device_pointer"')) `
        "raw CUDA virtual-address field was serialized"
}

function Invoke-GemmRun {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [bool]$ExpectOversubscribed,

        [Parameter(Mandatory = $true)]
        [bool]$RequireReuse
    )

    $reportPath = Join-Path $script:ResolvedOutputDirectory "$Name.json"
    Write-Host "Running $Name"
    & $script:GemmBenchPath @Arguments --compact-json --no-text --json $reportPath
    $exitCode = $LASTEXITCODE
    Assert-Condition ($exitCode -eq 0) "$Name exited with code $exitCode"
    Assert-NoResidualWorker -ExecutablePath $script:GemmBenchPath
    Assert-Condition (Test-Path -LiteralPath $reportPath -PathType Leaf) `
        "$Name did not create its report"
    Assert-JsonSchema -ReportPath $reportPath

    $rawJson = Get-Content -LiteralPath $reportPath -Raw
    $report = $rawJson | ConvertFrom-Json
    Assert-CommonReportSemantics -Report $report -RawJson $rawJson `
        -ExpectOversubscribed $ExpectOversubscribed -RequireReuse $RequireReuse
    return [pscustomobject]@{ Path = $reportPath; Report = $report }
}

function Get-OversubscribedSquareDimension {
    param(
        [Parameter(Mandatory = $true)]
        [uint64]$TotalVramBytes,

        [Parameter(Mandatory = $true)]
        [decimal]$Ratio
    )

    # Three square FP32 matrices consume 12*d*d bytes. Keep d odd to exercise tails.
    $targetBytes = [decimal]$TotalVramBytes * $Ratio
    $dimension = [uint64][Math]::Floor([Math]::Sqrt([double]($targetBytes / 12)))
    if (($dimension % 2) -eq 0) {
        $dimension--
    }
    Assert-Condition ($dimension -gt 1) "computed oversubscription dimension is invalid"
    $logicalBytes = [decimal]12 * [decimal]$dimension * [decimal]$dimension
    Assert-Condition ($logicalBytes -gt [decimal]$TotalVramBytes) `
        "computed $Ratio x case is not actually oversubscribed"
    return $dimension
}

$script:ProbePath = Resolve-XvramExecutable -Name "xvram-probe" -Override $ProbeExecutable
$script:GemmBenchPath = Resolve-XvramExecutable -Name "xvram-gemm-bench" `
    -Override $GemmBenchExecutable
Assert-Condition (@(Get-ResidualGemmBenchProcess -ExecutablePath $script:GemmBenchPath).Count -eq 0) `
    "a benchmark process is already running from $script:GemmBenchPath"
Assert-Condition (Test-Path -LiteralPath $script:SchemaPath -PathType Leaf) `
    "report schema is missing: $script:SchemaPath"

$script:ResolvedOutputDirectory = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $script:RepositoryRoot $OutputDirectory))
}
New-Item -ItemType Directory -Force -Path $script:ResolvedOutputDirectory | Out-Null

$probeReportPath = Join-Path $script:ResolvedOutputDirectory "probe.json"
Write-Host "Probing CUDA device $Device"
& $script:ProbePath --device $Device --skip-vmm-smoke --compact-json --no-text `
    --json $probeReportPath
$probeExitCode = $LASTEXITCODE
Assert-Condition ($probeExitCode -eq 0) "xvram-probe exited with code $probeExitCode"
$probe = Get-Content -LiteralPath $probeReportPath -Raw | ConvertFrom-Json
$selectedDevices = @($probe.cuda.devices | Where-Object { $_.ordinal -eq $Device })
Assert-Condition ($selectedDevices.Count -eq 1) "probe did not return CUDA device $Device"
$selectedDevice = $selectedDevices[0]
Assert-Condition ($selectedDevice.name -match $ExpectedDeviceNamePattern) `
    "device '$($selectedDevice.name)' does not match '$ExpectedDeviceNamePattern'"
$totalVramBytes = [uint64]$selectedDevice.total_memory_bytes
Assert-Condition ($totalVramBytes -gt 0) "probe reported zero total VRAM"

$baseArguments = @(
    "--device", "$Device",
    "--chunk-size", "64MiB",
    "--workspace", "4MiB",
    "--staging-slots", "4",
    "--device-headroom", "512MiB",
    "--passes", "1",
    "--timeout-seconds", "$TimeoutSeconds"
)
$manifestReports = @()

$cases = @(
    [pscustomobject]@{ Name = "suite"; Arguments = @(
        "--suite", "--m", "1025", "--n", "1023", "--k", "257") },
    [pscustomobject]@{ Name = "fp16-row-tail"; Arguments = @(
        "--m", "257", "--n", "193", "--k", "131", "--data-type", "fp16",
        "--compute-mode", "auto", "--layout-a", "row", "--layout-b", "row",
        "--layout-c", "row", "--op-a", "n", "--op-b", "n") },
    [pscustomobject]@{ Name = "bf16-column-tail"; Arguments = @(
        "--m", "259", "--n", "197", "--k", "137", "--data-type", "bf16",
        "--compute-mode", "auto", "--layout-a", "column", "--layout-b", "column",
        "--layout-c", "column", "--op-a", "t", "--op-b", "n") },
    [pscustomobject]@{ Name = "fp32-strict-tail"; Arguments = @(
        "--m", "263", "--n", "199", "--k", "139", "--data-type", "fp32",
        "--compute-mode", "strict", "--layout-a", "row", "--layout-b", "column",
        "--layout-c", "row", "--op-a", "n", "--op-b", "t") },
    [pscustomobject]@{ Name = "fp32-tf32-tail"; Arguments = @(
        "--m", "269", "--n", "211", "--k", "149", "--data-type", "fp32",
        "--compute-mode", "tf32", "--layout-a", "column", "--layout-b", "row",
        "--layout-c", "column", "--op-a", "t", "--op-b", "n") },
    [pscustomobject]@{ Name = "fp64-transpose-tail"; Arguments = @(
        "--m", "271", "--n", "223", "--k", "151", "--data-type", "fp64",
        "--compute-mode", "fp64", "--layout-a", "column", "--layout-b", "column",
        "--layout-c", "row", "--op-a", "t", "--op-b", "t") }
)

foreach ($case in $cases) {
    $run = Invoke-GemmRun -Name $case.Name -Arguments ($baseArguments + $case.Arguments) `
        -ExpectOversubscribed $false -RequireReuse $false
    $manifestReports += [ordered]@{ name = $case.Name; path = $run.Path }
}

$kSplitArguments = $baseArguments + @(
    "--m", "2049", "--n", "2049", "--k", "32769",
    "--data-type", "fp32", "--compute-mode", "strict",
    "--layout-a", "row", "--layout-b", "row", "--layout-c", "row",
    "--op-a", "n", "--op-b", "n", "--cache-target", "384MiB"
)
$kSplitRun = Invoke-GemmRun -Name "fp32-k-split" -Arguments $kSplitArguments `
    -ExpectOversubscribed $false -RequireReuse $true
$kSplitPlans = @($kSplitRun.Report.plans | Where-Object { $_.tile_k -lt $_.k })
Assert-Condition ($kSplitPlans.Count -gt 0) "K-split case did not split the K dimension"
$manifestReports += [ordered]@{ name = "fp32-k-split"; path = $kSplitRun.Path }

foreach ($ratio in @([decimal]"1.1", [decimal]"1.5")) {
    $dimension = Get-OversubscribedSquareDimension -TotalVramBytes $totalVramBytes -Ratio $ratio
    $tag = ([string]$ratio).Replace(".", "p")
    $arguments = $baseArguments + @(
        "--m", "$dimension", "--n", "$dimension", "--k", "$dimension",
        "--data-type", "fp32", "--compute-mode", "tf32",
        "--layout-a", "row", "--layout-b", "row", "--layout-c", "row",
        "--op-a", "n", "--op-b", "n", "--cache-target", "auto"
    )
    $run = Invoke-GemmRun -Name "oversubscribed-${tag}x" -Arguments $arguments `
        -ExpectOversubscribed $true -RequireReuse $true
    $manifestReports += [ordered]@{
        name = "oversubscribed-${tag}x"
        ratio = [string]$ratio
        dimension = $dimension
        path = $run.Path
    }
}

Assert-NoResidualWorker -ExecutablePath $script:GemmBenchPath
$manifest = [ordered]@{
    acceptance = "xvram.phase3.rtx3070"
    generated_at_utc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
    device_ordinal = $Device
    device_name = $selectedDevice.name
    total_vram_bytes = $totalVramBytes
    configuration = $Configuration
    reports = $manifestReports
    result = "completed"
}
$manifestPath = Join-Path $script:ResolvedOutputDirectory "acceptance-manifest.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host "Phase 3 RTX 3070 acceptance completed: $manifestPath"
