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
    [string]$OutputDirectory = "artifacts/phase2-rtx3070",

    [Parameter()]
    [ValidateRange(1, 86400)]
    [int]$TimeoutSeconds = 300,

    [Parameter()]
    [string]$ProbeExecutable,

    [Parameter()]
    [string]$CacheBenchExecutable,

    [Parameter()]
    [string]$ExpectedDeviceNamePattern = "RTX 3070"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$script:SchemaPath = Join-Path $script:RepositoryRoot "schemas/residency-cache-report-v1.schema.json"

function Assert-Condition {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if (-not $Condition) {
        throw "Phase 2 acceptance failed: $Message"
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
        throw "Phase 2 acceptance failed: $Context is missing '$Name'"
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

function Get-ResidualCacheBenchProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ExecutablePath
    )

    $expected = [System.IO.Path]::GetFullPath($ExecutablePath)
    $matches = @()
    foreach ($process in @(Get-Process -Name "xvram-cache-bench" -ErrorAction SilentlyContinue)) {
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
    $residual = @(Get-ResidualCacheBenchProcess -ExecutablePath $ExecutablePath)
    Assert-Condition ($residual.Count -eq 0) `
        "residual xvram-cache-bench process IDs: $($residual.Id -join ', ')"
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
        [string]$ExpectedPolicy,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedScenario,

        [Parameter(Mandatory = $true)]
        [bool]$ExpectOversubscribed
    )

    Assert-Condition ($Report.schema_version -eq 1) "schema_version must be 1"
    Assert-Condition ($Report.report_type -eq "xvram.residency_cache") `
        "report_type must be xvram.residency_cache"
    Assert-Condition ($Report.outcome.status -eq "completed") "outcome must be completed"
    Assert-Condition ($Report.outcome.exit_code -eq 0) "outcome.exit_code must be 0"
    Assert-Condition (@($Report.diagnostics).Count -eq 0) "diagnostics must be empty"
    Assert-Condition ($Report.configuration.policy -eq $ExpectedPolicy) `
        "reported policy must be $ExpectedPolicy"
    Assert-Condition ($Report.configuration.scenario -eq $ExpectedScenario) `
        "reported scenario must be $ExpectedScenario"

    $proofFields = @(
        "cache_smaller_than_logical",
        "handles_reused",
        "stable_virtual_addresses_verified",
        "set_access_after_map_verified",
        "event_boundaries_verified",
        "no_physical_aliases_verified",
        "dirty_writeback_verified",
        "staging_pool_bounded",
        "cache_target_respected",
        "all_workloads_match_cpu",
        "policies_match",
        "raw_virtual_addresses_omitted"
    )
    Assert-BooleanLedger -Ledger $Report.proof -Fields $proofFields -Context "proof"
    Assert-Condition ($Report.proof.logical_data_exceeds_vram -eq $ExpectOversubscribed) `
        "logical_data_exceeds_vram does not match the requested ratio"

    $cleanupFields = @(
        "complete",
        "transactions_drained",
        "prefetch_drained",
        "writebacks_completed",
        "events_drained",
        "events_destroyed",
        "streams_destroyed",
        "module_unloaded",
        "mappings_removed",
        "physical_handles_released",
        "device_allocations_released",
        "virtual_reservations_released",
        "pinned_staging_released",
        "host_backing_released",
        "context_destroyed",
        "trace_closed",
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
        "every unmap must have an event boundary"
    Assert-Condition ($Report.cache.physical_handle_create_count -eq
        $Report.cache.physical_handle_release_count) "physical handle lifecycle mismatch"
    Assert-Condition ($Report.cache.handle_reuse_count -gt 0) "handle reuse was not observed"
    Assert-Condition ($Report.cache.dirty_evictions -eq $Report.cache.writebacks_completed) `
        "dirty eviction/write-back accounting mismatch"
    Assert-Condition ($Report.cache.prefetch_issued -eq
        ($Report.cache.prefetch_useful + $Report.cache.prefetch_wasted +
         $Report.cache.prefetch_cancelled)) "prefetch terminal accounting mismatch"
    Assert-Condition ($Report.cache.resident_bytes_peak -le $Report.cache.target_bytes_maximum) `
        "resident peak exceeded cache target"
    Assert-Condition ($Report.cache.maximum_working_set_bytes -le
        $Report.cache.target_bytes_minimum) "working set exceeded the minimum live target"
    Assert-Condition ($Report.cache.pinned_staging_bytes -le
        ($Report.proof.chunk_bytes * $Report.configuration.staging_slots)) `
        "pinned staging exceeded the configured pool"

    $workloads = @($Report.workloads)
    $operationCount = [uint64]0
    foreach ($workload in $workloads) {
        Assert-Condition ($workload.status -eq "completed") `
            "workload $($workload.scenario)/$($workload.policy) did not complete"
        Assert-Condition ($workload.policy -eq $ExpectedPolicy) `
            "workload policy differs from requested policy"
        Assert-Condition ($workload.matches_cpu -eq $true) `
            "workload $($workload.scenario) does not match CPU"
        Assert-Condition ($workload.full_verification_completed -eq $true) `
            "workload $($workload.scenario) did not complete full verification"
        Assert-Condition ($workload.stable_addresses_verified -eq $true) `
            "workload $($workload.scenario) did not verify stable addresses"
        Assert-Condition ($workload.mismatch_count -eq 0) `
            "workload $($workload.scenario) reported a mismatch"
        Assert-Condition ($workload.unsafe_remap_count -eq 0) `
            "workload $($workload.scenario) reported an unsafe remap"
        Assert-Condition ($workload.unsafe_transition_count -eq 0) `
            "workload $($workload.scenario) reported an unsafe transition"
        Assert-Condition ($workload.expected_digest128 -eq $workload.output_digest128) `
            "workload $($workload.scenario) digest mismatch"
        Assert-Condition (($workload.cache_hits + $workload.cache_misses) -eq
            $workload.operations_retired) `
            "workload $($workload.scenario) hit/miss accounting mismatch"
        Assert-Condition ($workload.prefetch_issued -eq
            ($workload.prefetch_useful + $workload.prefetch_wasted +
             $workload.prefetch_cancelled)) `
            "workload $($workload.scenario) prefetch accounting mismatch"
        $operationCount += [uint64]$workload.operations_retired
    }
    Assert-Condition (($Report.cache.cache_hits + $Report.cache.cache_misses) -eq
        $operationCount) "global hit/miss accounting mismatch"

    Assert-Condition ($null -eq $Report.device.uuid) "UUID must be redacted by default"
    Assert-Condition ($null -eq $Report.device.luid) "LUID must be redacted by default"
    Assert-Condition ($null -eq $Report.device.pci_bus_id) `
        "PCI bus ID must be redacted by default"
    Assert-Condition (-not ($RawJson -match
        '"(?:raw_)?(?:cuda_)?virtual_address"|"logical_(?:base|address)"|"device_pointer"')) `
        "raw CUDA virtual-address field was serialized"
}

function Assert-SuiteSemantics {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Report,

        [Parameter(Mandatory = $true)]
        [bool]$ExpectOversubscribed
    )

    $expectedScenarios = @("random", "read-only", "reuse", "sequential", "write-heavy")
    $actualScenarios = @($Report.workloads | ForEach-Object { $_.scenario } | Sort-Object)
    Assert-Condition (($actualScenarios -join ",") -eq ($expectedScenarios -join ",")) `
        "suite must contain each of the five scenarios exactly once"

    $readOnly = @($Report.workloads | Where-Object { $_.scenario -eq "read-only" })[0]
    Assert-Condition ($readOnly.bytes_d2h -eq 0) "read-only workload performed D2H"

    $writeHeavy = @($Report.workloads | Where-Object { $_.scenario -eq "write-heavy" })[0]
    Assert-Condition ($writeHeavy.dirty_evictions -gt 0) `
        "write-heavy workload did not observe dirty eviction"
    Assert-Condition ($writeHeavy.writebacks_completed -gt 0) `
        "write-heavy workload did not complete dirty write-back"

    $reuse = @($Report.workloads | Where-Object { $_.scenario -eq "reuse" })[0]
    Assert-Condition ($reuse.cache_hit_rate -ge 0.85) `
        "reuse hit rate is below 85%: $($reuse.cache_hit_rate)"

    if ($ExpectOversubscribed) {
        Assert-Condition (($Report.cache.clean_evictions + $Report.cache.dirty_evictions) -gt 0) `
            "oversubscribed suite did not perform a real eviction"
        Assert-Condition ($Report.cache.handle_reuse_count -gt 0) `
            "oversubscribed suite did not reuse physical handles"
    }
}

function Invoke-CacheRun {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedPolicy,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedScenario,

        [Parameter(Mandatory = $true)]
        [bool]$ExpectOversubscribed
    )

    $reportPath = Join-Path $script:ResolvedOutputDirectory "$Name.json"
    Write-Host "Running $Name"
    & $script:CacheBenchPath @Arguments --compact-json --no-text --json $reportPath
    $exitCode = $LASTEXITCODE
    Assert-Condition ($exitCode -eq 0) "$Name exited with code $exitCode"
    Assert-NoResidualWorker -ExecutablePath $script:CacheBenchPath
    Assert-Condition (Test-Path -LiteralPath $reportPath -PathType Leaf) `
        "$Name did not create its report"
    Assert-JsonSchema -ReportPath $reportPath

    $rawJson = Get-Content -LiteralPath $reportPath -Raw
    $report = $rawJson | ConvertFrom-Json
    Assert-CommonReportSemantics -Report $report -RawJson $rawJson `
        -ExpectedPolicy $ExpectedPolicy -ExpectedScenario $ExpectedScenario `
        -ExpectOversubscribed $ExpectOversubscribed
    if ($ExpectedScenario -eq "suite") {
        Assert-SuiteSemantics -Report $report -ExpectOversubscribed $ExpectOversubscribed
    }
    return [pscustomobject]@{ Path = $reportPath; Report = $report }
}

function Get-AlignedLogicalBytes {
    param(
        [Parameter(Mandatory = $true)]
        [uint64]$TotalBytes,

        [Parameter(Mandatory = $true)]
        [decimal]$Ratio
    )

    $scaled = [decimal]$TotalBytes * $Ratio
    return [uint64]([decimal]::Floor($scaled / 4) * 4)
}

$script:ProbePath = Resolve-XvramExecutable -Name "xvram-probe" -Override $ProbeExecutable
$script:CacheBenchPath = Resolve-XvramExecutable -Name "xvram-cache-bench" `
    -Override $CacheBenchExecutable
Assert-Condition (@(Get-ResidualCacheBenchProcess -ExecutablePath $script:CacheBenchPath).Count -eq 0) `
    "a benchmark process is already running from $script:CacheBenchPath"
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

$ratios = @(
    [pscustomobject]@{ Name = "0.8"; Value = [decimal]"0.8"; Oversubscribed = $false },
    [pscustomobject]@{ Name = "1.1"; Value = [decimal]"1.1"; Oversubscribed = $true },
    [pscustomobject]@{ Name = "1.5"; Value = [decimal]"1.5"; Oversubscribed = $true },
    [pscustomobject]@{ Name = "2.0"; Value = [decimal]"2.0"; Oversubscribed = $true }
)
$reports = @{}
$manifestReports = @()

foreach ($ratio in $ratios) {
    $logicalBytes = Get-AlignedLogicalBytes -TotalBytes $totalVramBytes -Ratio $ratio.Value
    $tag = $ratio.Name.Replace(".", "p")
    foreach ($policy in @("clock", "lru")) {
        $name = "suite-${tag}x-$policy"
        $arguments = @(
            "--device", "$Device",
            "--logical-size", "$logicalBytes",
            "--chunk-size", "64MiB",
            "--cache-target", "auto",
            "--staging-slots", "4",
            "--policy", $policy,
            "--prefetch-distance", "2",
            "--scenario", "suite",
            "--passes", "2",
            "--device-headroom", "512MiB",
            "--budget-poll-ms", "100",
            "--stall-timeout-ms", "5000",
            "--timeout-seconds", "$TimeoutSeconds",
            "--seed", "0x585652414D503032"
        )
        $run = Invoke-CacheRun -Name $name -Arguments $arguments -ExpectedPolicy $policy `
            -ExpectedScenario "suite" -ExpectOversubscribed $ratio.Oversubscribed
        $reports["$tag/$policy"] = $run.Report
        $manifestReports += [ordered]@{
            name = $name
            ratio = $ratio.Name
            policy = $policy
            scenario = "suite"
            logical_bytes = $logicalBytes
            path = $run.Path
        }
    }

    $clockReport = $reports["$tag/clock"]
    $lruReport = $reports["$tag/lru"]
    foreach ($scenario in @("sequential", "reuse", "random", "read-only", "write-heavy")) {
        $clockWorkload = @($clockReport.workloads | Where-Object { $_.scenario -eq $scenario })[0]
        $lruWorkload = @($lruReport.workloads | Where-Object { $_.scenario -eq $scenario })[0]
        Assert-Condition ($clockWorkload.output_digest128 -eq $lruWorkload.output_digest128) `
            "CLOCK/LRU digest mismatch at $($ratio.Name)x for $scenario"
    }
}

$pressureLogicalBytes = Get-AlignedLogicalBytes -TotalBytes $totalVramBytes `
    -Ratio ([decimal]"1.5")
$pressureArguments = @(
    "--device", "$Device",
    "--logical-size", "$pressureLogicalBytes",
    "--chunk-size", "64MiB",
    "--cache-target", "auto",
    "--staging-slots", "4",
    "--policy", "clock",
    "--prefetch-distance", "2",
    "--scenario", "budget-pressure",
    "--passes", "2",
    "--pressure-size", "auto",
    "--device-headroom", "512MiB",
    "--budget-poll-ms", "100",
    "--stall-timeout-ms", "5000",
    "--timeout-seconds", "$TimeoutSeconds",
    "--seed", "0x585652414D503032"
)
$pressureRun = Invoke-CacheRun -Name "budget-pressure-1p5x-clock" `
    -Arguments $pressureArguments -ExpectedPolicy "clock" `
    -ExpectedScenario "budget-pressure" -ExpectOversubscribed $true
Assert-Condition (@($pressureRun.Report.workloads).Count -eq 1) `
    "budget-pressure report must contain exactly one workload"
Assert-Condition ($pressureRun.Report.cache.target_shrink_count -gt 0) `
    "budget-pressure run did not observe target shrink"
Assert-Condition ($pressureRun.Report.cache.target_grow_count -gt 0) `
    "budget-pressure run did not observe target grow"
$manifestReports += [ordered]@{
    name = "budget-pressure-1p5x-clock"
    ratio = "1.5"
    policy = "clock"
    scenario = "budget-pressure"
    logical_bytes = $pressureLogicalBytes
    path = $pressureRun.Path
}

Assert-NoResidualWorker -ExecutablePath $script:CacheBenchPath
$manifest = [ordered]@{
    acceptance = "xvram.phase2.rtx3070"
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

Write-Host "Phase 2 RTX 3070 acceptance completed: $manifestPath"
