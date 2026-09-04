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
    [string]$OutputDirectory = "artifacts/phase5-compression-rtx3070",

    [Parameter()]
    [ValidateRange(1, 86400)]
    [int]$TimeoutSeconds = 900,

    [Parameter()]
    [string]$ProbeExecutable,

    [Parameter()]
    [string]$CompressionBenchExecutable,

    [Parameter()]
    [string]$Python = "python",

    [Parameter()]
    [string]$ExpectedDeviceNamePattern = "^NVIDIA GeForce RTX 3070$",

    [Parameter()]
    [uint64]$ExpectedTotalVramBytes = 8589410304,

    [Parameter()]
    [ValidatePattern("^[0-9a-fA-F]{64}$")]
    [string]$ExpectedNvcompSha256 =
        "86af6147413d09233c7ac78c968163a1b529c4e22f20c9fd8b8bab7894a29c35",

    [Parameter()]
    [switch]$IncludeIdentifiers
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
. (Join-Path $PSScriptRoot "acceptance-provenance.ps1")
$script:ReportSchemaPath = Join-Path $script:RepositoryRoot `
    "schemas/adaptive-compression-report-v1.schema.json"
$script:TraceSchemaPath = Join-Path $script:RepositoryRoot `
    "schemas/compression-trace-record-v1.schema.json"

function Assert-Condition {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if (-not $Condition) {
        throw "Phase 5 compression acceptance failed: $Message"
    }
}

function Resolve-XvramExecutable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter()]
        [string]$Override
    )

    if (-not [string]::IsNullOrWhiteSpace($Override)) {
        Assert-Condition (Test-Path -LiteralPath $Override -PathType Leaf) `
            "executable override does not exist: $Override"
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
    throw "Phase 5 compression acceptance failed: unable to locate $Name.exe below $buildRoot"
}

function Resolve-Python {
    $requested = @(Get-Command $Python -CommandType Application `
        -ErrorAction SilentlyContinue)
    Assert-Condition ($requested.Count -gt 0) `
        "Python is required for JSON Schema validation"
    return $requested[0].Source
}

function Get-ResidualCompressionBenchProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ExecutablePath
    )

    $expected = [System.IO.Path]::GetFullPath($ExecutablePath)
    $matches = @()
    foreach ($process in @(Get-Process -Name "xvram-compression-bench" `
            -ErrorAction SilentlyContinue)) {
        try {
            if (-not [string]::IsNullOrWhiteSpace($process.Path) -and
                [string]::Equals(
                    [System.IO.Path]::GetFullPath($process.Path),
                    $expected,
                    [System.StringComparison]::OrdinalIgnoreCase)) {
                $matches += $process
            }
        } catch {
            # An inaccessible unrelated process is not identified as this benchmark.
        }
    }
    return $matches
}

function Assert-NoResidualWorker {
    Start-Sleep -Milliseconds 500
    $residual = @(Get-ResidualCompressionBenchProcess `
        -ExecutablePath $script:CompressionBenchPath)
    if ($residual.Count -ne 0) {
        $processIds = @($residual | ForEach-Object { $_.Id })
        throw "Phase 5 compression acceptance failed: residual benchmark process IDs: " +
            ($processIds -join ", ")
    }
}

function Assert-JsonContracts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportPath,

        [Parameter()]
        [string]$TracePath,

        [Parameter()]
        [switch]$AllowMissingTrace
    )

    Assert-Condition (Test-Path -LiteralPath $ReportPath -PathType Leaf) `
        "report was not created: $ReportPath"
    Assert-Condition ((Get-Item -LiteralPath $ReportPath).Length -gt 0) `
        "report is empty: $ReportPath"

    $validatorPath = Join-Path $script:RepositoryRoot `
        "scripts/validate_phase5_compression_trace.py"
    Assert-Condition (Test-Path -LiteralPath $validatorPath -PathType Leaf) `
        "compression trace validator is missing"
    $arguments = @(
        $validatorPath,
        "--report-schema", $script:ReportSchemaPath,
        "--trace-schema", $script:TraceSchemaPath,
        "--report", $ReportPath
    )
    if (-not [string]::IsNullOrWhiteSpace($TracePath)) {
        $arguments += @("--trace", $TracePath)
    }
    if ($AllowMissingTrace) {
        $arguments += "--allow-missing-trace"
    }
    & $script:PythonPath @arguments
    Assert-Condition ($LASTEXITCODE -eq 0) "schema/trace validation failed for $ReportPath"
}

function Assert-AllTrue {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Object,

        [Parameter(Mandatory = $true)]
        [string[]]$Fields,

        [Parameter(Mandatory = $true)]
        [string]$Context
    )

    foreach ($field in $Fields) {
        $property = $Object.PSObject.Properties[$field]
        Assert-Condition ($null -ne $property) "$Context is missing $field"
        Assert-Condition ($property.Value -eq $true) "$Context.$field must be true"
    }
}

function Assert-CommonReportSemantics {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Report,

        [Parameter(Mandatory = $true)]
        [string]$RawJson,

        [Parameter(Mandatory = $true)]
        [uint64]$ExpectedLogicalBytes
    )

    Assert-Condition ($Report.schema_version -eq 1) "schema_version must be 1"
    Assert-Condition ($Report.report_type -eq "xvram.adaptive_compression") `
        "report_type must be xvram.adaptive_compression"
    Assert-Condition ($Report.outcome.status -eq "completed") "outcome must be completed"
    Assert-Condition ($Report.outcome.exit_code -eq 0) "outcome.exit_code must be zero"
    Assert-XvramAcceptanceReportRevision -Observed $Report.build.git_commit `
        -Source $script:SourceProvenance
    Assert-Condition ($Report.build.build_type -eq $Configuration) `
        "report binary configuration differs from the requested gate"
    Assert-Condition (@($Report.diagnostics).Count -eq 0) "diagnostics must be empty"
    Assert-Condition ([uint64]$Report.configuration.effective_logical_bytes -eq `
        $ExpectedLogicalBytes) "effective logical size differs from the requested matrix"
    Assert-Condition ([uint64]$Report.device.total_memory_bytes -eq `
        $script:TotalVramBytes) "report total VRAM changed during the matrix"
    Assert-Condition ($Report.device.driver_model -eq "wddm") `
        "RTX 3070 hardware gate must execute under WDDM"
    Assert-Condition ($Report.device.vmm_supported -eq $true -and `
        $Report.device.uva_supported -eq $true) "hardware gate requires VMM and UVA"

    Assert-AllTrue -Object $Report.proof -Context "proof" -Fields @(
        "logical_data_exceeds_vram",
        "cache_smaller_than_logical",
        "no_expansion_stored",
        "generation_atomicity_verified",
        "write_admission_verified",
        "stable_virtual_addresses_verified",
        "maps_match_set_access",
        "event_boundaries_verified",
        "host_budget_respected",
        "device_budget_respected",
        "all_workloads_match_reference",
        "path_digests_match",
        "policy_digests_match",
        "raw_virtual_addresses_omitted"
    )
    Assert-AllTrue -Object $Report.cleanup -Context "cleanup" -Fields @(
        "complete",
        "operations_drained",
        "codec_slots_drained",
        "events_drained",
        "events_destroyed",
        "streams_destroyed",
        "mappings_removed",
        "physical_handles_released",
        "codec_workspace_released",
        "virtual_reservations_released",
        "pinned_staging_released",
        "spill_reservations_released",
        "host_backing_released",
        "context_released",
        "trace_closed",
        "worker_terminated"
    )

    Assert-Condition ($Report.telemetry.unsafe_remap_count -eq 0) `
        "unsafe_remap_count must be zero"
    Assert-Condition ($Report.telemetry.unsafe_transition_count -eq 0) `
        "unsafe_transition_count must be zero"
    Assert-Condition ($Report.telemetry.mapping_count -eq `
        $Report.telemetry.set_access_count) "every mapping must have SetAccess"
    Assert-Condition ($Report.telemetry.mapping_count -eq `
        $Report.telemetry.unmap_count) "every completed mapping must be unmapped"
    Assert-Condition ($Report.telemetry.event_record_count -eq `
        $Report.telemetry.event_retire_count) "event generations do not reconcile"
    Assert-Condition ($Report.telemetry.pcie_h2d_bytes -eq `
        ($Report.telemetry.pcie_h2d_payload_bytes + `
         $Report.telemetry.pcie_h2d_metadata_bytes)) `
        "H2D PCIe total does not reconcile with payload and metadata"
    Assert-Condition ($Report.telemetry.pcie_d2h_bytes -eq `
        ($Report.telemetry.pcie_d2h_payload_bytes + `
         $Report.telemetry.pcie_d2h_metadata_bytes)) `
        "D2H PCIe total does not reconcile with payload and metadata"
    Assert-Condition ($Report.telemetry.pcie_h2d_payload_bytes -le `
        $Report.telemetry.logical_h2d_bytes) "H2D payload exceeds logical attempt bytes"
    Assert-Condition ($Report.telemetry.pcie_d2h_payload_bytes -le `
        $Report.telemetry.logical_d2h_bytes) "D2H payload exceeds logical attempt bytes"
    Assert-Condition ($Report.telemetry.rejected_candidate_logical_d2h_bytes -le `
        $Report.telemetry.logical_d2h_bytes) `
        "rejected candidate D2H bytes exceed logical attempts"
    Assert-Condition ($Report.backing.host_budget_bytes_peak -le `
        $Report.backing.host_store_cap_bytes) "full host budget exceeded its cap"
    Assert-Condition ($Report.backing.host_bytes_peak -le `
        $Report.backing.host_budget_bytes_peak) `
        "authoritative backing peak exceeds the charged host budget"
    Assert-Condition ($Report.codec.workspace_bytes_peak -le `
        $Report.configuration.compression_scratch_cap_bytes) `
        "codec workspace exceeded its cap"
    Assert-Condition ($Report.codec.codec_slots_peak -le `
        $Report.configuration.codec_slots) "codec slot count exceeded its configured bound"
    Assert-Condition ($Report.codec.device_slot_bytes_peak -le `
        $Report.codec.device_slot_capacity_bytes) `
        "codec device slots exceeded their runtime-derived capacity"
    Assert-Condition (([decimal]$Report.codec.device_slot_capacity_bytes + `
        [decimal]$Report.codec.workspace_bytes_peak) -le `
        [decimal]$Report.telemetry.device_reserve_bytes_peak) `
        "codec slot/workspace capacity was not charged to the device reserve"
    Assert-Condition ($Report.telemetry.device_budget_violation_count -eq 0) `
        "runtime observed a managed-device budget violation"
    Assert-Condition ($Report.telemetry.safe_device_budget_bytes_minimum -gt 0) `
        "runtime did not publish a live safe-device budget"
    Assert-Condition ($Report.telemetry.handle_reuse_count -gt 0) `
        "oversubscribed run did not reuse physical VMM handles"
    Assert-Condition ($Report.backing.generations_created -eq `
        ($Report.backing.generations_committed + $Report.backing.generations_discarded)) `
        "host generations do not reconcile"
    Assert-Condition ($Report.backing.host_bytes_current -eq `
        ($Report.backing.raw_bytes_current + $Report.backing.compressed_bytes_current)) `
        "authoritative host bytes do not reconcile"
    Assert-Condition ($Report.backing.atomic_commit_failures -eq 0) `
        "an atomic backing commit failed"
    Assert-Condition ($Report.codec.verification_failures -eq 0) `
        "codec verification failed"

    [uint64]$rawDecisions = 0
    [uint64]$cpuDecisions = 0
    [uint64]$gpuDecisions = 0
    [uint64]$neverCompressDecisions = 0
    [uint64]$fallbacks = 0
    foreach ($workload in @($Report.workloads)) {
        Assert-Condition ($workload.status -eq "completed") `
            "workload $($workload.scenario) did not complete"
        Assert-Condition ($workload.mismatch_count -eq 0) `
            "workload $($workload.scenario) reported a mismatch"
        Assert-Condition ($workload.expected_digest128 -eq $workload.output_digest128) `
            "workload $($workload.scenario) digest mismatch"
        Assert-Condition ($workload.mappings -eq $workload.set_access -and `
            $workload.mappings -eq $workload.unmaps) `
            "workload $($workload.scenario) mapping accounting mismatch"
        Assert-Condition ($workload.events_recorded -eq $workload.events_retired) `
            "workload $($workload.scenario) event accounting mismatch"
        Assert-Condition ($workload.unsafe_remaps -eq 0 -and `
            $workload.unsafe_transitions -eq 0) `
            "workload $($workload.scenario) reported unsafe activity"
        $rawDecisions += [uint64]$workload.raw_path_decisions
        $cpuDecisions += [uint64]$workload.cpu_lz4_gpu_decisions
        $gpuDecisions += [uint64]$workload.gpu_lz4_decisions
        $neverCompressDecisions += [uint64]$workload.never_compress_decisions
        $fallbacks += [uint64]$workload.fallback_count
    }
    Assert-Condition ($rawDecisions -eq $Report.codec.raw_path_decisions) `
        "raw path decisions do not reconcile"
    Assert-Condition ($cpuDecisions -eq $Report.codec.cpu_lz4_gpu_decisions) `
        "CPU LZ4/GPU decode decisions do not reconcile"
    Assert-Condition ($gpuDecisions -eq $Report.codec.gpu_lz4_decisions) `
        "GPU LZ4 decisions do not reconcile"
    Assert-Condition ($neverCompressDecisions -eq $Report.codec.never_compress_decisions) `
        "never-compress decisions do not reconcile"
    Assert-Condition ($fallbacks -eq $Report.codec.fallback_count) `
        "codec fallback decisions do not reconcile"
    Assert-Condition ($Report.codec.never_compress_decisions -le `
        $Report.codec.raw_path_decisions) "never-compress decisions exceed raw decisions"

    if (-not $IncludeIdentifiers) {
        Assert-Condition ($null -eq $Report.device.uuid) "UUID was not redacted"
        Assert-Condition ($null -eq $Report.device.luid) "LUID was not redacted"
        Assert-Condition ($null -eq $Report.device.pci_bus_id) "PCI bus ID was not redacted"
    }
    Assert-Condition (-not ($RawJson -match `
        '"(?:raw_)?(?:cuda_)?virtual_address"|"logical_(?:base|address)"|"device_pointer"|"stream_handle"')) `
        "report serialized an address, pointer, or stream handle"
}

function Get-AlignedLogicalBytes {
    param(
        [Parameter(Mandatory = $true)]
        [decimal]$Ratio
    )

    $scaled = [decimal]$script:TotalVramBytes * $Ratio
    return [uint64]([decimal]::Floor($scaled / 4) * 4)
}

function Get-RunDigest {
    param([Parameter(Mandatory = $true)][object]$Report)

    $parts = @($Report.workloads | ForEach-Object {
        "$($_.scenario):$($_.output_digest128)"
    })
    return $parts -join "|"
}

function Invoke-CompressionRun {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [uint64]$LogicalBytes,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter()]
        [switch]$AllowPreflightSkip
    )

    $reportPath = Join-Path $script:ResolvedOutputDirectory "$Name.json"
    $tracePath = Join-Path $script:ResolvedOutputDirectory "$Name.jsonl"
    foreach ($path in @($reportPath, $tracePath)) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Remove-Item -LiteralPath $path -Force
        }
    }

    $identifierArguments = if ($IncludeIdentifiers) { @("--include-identifiers") } else { @() }
    $common = @(
        "--device", "$Device",
        "--logical-size", "${LogicalBytes}B",
        "--chunk-size", "64MiB",
        "--cache-target", "auto",
        "--host-store-cap", "auto",
        "--host-headroom", "auto",
        "--device-headroom", "512MiB",
        "--compression-scratch-cap", "256MiB",
        "--codec-slots", "2",
        "--codec-workers", "2",
        "--staging-slots", "4",
        "--prefetch-distance", "2",
        "--budget-poll-ms", "100",
        "--stall-timeout-ms", "5000",
        "--timeout-seconds", "$TimeoutSeconds",
        "--seed", "0x585652414D503035",
        "--warmup-passes", "2",
        "--measurement-passes", "5",
        "--trace", $tracePath,
        "--compact-json",
        "--no-text",
        "--json", $reportPath
    )

    Write-Host "Running $Name..."
    & $script:CompressionBenchPath @common @Arguments @identifierArguments
    $exitCode = $LASTEXITCODE
    Assert-NoResidualWorker

    if ($AllowPreflightSkip -and $exitCode -eq 23) {
        Assert-JsonContracts -ReportPath $reportPath -TracePath $tracePath -AllowMissingTrace
        $skipped = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        Assert-Condition ($skipped.outcome.status -eq "skipped") `
            "$Name exit 23 did not report a skipped preflight"
        Assert-Condition ($skipped.outcome.reason -eq "insufficient_host_memory" -and `
            $skipped.outcome.stage -eq "preflight" -and `
            $skipped.outcome.operation -in @("logical_size", "host_store_cap")) `
            "$Name may skip only a live host-memory preflight failure"
        Assert-XvramAcceptanceReportRevision -Observed $skipped.build.git_commit `
            -Source $script:SourceProvenance
        Assert-Condition ($skipped.build.build_type -eq $Configuration) `
            "$Name skipped report was produced by a differently configured binary"
        return [pscustomobject]@{
            Name = $Name
            Path = $reportPath
            Trace = if (Test-Path -LiteralPath $tracePath) { $tracePath } else { $null }
            Report = $skipped
            Skipped = $true
        }
    }

    Assert-Condition ($exitCode -eq 0) "$Name exited with code $exitCode"
    Assert-JsonContracts -ReportPath $reportPath -TracePath $tracePath
    $raw = Get-Content -LiteralPath $reportPath -Raw
    $report = $raw | ConvertFrom-Json
    Assert-CommonReportSemantics -Report $report -RawJson $raw `
        -ExpectedLogicalBytes $LogicalBytes
    return [pscustomobject]@{
        Name = $Name
        Path = $reportPath
        Trace = $tracePath
        Report = $report
        Skipped = $false
    }
}

$script:ResolvedOutputDirectory = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    [System.IO.Path]::GetFullPath($OutputDirectory)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $script:RepositoryRoot $OutputDirectory))
}
New-Item -ItemType Directory -Force -Path $script:ResolvedOutputDirectory | Out-Null
$manifestPath = Initialize-XvramAcceptanceManifest -OutputDirectory $script:ResolvedOutputDirectory
$script:SourceProvenance = Get-XvramAcceptanceSource -RepositoryRoot $script:RepositoryRoot
$script:PythonPath = Resolve-Python
$script:ProbePath = Resolve-XvramExecutable -Name "xvram-probe" -Override $ProbeExecutable
$script:CompressionBenchPath = Resolve-XvramExecutable -Name "xvram-compression-bench" `
    -Override $CompressionBenchExecutable
Assert-Condition (Test-Path -LiteralPath $script:ReportSchemaPath -PathType Leaf) `
    "report schema is missing"
Assert-Condition (Test-Path -LiteralPath $script:TraceSchemaPath -PathType Leaf) `
    "trace schema is missing"
Assert-Condition (@(Get-ResidualCompressionBenchProcess `
    -ExecutablePath $script:CompressionBenchPath).Count -eq 0) `
    "a compression benchmark process is already running"

$binaryProvenance = [ordered]@{
    probe = Get-XvramAcceptanceBinary -Path $script:ProbePath
    benchmark = Get-XvramAcceptanceBinary -Path $script:CompressionBenchPath
    nvcomp_gpu = Get-XvramAcceptanceBinary -Path (Join-Path `
        (Split-Path $script:CompressionBenchPath -Parent) "nvcomp64_5.dll")
    nvcomp_cpu = Get-XvramAcceptanceBinary -Path (Join-Path `
        (Split-Path $script:CompressionBenchPath -Parent) "nvcomp_cpu64_5.dll")
}

$probeReportPath = Join-Path $script:ResolvedOutputDirectory "probe.json"
& $script:ProbePath --device $Device --skip-vmm-smoke --compact-json --no-text `
    --json $probeReportPath
Assert-Condition ($LASTEXITCODE -eq 0) "xvram-probe failed"
$probe = Get-Content -LiteralPath $probeReportPath -Raw | ConvertFrom-Json
$selected = @($probe.cuda.devices | Where-Object { $_.ordinal -eq $Device })
Assert-Condition ($selected.Count -eq 1) "probe did not identify CUDA device $Device"
$deviceInfo = $selected[0]
Assert-Condition ($deviceInfo.name -match $ExpectedDeviceNamePattern) `
    "device '$($deviceInfo.name)' does not match '$ExpectedDeviceNamePattern'"
$script:TotalVramBytes = [uint64]$deviceInfo.total_memory_bytes
Assert-Condition ($script:TotalVramBytes -eq $ExpectedTotalVramBytes) `
    "total VRAM is $($script:TotalVramBytes), expected $ExpectedTotalVramBytes"

$size1p1 = Get-AlignedLogicalBytes -Ratio ([decimal]"1.1")
$size1p5 = Get-AlignedLogicalBytes -Ratio ([decimal]"1.5")
$size2p5 = Get-AlignedLogicalBytes -Ratio ([decimal]"2.5")
$size3 = Get-AlignedLogicalBytes -Ratio ([decimal]"3.0")
$size4 = Get-AlignedLogicalBytes -Ratio ([decimal]"4.0")
$runs = @()

$parityRaw = Invoke-CompressionRun -Name "forced-raw-1p1x" -LogicalBytes $size1p1 `
    -Arguments @("--compression-policy", "capacity", "--path", "raw", "--codec", "lz4",
        "--policy", "clock", "--scenario", "compressible-read", "--passes", "2")
$parityCpu = Invoke-CompressionRun -Name "forced-cpu-lz4-gpu-1p1x" `
    -LogicalBytes $size1p1 -Arguments @("--compression-policy", "capacity", "--path",
        "cpu-lz4-gpu", "--codec", "lz4", "--policy", "clock", "--scenario",
        "compressible-read", "--passes", "2")
$parityGpu = Invoke-CompressionRun -Name "forced-gpu-lz4-1p1x" `
    -LogicalBytes $size1p1 -Arguments @("--compression-policy", "capacity", "--path",
        "gpu-lz4", "--codec", "lz4", "--policy", "clock", "--scenario",
        "compressible-read", "--passes", "2")
$runs += @($parityRaw, $parityCpu, $parityGpu)

$parityDigests = @(@($parityRaw, $parityCpu, $parityGpu) | ForEach-Object {
    Get-RunDigest -Report $_.Report
} | Select-Object -Unique)
Assert-Condition ($parityDigests.Count -eq 1) "forced raw/CPU/GPU path digests differ"
Assert-Condition ($parityRaw.Report.codec.raw_path_decisions -gt 0 -and `
    $parityRaw.Report.codec.cpu_lz4_gpu_decisions -eq 0 -and `
    $parityRaw.Report.codec.gpu_lz4_decisions -eq 0) `
    "forced raw run did not stay on the raw path"
Assert-Condition ($parityCpu.Report.codec.cpu_lz4_gpu_decisions -gt 0 -and `
    $parityCpu.Report.codec.cpu_encode_operations -gt 0 -and `
    $parityCpu.Report.codec.gpu_decode_operations -gt 0) `
    "forced CPU-LZ4/GPU-decode run did not execute its requested path"
Assert-Condition ($parityGpu.Report.codec.gpu_lz4_decisions -gt 0 -and `
    $parityGpu.Report.codec.gpu_encode_operations -gt 0 -and `
    $parityGpu.Report.codec.gpu_decode_operations -gt 0) `
    "forced nvCOMP GPU run did not execute GPU encode/decode"
foreach ($compressedParity in @($parityCpu, $parityGpu)) {
    Assert-Condition ($compressedParity.Report.codec.cpu_codec_version -eq "1.10.0") `
        "CPU LZ4 version is not pinned to 1.10.0"
    Assert-Condition ($compressedParity.Report.codec.nvcomp_available -eq $true -and `
        $compressedParity.Report.codec.nvcomp_version -eq "5.3.0.16" -and `
        $compressedParity.Report.codec.nvcomp_library_source -eq "app_local") `
        "nvCOMP was not loaded from the pinned app-local package"
    Assert-Condition ($compressedParity.Report.codec.nvcomp_library_sha256 -eq `
        $ExpectedNvcompSha256.ToLowerInvariant()) "nvCOMP library hash differs from the pin"
}

$incompressible = Invoke-CompressionRun -Name "incompressible-adaptive-1p5x" `
    -LogicalBytes $size1p5 -Arguments @("--compression-policy", "adaptive", "--path", "auto",
        "--codec", "lz4", "--policy", "clock", "--scenario", "incompressible-read",
        "--passes", "2")
$runs += $incompressible
Assert-Condition ($incompressible.Report.proof.no_expansion_stored -eq $true) `
    "incompressible adaptive stored expansion"
Assert-Condition ($incompressible.Report.codec.raw_path_decisions -gt 0) `
    "incompressible adaptive never selected raw"
Assert-Condition ($incompressible.Report.codec.raw_path_decisions -gt `
    ($incompressible.Report.codec.cpu_lz4_gpu_decisions + `
     $incompressible.Report.codec.gpu_lz4_decisions)) `
    "incompressible adaptive did not predominantly select raw"

$perfRaw = Invoke-CompressionRun -Name "compressible-raw-3x" -LogicalBytes $size3 `
    -Arguments @("--compression-policy", "capacity", "--path", "raw", "--codec", "lz4",
        "--policy", "clock", "--scenario", "compressible-read", "--passes", "2")
$perfCompressed = Invoke-CompressionRun -Name "compressible-gpu-lz4-3x" `
    -LogicalBytes $size3 -Arguments @("--compression-policy", "capacity", "--path", "gpu-lz4",
        "--codec", "lz4", "--policy", "clock", "--scenario", "compressible-read",
        "--passes", "2")
$runs += @($perfRaw, $perfCompressed)
Assert-Condition ($perfCompressed.Report.proof.authoritative_compressed_backing_verified -eq `
    $true) "compressible run did not retain authoritative compressed backing"
Assert-Condition ($perfCompressed.Report.proof.compressed_h2d_reduction_verified -eq $true) `
    "compressible run did not reduce PCIe H2D bytes"
Assert-Condition ($perfCompressed.Report.telemetry.compressed_transfer_timing.median_ms -lt `
    $perfRaw.Report.telemetry.raw_transfer_timing.median_ms) `
    "median compressed end-to-end transfer was not faster than raw"

$reuse = Invoke-CompressionRun -Name "reuse-capacity-3x" -LogicalBytes $size3 `
    -Arguments @("--compression-policy", "capacity", "--path", "cpu-lz4-gpu", "--codec",
        "lz4", "--policy", "clock", "--scenario", "reuse", "--passes", "2")
$runs += $reuse
Assert-Condition ($reuse.Report.backing.lz4_chunks -gt 0) `
    "reuse run contains no authoritative LZ4 chunks"
Assert-Condition ($reuse.Report.telemetry.pcie_h2d_bytes -lt `
    $reuse.Report.telemetry.logical_h2d_bytes) "reuse run did not reduce H2D bytes"

$dirty = Invoke-CompressionRun -Name "dirty-writeback-gpu-3x" -LogicalBytes $size3 `
    -Arguments @("--compression-policy", "capacity", "--path", "gpu-lz4", "--codec", "lz4",
        "--policy", "clock", "--scenario", "dirty-writeback", "--passes", "2")
$runs += $dirty
Assert-Condition ($dirty.Report.codec.gpu_encode_operations -gt 0) `
    "dirty write-back did not execute GPU encode"
Assert-Condition ($dirty.Report.proof.gpu_encode_before_d2h_verified -eq $true) `
    "dirty write-back did not prove GPU encode before D2H"

$mixedClock = Invoke-CompressionRun -Name "mixed-adaptive-clock-3x" `
    -LogicalBytes $size3 -Arguments @("--compression-policy", "adaptive", "--path", "auto",
        "--codec", "lz4", "--policy", "clock", "--scenario", "mixed", "--passes", "2")
$runs += $mixedClock
Assert-Condition ($mixedClock.Report.codec.raw_path_decisions -gt 0) `
    "mixed workload observed no raw decision"
Assert-Condition (($mixedClock.Report.codec.cpu_lz4_gpu_decisions + `
    $mixedClock.Report.codec.gpu_lz4_decisions) -gt 0) `
    "mixed workload observed no compressed decision"
Assert-Condition ($mixedClock.Report.codec.calibration_samples -ge 2) `
    "mixed workload did not calibrate the cost model"

$budget = Invoke-CompressionRun -Name "budget-pressure-capacity-2p5x" `
    -LogicalBytes $size2p5 -Arguments @("--compression-policy", "capacity", "--path", "auto",
        "--codec", "lz4", "--policy", "clock", "--scenario", "budget-pressure",
        "--passes", "2")
$runs += $budget
Assert-Condition ($budget.Report.telemetry.target_shrink_count -gt 0) `
    "budget-pressure run observed no target shrink"
Assert-Condition ($budget.Report.telemetry.target_grow_count -gt 0) `
    "budget-pressure run observed no target grow"

$mixedLru = Invoke-CompressionRun -Name "mixed-adaptive-lru-3x" `
    -LogicalBytes $size3 -Arguments @("--compression-policy", "adaptive", "--path", "auto",
        "--codec", "lz4", "--policy", "lru", "--scenario", "mixed", "--passes", "2")
$runs += $mixedLru
Assert-Condition ((Get-RunDigest -Report $mixedClock.Report) -eq `
    (Get-RunDigest -Report $mixedLru.Report)) "CLOCK/LRU final digests differ"

$capacityStress = Invoke-CompressionRun -Name "capacity-stress-4x" -LogicalBytes $size4 `
    -Arguments @("--compression-policy", "capacity", "--path", "auto", "--codec", "lz4",
        "--policy", "clock", "--scenario", "compressible-read", "--passes", "2") `
    -AllowPreflightSkip
$runs += $capacityStress
if ($capacityStress.Skipped) {
    Write-Warning "4x capacity stress skipped because live host preflight returned exit 23"
} else {
    Assert-Condition ($capacityStress.Report.proof.authoritative_compressed_backing_verified -eq `
        $true) "4x capacity stress did not retain compressed backing"
}

Assert-NoResidualWorker
$sourceAtCompletion = Assert-XvramAcceptanceSourceUnchanged `
    -RepositoryRoot $script:RepositoryRoot -Expected $script:SourceProvenance
foreach ($binary in $binaryProvenance.Values) {
    Assert-XvramAcceptanceBinaryUnchanged -Expected $binary
}
$manifestRuns = @($runs | ForEach-Object {
    [ordered]@{
        name = $_.Name
        report = $_.Path
        trace = $_.Trace
        status = if ($_.Skipped) { "preflight_skipped" } else { "completed" }
        logical_bytes = $_.Report.configuration.effective_logical_bytes
        output_digest = if ($_.Skipped) { $null } else { Get-RunDigest -Report $_.Report }
    }
})
$manifest = [ordered]@{
    acceptance = "xvram.phase5.compression.rtx3070"
    generated_at_utc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
    configuration = $Configuration
    device_ordinal = $Device
    device_name = $deviceInfo.name
    total_vram_bytes = $script:TotalVramBytes
    report_schema = $script:ReportSchemaPath
    trace_schema = $script:TraceSchemaPath
    source = $script:SourceProvenance
    source_at_completion = $sourceAtCompletion
    binaries = $binaryProvenance
    validation = [ordered]@{
        status = "passed"
        report_head_and_configuration_matched = $true
        source_snapshot_unchanged = $true
        binary_hashes_unchanged = $true
        dirty_working_tree_allowed_and_recorded = $true
    }
    runs = $manifestRuns
    capacity_stress = if ($capacityStress.Skipped) { "preflight_skipped" } else { "completed" }
    result = "completed"
}
$manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Host "Phase 5 compression RTX 3070 acceptance completed: $manifestPath"
