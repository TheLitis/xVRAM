[CmdletBinding()]
param(
    [string]$BuildDirectory = "build/phase6a-release",
    [ValidateSet("Debug", "Release")][string]$Configuration = "Release",
    [ValidateRange(0, 64)][int]$Device = 0,
    [string]$OutputDirectory = "artifacts/phase6a-rtx3070",
    [ValidateRange(1, 86400)][int]$TimeoutSeconds = 900,
    [string]$ProbeExecutable,
    [string]$CompatBenchExecutable,
    [string]$Python = "python",
    [Parameter(Mandatory = $true)][string]$CublasLibrary,
    [Parameter(Mandatory = $true)][string]$CublasLtLibrary,
    [string]$ExpectedDeviceNamePattern = "^NVIDIA GeForce RTX 3070$",
    [uint64]$ExpectedTotalVramBytes = 8589410304
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
. (Join-Path $PSScriptRoot "acceptance-provenance.ps1")

function Assert-Gate {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw "Phase 6a acceptance: $Message" }
}

function Resolve-GateExecutable {
    param([string]$Explicit, [string]$Name)
    if ($Explicit) { return (Resolve-Path -LiteralPath $Explicit).Path }
    foreach ($relative in @("$BuildDirectory/$Configuration/$Name.exe", "$BuildDirectory/$Name.exe")) {
        $candidate = Join-Path $repository $relative
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return (Resolve-Path -LiteralPath $candidate).Path }
    }
    throw "Phase 6a acceptance: missing $Name executable"
}

function Assert-NoCompatWorker {
    param([string]$Executable)
    $residual = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ExecutablePath -and [string]::Equals($_.ExecutablePath, $Executable, [StringComparison]::OrdinalIgnoreCase)
    })
    Assert-Gate ($residual.Count -eq 0) "residual controller/worker process after completion"
}

$probe = Resolve-GateExecutable $ProbeExecutable "xvram-probe"
$bench = Resolve-GateExecutable $CompatBenchExecutable "xvram-compat-bench"
$core = (Resolve-Path -LiteralPath $CublasLibrary).Path
$lt = (Resolve-Path -LiteralPath $CublasLtLibrary).Path
$outputRoot = [IO.Path]::GetFullPath((Join-Path $repository $OutputDirectory))
if (-not (Test-Path -LiteralPath $outputRoot)) { $null = New-Item -ItemType Directory -Path $outputRoot }
$source = Get-XvramAcceptanceSource -RepositoryRoot $repository
$binary = Get-XvramAcceptanceBinary -Path $bench
$coreBinary = Get-XvramAcceptanceBinary -Path $core
$ltBinary = Get-XvramAcceptanceBinary -Path $lt
$manifestPath = Initialize-XvramAcceptanceManifest -OutputDirectory $outputRoot
$probePath = Join-Path $outputRoot "probe.json"
& $probe --device $Device --json $probePath --no-text --compact-json
Assert-Gate ($LASTEXITCODE -eq 0) "capability probe failed"
$probeReport = Get-Content -LiteralPath $probePath -Raw | ConvertFrom-Json
$deviceReport = @($probeReport.devices | Where-Object { $_.ordinal -eq $Device })
Assert-Gate ($deviceReport.Count -eq 1) "requested probe device absent"
Assert-Gate ($deviceReport[0].name -match $ExpectedDeviceNamePattern) "hardware gate requires the requested RTX 3070"
$total = [uint64]$deviceReport[0].total_memory_bytes
Assert-Gate ($total -eq $ExpectedTotalVramBytes) "GPU memory size differs from expected hardware"

$cases = [Collections.Generic.List[object]]::new()
$cases.Add([pscustomobject]@{ name = "small-suite"; arguments = @("--scenario", "suite", "--m", "37", "--n", "19", "--k", "67", "--padding", "7", "--offset-elements", "17"); ratio = $null; policy = "clock" })
$cases.Add([pscustomobject]@{ name = "split-k"; arguments = @("--scenario", "gemm", "--m", "512", "--n", "32", "--k", "65536", "--chunk-size", "2MiB", "--cache-target", "32MiB", "--padding", "3", "--offset-elements", "11"); ratio = $null; policy = "clock" })
foreach ($ratio in @(1.1, 1.5, 2.0)) {
    foreach ($policy in @("clock", "lru")) {
        $bytes = [uint64][Math]::Floor([double]$total * $ratio)
        $name = "oversubscribed-$($ratio.ToString('0.0', [Globalization.CultureInfo]::InvariantCulture))-$policy"
        $cases.Add([pscustomobject]@{ name = $name; arguments = @("--scenario", "gemm", "--logical-size", "$bytes"); ratio = $ratio; policy = $policy })
    }
}
$reports = [Collections.Generic.List[object]]::new()
$validator = Join-Path $repository "tests/contract/compat_contract.py"
$schema = Join-Path $repository "schemas/cuda-compat-v1.schema.json"
$traceSchema = Join-Path $repository "schemas/cuda-compat-trace-v1.schema.json"
foreach ($case in $cases) {
    Assert-XvramAcceptanceBinaryUnchanged -Expected $binary
    Assert-XvramAcceptanceBinaryUnchanged -Expected $coreBinary
    Assert-XvramAcceptanceBinaryUnchanged -Expected $ltBinary
    $reportPath = Join-Path $outputRoot "$($case.name).json"
    $tracePath = Join-Path $outputRoot "$($case.name).jsonl"
    $stderrPath = Join-Path $outputRoot "$($case.name).stderr.txt"
    $arguments = @("--device", "$Device", "--policy", $case.policy, "--passes", "2", "--alpha", "1.25", "--beta", "0.5", "--timeout-seconds", "$TimeoutSeconds", "--json", $reportPath, "--trace", $tracePath, "--no-text", "--compact-json", "--cublas-library", $core, "--cublas-lt-library", $lt) + $case.arguments
    Write-Host "Phase 6a hardware case: $($case.name)"
    $started = [DateTime]::UtcNow
    & $bench @arguments 2> $stderrPath
    $exitCode = $LASTEXITCODE
    Assert-NoCompatWorker -Executable $bench
    Assert-Gate ($exitCode -eq 0) "$($case.name) exited $exitCode; inspect its JSON and stderr"
    & $Python $validator --schema $schema --trace-schema $traceSchema --report $reportPath --trace $tracePath
    Assert-Gate ($LASTEXITCODE -eq 0) "$($case.name) schema/semantic/trace validation failed"
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    Assert-XvramAcceptanceReportRevision -Observed $report.build.git_commit -Source $source
    Assert-Gate ($report.device.name -match $ExpectedDeviceNamePattern) "benchmark device mismatch"
    Assert-Gate ($report.device.cublas_library_source -eq 3) "explicit hash-observed cuBLAS pair was not selected"
    if ($case.name -eq "small-suite") {
        Assert-Gate ($report.workloads.Count -eq 5) "small suite must exercise NN/NT/TN/TT plus explicit padded geometry"
        foreach ($workload in $report.workloads) {
            Assert-Gate ($workload.native_baseline_equal -eq $true) "small suite lacks ordinary cuBLAS equality"
            Assert-Gate ($workload.padding_elements_checked -gt 0) "small suite did not verify padding"
        }
    }
    if ($case.name -eq "split-k") {
        Assert-Gate ($report.workloads[0].tiles_retired -gt 2) "constrained cache did not split GEMM into multiple K-panels"
    }
    if ($null -ne $case.ratio) {
        $workload = $report.workloads[0]
        $requested = [uint64][Math]::Floor([double]$total * $case.ratio)
        $planeBytes = [uint64](4 * (4096 + 16))
        Assert-Gate ($workload.logical_bytes -le $requested -and $requested - $workload.logical_bytes -lt $planeBytes) "logical size does not cover requested ratio to one K-plane precision"
        Assert-Gate ($workload.logical_bytes -gt $total) "oversubscribed case did not exceed physical VRAM"
        Assert-Gate ($workload.evictions -gt 0 -and $workload.handle_reuses -gt 0) "oversubscribed case lacks real eviction/reuse"
    }
    $reports.Add([pscustomobject][ordered]@{
        name = $case.name; policy = $case.policy; ratio = $case.ratio; exit_code = $exitCode
        report = $reportPath; report_sha256 = (Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash.ToLowerInvariant()
        trace = $tracePath; trace_sha256 = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant()
        elapsed_seconds = ([DateTime]::UtcNow - $started).TotalSeconds
        workloads = @($report.workloads | Select-Object name, m, n, k, logical_bytes, tiles_retired, digest, reference_digest, max_absolute_error, max_relative_error)
    })
}
foreach ($ratio in @(1.1, 1.5, 2.0)) {
    $pair = @($reports | Where-Object { $null -ne $_.ratio -and $_.ratio -eq $ratio })
    Assert-Gate ($pair.Count -eq 2) "missing policy pair"
    Assert-Gate ($pair[0].workloads[0].reference_digest -eq $pair[1].workloads[0].reference_digest) "policy pair does not have identical reference data"
    # FP32 kernels may choose distinct reduction orders. Each policy must pass the full numeric
    # reference; record actual digests without imposing an additional bitwise-equality gate.
}
$finalSource = Assert-XvramAcceptanceSourceUnchanged -RepositoryRoot $repository -Expected $source
Assert-XvramAcceptanceBinaryUnchanged -Expected $binary
Assert-XvramAcceptanceBinaryUnchanged -Expected $coreBinary
Assert-XvramAcceptanceBinaryUnchanged -Expected $ltBinary
$manifest = [ordered]@{
    schema_version = 1; report_type = "xvram.phase6a_acceptance"; completed = $true
    source = $source; final_source = $finalSource; binary = $binary
    cublas = $coreBinary; cublas_lt = $ltBinary; total_vram_bytes = $total
    report_count = $reports.Count; reports = @($reports)
    numerical_absolute_tolerance = 0.0001; numerical_relative_tolerance = 0.00002
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
}
$manifest | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Host "Phase 6a acceptance completed: $manifestPath"
