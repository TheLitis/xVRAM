[CmdletBinding()]
param(
    [Parameter()]
    [string]$BuildDirectory = "build/phase4b",

    [Parameter()]
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",

    [Parameter()]
    [ValidateRange(0, 64)]
    [int]$Device = 0,

    [Parameter()]
    [string]$OutputDirectory = "artifacts/phase4b-rtx3070",

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
    [switch]$IncludeIdentifiers
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$script:ReportSchemaPath = Join-Path $script:RepositoryRoot `
    "schemas/pytorch-inference-report-v1.schema.json"
$script:TraceSchemaPath = Join-Path $script:RepositoryRoot `
    "schemas/pytorch-inference-trace-record-v1.schema.json"

function Assert-Condition {
    param(
        [Parameter(Mandatory = $true)]
        [bool]$Condition,

        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if (-not $Condition) {
        throw "Phase 4b acceptance failed: $Message"
    }
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter()]
        [string[]]$Arguments = @(),

        [Parameter()]
        [string]$Context = "command"
    )

    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Phase 4b acceptance failed: $Context exited with code $exitCode"
    }
}

function Resolve-Executable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Requested,

        [Parameter()]
        [string[]]$Candidates = @(),

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (Test-Path -LiteralPath $Requested -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Requested).Path
    }
    $commands = @(Get-Command $Requested -CommandType Application `
        -ErrorAction SilentlyContinue)
    if ($commands.Count -gt 0) {
        return $commands[0].Source
    }
    foreach ($candidate in $Candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and
            (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "Phase 4b acceptance failed: unable to locate $Description"
}

function Get-VisualStudioInstallation {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} `
        "Microsoft Visual Studio/Installer/vswhere.exe"
    Assert-Condition (Test-Path -LiteralPath $vswhere -PathType Leaf) `
        "Visual Studio Installer/vswhere.exe is unavailable"
    $installation = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    Assert-Condition ($LASTEXITCODE -eq 0) "vswhere failed"
    $paths = @($installation | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    Assert-Condition ($paths.Count -eq 1) `
        "a single Visual Studio installation with the x64 C++ toolchain was not found"
    return [System.IO.Path]::GetFullPath($paths[0].Trim())
}

function Import-VisualStudioEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InstallationPath
    )

    $developerCommand = Join-Path $InstallationPath "Common7/Tools/VsDevCmd.bat"
    Assert-Condition (Test-Path -LiteralPath $developerCommand -PathType Leaf) `
        "VsDevCmd.bat is unavailable at $developerCommand"
    $commandLine = "`"$developerCommand`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
    $environmentLines = & $env:ComSpec /d /s /c $commandLine
    Assert-Condition ($LASTEXITCODE -eq 0) "VsDevCmd.bat failed"
    foreach ($line in $environmentLines) {
        $separator = $line.IndexOf("=")
        if ($separator -le 0) {
            continue
        }
        $name = $line.Substring(0, $separator)
        $value = $line.Substring($separator + 1)
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
    Assert-Condition ($null -ne (Get-Command cl.exe -ErrorAction SilentlyContinue)) `
        "the Visual Studio x64 compiler environment was not initialized"
}

function Assert-ToolVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter()]
        [version]$MinimumVersion
    )

    $output = @(& $Executable --version)
    Assert-Condition ($LASTEXITCODE -eq 0) "$Name --version failed"
    $joined = $output -join "`n"
    $match = [regex]::Match($joined, "[0-9]+[.][0-9]+(?:[.][0-9]+)?")
    Assert-Condition $match.Success "$Name did not report a parseable version"
    $observed = [version]$match.Value
    if ($null -ne $MinimumVersion) {
        Assert-Condition ($observed -ge $MinimumVersion) `
            "$Name $observed is older than required $MinimumVersion"
    }
    Write-Host "${Name}: $observed ($Executable)"
}

function Resolve-RuntimeLibrary {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BuildRoot,

        [Parameter()]
        [string]$Override
    )

    if (-not [string]::IsNullOrWhiteSpace($Override)) {
        Assert-Condition (Test-Path -LiteralPath $Override -PathType Leaf) `
            "runtime DLL override does not exist: $Override"
        return (Resolve-Path -LiteralPath $Override).Path
    }
    $candidates = @(
        (Join-Path $BuildRoot "xvram_torch_runtime.dll"),
        (Join-Path (Join-Path $BuildRoot $Configuration) "xvram_torch_runtime.dll")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    $discovered = @(Get-ChildItem -LiteralPath $BuildRoot -Filter `
        "xvram_torch_runtime.dll" -File -Recurse -ErrorAction SilentlyContinue)
    Assert-Condition ($discovered.Count -eq 1) `
        "xvram_torch_runtime.dll was not uniquely discoverable below $BuildRoot"
    return $discovered[0].FullName
}

function Get-XvramTorchWorkers {
    $workers = @()
    try {
        $processes = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    } catch {
        throw "Phase 4b acceptance failed: unable to inspect worker processes: $($_.Exception.Message)"
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
        $identifiers = @($workers | ForEach-Object { $_.ProcessId })
        throw "Phase 4b acceptance failed: residual xvram.torch_bench worker process IDs: " +
            ($identifiers -join ", ")
    }
}

function Assert-RunArtifacts {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportPath,

        [Parameter(Mandatory = $true)]
        [string]$TracePath,

        [Parameter(Mandatory = $true)]
        [object]$Expected,

        [Parameter(Mandatory = $true)]
        [object]$Hardware
    )

    Assert-Condition (Test-Path -LiteralPath $ReportPath -PathType Leaf) `
        "report was not created: $ReportPath"
    Assert-Condition ((Get-Item -LiteralPath $ReportPath).Length -gt 0) `
        "report is empty: $ReportPath"
    Assert-Condition (Test-Path -LiteralPath $TracePath -PathType Leaf) `
        "trace was not created: $TracePath"
    Assert-Condition ((Get-Item -LiteralPath $TracePath).Length -gt 0) `
        "trace is empty: $TracePath"

    $validator = @'
import base64
import json
import math
import pathlib
import re
import sys

import jsonschema

report_schema_path, trace_schema_path, report_path, trace_path = map(pathlib.Path, sys.argv[1:5])
expected = json.loads(base64.b64decode(sys.argv[5]).decode("utf-8"))
hardware = json.loads(base64.b64decode(sys.argv[6]).decode("utf-8"))

def require(condition, message):
    if not condition:
        raise AssertionError(message)

report_schema = json.loads(report_schema_path.read_text(encoding="utf-8"))
trace_schema = json.loads(trace_schema_path.read_text(encoding="utf-8"))
jsonschema.Draft202012Validator.check_schema(report_schema)
jsonschema.Draft202012Validator.check_schema(trace_schema)
report = json.loads(report_path.read_text(encoding="utf-8"))
jsonschema.Draft202012Validator(report_schema).validate(report)

raw_trace = trace_path.read_text(encoding="utf-8")
require(raw_trace.endswith("\n"), "trace JSONL is not newline terminated")
lines = [line for line in raw_trace.splitlines() if line.strip()]
require(lines, "trace contains no records")
records = [json.loads(line) for line in lines]
trace_validator = jsonschema.Draft202012Validator(trace_schema)
for record in records:
    trace_validator.validate(record)

sequences = [record["sequence"] for record in records]
require(sequences[0] == 1, "trace sequence does not start at one")
require(all(left < right for left, right in zip(sequences, sequences[1:])),
        "trace sequences are not strictly increasing")
timestamps = [record["monotonic_ns"] for record in records]
require(all(left <= right for left, right in zip(timestamps, timestamps[1:])),
        "trace monotonic timestamps moved backwards")
require(records[-1]["kind"] == "verification" and
        records[-1]["operation"] == "trace_complete",
        "trace has no terminal completion record")
require(any(record["kind"] == "lease" for record in records),
        "trace contains no lease observation")

forbidden_keys = {
    "cuda_va", "raw_va", "virtual_address", "virtual_address_base",
    "native_stream_handle", "stream_handle", "device_pointer",
    "logical_address", "logical_base", "cuda_address",
}
forbidden_text = re.compile(
    r"\b(?:raw_va|cuda_va|virtual_address|native_stream_handle|stream_handle|"
    r"device_pointer|logical_address|logical_base|cuda_address)\b",
    re.IGNORECASE,
)

def assert_private(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            require(lowered not in forbidden_keys and "virtual_address" not in lowered,
                    f"{path} exposes forbidden field {key}")
            assert_private(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_private(child, f"{path}[{index}]")
    elif isinstance(value, str):
        require(forbidden_text.search(value) is None,
                f"{path} exposes a raw address or stream handle")

assert_private(report)
assert_private(records)

require(report["schema_version"] == 1 and
        report["report_type"] == "xvram.pytorch_inference", "wrong report contract")
require(report["outcome"]["status"] == "completed" and
        report["outcome"]["exit_code"] == 0, "run did not complete")
require(not report["diagnostics"], "completed report contains diagnostics")
require(report["system"]["os"] == "Windows", "report was not produced on Windows")
require(report["build"]["torch_version"] == hardware["torch_version"],
        "report PyTorch version differs from preflight")
require(report["build"]["cuda_version"] == hardware["cuda_version"],
        "report CUDA version differs from preflight")

device = report["device"]
require(device["ordinal"] == expected["device"], "wrong report device ordinal")
expected_device_name = hardware["device_name"] if expected["include_identifiers"] else "redacted"
require(device["name"] == expected_device_name,
        "report device-name redaction differs from the CLI request")
require(device["total_vram_bytes"] == hardware["total_vram_bytes"],
        "report VRAM differs from preflight")
require(device["compute_capability"] == hardware["compute_capability"],
        "report compute capability differs from preflight")
require(device["identifiers_included"] == expected["include_identifiers"],
        "identifier privacy flag differs from the CLI request")

configuration = report["configuration"]
exact_configuration = {
    "device": expected["device"],
    "model": expected["model"],
    "layers": expected["layers"],
    "batch": 1,
    "sequence": expected["sequence"],
    "hidden": 4096,
    "intermediate": 11008,
    "heads": 32,
    "dtype": "float16",
    "policy": expected["policy"],
    "cache_target_bytes": 0,
    "chunk_size_bytes": 64 * 1024**2,
    "device_headroom_bytes": 512 * 1024**2,
    "scratch_cap_bytes": 512 * 1024**2,
    "prefetch_distance": expected["prefetch_distance"],
    "sdpa_backend": "math",
    "seed": "0x585652414d503034",
    "include_identifiers": expected["include_identifiers"],
}
for key, value in exact_configuration.items():
    require(configuration.get(key) == value, f"configuration.{key} differs from the gate")
require(math.isclose(configuration["model_ratio"], expected["ratio"],
                     rel_tol=0.0, abs_tol=1.0e-12), "configuration.model_ratio differs")

model = report["model"]
for key in ("layers", "batch", "sequence", "hidden", "intermediate", "heads"):
    require(model[key] == exact_configuration[key], f"model.{key} differs from configuration")
require(model["kind"] == expected["model"], "model.kind differs from configuration")
require(model["logical_bytes"] > 0 and model["parameter_bytes"] > 0 and
        model["activation_bytes"] > 0, "model byte accounting is empty")
if expected["parameter_bytes"] is not None:
    require(model["parameter_bytes"] == expected["parameter_bytes"],
            "parameter byte count differs from the exact Llama contract")
require(model["activation_bytes"] == expected["activation_bytes"],
        "activation byte count differs from the exact static graph contract")
require(model["logical_bytes"] == expected["logical_bytes"],
        "logical byte count differs from the exact static graph contract")
if expected["model"] == "llama2-like":
    observed_ratio = model["parameter_bytes"] / hardware["total_vram_bytes"]
    require(math.isclose(observed_ratio, expected["ratio"], rel_tol=0.0, abs_tol=0.001),
            f"observed parameter/VRAM ratio {observed_ratio:.9f} differs from the gate")
require(model["logical_bytes"] >= model["parameter_bytes"],
        "logical bytes are smaller than parameter bytes")

oversubscribed = model["logical_bytes"] > device["total_vram_bytes"]
require(oversubscribed == expected["oversubscribed"],
        "observed logical oversubscription differs from the requested case")

graph = report["graph"]
sha256 = re.compile(r"^[0-9a-f]{64}$")
require(sha256.fullmatch(graph["hash"]) is not None, "graph hash is not SHA-256")
require(graph["backend_hash"] == graph["hash"], "backend graph hash differs")
require(sha256.fullmatch(graph["allowlist_hash"]) is not None,
        "allowlist hash is not SHA-256")
require(graph["node_count"] > 0 and graph["region_count"] > 0,
        "captured graph is empty")
require(not graph["unsupported_nodes"], "graph contains unsupported nodes")

execution = report["execution"]
leases = (execution["leases_acquired"], execution["leases_sealed"],
          execution["leases_retired"])
require(leases[0] == leases[1] == leases[2],
        "lease lifecycle does not reconcile")
require(execution["tiled_gemm_regions"] >= 0 and
        execution["tiled_gemm_tiles"] >= execution["tiled_gemm_regions"],
        "tiled GEMM regions/tiles do not reconcile")
require(execution["events_recorded"] > 0 and
        execution["events_recorded"] == execution["events_retired"],
        "event generations do not reconcile")
require(execution["regions_completed"] == graph["region_count"] and
        leases[2] + execution["tiled_gemm_regions"] == execution["regions_completed"],
        "lease/tiled regions were not fully retired")
require(execution["events_retired"] >=
        leases[2] + execution["tiled_gemm_tiles"],
        "execution boundaries have fewer retired events than work items")
timings = execution["region_timings_ms"]
require(len(timings) == execution["regions_completed"] and
        all(math.isfinite(value) and value >= 0 for value in timings) and any(value > 0 for value in timings),
        "per-region timing observations are incomplete")
require((execution["regions_completed"] == execution["tiled_gemm_regions"] or
         execution["live_views_peak"] > 0) and execution["live_views_final"] == 0,
        "managed view lifecycle does not reconcile")
require(execution["scratch_cap_bytes"] == 512 * 1024**2 and
        execution["scratch_peak_bytes"] <= execution["scratch_cap_bytes"],
        "scratch usage exceeded the fixed cap")
require(execution["prefetch_distance"] == expected["prefetch_distance"],
        "execution prefetch distance differs")
require(execution["trace_complete"] and execution["trace_records"] == len(records),
        "trace accounting does not reconcile")

cache = report["cache"]
require(0 < cache["target_bytes"] <=
        device["total_vram_bytes"] - configuration["device_headroom_bytes"] -
        configuration["scratch_cap_bytes"], "cache target exceeds the safe device bound")
require(cache["chunk_bytes"] == 64 * 1024**2, "cache chunk size differs")
require(cache["resident_bytes"] == 0 and
        0 < cache["resident_peak_bytes"] <= cache["target_bytes"],
        "cache residency did not remain bounded or drain")
require(cache["maps"] > 0 and cache["maps"] == cache["set_access"] == cache["unmaps"],
        "map/SetAccess/unmap lifecycle does not reconcile")
require(cache["unsafe_remaps"] == 0 and cache["unsafe_transitions"] == 0,
        "unsafe cache activity was observed")
require(cache["hits"] + cache["misses"] > 0 and cache["h2d_bytes"] > 0,
        "cache performed no observable demand traffic")
require(cache["weight_d2h_bytes"] == 0, "read-only weights were written back")
require(cache["prefetch_submitted"] == cache["prefetch_retired"],
        "prefetch requests did not fully retire")
if expected["prefetch_distance"] == 0:
    require(cache["prefetch_submitted"] == 0, "distance-zero run submitted prefetch")
else:
    require(cache["prefetch_submitted"] > 0, "prefetch-enabled run observed no prefetch")
if expected["oversubscribed"]:
    require(cache["evictions"] > 0 and cache["frame_reuses"] > 0,
            "oversubscribed run observed no eviction/frame reuse")

verification = report["verification"]
require(sha256.fullmatch(verification["reference_digest"]) is not None and
        verification["reference_digest"] == verification["output_digest"],
        "reference/output digests differ")
require(verification["mismatch_count"] == 0 and
        verification["max_abs_error"] == 0 and verification["max_rel_error"] == 0,
        "reference equality was not exact")

proof = report["proof"]
common_proof = (
    "stable_addresses", "maps_match_set_access", "event_safe",
    "zero_weight_writeback", "bounded_scratch", "reference_matches",
    "zero_live_views", "full_cleanup",
)
for field in common_proof:
    require(proof[field] is True, f"proof.{field} is false")
require(proof["logical_exceeds_vram"] == expected["oversubscribed"],
        "proof.logical_exceeds_vram differs from observed bytes")
require(proof["real_oversubscription"] == expected["oversubscribed"],
        "proof.real_oversubscription differs from the gate")
for field, value in report["cleanup"].items():
    require(value is True, f"cleanup.{field} is false")
'@

    $expectedJson = $Expected | ConvertTo-Json -Depth 8 -Compress
    $hardwareJson = $Hardware | ConvertTo-Json -Depth 8 -Compress
    $expectedArgument = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($expectedJson))
    $hardwareArgument = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($hardwareJson))
    $validator | & $script:PythonPath - $script:ReportSchemaPath `
        $script:TraceSchemaPath $ReportPath $TracePath $expectedArgument $hardwareArgument
    Assert-Condition ($LASTEXITCODE -eq 0) `
        "strict report/trace validation failed for $($Expected.name)"
}

Assert-Condition ([Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) `
    "the mandatory hardware gate must run on Windows"
Assert-Condition ([Environment]::Is64BitProcess) "the hardware gate requires a 64-bit process"
Assert-Condition (Test-Path -LiteralPath $script:ReportSchemaPath -PathType Leaf) `
    "report schema is missing"
Assert-Condition (Test-Path -LiteralPath $script:TraceSchemaPath -PathType Leaf) `
    "trace schema is missing"

$visualStudio = Get-VisualStudioInstallation
Import-VisualStudioEnvironment -InstallationPath $visualStudio
$cmakeCandidates = @(
    (Join-Path $visualStudio `
        "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe")
)
$ninjaCandidates = @(
    (Join-Path $visualStudio `
        "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe")
)
$script:CMakePath = Resolve-Executable -Requested $CMake -Candidates $cmakeCandidates `
    -Description "CMake (PATH or Visual Studio bundled copy)"
$script:NinjaPath = Resolve-Executable -Requested $Ninja -Candidates $ninjaCandidates `
    -Description "Ninja (PATH or Visual Studio bundled copy)"
$script:PythonPath = Resolve-Executable -Requested $Python -Description "Python"
Assert-ToolVersion -Executable $script:CMakePath -Name "CMake" `
    -MinimumVersion ([version]"4.3.1")
Assert-ToolVersion -Executable $script:NinjaPath -Name "Ninja"

$nvidiaSmiCandidates = @(
    (Join-Path $env:windir "System32/nvidia-smi.exe"),
    (Join-Path $env:ProgramW6432 "NVIDIA Corporation/NVSMI/nvidia-smi.exe")
)
$nvidiaSmi = Resolve-Executable -Requested "nvidia-smi.exe" `
    -Candidates $nvidiaSmiCandidates -Description "nvidia-smi"
$driverModelOutput = @(& $nvidiaSmi --id=$Device --query-gpu=driver_model.current `
    --format=csv,noheader)
Assert-Condition ($LASTEXITCODE -eq 0) "nvidia-smi driver-model query failed"
$driverModels = @($driverModelOutput | ForEach-Object { $_.Trim() } |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
Assert-Condition ($driverModels.Count -eq 1 -and $driverModels[0] -eq "WDDM") `
    "CUDA device $Device is not running under WDDM"

$previousPythonPath = $env:PYTHONPATH
$previousRuntimeLibrary = $env:XVRAM_TORCH_RUNTIME_LIBRARY
try {
    $env:PYTHONPATH = Join-Path $script:RepositoryRoot "python"

    $hardwareProbe = @'
import json
import re
import struct
import sys

import jsonschema
import torch

device = int(sys.argv[1])
expected_pattern = sys.argv[2]
expected_vram = int(sys.argv[3])
version_match = re.match(r"^(\d+)[.](\d+)", torch.__version__)
assert version_match is not None, f"unparseable PyTorch version: {torch.__version__}"
assert tuple(map(int, version_match.groups())) >= (2, 13), torch.__version__
assert torch.version.cuda == "13.0", f"expected PyTorch CUDA 13.0, got {torch.version.cuda}"
assert struct.calcsize("P") == 8, "Python must be 64-bit"
assert torch.cuda.is_available(), "CUDA is unavailable"
assert 0 <= device < torch.cuda.device_count(), "CUDA device ordinal is unavailable"
torch.cuda.init()
properties = torch.cuda.get_device_properties(device)
assert re.search(expected_pattern, properties.name), properties.name
assert properties.total_memory == expected_vram, (
    f"expected {expected_vram} bytes VRAM, got {properties.total_memory}"
)
print(json.dumps({
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "device_name": properties.name,
    "total_vram_bytes": properties.total_memory,
    "compute_capability": f"{properties.major}.{properties.minor}",
}, separators=(",", ":")))
'@
    $hardwareOutput = @($hardwareProbe | & $script:PythonPath - $Device `
        $ExpectedDeviceNamePattern $ExpectedTotalVramBytes)
    Assert-Condition ($LASTEXITCODE -eq 0) "PyTorch/CUDA/RTX 3070 preflight failed"
    $hardwareLines = @($hardwareOutput |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    Assert-Condition ($hardwareLines.Count -ge 1) "hardware preflight returned no data"
    $hardware = $hardwareLines[-1] | ConvertFrom-Json
    Write-Host ("Hardware gate: $($hardware.device_name), " +
        "$($hardware.total_vram_bytes) bytes, PyTorch $($hardware.torch_version), " +
        "CUDA $($hardware.cuda_version), WDDM")

    $resolvedBuild = if ([System.IO.Path]::IsPathRooted($BuildDirectory)) {
        [System.IO.Path]::GetFullPath($BuildDirectory)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $script:RepositoryRoot $BuildDirectory))
    }
    New-Item -ItemType Directory -Force -Path $resolvedBuild | Out-Null
    $cachePath = Join-Path $resolvedBuild "CMakeCache.txt"
    if (Test-Path -LiteralPath $cachePath -PathType Leaf) {
        $generatorLine = Get-Content -LiteralPath $cachePath |
            Where-Object { $_ -like "CMAKE_GENERATOR:INTERNAL=*" } |
            Select-Object -First 1
        Assert-Condition ($null -eq $generatorLine -or $generatorLine -eq `
            "CMAKE_GENERATOR:INTERNAL=Ninja") `
            "build directory was configured with a non-Ninja generator: $resolvedBuild"
    }

    Invoke-Checked -Executable $script:CMakePath -Context "CMake configure" -Arguments @(
        "-S", $script:RepositoryRoot,
        "-B", $resolvedBuild,
        "-G", "Ninja",
        "-DCMAKE_MAKE_PROGRAM:FILEPATH=$($script:NinjaPath)",
        "-DCMAKE_BUILD_TYPE=$Configuration",
        "-DPython3_EXECUTABLE:FILEPATH=$($script:PythonPath)",
        "-DXVRAM_BUILD_TESTS=ON",
        "-DXVRAM_BUILD_TORCH_RUNTIME=ON",
        "-DXVRAM_FETCH_CUDA_HEADERS=ON",
        "-DXVRAM_WARNINGS_AS_ERRORS=ON"
    )
    Invoke-Checked -Executable $script:CMakePath -Context "native runtime build" -Arguments @(
        "--build", $resolvedBuild,
        "--config", $Configuration,
        "--target", "xvram_torch_runtime", "xvram-torch-runtime-control-tests",
        "--parallel"
    )

    $ctestPath = Join-Path (Split-Path $script:CMakePath -Parent) "ctest.exe"
    $ctestPath = Resolve-Executable -Requested $ctestPath -Description "CTest"
    Invoke-Checked -Executable $ctestPath -Context "native runtime contract tests" -Arguments @(
        "--test-dir", $resolvedBuild,
        "-C", $Configuration,
        "--output-on-failure",
        "--no-tests=error",
        "-R", "^xvram[.]torch-runtime[.](control|export-contract|stable-load)$"
    )

    $runtimePath = Resolve-RuntimeLibrary -BuildRoot $resolvedBuild -Override $RuntimeLibrary
    $env:XVRAM_TORCH_RUNTIME_LIBRARY = $runtimePath
    $torchLibraryProbe = @'
import pathlib
import torch
print(pathlib.Path(torch.__file__).resolve().parent / "lib")
'@
    $torchLibraryOutput = @($torchLibraryProbe | & $script:PythonPath -)
    Assert-Condition ($LASTEXITCODE -eq 0 -and $torchLibraryOutput.Count -ge 1) `
        "PyTorch library directory discovery failed"
    $torchLibraryDirectory = $torchLibraryOutput[-1].Trim()
    Assert-Condition (Test-Path -LiteralPath $torchLibraryDirectory -PathType Container) `
        "PyTorch library directory is missing: $torchLibraryDirectory"
    $env:PATH = (Split-Path $runtimePath -Parent) + ";" + $torchLibraryDirectory + ";" +
        $env:PATH

    $runtimeProbe = @'
import ctypes
import os
import pathlib

import torch
from xvram.torch_native import find_runtime_library
from xvram.torch_runtime import run_benchmark_plan

path = find_runtime_library()
assert path == pathlib.Path(os.environ["XVRAM_TORCH_RUNTIME_LIBRARY"]).resolve()
torch.ops.load_library(str(path))
native = ctypes.CDLL(str(path))
assert getattr(native, "xvram_torch_runtime_get_api", None) is not None
assert torch.cuda.is_available()
torch.cuda.init()
assert hasattr(torch.ops.xvram_internal._wrap_resolved_v1, "default")
assert callable(run_benchmark_plan)
print(path)
'@
    $runtimeProbeOutput = @($runtimeProbe | & $script:PythonPath -)
    Assert-Condition ($LASTEXITCODE -eq 0 -and $runtimeProbeOutput.Count -ge 1) `
        "built xvram_torch_runtime failed its Python load/entry-point preflight"
    Write-Host "Native runtime: $runtimePath"

    $resolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
        [System.IO.Path]::GetFullPath($OutputDirectory)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $script:RepositoryRoot $OutputDirectory))
    }
    New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
    Assert-NoResidualWorker

    $runs = @(
        [pscustomobject]@{ Name = "clock-16-p2"; Layers = 16; Ratio = 0.815;
            Policy = "clock"; Prefetch = 2; Sequence = 32; Model = "llama2-like";
            ParameterBytes = [uint64]7000563712; ActivationBytes = [uint64]4948224;
            LogicalBytes = [uint64]7005520128; Oversubscribed = $false },
        [pscustomobject]@{ Name = "clock-23-p2"; Layers = 23; Ratio = 1.145;
            Policy = "clock"; Prefetch = 2; Sequence = 32; Model = "llama2-like";
            ParameterBytes = [uint64]9833930752; ActivationBytes = [uint64]4948224;
            LogicalBytes = [uint64]9838887168; Oversubscribed = $true },
        [pscustomobject]@{ Name = "clock-31-p2"; Layers = 31; Ratio = 1.522;
            Policy = "clock"; Prefetch = 2; Sequence = 32; Model = "llama2-like";
            ParameterBytes = [uint64]13072064512; ActivationBytes = [uint64]4948224;
            LogicalBytes = [uint64]13077020928; Oversubscribed = $true },
        [pscustomobject]@{ Name = "lru-31-p2"; Layers = 31; Ratio = 1.522;
            Policy = "lru"; Prefetch = 2; Sequence = 32; Model = "llama2-like";
            ParameterBytes = [uint64]13072064512; ActivationBytes = [uint64]4948224;
            LogicalBytes = [uint64]13077020928; Oversubscribed = $true },
        [pscustomobject]@{ Name = "clock-31-p0"; Layers = 31; Ratio = 1.522;
            Policy = "clock"; Prefetch = 0; Sequence = 32; Model = "llama2-like";
            ParameterBytes = [uint64]13072064512; ActivationBytes = [uint64]4948224;
            LogicalBytes = [uint64]13077020928; Oversubscribed = $true },
        [pscustomobject]@{ Name = "attention-smoke"; Layers = 2; Ratio = 0.1;
            Policy = "clock"; Prefetch = 2; Sequence = 128; Model = "operator-smoke";
            ParameterBytes = [uint64]1333829632; ActivationBytes = [uint64]19792896;
            LogicalBytes = [uint64]1353655296; Oversubscribed = $false }
    )
    $identifierArguments = @()
    if ($IncludeIdentifiers) {
        $identifierArguments = @("--include-identifiers")
    }
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
            "--prefetch-distance", "$($run.Prefetch)",
            "--sdpa-backend", "math",
            "--seed", "0x585652414D503034",
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
            ratio = [double]::Parse(
                $ratioArgument, [Globalization.CultureInfo]::InvariantCulture)
            layers = [int]$run.Layers
            sequence = [int]$run.Sequence
            policy = $run.Policy
            prefetch_distance = [int]$run.Prefetch
            parameter_bytes = $run.ParameterBytes
            activation_bytes = $run.ActivationBytes
            logical_bytes = $run.LogicalBytes
            oversubscribed = [bool]$run.Oversubscribed
            include_identifiers = [bool]$IncludeIdentifiers
        }
        Assert-RunArtifacts -ReportPath $reportPath -TracePath $tracePath `
            -Expected $expected -Hardware $hardware
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $validated[$run.Name] = $report
        $manifestReports += [ordered]@{
            name = $run.Name
            report = $reportPath
            trace = $tracePath
            parameter_bytes = $report.model.parameter_bytes
            logical_bytes = $report.model.logical_bytes
            output_digest = $report.verification.output_digest
            elapsed_ms = [double](($report.execution.region_timings_ms |
                Measure-Object -Sum).Sum)
            prefetch_submitted = $report.cache.prefetch_submitted
        }
    }

    $thirtyOneNames = @("clock-31-p2", "lru-31-p2", "clock-31-p0")
    $thirtyOneDigests = @($thirtyOneNames | ForEach-Object {
        $validated[$_].verification.output_digest
    } | Select-Object -Unique)
    Assert-Condition ($thirtyOneDigests.Count -eq 1) `
        "CLOCK/LRU/prefetch-distance 31-layer output digests differ"
    $thirtyOneGraphs = @($thirtyOneNames | ForEach-Object {
        $validated[$_].graph.hash
    } | Select-Object -Unique)
    Assert-Condition ($thirtyOneGraphs.Count -eq 1) `
        "CLOCK/LRU backend captures differ for the identical 31-layer graph"
    Assert-Condition ($validated["clock-31-p0"].cache.prefetch_submitted -eq 0) `
        "distance-zero comparison submitted prefetch"
    Assert-Condition ($validated["clock-31-p2"].cache.prefetch_submitted -gt 0) `
        "distance-two comparison observed no prefetch"
    Assert-NoResidualWorker

    $manifest = [ordered]@{
        acceptance = "xvram.phase4b.rtx3070"
        generated_at_utc = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
        build_configuration = $Configuration
        runtime_library = $runtimePath
        device = $hardware
        driver_model = "WDDM"
        reports = $manifestReports
        result = "completed"
    }
    $manifestPath = Join-Path $resolvedOutput "acceptance-manifest.json"
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath `
        -Encoding utf8
    Write-Host "Phase 4b RTX 3070 acceptance completed: $manifestPath"
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:XVRAM_TORCH_RUNTIME_LIBRARY = $previousRuntimeLibrary
}
