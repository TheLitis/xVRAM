[CmdletBinding()]
param(
    [string]$BuildDirectory = "build/phase4",
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    [string]$Python = "python",
    [string]$CMake = "cmake"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResolvedBuild = [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $BuildDirectory))

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($Arguments -join ' ')"
    }
}

if (-not (Get-Command $CMake -ErrorAction SilentlyContinue)) {
    $BundledCMake = Get-ChildItem "${env:ProgramFiles}\Microsoft Visual Studio" `
        -Filter cmake.exe -Recurse -ErrorAction SilentlyContinue |
        Where-Object FullName -Match "CommonExtensions\\Microsoft\\CMake\\CMake\\bin" |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $BundledCMake) {
        throw "CMake was not found on PATH or in a Visual Studio installation"
    }
    $CMake = $BundledCMake
}

Invoke-Checked -Executable $Python -Arguments @(
    "-c", "import jsonschema, torch; assert torch.cuda.is_available()"
)
Invoke-Checked -Executable $CMake -Arguments @(
    "-S", $RepositoryRoot, "-B", $ResolvedBuild, "-A", "x64",
    "-DXVRAM_BUILD_TESTS=ON", "-DXVRAM_FETCH_CUDA_HEADERS=OFF",
    "-DXVRAM_WARNINGS_AS_ERRORS=ON"
)
Invoke-Checked -Executable $CMake -Arguments @(
    "--build", $ResolvedBuild, "--config", $Configuration,
    "--target", "xvram_torch_allocator", "xvram-torch-allocator-tests", "--parallel"
)

$CTest = Join-Path (Split-Path $CMake -Parent) "ctest.exe"
if (-not (Test-Path -LiteralPath $CTest)) {
    $CTest = "ctest"
}
Invoke-Checked -Executable $CTest -Arguments @(
    "--test-dir", $ResolvedBuild, "-C", $Configuration, "--output-on-failure",
    "--no-tests=error", "-R", "^xvram[.]torch"
)

$AllocatorLibrary = Join-Path $ResolvedBuild "$Configuration/xvram_torch_allocator.dll"
if (-not (Test-Path -LiteralPath $AllocatorLibrary)) {
    throw "Allocator DLL was not produced at $AllocatorLibrary"
}

$PreviousPythonPath = $env:PYTHONPATH
$PreviousAllocator = $env:XVRAM_TORCH_ALLOCATOR_LIBRARY
try {
    $env:PYTHONPATH = Join-Path $RepositoryRoot "python"
    $env:XVRAM_TORCH_ALLOCATOR_LIBRARY = (Resolve-Path $AllocatorLibrary).Path
    Invoke-Checked -Executable $Python -Arguments @(
        (Join-Path $RepositoryRoot "tests/python/test_torch_allocator_integration.py"), "-v"
    )
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
    $env:XVRAM_TORCH_ALLOCATOR_LIBRARY = $PreviousAllocator
}

Write-Host "Phase 4a PyTorch MemPool acceptance completed successfully."
