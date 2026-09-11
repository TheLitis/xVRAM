# xVRAM Developer Preview

**Run supported CUDA workloads with operand data larger than physical VRAM.**

xVRAM is an experimental Windows-first SDK for CUDA developers. System RAM is the
backing store; a bounded VRAM cache is where GPU work executes. This preview focuses
on the existing explicit, synchronous FP32 CUDA/cuBLAS integration (Phase 6a).
Applications must integrate the SDK. It is **not** a driver replacement or a way to
run any unchanged application, PyTorch model or llama.cpp binary transparently.

## Get a runnable version

Get the **Windows x64 ZIP** and `SHA256SUMS.txt` from
[GitHub Releases](https://github.com/TheLitis/xVRAM/releases). Extract the ZIP to a
new folder. The SDK, CLI tools, app-local cuBLAS/nvCOMP libraries, licenses, CMake
package, example source and report validator are included. NVIDIA's driver is not.
No OpenAI API key, account, service, administrator access or driver-setting changes
are needed to run xVRAM.

Requirements: 64-bit Windows 10/11, an NVIDIA GPU with CUDA VMM support, a driver
compatible with the packaged CUDA 13 libraries, Python 3.10 or newer, and the
Microsoft Visual C++ v14 x64 runtime. The recorded hardware baseline is an RTX 3070
with 8 GB VRAM. Compatibility with every NVIDIA GPU is not claimed. Large runs
need available RAM **in addition to** the logical data size: the runtime reserves
at least 4 GiB or 25% of physical RAM, plus staging/workspace. Insufficient budget
produces an error, not a successful proof.

From the extracted package directory in PowerShell:

```powershell
# Compare the ZIP's SHA256 with the separately published SHA256SUMS.txt.
Get-FileHash -Algorithm SHA256 ..\xvram-preview-2026.09.11.1-windows-x64.zip

# Keep the virtual environment outside the immutable package.
py -3 -m venv ..\xvram-preview-env
..\xvram-preview-env\Scripts\python.exe -m pip install -r requirements-preview.txt
..\xvram-preview-env\Scripts\python.exe preview.py verify
..\xvram-preview-env\Scripts\python.exe preview.py smoke
```

`verify` checks file integrity only; it is **not a GPU test or a signature**. `smoke`
executes the real small padded FP32 suite with numerical, rejection, event/resource
and trace checks. It is deliberately not an oversubscription claim. Never run the
validator with `-O` or `PYTHONOPTIMIZE`: it refuses to disable the existing contract
assertions. The helper does not install software, fetch models or upload results.

For the larger-than-VRAM demonstration, first close other GPU-heavy programs:

```powershell
..\xvram-preview-env\Scripts\python.exe preview.py oversubscribe
```

`auto` requests the benchmark's conservative 1.5x-VRAM sizing; actual rounded
operand bytes are read from the report, not inferred from the request. To request
a particular amount on a machine with sufficient RAM:

```powershell
..\xvram-preview-env\Scripts\python.exe preview.py oversubscribe --logical-size 16GiB
```

Every invocation creates a **new result directory next to the package**. Nothing
is overwritten. Open `report.html` in that directory. It displays a passing GPU
result only after the report schema, established Phase 6a semantic checks, trace,
source revision and process status pass; the large-data mode also verifies real
oversubscription, eviction and reuse. No GPU, missing libraries, timeout, OOM or
incomplete cleanup stay failures/skips, never passes. Large runs may take minutes.
The native controller isolates its worker and enforces its existing deadlines;
this wrapper does not increase TDR or change safety limits.

Keep `report.json`, `trace.jsonl`, `run.json`, stdout/stderr and the package checksum
together. The run manifest hashes result files and records the exact command.
Review before sharing: it contains local paths and hardware/software information.
A reviewer can validate the raw files without a GPU:

```powershell
..\xvram-preview-env\Scripts\python.exe preview.py validate `
  --report ..\my-run\report.json --trace ..\my-run\trace.jsonl `
  --require-oversubscription
```

This validates supplied evidence; it does not independently attest to a GPU run or
authenticate a modified manifest. Use the source/CI/release links for provenance.

## Integrate the SDK

`bin\xvram-preview-gemm.exe --run` is a small standalone example checking every
output against a full CPU FP64 reference. It consumes only installed public C
headers and `xVRAM::cuda_compat`, not private runtime headers or NVIDIA development
headers. It is a small integration example, **not** a large-data benchmark. Its
source is in `examples/preview-gemm`.

To rebuild it, use a Visual Studio developer shell with CMake 3.25+:

```powershell
cmake -S examples\preview-gemm -B ..\xvram-example-build -A x64 `
  "-DCMAKE_PREFIX_PATH=$((Get-Location).Path)"
cmake --build ..\xvram-example-build --config Release
$env:PATH = "$((Get-Location).Path)\bin;$env:PATH"
..\xvram-example-build\Release\xvram-preview-gemm.exe --run
```

The included runtime DLLs must remain discoverable; do not copy just the EXE.
The CUDA-name C++ facade is a separate integration choice and requires official
NVIDIA headers. See `docs/cuda-compat.md` for its exact boundary.

## Build the SDK from source

In the repository, not the downloaded binary package:

```powershell
py -3 -m pip install -r tests\requirements.txt
cmake -S . -B build\preview -A x64 -DXVRAM_BUILD_TESTS=ON `
  -DXVRAM_WARNINGS_AS_ERRORS=ON -DXVRAM_BUILD_CUDA_COMPAT=ON `
  -DXVRAM_FETCH_CUBLAS_REDIST=ON -DXVRAM_BUILD_TORCH_RUNTIME=OFF
cmake --build build\preview --config Release --parallel
ctest --test-dir build\preview -C Release --output-on-failure
cmake --install build\preview --config Release --prefix build\install
cmake -S examples\preview-gemm -B build\example -A x64 `
  "-DCMAKE_PREFIX_PATH=$((Resolve-Path build\install).Path)"
cmake --build build\example --config Release
cmake --install build\example --config Release --prefix build\install
```

First configuration downloads the repository's hash-pinned dependencies. A local
build is not the published release until its provenance and tests are recorded.
Linux source builds remain covered by existing CI; this downloadable package is
Windows x64 only. PyTorch inference and the transparent-compatibility audit are not
part of this package's promised demonstration.

## Scope and evidence

Supported: synchronous host-only integration, managed allocation/copies and positive
column-major FP32 SGEMM with supported N/T layouts, using the existing tiled runtime.
Not supported: arbitrary kernels, training, dynamic-shape general inference, CUDA
Graph capture, native-stream interoperation or unchanged-executable interception.
Compression is disabled for this profile. More system RAM is not GDDR-speed VRAM;
no general speedup or hardware-capacity multiplier is promised.

Read [evidence](https://github.com/TheLitis/xVRAM/blob/main/docs/preview/EVIDENCE.md),
[development with Astra](https://github.com/TheLitis/xVRAM/blob/main/docs/preview/ASTRA.md)
and [the architecture](https://github.com/TheLitis/xVRAM/blob/main/docs/architecture.md).
The SDK remains `0.1.0-dev`; the preview tag identifies a distribution milestone,
not a frozen production ABI. Apache-2.0 applies to xVRAM; included third-party
components retain their own notices and terms.
