# Privacy

`xvram-probe` is a local diagnostic tool. It does not upload reports or telemetry.

By default, the report redacts GPU UUID, CUDA LUID, DXGI LUID, and both textual and
numeric PCI location fields after they have been used locally to correlate CUDA, NVML,
and DXGI observations. The report does
not collect usernames, hostnames, command lines, or environment variables.

The default report is still not anonymous. It contains a UTC timestamp, exact operating
system and NVIDIA driver versions, compiler/build information, GPU model and
capabilities, system-memory quantities, and current process video-memory budgets. This
combination may identify or fingerprint a machine. Review every report before sharing
it publicly.

`--include-identifiers` deliberately includes the stable hardware and topology
identifiers listed above. Use that option only for private local correlation or when a
trusted maintainer specifically needs it.

xVRAM currently has no telemetry endpoint and performs no background network requests.
At configure time, CMake may download hash-pinned official NVIDIA header archives when
CUDA 13.x and NVML development headers are not installed. Disable this explicitly with
`-DXVRAM_FETCH_CUDA_HEADERS=OFF` for an offline build.
