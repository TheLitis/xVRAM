# Capability report schema

`xvram-probe` emits UTF-8 JSON described by
[`schemas/capability-report-v1.schema.json`](../schemas/capability-report-v1.schema.json).

The top-level `schema_version` is an integer major version. Consumers must reject an
unknown major version. Additive fields require a schema update and consumer review;
renames or semantic changes require a new major version.

Machine-readable quantities are raw units:

- memory and granularity are bytes;
- CUDA clock attributes are kHz as returned by the Driver API;
- transfer rates are GiB/s in the current pre-release benchmark object;
- every transfer result includes the selected CUDA device ordinal;
- unavailable observations are `null`, not the strings `N/A` or `unsupported`.

On Windows, `driver_model` is the current NVML mode. The `tcc` report value corresponds
to NVML's legacy `NVML_DRIVER_WDM` enum; `pending_driver_model` is the post-reboot target
and never overrides decisions based on the current mode.

`attributes.host_numa_id` is CUDA's closest host NUMA node for the device. A value of
`-1` means that the host does not expose NUMA; in that case no synthetic node `0` is
invented and the HOST_NUMA granularity observation is omitted.

The default report is sanitized, not anonymous. GPU UUID, CUDA LUID, DXGI LUID, and PCI
location fields are used internally for adapter correlation and then removed or
serialized as `null`.
`--include-identifiers` opts into those hardware/topology identifiers. Reports still
contain a timestamp, exact OS and driver versions, GPU model, RAM quantities, and live
DXGI budgets. Review a report before sharing it. The probe never includes a username,
hostname, command line, or environment variables.

See [`PRIVACY.md`](../PRIVACY.md) for the complete disclosure boundary.

Unsupported capabilities and optional query failures do not invalidate the report. They
are represented by `null` fields and structured diagnostics. Per-device diagnostics
carry `device_ordinal`, so multi-GPU failures remain attributable. CLI I/O or
serialization failure is distinct and returns a non-zero process exit code.
