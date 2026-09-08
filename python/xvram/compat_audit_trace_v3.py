"""Opt-in module evidence; v1/v2 validators remain unchanged."""
from .compat_audit_analysis import AuditInputError, _ENVELOPE, _uint, _SHA256, validate_trace_record

EXTRA = {
    "resource": {"module_generation", "module_copy_queued"},
    "api_enter": {"module_generation", "function_generation"},
    "api_exit": {"module_generation", "function_generation"},
    "activity": {"module_generation", "function_index"},
    "summary": {"module_copies_submitted", "module_copies_retired", "module_copy_active_bytes",
                "module_copy_total_bytes", "module_copy_peak_bytes", "module_hash_records"},
}
HASH_FIELDS = {"context_id", "module_id", "module_generation", "source_sequence", "size_bytes", "sha256"}

def validate_record(record):
    kind = record.get("kind")
    if kind == "module_hash":
        if set(record) != _ENVELOPE | HASH_FIELDS or record.get("report_type") != "xvram.cuda_compat_audit_trace":
            raise AuditInputError("invalid_module_hash_fields")
        for key in ("sequence", "timestamp_ns", "context_id", "module_id", "module_generation", "source_sequence", "size_bytes"):
            _uint(record[key])
        if (not all(record[key] > 0 for key in ("sequence", "module_id", "module_generation", "source_sequence", "size_bytes"))
                or record["source_sequence"] >= record["sequence"] or record["size_bytes"] > 64*1024*1024
                or not isinstance(record["sha256"], str) or not _SHA256.fullmatch(record["sha256"])):
            raise AuditInputError("invalid_module_hash")
        return
    extra = EXTRA.get(kind, set())
    base = {key: value for key, value in record.items() if key not in extra}
    base["schema_version"] = 2
    validate_trace_record(base)
    for key in record.keys() & extra:
        if key == "module_copy_queued":
            if type(record[key]) is not bool:
                raise AuditInputError("invalid_module_copy_boolean")
        else:
            _uint(record[key])
    if kind == "summary":
        if not extra <= record.keys():
            raise AuditInputError("missing_module_summary")
        if (record["module_copy_active_bytes"] > 128*1024*1024 or
                record["module_copy_peak_bytes"] > 128*1024*1024 or
                record["module_copy_total_bytes"] > 512*1024*1024 or
                record["module_copies_retired"] > record["module_copies_submitted"] or
                record["module_hash_records"] > record["module_copies_retired"]):
            raise AuditInputError("invalid_module_summary_bounds")
    elif kind == "resource" and record.keys() & extra:
        if record.get("resource_kind") != "module" or record.get("operation") not in ("load", "unload"):
            raise AuditInputError("unexpected_module_lifetime")
        if "module_copy_queued" in record and record["operation"] != "load":
            raise AuditInputError("unexpected_module_copy")
    elif kind == "activity" and record.keys() & extra:
        if record.get("activity_kind") != "function" or record.get("module_generation") != 0:
            raise AuditInputError("buffered_function_generation_not_proven")
    elif kind in ("api_enter", "api_exit") and record.keys() & extra:
        if record.get("domain") != "driver" or not record.get("module_id"):
            raise AuditInputError("unexpected_native_generation")
