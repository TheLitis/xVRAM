"""Offline module/lookup evidence. Never infer the missing native/CUPTI bridge."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from .compat_audit_analysis import AuditInputError, _trace_records, MAX_TRACE_BYTES, MAX_STATE_ITEMS
from .compat_audit_binary import _file_bytes, _file_hash, _remove_scratch
from .compat_audit_binary_contract import validate_report as validate_binary
from .compat_audit_bindings_contract import empty_report, validate_report
from .compat_audit_correlations import correlate_trace
from .compat_audit_sources import _json_bytes

def analyze(trace, binary_evidence):
    binary_bytes = _file_bytes(binary_evidence, 4*1024*1024)
    binary = _json_bytes(binary_bytes)
    validate_binary(binary)
    if binary["outcome"]["status"] != "completed":
        raise AuditInputError("binary_evidence_not_completed")
    static = defaultdict(list)
    for module in binary["modules"]:
        static[(module["sha256"],module["size_bytes"])].append(module["module_index"])
    report=empty_report()
    digest=hashlib.sha256()
    modules, by_source, live, generations, hashes, lookups, native_live, links = {}, {}, {}, {}, {}, {}, {}, {}
    issues=set(); counters=Counter(); pending={}
    sequence=0; stamp=0; summary=None; session=False
    for record in _trace_records(trace, digest=digest):
        if record["schema_version"] != 3: raise AuditInputError("bindings_require_trace_v3")
        if record["sequence"] != sequence+1 or record["timestamp_ns"] < stamp:
            raise AuditInputError("non_monotonic_binding_trace")
        if summary is not None: raise AuditInputError("records_after_binding_summary")
        sequence=record["sequence"]; stamp=record["timestamp_ns"]
        kind=record["kind"]
        if sequence==1 and kind!="session": raise AuditInputError("missing_binding_session")
        if sum(map(len,(modules,by_source,live,generations,hashes,lookups,native_live,links,pending)))>MAX_STATE_ITEMS:
            raise AuditInputError("binding_state_limit")
        if kind=="session":
            if session: raise AuditInputError("duplicate_binding_session")
            session=True
        elif kind=="summary": summary=record
        elif kind=="gap": issues.add("collector_gap")
        elif kind=="resource" and record["resource_kind"]=="module":
            base=(record.get("context_id",0),record.get("module_id",0))
            generation=record.get("module_generation",0)
            key=(*base,generation)
            if not generation or not base[1]:
                issues.add("missing_module_lifetime"); continue
            if record["operation"]=="load":
                if base in live or key in modules or generation!=generations.get(base,0)+1:
                    issues.add("contradictory_module_lifetime"); continue
                generations[base]=generation; live[base]=key
                modules[key]={"context_id":base[0],"module_id":base[1],"module_generation":generation,
                              "source_sequence":sequence,"unload_sequence":None,"size_bytes":record.get("size_bytes",0),
                              "sha256":None,"static_module_indices":[]}
                by_source[sequence]=key
                if not record.get("module_copy_queued"): issues.add("module_copy_not_queued")
                else:
                    counters["queued"]+=1
                    counters["copy_bytes"]+=record.get("size_bytes",0)
            elif record["operation"]=="unload":
                if live.get(base)!=key: issues.add("unmatched_module_unload")
                else:
                    modules[key]["unload_sequence"]=sequence; del live[base]
        elif kind=="module_hash":
            counters["hash_records"]+=1
            key=(record["context_id"],record["module_id"],record["module_generation"])
            if by_source.get(record["source_sequence"])!=key or key in hashes or key not in modules:
                issues.add("unmatched_or_duplicate_module_hash"); continue
            hashes[key]=record["sha256"]
            if modules[key]["size_bytes"]!=record["size_bytes"]:
                issues.add("module_hash_size_mismatch"); continue
            modules[key]["sha256"]=record["sha256"]
            modules[key]["static_module_indices"]=sorted(static.get((record["sha256"],record["size_bytes"]),[]))
        elif kind=="activity" and record["activity_kind"]=="function":
            counters["function_activities"]+=1
            # These records have no correlated launch and no proven generation.
        elif kind in ("api_enter","api_exit"):
            invocation=(record["thread_id"],record["domain"],record["correlation_id"])
            if kind=="api_enter":
                if invocation in pending: issues.add("duplicate_native_invocation")
                pending[invocation]=record; continue
            entered=pending.pop(invocation,None)
            if entered is None or entered["symbol"]!=record["symbol"]:
                issues.add("unpaired_native_invocation"); continue
            if record["domain"]!="driver" or record["status"]!=0: continue
            context=record.get("context_id",0); module=record.get("module_id",0); gen=record.get("module_generation",0)
            symbol=record["symbol"]
            if symbol in ("cuModuleLoad","cuModuleLoadData","cuModuleLoadDataEx","cuModuleLoadFatBinary"):
                if not context or not module or not gen or (context,module) in native_live:
                    issues.add("invalid_native_module_load")
                else: native_live[(context,module)]=gen
            elif symbol=="cuModuleUnload":
                if native_live.pop((context,module),None)!=gen: issues.add("invalid_native_module_unload")
            elif symbol=="cuModuleGetFunction":
                counters["native_function_lookups"]+=1
                function=record.get("function_id",0); fg=record.get("function_generation",0)
                if not function or not fg or not gen or native_live.get((context,module))!=gen:
                    issues.add("lookup_without_native_module_lifetime"); continue
                if entered.get("module_id")!=module or entered.get("module_generation")!=gen or entered.get("kernel_name")!=record.get("kernel_name"):
                    issues.add("lookup_input_changed"); continue
                lookups[(context,function,fg)]=(module,gen,sequence,record.get("kernel_name"))
            elif record.get("op")=="launch":
                function=record.get("function_id",0); fg=record.get("function_generation",0)
                identity=(context,function,fg); lookup=lookups.get(identity)
                if lookup is None: continue
                if lookup[:2]!=(module,gen) or native_live.get((context,module))!=gen:
                    issues.add("launch_lookup_lifetime_mismatch"); continue
                if lookup[3]!=record.get("kernel_name"):
                    issues.add("launch_lookup_name_mismatch"); continue
                if any(entered.get(k)!=record.get(k) for k in ("function_id","function_generation","module_id","module_generation")):
                    issues.add("launch_identity_changed"); continue
                if identity not in links:
                    if len(links)>=10000: raise AuditInputError("native_link_limit")
                    links[identity]={"context_id":context,"function_id":function,"function_generation":fg,
                                     "module_id":module,"module_generation":gen,"lookup_sequence":lookup[2],
                                     "first_launch_sequence":sequence,"last_launch_sequence":sequence,"launches":0}
                links[identity]["last_launch_sequence"]=sequence; links[identity]["launches"]+=1
    if not session: raise AuditInputError("empty_binding_trace")
    if pending: issues.add("unclosed_native_invocation")
    if summary is None: issues.add("missing_terminal_summary")
    else:
        if (summary["module_copies_submitted"]!=counters["queued"] or
                summary["module_hash_records"]!=counters["hash_records"] or
                summary["module_copies_retired"]!=summary["module_hash_records"] or
                summary["module_copies_submitted"]!=summary["module_copies_retired"] or
                summary["module_copy_total_bytes"]!=counters["copy_bytes"] or summary["module_copy_active_bytes"]):
            issues.add("incomplete_module_hash_accounting")
    correlation=correlate_trace(trace)
    issues.update(item["code"] for item in correlation["issues"]+correlation["trace_issues"])
    if _file_hash(trace,MAX_TRACE_BYTES)!=digest.hexdigest() or _file_hash(binary_evidence,4*1024*1024)!=hashlib.sha256(binary_bytes).hexdigest():
        raise AuditInputError("binding_input_changed")
    report["provenance"]={"trace_sha256":digest.hexdigest(),"binary_evidence_sha256":hashlib.sha256(binary_bytes).hexdigest()}
    report["modules"]=list(modules.values())
    report["native_links"]=list(links.values()) if correlation["launch_records_reconciled"] else []
    report["coverage"].update(gpu_activities=correlation["gpu_kernel_activities"],
        correlated_gpu_activities=correlation["correlated_gpu_activities"], loaded_modules=len(modules),
        hashed_modules=sum(m["sha256"] is not None for m in modules.values()),
        static_module_matches=sum(bool(m["static_module_indices"]) for m in modules.values()),
        function_activities=counters["function_activities"],native_function_lookups=counters["native_function_lookups"],
        native_launch_links=sum(link["launches"] for link in report["native_links"]))
    report["integrity"]={"input_valid":not issues,"launch_records_reconciled":correlation["launch_records_reconciled"],
                         "trace_complete":correlation["trace_complete"] and not issues,"issues":sorted(issues)}
    report["outcome"]={"status":"completed","exit_code":0}
    validate_report(report,allow_pending_cleanup=True)
    return report

def _worker(config_path):
    report=empty_report()
    try:
        config=_json_bytes(_file_bytes(config_path,1024*1024))
        report=analyze(Path(config["trace"]),Path(config["binary_evidence"]))
    except (OSError,ValueError):
        report["outcome"]={"status":"rejected","exit_code":23}
        report["diagnostics"]=["binding_input_rejected"]
    except Exception:
        report["diagnostics"]=["binding_worker_failed"]
    validate_report(report,allow_pending_cleanup=True)
    with Path(config_path).with_name("result.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=True,allow_nan=False)
    return report["outcome"]["exit_code"]

def main(argv=None):
    from .compat_audit import _Parser, clean_capture_environment
    from .compat_audit_capture import run_process
    args=list(sys.argv[1:] if argv is None else argv)
    if len(args)==2 and args[0]=="--worker": return _worker(Path(args[1]))
    parser=_Parser(description=__doc__)
    parser.add_argument("--trace",type=Path,required=True)
    parser.add_argument("--binary-evidence",type=Path,required=True)
    parser.add_argument("--json",type=Path,required=True,help="new report path; never overwrite")
    parser.add_argument("--timeout-seconds",type=int,choices=range(1,901),default=300,metavar="N")
    options=parser.parse_args(args)
    if options.json.exists(): return 74
    report=empty_report(); scratch=None; safe=False
    try:
        scratch=Path(tempfile.mkdtemp(prefix="xvram-binary-audit-bindings-"))
        config=scratch/"plan.json"
        with config.open("x",encoding="utf-8") as stream:
            json.dump({"trace":str(options.trace.absolute()),"binary_evidence":str(options.binary_evidence.absolute())},stream)
        capture=run_process([sys.executable,"-m","xvram.compat_audit_bindings","--worker",str(config)],
            output_dir=scratch/"control",cwd=scratch,timeout_seconds=options.timeout_seconds,
            environment=clean_capture_environment())
        cleanup={"worker_reaped":capture["controller_reaped"],"process_tree_drained":capture["process_tree_drained"],"scratch_removed":False}
        safe=cleanup["worker_reaped"] and cleanup["process_tree_drained"]
        if capture["timed_out"]:
            report["outcome"]={"status":"timeout","exit_code":26}; report["diagnostics"]=["binding_worker_timeout"]
        else:
            try:
                candidate=_json_bytes(_file_bytes(scratch/"result.json",4*1024*1024))
                validate_report(candidate,allow_pending_cleanup=True)
                if candidate["outcome"]["exit_code"]!=capture["exit_code"]: raise ValueError("exit mismatch")
                report=candidate
            except (OSError,ValueError): report["diagnostics"]=["invalid_binding_worker_result"]
        report["cleanup"]=cleanup
        if not safe or capture.get("output_truncated") or (capture["errors"] and not capture["timed_out"]):
            report["outcome"]={"status":"failed","exit_code":27}; report["diagnostics"].append("binding_containment_failed")
    except (OSError,ValueError):
        report["outcome"]={"status":"failed","exit_code":27}; report["diagnostics"].append("binding_controller_failed")
    finally:
        if scratch is not None and safe:
            try:
                _remove_scratch(scratch,scratch.parent)
                report["cleanup"]["scratch_removed"]=True
            except (OSError,ValueError):
                report["outcome"]={"status":"failed","exit_code":27}; report["diagnostics"].append("binding_cleanup_failed")
    try: validate_report(report)
    except ValueError:
        cleanup=report["cleanup"]; report=empty_report(); report["cleanup"]=cleanup
        report["diagnostics"]=["invalid_binding_final_report"]
    try:
        with options.json.open("x",encoding="utf-8") as stream:
            json.dump(report,stream,ensure_ascii=True,allow_nan=False,indent=2); stream.write("\n")
    except OSError: return 74
    return report["outcome"]["exit_code"]

if __name__=="__main__":
    raise SystemExit(main())
