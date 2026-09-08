"""Strict sampled identity evidence; sampled coverage is never execution GO.

This parser deliberately accepts no caller-provided 'proven' field. A complete
sample set still cannot authenticate unsampled uses or establish unperturbed
ordering. Native errors and missing footers fail closed, retaining no success.
"""
from collections import Counter
from pathlib import Path
import re

from .compat_audit_identity import records, digest, _fields, _false, _uint
from .compat_launch_census import analyze as analyze_census
from .compat_launch_probe import analyze_trace


def analyze_rows(rows):
    images, pending, samples = {}, [], {}
    counts = Counter(records=0, cubin_observations=0, collections=0,
                     pc_records=0, sampled_calls=0, unsampled_calls=0,
                     dropped_samples=0)
    summary = None
    preflight = enabled = False
    for row in rows:
        if summary is not None:
            raise ValueError('pc_after_summary')
        counts['records'] += 1
        kind = row['kind']
        if counts['records'] == 1 and kind != 'session':
            raise ValueError('pc_session_missing')
        if kind == 'session':
            _fields(row, 'serialized_diagnostic terminal_complete sampling_period hardware_buffer_bytes configured_buffer_pcs result_buffer_pcs')
            _false(row['terminal_complete'])
            if row['serialized_diagnostic'] is not True or counts['records'] != 1:
                raise ValueError('pc_session')
            _uint(row['sampling_period'], 5, 31)
            _uint(row['hardware_buffer_bytes'], 1, 512*1024*1024)
            _uint(row['configured_buffer_pcs'], 1, 4096)
            _uint(row['result_buffer_pcs'], 1, 4096)
        elif kind == 'preflight':
            _fields(row, 'context_present errors copies_submitted')
            if preflight or row['context_present'] is not True or _uint(row['errors']):
                raise ValueError('pc_preflight')
            _uint(row['copies_submitted'], maximum=100000)
            preflight = True
        elif kind == 'enabled':
            _fields(row, 'stall_reasons')
            if not preflight or enabled:
                raise ValueError('pc_enable_order')
            _uint(row['stall_reasons'], 1, 128)
            enabled = True
        elif kind == 'cubin':
            _fields(row, 'cubin_cookie bytes sha256')
            key = _uint(row['cubin_cookie'], 1)
            size = _uint(row['bytes'], 1, 64*1024*1024)
            sha = row['sha256']
            if type(sha) is not str or not re.fullmatch('[0-9a-f]{64}', sha):
                raise ValueError('pc_hash')
            if key in images and images[key] != (sha, size):
                raise ValueError('pc_crc_collision')
            if len(images) >= 4096 and key not in images:
                raise ValueError('pc_image_limit')
            images[key] = sha, size
            counts['cubin_observations'] += 1
        elif kind == 'sample_identity':
            _fields(row, 'observed_after_call correlation_id cubin_cookie function_index name')
            call = _uint(row['observed_after_call'], 1, 100000)
            corr = _uint(row['correlation_id'], 1, 2**32-1)
            key = _uint(row['cubin_cookie'], 1)
            index = _uint(row['function_index'], maximum=2**32-1)
            name = row['name']
            if not enabled or call != counts['collections'] + 1 or key not in images:
                raise ValueError('pc_unbound_or_late_sample')
            if type(name) is not str or not name or len(name.encode('utf-8')) > 4096 or any(ord(c) < 32 for c in name):
                raise ValueError('pc_name')
            item = (corr, images[key][0], index, name)
            if item in pending or len(pending) >= 4096:
                raise ValueError('pc_duplicate_or_excess_samples')
            pending.append(item)
        elif kind == 'collection':
            _fields(row, 'observed_after_call pc_records dropped_samples')
            call = _uint(row['observed_after_call'], 1, 100000)
            pcs = _uint(row['pc_records'], maximum=64*4096)
            dropped = _uint(row['dropped_samples'])
            if not enabled or call != counts['collections'] + 1 or len(pending) > pcs or bool(pcs) != bool(pending):
                raise ValueError('pc_collection_reconciliation')
            # One native launch cannot be silently replaced by several candidate
            # identities, even when their names happen to match.
            if len(pending) > 1:
                raise ValueError('pc_ambiguous_launch')
            samples[call] = pending[0] if pending else None
            counts['collections'] += 1
            counts['pc_records'] += pcs
            counts['dropped_samples'] += dropped
            counts['sampled_calls' if pending else 'unsampled_calls'] += 1
            pending = []
        elif kind == 'summary':
            _fields(row, 'errors copies_submitted copies_retired terminal_complete sampler_cleanup_proven')
            _false(row['terminal_complete'])
            if type(row['sampler_cleanup_proven']) is not bool:
                raise ValueError('pc_cleanup_boolean')
            if (_uint(row['errors']) or pending or not counts['collections']
                    or _uint(row['copies_submitted']) != counts['cubin_observations']
                    or _uint(row['copies_retired']) != counts['cubin_observations']):
                raise ValueError('pc_incomplete_summary')
            summary = row
        else:
            raise ValueError('pc_native_error_or_unknown_record')
    if summary is None or counts['dropped_samples']:
        raise ValueError('pc_incomplete_or_dropped')
    return dict(counts), samples


def correlate(samples, api_rows, launch_names):
    """Join actual driver API instances, never names or timestamps alone."""
    entered, returned, correlations = {}, set(), {}
    for row in api_rows:
        if row.get('domain') != 'driver' or row.get('symbol') != 'cuLaunchKernel':
            continue
        call = _uint(row['probe_call_id'], 1, 100000)
        api = _uint(row['api_id'], 1, 2000000)
        corr = _uint(row['correlation_id'], 1, 2**32-1)
        if row['kind'] == 'api_enter':
            if api in entered or call in correlations:
                raise ValueError('pc_duplicate_native_launch')
            entered[api] = call, corr
            correlations[call] = corr
        elif row['kind'] == 'api_exit':
            if api in returned or entered.get(api) != (call, corr) or row['result'] != 0:
                raise ValueError('pc_native_exit')
            returned.add(api)
    if set(entered) != returned or set(samples) != set(correlations) or set(samples) != set(launch_names):
        raise ValueError('pc_launch_coverage_mismatch')
    used = set()
    result = []
    for call, sample in samples.items():
        if sample is None:
            continue
        corr, sha, index, name = sample
        if correlations[call] != corr or name != launch_names[call] or corr in used:
            raise ValueError('pc_launch_identity_mismatch')
        used.add(corr)
        result.append(dict(call_id=call, cubin_sha256=sha, function_index=index, name=name))
    return result


def analyze(path, census_path, launch_path):
    paths = (path, census_path, launch_path)
    caps = (64*1024*1024, 1024*1024*1024, 128*1024*1024)
    before = [digest(p, cap) for p, cap in zip(paths, caps)]
    counts, samples = analyze_rows(records(path, 'xvram.cuda_pc_witness'))
    launch = analyze_trace(Path(launch_path))
    census = analyze_census(Path(census_path), counts['collections'])
    observed = census['counts']
    if (launch['counts']['calls_returned'] != counts['collections'] or launch['counts']['native_errors']
            or census['sha256'] != before[1]
            or any(observed[key] for key in ('open_api_calls', 'buffers_outstanding', 'unmarked_kernels', 'uncorrelated_kernels'))
            or any(observed[key] != counts['collections'] for key in
                   ('gpu_kernels_observed', 'marked_kernels', 'probe_calls_with_gpu_activity'))):
        raise ValueError('pc_census_incomplete')
    names = {}
    for row in records(launch_path, 'xvram.cuda_launch_probe', cap=caps[2], versions=(2,)):
        if row['kind'] == 'begin':
            call = _uint(row['call_id'], 1, 100000)
            if call in names:
                raise ValueError('pc_duplicate_probe_call')
            names[call] = row['kernel_name']
    bindings = correlate(samples, records(census_path, 'xvram.cuda_launch_census', cap=caps[1],
                                          versions=(2, 3), max_records=4000000), names)
    if before != [digest(p, cap) for p, cap in zip(paths, caps)]:
        raise ValueError('pc_inputs_changed')
    return dict(counts=counts, sampled_bindings=bindings,
                provenance=dict(pc_trace=before[0], census_trace=before[1], launch_trace=before[2]),
                proof=dict(memory_bounds=False, cubin_binding=False, device_ordering=False, trace_completeness=False))
