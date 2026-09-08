"""Validate normalized device-allocation observations, never tensor bounds.

This diagnostic reconstructs allocation/mapping lifetimes and joins every
observed memory API to the independently validated Driver API census. It does
not establish GPU retirement, host-pinned coverage, or end-of-process closure.
"""
from pathlib import Path
import re

from .compat_audit_identity import records, digest, _fields, _uint
from .compat_launch_census import analyze as analyze_census

CAP = 256 * 1024 * 1024
CENSUS_CAP = 1024 * 1024 * 1024
MAX_RECORDS = 4_000_000
MAX_API_ID = 2_000_000
IDENTITY = 'object_id generation bytes'
MAPPING = 'allocation_id generation offset_bytes bytes mapping_id physical_id physical_generation physical_offset_bytes'
RESOLUTION = 'allocation_id generation offset_bytes allocation_bytes known mapped readable writable'
SUMMARY = ('errors api_pairs failed_apis open_calls revision live_allocations live_reservations '
           'live_handles live_mappings allocation_bytes reservation_bytes physical_bytes mapped_bytes '
           'tensor_bounds_proven terminal_complete')
NUMERIC = set((IDENTITY + ' ' + MAPPING + ' ' + SUMMARY +
               ' api_id result revision descriptor_count device flags allocation_bytes '
               'source_allocation_id source_generation source_offset_bytes source_allocation_bytes').split())
NUMERIC -= {'tensor_bounds_proven', 'terminal_complete'}
NUMERIC.add('null_free_noops')
NUMERIC.update(('mapping_count', 'index'))
ALLOC = {'cuMemAlloc_v2', 'cuMemAddressReserve'}
FREE = {'cuMemFree_v2', 'cuMemAddressFree'}
ERROR_METADATA = ('category symbol callback_stage parameters_available input_kind null_input known_identity '
                  'allocation_id generation offset_bytes allocation_bytes revision')


def error_details(row):
    """Return normalized first-failure details, without accepting its trace.

Older v1 error rows lack input diagnostics and remain readable; this never
turns an incomplete observer into a successful memory or coverage proof.
"""
    has_metadata = 'category' in row
    _fields(row, 'api_id reason ' + (ERROR_METADATA if has_metadata else ''))
    _uint(row['api_id'], 1, MAX_API_ID)
    if row['kind'] != 'error' or row['reason'] != 'observer_state_or_unsupported_api':
        raise ValueError('memory_error_framing')
    if not has_metadata:
        return dict(api_id=row['api_id'], category='memory_unspecified_observer_failure')
    if (not isinstance(row['category'], str) or not re.fullmatch(r'memory_[a-z_]{1,96}', row['category'])
            or not isinstance(row['symbol'], str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,255}', row['symbol'])
            or row['callback_stage'] not in ('enter', 'exit', 'unknown')
            or row['input_kind'] not in ('none', 'address', 'handle')):
        raise ValueError('memory_error_metadata')
    for flag in ('parameters_available', 'null_input', 'known_identity'):
        if type(row[flag]) is not bool:
            raise ValueError('memory_error_boolean')
    for name in ('allocation_id', 'generation', 'offset_bytes', 'allocation_bytes', 'revision'):
        _uint(row[name])
    if not row['parameters_available'] and row['input_kind'] != 'none':
        raise ValueError('memory_error_unknown_parameters')
    if row['input_kind'] == 'none' and row['null_input']:
        raise ValueError('memory_error_missing_input')
    if row['known_identity']:
        if (row['input_kind'] != 'address' or row['null_input'] or not row['allocation_id']
                or not row['generation'] or not row['allocation_bytes']
                or row['offset_bytes'] >= row['allocation_bytes']):
            raise ValueError('memory_error_identity')
    elif any(row[name] for name in ('allocation_id', 'generation', 'offset_bytes', 'allocation_bytes')):
        raise ValueError('memory_error_unknown_identity')
    return {name: row[name] for name in ('api_id ' + ERROR_METADATA).split()}


def memory_api(symbol):
    if symbol == 'cuMemFreeHost' or symbol.startswith('cuMemAllocHost'):
        return False
    return symbol.startswith(('cuMemAlloc', 'cuMemFree', 'cuMemAddress', 'cuMemCreate',
                              'cuMemRelease', 'cuMemMap', 'cuMemUnmap', 'cuMemSetAccess',
                              'cuMemImport', 'cuMemExport', 'cuMemRetain', 'cuMemcpy', 'cuMemset'))


def transfer(symbol):
    for base in ('cuMemcpyHtoD', 'cuMemcpyDtoH', 'cuMemcpyDtoD'):
        if symbol in {base + suffix for suffix in ('_v2', '_v2_ptds', 'Async_v2', 'Async_v2_ptsz')}:
            return 'd2d' if base == 'cuMemcpyDtoD' else 'copy'
    for base in ('cuMemsetD8', 'cuMemsetD16', 'cuMemsetD32'):
        if symbol in {base + suffix for suffix in ('_v2', '_v2_ptds', 'Async', 'Async_ptsz')}:
            return 'memset'
    return None


def analyze_rows(rows):
    allocations, physical, mappings, calls = {}, {}, {}, {}
    revision = failed = null_free_noops = last_api = last_object = 0
    descriptors = None
    unmap_segments = None
    unmap_calls = unmapped_mappings = 0
    summary = None

    def interval(offset, size, limit):
        _uint(offset); _uint(size, 1); _uint(limit, 1)
        if offset > limit or size > limit - offset:
            raise ValueError('memory_range_overflow_or_bounds')

    def new_id(value):
        nonlocal last_object
        _uint(value, 1)
        if value <= last_object:
            raise ValueError('memory_object_id_reuse')
        last_object = value

    def owner(row, prefix=''):
        value = allocations.get(_uint(row[prefix + 'allocation_id'], 1))
        if not value or value['generation'] != _uint(row[prefix + 'generation'], 1):
            raise ValueError('memory_unknown_allocation_generation')
        interval(row[prefix + 'offset_bytes'], row['bytes'], value['bytes'])
        return value

    def covered(allocation, offset, size):
        pieces = sorted((m['offset_bytes'], m['offset_bytes'] + m['bytes'])
                        for m in mappings.values() if m['allocation_id'] == allocation)
        cursor = offset
        for lo, hi in pieces:
            if lo > cursor:
                break
            if hi > cursor:
                cursor = hi
            if cursor >= offset + size:
                return True
        return False

    def resolution(row, prefix=''):
        value = owner(row, prefix)
        if row[prefix + 'allocation_bytes'] != value['bytes']:
            raise ValueError('memory_allocation_size_changed')
        for flag in ('known', 'mapped', 'readable', 'writable'):
            if type(row[prefix + flag]) is not bool:
                raise ValueError('memory_resolution_boolean')
        if not row[prefix + 'known'] or not row[prefix + 'mapped']:
            raise ValueError('memory_transfer_unknown')
        if value['reservation'] and not covered(row[prefix + 'allocation_id'],
                                               row[prefix + 'offset_bytes'], row['bytes']):
            raise ValueError('memory_transfer_unmapped')

    def retire_physical(key):
        if not physical[key]['retained'] and not any(m['physical_id'] == key for m in mappings.values()):
            del physical[key]

    for number, row in enumerate(rows, 1):
        if (row.get('schema_version') != 1 or type(row.get('schema_version')) is not int
                or row.get('record_type') != 'xvram.cuda_memory_witness'
                or type(row.get('sequence')) is not int or row['sequence'] != number
                or number > MAX_RECORDS or summary is not None):
            raise ValueError('memory_framing_or_after_summary')
        kind = row.get('kind')
        for name in NUMERIC.intersection(row):
            _uint(row[name])
        if number == 1 and kind != 'session':
            raise ValueError('memory_session_missing')
        if kind == 'session':
            _fields(row, 'scope host_pinned_observed tensor_bounds_proven terminal_complete')
            if (number != 1 or row['scope'] != 'device_allocations_and_vmm_only'
                    or any(row[k] is not False for k in ('host_pinned_observed', 'tensor_bounds_proven', 'terminal_complete'))):
                raise ValueError('memory_session_claim')
            continue
        if kind == 'error':
            details = error_details(row)
            raise ValueError('memory_observer_error:' + details['category'])
        if unmap_segments is not None:
            if kind != 'unmap_segment':
                raise ValueError('memory_missing_unmap_segments')
            _fields(row, 'api_id index revision ' + MAPPING)
            operation, expected, index = unmap_segments
            if (row['api_id'] != operation['api_id'] or row['revision'] != revision or row['index'] != index
                    or any(row[key] != expected[index][key] for key in MAPPING.split())):
                raise ValueError('memory_unmap_segment_join')
            if index+1 == len(expected):
                # Commit the reconstruction only after every exact segment is
                # present, ordered and bound to the single native API result.
                for segment in expected:
                    del mappings[segment['mapping_id']]
                    retire_physical(segment['physical_id'])
                unmap_segments = None
            else:
                unmap_segments = operation, expected, index+1
            continue
        if descriptors is not None:
            if kind != 'access_descriptor':
                raise ValueError('memory_missing_access_descriptors')
            _fields(row, 'api_id allocation_id generation offset_bytes bytes device flags revision')
            operation, remaining, devices = descriptors
            for name in ('api_id', 'allocation_id', 'generation', 'offset_bytes', 'bytes', 'revision'):
                if row[name] != operation[name]:
                    raise ValueError('memory_access_descriptor_join')
            device = _uint(row['device'], maximum=2**31-1)
            if device in devices or type(row['flags']) is not int or row['flags'] not in (0, 1, 3):
                raise ValueError('memory_access_descriptor_device_or_flags')
            devices.add(device)
            descriptors = (operation, remaining - 1, devices) if remaining > 1 else None
            continue
        if kind == 'summary':
            _fields(row, SUMMARY + (' null_free_noops' if 'null_free_noops' in row else ''))
            for name in SUMMARY.split():
                if name not in ('tensor_bounds_proven', 'terminal_complete'):
                    _uint(row[name])
            if (row['errors'] or row['open_calls'] or row['tensor_bounds_proven'] is not False
                    or row['terminal_complete'] is not False):
                raise ValueError('memory_incomplete_or_false_claim')
            expected = dict(api_pairs=len(calls), failed_apis=failed, revision=revision,
                            live_allocations=sum(not a['reservation'] for a in allocations.values()),
                            live_reservations=sum(a['reservation'] for a in allocations.values()),
                            live_handles=sum(p['retained'] for p in physical.values()),
                            live_mappings=len(mappings),
                            allocation_bytes=sum(a['bytes'] for a in allocations.values() if not a['reservation']),
                            reservation_bytes=sum(a['bytes'] for a in allocations.values() if a['reservation']),
                            physical_bytes=sum(p['bytes'] for p in physical.values()),
                            mapped_bytes=sum(m['bytes'] for m in mappings.values()))
            if any(row[key] != value for key, value in expected.items()):
                raise ValueError('memory_summary_reconciliation')
            if row.get('null_free_noops', 0) != null_free_noops:
                raise ValueError('memory_null_free_reconciliation')
            expected['null_free_noops'] = null_free_noops
            expected['unmap_calls'] = unmap_calls
            expected['unmapped_mappings'] = unmapped_mappings
            summary = expected
            continue
        if kind != 'operation':
            raise ValueError('memory_unknown_record')
        api = _uint(row.get('api_id'), 1, MAX_API_ID)
        symbol, result = row.get('symbol'), _uint(row.get('result'), maximum=2**31-1)
        if api <= last_api or not isinstance(symbol, str) or not memory_api(symbol):
            raise ValueError('memory_api_identity')
        last_api = api
        calls[api] = symbol, result
        base = 'api_id symbol result revision'
        if result:
            _fields(row, base)
            failed += 1
        elif symbol in ALLOC:
            _fields(row, base + ' ' + IDENTITY)
            new_id(row['object_id']); _uint(row['generation'], 1); _uint(row['bytes'], 1)
            allocations[row['object_id']] = dict(generation=row['generation'], bytes=row['bytes'],
                                                  reservation=symbol == 'cuMemAddressReserve')
            revision += 1
        elif symbol in FREE:
            if symbol == 'cuMemFree_v2' and 'null_input' in row:
                _fields(row, base + ' null_input no_op')
                if row['null_input'] is not True or row['no_op'] is not True:
                    raise ValueError('memory_null_free_observation')
                null_free_noops += 1
            else:
                _fields(row, base + ' ' + IDENTITY)
                value = allocations.get(row['object_id'])
                if (not value or value['generation'] != row['generation'] or value['bytes'] != row['bytes']
                        or value['reservation'] != (symbol == 'cuMemAddressFree')
                        or any(m['allocation_id'] == row['object_id'] for m in mappings.values())):
                    raise ValueError('memory_free_lifetime')
                del allocations[row['object_id']]; revision += 1
        elif symbol == 'cuMemCreate':
            _fields(row, base + ' ' + IDENTITY + ' device')
            new_id(row['object_id']); _uint(row['generation'], 1); _uint(row['bytes'], 1)
            _uint(row['device'], maximum=2**31-1)
            physical[row['object_id']] = dict(generation=row['generation'], bytes=row['bytes'], retained=True)
            revision += 1
        elif symbol == 'cuMemRelease':
            _fields(row, base + ' ' + IDENTITY)
            value = physical.get(row['object_id'])
            if (not value or not value['retained'] or value['generation'] != row['generation']
                    or value['bytes'] != row['bytes']):
                raise ValueError('memory_handle_lifetime')
            value['retained'] = False; retire_physical(row['object_id']); revision += 1
        elif symbol == 'cuMemUnmap' and 'mapping_count' in row:
            _fields(row, base + ' allocation_id generation offset_bytes bytes mapping_count')
            value = owner(row)
            if not value['reservation']:
                raise ValueError('memory_mapping_requires_reservation')
            count = _uint(row['mapping_count'], 2, 4096)
            lo, hi = row['offset_bytes'], row['offset_bytes']+row['bytes']
            segments = sorted((m for m in mappings.values() if m['allocation_id'] == row['allocation_id']
                               and lo < m['offset_bytes']+m['bytes'] and m['offset_bytes'] < hi),
                              key=lambda item: item['offset_bytes'])
            cursor = lo
            for segment in segments:
                if segment['offset_bytes'] != cursor or segment['generation'] != row['generation']:
                    raise ValueError('memory_compound_unmap_gap_or_partial')
                cursor += segment['bytes']
            if cursor != hi or len(segments) != count:
                raise ValueError('memory_compound_unmap_count_or_tail')
            revision += count
            unmap_calls += 1; unmapped_mappings += count
            unmap_segments = row, segments, 0
        elif symbol in ('cuMemMap', 'cuMemUnmap'):
            _fields(row, base + ' ' + MAPPING)
            value = owner(row)
            if not value['reservation']:
                raise ValueError('memory_mapping_requires_reservation')
            if symbol == 'cuMemMap':
                new_id(row['mapping_id'])
                source = physical.get(row['physical_id'])
                if (not source or not source['retained'] or source['generation'] != row['physical_generation']
                        or row['physical_offset_bytes'] != 0 or row['bytes'] > source['bytes']):
                    raise ValueError('memory_mapping_physical_bounds')
                lo, hi = row['offset_bytes'], row['offset_bytes'] + row['bytes']
                if any(m['allocation_id'] == row['allocation_id'] and lo < m['offset_bytes'] + m['bytes']
                       and m['offset_bytes'] < hi for m in mappings.values()):
                    raise ValueError('memory_mapping_overlap')
                mappings[row['mapping_id']] = {key: row[key] for key in MAPPING.split()}
            else:
                value = mappings.get(row['mapping_id'])
                if not value or any(value[key] != row[key] for key in MAPPING.split()):
                    raise ValueError('memory_unmap_not_exact')
                del mappings[row['mapping_id']]; retire_physical(row['physical_id'])
                unmap_calls += 1; unmapped_mappings += 1
            revision += 1
        elif symbol == 'cuMemSetAccess':
            _fields(row, base + ' allocation_id generation offset_bytes bytes descriptor_count')
            value = owner(row)
            if not value['reservation'] or not covered(row['allocation_id'], row['offset_bytes'], row['bytes']):
                raise ValueError('memory_access_unmapped')
            count = _uint(row['descriptor_count'], 1, 16)
            revision += count
            descriptors = row, count, set()
        elif transfer(symbol):
            _uint(row['bytes'])
            if type(row['zero_bytes']) is not bool or row['zero_bytes'] != (row['bytes'] == 0):
                raise ValueError('memory_zero_transfer')
            fields = base + ' bytes zero_bytes'
            if row['bytes']:
                fields += ' ' + RESOLUTION
                if transfer(symbol) == 'd2d':
                    fields += ' ' + ' '.join('source_' + key for key in RESOLUTION.split())
                _fields(row, fields); resolution(row)
                if transfer(symbol) == 'd2d':
                    resolution(row, 'source_')
            else:
                _fields(row, fields)
        else:
            raise ValueError('memory_unsupported_api')
        if _uint(row['revision']) != revision:
            raise ValueError('memory_revision')
    if summary is None or descriptors is not None or unmap_segments is not None:
        raise ValueError('memory_summary_missing')
    return summary, calls


def analyze(path, census_path, kernel_count):
    before = digest(path, CAP), digest(census_path, CENSUS_CAP)
    counts, calls = analyze_rows(records(path, 'xvram.cuda_memory_witness', cap=CAP, max_records=MAX_RECORDS))
    census = analyze_census(Path(census_path), kernel_count)
    if census['sha256'] != before[1] or any(census['counts'][key] for key in
            ('open_api_calls', 'buffers_outstanding', 'uncorrelated_kernels', 'unmarked_kernels')):
        raise ValueError('memory_census_incomplete')
    observed = {}
    for row in records(census_path, 'xvram.cuda_launch_census', cap=CENSUS_CAP,
                       versions=(2, 3), max_records=MAX_RECORDS):
        if row['kind'] == 'api_exit' and row['domain'] == 'driver' and memory_api(row['symbol']):
            if row['api_id'] in observed:
                raise ValueError('memory_census_duplicate_api')
            observed[row['api_id']] = row['symbol'], row['result']
    if calls != observed:
        raise ValueError('memory_census_api_coverage')
    if before != (digest(path, CAP), digest(census_path, CENSUS_CAP)):
        raise ValueError('memory_inputs_changed')
    return dict(counts=counts, provenance=dict(memory_trace=before[0], census_trace=before[1]),
                memory_bounds=False, cubin_binding=False, execution_order=False, terminal_complete=False,
                host_pinned_observed=False)
