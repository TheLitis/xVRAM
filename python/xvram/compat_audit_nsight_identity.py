"""Private exact-process binding for the pinned Nsight SQLite ETW export.

Only normalized IDs/counts/relative timestamps leave this module. Process names
are never identity evidence. SQLite timestamp/rawTimestamp are not assumed to
be QPC ticks: no conversion, reap timestamp or successful lifetime proof is
manufactured from their numerical proximity to a controller clock.
"""
from collections import defaultdict
from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

from .compat_audit_owned_identity import PrivateIdentity
from .compat_launch_probe import _object

PROCESS = '22fb2cd6-0e7b-422b-a0c7-2fad1fd0e716'
DXG = '802ec45a-1e99-4b83-9920-87c98277ba9d'
MAX_BYTES, MAX_EVENTS = 1024*1024*1024, 1000000
PROCESS_START = 'ProcessID ProcessSequenceNumber CreateTime ParentProcessID ParentProcessSequenceNumber SessionID Flags ProcessTokenElevationType ProcessTokenIsElevated'.split()
PROCESS_STOP = 'ProcessID ProcessSequenceNumber CreateTime ExitTime ExitCode TokenElevationType HandleCount CommitCharge CommitPeak CPUCycleCount ReadOperationCount WriteOperationCount ReadTransferKiloBytes WriteTransferKiloBytes HardFaultCount ImageName'.split()
DEVICE = 'hProcessId pDxgAdapter ClientType hDevice RequestVSync DisableGpuTimeout hThunkHandle DxgProcess HostDeviceHandle VirtualGpu'.split()
CONTEXT = 'hDevice NodeOrdinal EngineAffinity DmaBufferSize DmaBufferSegmentSet DmaBufferPrivateDataSize AllocationListSize PatchLocationListSize Flags hContext ContextHandle ParentDxgContext'.split()
QUEUE = 'hContext hHwQueue ParentDxgHwQueue'.split()
VERSIONS = {(PROCESS, 1): (4, 1, PROCESS_START), (PROCESS, 2): (2, 2, PROCESS_STOP)}
for _event, _version, _opcode, _fields in ((27, 2, 1, DEVICE), (28, 2, 2, DEVICE), (29, 2, 3, DEVICE),
                                        (30, 0, 1, CONTEXT), (31, 0, 2, CONTEXT), (32, 0, 3, CONTEXT),
                                        (422, 0, 1, QUEUE), (423, 0, 2, QUEUE), (424, 0, 3, QUEUE)):
    VERSIONS[DXG, _event] = _version, _opcode, _fields


def _unsigned(value, bits=64, positive=False):
    if type(value) is not int or not int(positive) <= value < 2**bits:
        raise ValueError('nsight_integer_type_or_range')
    return value


def _decimal(value, bits=64, positive=False):
    if type(value) is not str or not re.fullmatch(r'0|[1-9][0-9]{0,19}', value):
        raise ValueError('nsight_identity_requires_exact_decimal_text')
    return _unsigned(int(value), bits, positive)


def _hex(value, positive=False):
    if type(value) is not str or not re.fullmatch(r'0x[0-9a-fA-F]{1,16}', value):
        raise ValueError('nsight_handle_requires_exact_hex_text')
    return _unsigned(int(value, 16), positive=positive)


def _digest(path):
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError('nsight_sqlite_capacity')
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    """Open only an immutable finalized SQLite snapshot, never execute writes."""
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError('nsight_sqlite_file')
    if any(Path(str(path)+suffix).exists() for suffix in ('-wal', '-journal', '-shm')):
        raise ValueError('nsight_sqlite_live_sidecar')
    before = _digest(path)
    connection = sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
    started = time.monotonic()
    connection.set_progress_handler(lambda: int(time.monotonic()-started > 20), 10000)
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('PRAGMA trusted_schema=OFF')
        required = {'ETW_PROVIDERS', 'GENERIC_EVENT_TYPES', 'ETW_EVENTS', 'META_DATA_EXPORT'}
        tables = {name: sql for name, kind, sql in connection.execute('SELECT name,type,sql FROM sqlite_master')
                  if name in required and kind == 'table' and type(sql) is str and not sql.upper().startswith('CREATE VIRTUAL')}
        if set(tables) != required:
            raise ValueError('nsight_sqlite_schema')
        # Only an explicit safe metadata allowlist is read. Capture environment,
        # commands, users and system environment tables must never be loaded.
        names = ('EXPORT_PRODUCT_NAME', 'EXPORT_PRODUCT_VERSION', 'EXPORT_SCHEMA_VERSION', 'EXPORT_PLATFORM', 'EXPORT_ARCH',
                 'EXPORT_PARAM_TIME_NORMALIZE', 'EXPORT_PARAM_TIME_SHIFT', 'EXPORT_CONFIG_TIME_SHIFT')
        metadata = connection.execute('SELECT name,value FROM META_DATA_EXPORT WHERE name IN ('+','.join('?' for _ in names)+')', names).fetchall()
        if len(metadata) != len(names) or len(dict(metadata)) != len(names):
            raise ValueError('nsight_export_metadata')
        metadata = dict(metadata)
        expected = dict(zip(names, ('NVIDIA Nsight Systems', '2026.1.3.425', '3.24.18', 'windows-desktop', 'x64', 'false', '0', '0')))
        if metadata != expected:
            raise ValueError('nsight_export_profile_unreviewed')
        providers = connection.execute('SELECT providerId,guid FROM ETW_PROVIDERS').fetchall()
        selected = {}
        for key, guid in providers:
            if type(guid) is not str:
                raise ValueError('nsight_provider_guid')
            if guid.lower() in (PROCESS, DXG):
                if guid.lower() in selected:
                    raise ValueError('nsight_duplicate_provider')
                selected[guid.lower()] = _unsigned(key)
        if set(selected) != {PROCESS, DXG}:
            raise ValueError('nsight_required_provider_missing')
        if connection.execute('SELECT 1 FROM ETW_EVENTS e LEFT JOIN GENERIC_EVENT_TYPES t ON e.typeId=t.typeId WHERE t.typeId IS NULL LIMIT 1').fetchone():
            raise ValueError('nsight_event_type_missing')
        sql = ('SELECT p.guid,t.etwGuid,t.etwEventId,t.etwVersion,e.opcode,e.timestamp,e.rawTimestamp,e.data '
               'FROM ETW_EVENTS e JOIN GENERIC_EVENT_TYPES t ON t.typeId=e.typeId '
               'JOIN ETW_PROVIDERS p ON p.providerId=t.etwProviderId '
               'WHERE (t.etwProviderId=? AND t.etwEventId IN (1,2)) OR '
               '(t.etwProviderId=? AND t.etwEventId IN (27,28,29,30,31,32,422,423,424)) ORDER BY e.timestamp,e.rowid')
        events, total = [], 0
        for provider, guid, event, version, opcode, timestamp, raw_time, data in connection.execute(sql, (selected[PROCESS], selected[DXG])):
            if len(events) >= MAX_EVENTS or type(data) is not str or len(data) > 65536:
                raise ValueError('nsight_event_capacity')
            total += len(data)
            if total > 256*1024*1024:
                raise ValueError('nsight_payload_capacity')
            provider = provider.lower()
            if type(guid) is not str or guid.lower() != provider:
                raise ValueError('nsight_event_provider_disagreement')
            _unsigned(event, 16); _unsigned(version, 8); _unsigned(opcode, 8)
            if (provider, event) not in VERSIONS or (version, opcode) != VERSIONS[provider, event][:2]:
                raise ValueError('nsight_event_version_or_opcode')
            # Nsight's relative epoch can put pre-capture/rundown events below
            # zero. Preserve the signed timestamps rather than dropping rows.
            if any(type(value) is not int or not -2**63 <= value < 2**63 for value in (timestamp, raw_time)):
                raise ValueError('nsight_timestamp_type_or_range')
            payload = json.loads(data, object_pairs_hook=_object,
                                 parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nsight_payload_nonfinite')))
            if type(payload) is not dict or set(payload) != set(VERSIONS[provider, event][2]):
                raise ValueError('nsight_event_payload_fields')
            if provider == PROCESS and event == 2 and type(payload.get('ImageName')) is str:
                # This ignored descriptive field is a NUL-terminated ETW
                # string, never an input to the process identity join.
                payload['ImageName'] = payload['ImageName'].removesuffix('\0')
            if any(type(value) is not str or len(value) > 4096 or '\0' in value for value in payload.values()):
                raise ValueError('nsight_event_payload_text')
            events.append(dict(provider=provider, event=event, timestamp=timestamp, payload=payload))
    except sqlite3.Error as error:
        raise ValueError('nsight_sqlite_read_or_decode') from error
    finally:
        connection.close()
    if before != _digest(path):
        raise ValueError('nsight_sqlite_changed')
    return before, events


def _bind(events, private):
    if type(private) is not PrivateIdentity:
        raise TypeError('nsight_requires_owned_private_identity')
    _unsigned(private.native_pid, 32, True)
    _unsigned(private.creation_filetime, 63, True); _unsigned(private.exit_filetime, 63, True)
    if private.exit_filetime < private.creation_filetime:
        raise ValueError('nsight_private_process_times')
    process_rows = []
    for row in events:
        if row['provider'] != PROCESS:
            continue
        p = row['payload']
        pid = _decimal(p['ProcessID'], 32, True)
        creation = _decimal(p['CreateTime'], 63, True)
        sequence = _decimal(p['ProcessSequenceNumber'], positive=True)
        if row['event'] == 1:
            _decimal(p['ParentProcessID'], 32); _decimal(p['ParentProcessSequenceNumber'])
        else:
            _decimal(p['ExitTime'], 63, True); _decimal(p['ExitCode'], 32)
        process_rows.append((row, pid, creation, sequence))
    matched = [(row, sequence) for row, pid, creation, sequence in process_rows
               if pid == private.native_pid and creation == private.creation_filetime]
    starts = [(row, seq) for row, seq in matched if row['event'] == 1]
    stops = [(row, seq) for row, seq in matched if row['event'] == 2]
    if len(starts) != 1 or len(stops) != 1:
        raise ValueError('nsight_exact_owned_process_pair_missing_or_ambiguous')
    start, seq = starts[0]; stop, stop_seq = stops[0]
    if (seq != stop_seq or _decimal(stop['payload']['ExitTime'], 63, True) != private.exit_filetime or
            _decimal(stop['payload']['ExitCode'], 32) != 0 or start['timestamp'] >= stop['timestamp']):
        raise ValueError('nsight_owned_process_generation_or_exit_mismatch')
    for row, pid, creation, _ in process_rows:
        if pid == private.native_pid and creation != private.creation_filetime and start['timestamp'] <= row['timestamp'] <= stop['timestamp']:
            raise ValueError('nsight_overlapping_process_generation')
    descendants = sum(row['event'] == 1 and _decimal(row['payload']['ParentProcessID'], 32) == private.native_pid
                      and _decimal(row['payload']['ParentProcessSequenceNumber']) == seq for row, _, _, _ in process_rows)
    return start, stop, descendants


def _pairs(starts, all_starts, all_stops, key):
    """Pair exact raw identities privately, rejecting missing/duplicate stops."""
    result, ids, generations, last_stop = [], {}, defaultdict(int), {}
    start_times, stop_rows = defaultdict(list), defaultdict(list)
    for row in all_starts:
        start_times[key(row)].append(row['timestamp'])
    for row in all_stops:
        stop_rows[key(row)].append(row)
    for values in start_times.values():
        values.sort()
    for values in stop_rows.values():
        values.sort(key=lambda row: row['timestamp'])
    stop_times = {identity: [row['timestamp'] for row in values] for identity, values in stop_rows.items()}
    for start in sorted(starts, key=lambda row: row['timestamp']):
        identity = key(start)
        same = start_times[identity]
        left, right = bisect_left(same, start['timestamp']), bisect_right(same, start['timestamp'])
        if right-left != 1:
            raise ValueError('nsight_duplicate_resource_start')
        upper = same[right] if right < len(same) else 2**63
        times = stop_times.get(identity, [])
        stops = stop_rows[identity][bisect_right(times,start['timestamp']):bisect_left(times,upper)]
        if len(stops) != 1 or (identity in last_stop and start['timestamp'] <= last_stop[identity]):
            raise ValueError('nsight_resource_stop_missing_ambiguous_or_aba')
        if identity not in ids:
            ids[identity] = len(ids)+1
        generations[identity] += 1
        last_stop[identity] = stops[0]['timestamp']
        result.append(dict(start=start, stop=stops[0], id=ids[identity], generation=generations[identity]))
    return result


def _resources(events, private, start, stop):
    groups = defaultdict(list)
    for row in events:
        if row['provider'] == DXG:
            groups[row['event']].append(row)
    if sum(len(rows) for rows in groups.values()) > 65536:
        raise ValueError('nsight_resource_capacity')
    # Foreign/system rundown rows legitimately use a null namespace. Parse
    # them exactly, but require non-null identities only after ownership joins.
    h = lambda row, field: _hex(row['payload'][field])
    def containing(objects, row, field, parent_field):
        return [obj for obj in objects if h(obj['start'], parent_field) == h(row, field)
                and obj['start']['timestamp'] <= row['timestamp'] <= obj['stop']['timestamp']]
    device_starts = [row for row in groups[27] if h(row, 'hProcessId') == private.native_pid
                     and start['timestamp'] <= row['timestamp'] <= stop['timestamp']]
    if any(h(row, 'hProcessId') == private.native_pid and start['timestamp'] <= row['timestamp'] <= stop['timestamp'] for row in groups[29]):
        raise ValueError('nsight_owned_device_rundown_not_birth')
    # DxgProcess is zero at the observed Device Stop, not a lifetime key.
    # Keep the process-qualified hDevice identity and its temporal generation.
    device_key = lambda row: (h(row, 'hDevice'), h(row, 'hProcessId'))
    if any(not all(device_key(row)) for row in device_starts):
        raise ValueError('nsight_owned_device_null_identity')
    devices = _pairs(device_starts, groups[27], groups[28], device_key)
    used_device_stops = {id(obj['stop']) for obj in devices}
    if any(h(row, 'hProcessId') == private.native_pid and
           start['timestamp'] <= row['timestamp'] <= stop['timestamp'] and id(row) not in used_device_stops for row in groups[28]):
        raise ValueError('nsight_orphan_owned_device_stop')
    context_starts = []
    context_owner = {}
    for row in groups[30]+groups[32]:
        owner = containing(devices, row, 'hDevice', 'hDevice')
        if not owner:
            continue
        if len(owner) != 1 or row['event'] == 32:
            raise ValueError('nsight_context_owner_or_rundown')
        if not h(row, 'hContext'):
            raise ValueError('nsight_owned_context_null_identity')
        if _hex(row['payload']['ParentDxgContext']) != 0:
            raise ValueError('nsight_context_parent_namespace_unresolved')
        context_starts.append(row); context_owner[id(row)] = owner[0]
    contexts = _pairs(context_starts, groups[30], groups[31], lambda row: h(row, 'hContext'))
    for context in contexts:
        owner = context_owner[id(context['start'])]; context['owner'] = owner
        if (h(context['stop'], 'hDevice') != h(owner['start'], 'hDevice') or
                _hex(context['stop']['payload']['ParentDxgContext']) != 0 or
                context['stop']['timestamp'] > owner['stop']['timestamp']):
            raise ValueError('nsight_context_stop_owner_or_lifetime')
    queue_starts, queue_owner = [], {}
    for row in groups[422]+groups[424]:
        owner = containing(contexts, row, 'hContext', 'hContext')
        if not owner:
            continue
        if len(owner) != 1 or row['event'] == 424:
            raise ValueError('nsight_queue_owner_or_rundown')
        if not h(row, 'ParentDxgHwQueue'):
            raise ValueError('nsight_owned_queue_null_identity')
        # hHwQueue may be zero even at Start; validate its representation only.
        _hex(row['payload']['hHwQueue'])
        queue_starts.append(row); queue_owner[id(row)] = owner[0]
    queues = _pairs(queue_starts, groups[422], groups[423], lambda row: (h(row, 'hContext'), h(row, 'ParentDxgHwQueue')))
    for queue in queues:
        owner = queue_owner[id(queue['start'])]; queue['owner'] = owner
        _hex(queue['stop']['payload']['hHwQueue'])
        if queue['stop']['timestamp'] > owner['stop']['timestamp']:
            raise ValueError('nsight_queue_stop_after_context')
    for rows, objects, parents, field, parent_field in ((groups[31], contexts, devices, 'hDevice', 'hDevice'),
                                                      (groups[423], queues, contexts, 'hContext', 'hContext')):
        used = {id(obj['stop']) for obj in objects}
        if any(containing(parents, row, field, parent_field) and id(row) not in used for row in rows):
            raise ValueError('nsight_orphan_owned_resource_stop')
    if not devices or not contexts or not queues:
        raise ValueError('nsight_owned_resource_graph_missing')
    normalized = [dict(kind='process_start', timestamp_ns=start['timestamp'], process_id=1, process_generation=1,
                       parent_process_id=0, parent_process_generation=0),
                  dict(kind='process_stop', timestamp_ns=stop['timestamp'], process_id=1, process_generation=1, exit_code=0)]
    for category, objects in (('device', devices), ('context', contexts), ('queue', queues)):
        for obj in objects:
            fields = {category+'_id': obj['id'], category+'_generation': obj['generation']}
            if category == 'device':
                fields.update(process_id=1, process_generation=1)
            else:
                parent = 'device' if category == 'context' else 'context'
                fields.update({parent+'_id': obj['owner']['id'], parent+'_generation': obj['owner']['generation'],
                               'parent_'+category+'_id': 0, 'parent_'+category+'_generation': 0})
            for kind in ('start', 'stop'):
                normalized.append(dict(kind=category+'_'+kind, timestamp_ns=obj[kind]['timestamp'], **fields))
    normalized.sort(key=lambda row: row['timestamp_ns'])
    for sequence, row in enumerate(normalized, 1):
        row['sequence'] = sequence
    return normalized, dict(devices=len(devices), contexts=len(contexts), queues=len(queues))


def analyze(sqlite_path, private_identity):
    """Return public-safe partial evidence; no common-clock proof is inferred."""
    digest, events = _read(Path(sqlite_path))
    start, stop, descendants = _bind(events, private_identity)
    normalized, counts = _resources(events, private_identity, start, stop)
    gaps = ['nsight_relative_ns_to_controller_qpc_transform_unverified',
            'etl_loss_counters_and_closed_capture_not_verified_by_sqlite',
            'resource_stop_success_semantics_unproven', 'cupti_channel_lifetime_unproven']
    if descendants:
        gaps.append('owned_descendant_process_graph_not_normalized')
    return dict(observation='owned_process_bound_with_partial_wddm_graph', sqlite_sha256=digest,
                owned_process_exactly_bound=True, process_identity_source='original_owned_popen_handle_and_exact_etw_filetimes',
                process_generations=1, descendant_processes_observed=descendants,
                observed_resource_counts=counts, normalized_events=normalized,
                clock_domain='nsight_sqlite_relative_ns', common_clock_verified=False,
                full_capture_lifetimes_validated=False, unresolved=gaps,
                proof=dict.fromkeys(('memory_bounds', 'cubin_binding', 'device_ordering', 'trace_completeness'), False))
