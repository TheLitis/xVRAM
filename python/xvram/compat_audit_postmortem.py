"""Controller-owned, post-reap observation ledger (not an execution sandbox).

No positive claim may be made from a live snapshot. Callback counters are
updated before file I/O and survive worker termination. They supplement, never
replace, strict trace validation, kernel coverage, and device retirement.
"""
import mmap
import os
import secrets
import struct

FIELDS = ('magic version bytes armed sealed finished callbacks_open apis_open '
          'late_entries errors entered exited gpu_drained activity_drained '
          'census_closed retained_names sampler_disabled activity_callbacks_open '
          'activity_outstanding late_activity').split()
FORMAT = struct.Struct('<' + 'q'*len(FIELDS))
MAGIC = 0x3154504D415658


def decode(data, capture, *, sampled=False):
    if type(sampled) is not bool:
        raise ValueError('postmortem_sampled_boolean')
    if len(data) != FORMAT.size:
        raise ValueError('postmortem_size')
    row = dict(zip(FIELDS, FORMAT.unpack(data)))
    if (row['magic'], row['version'], row['bytes']) != (MAGIC, 1, FORMAT.size):
        raise ValueError('postmortem_header')
    if (capture.get('controller_reaped') is not True or capture.get('process_tree_drained') is not True
            or capture.get('timed_out') is not False or type(capture.get('exit_code')) is not int
            or capture['exit_code'] != 0 or capture.get('errors') != []):
        raise ValueError('postmortem_worker_not_cleanly_reaped')
    for field in ('armed', 'sealed', 'finished', 'gpu_drained', 'activity_drained', 'census_closed'):
        if row[field] != 1:
            raise ValueError('postmortem_' + field)
    for field in ('callbacks_open', 'apis_open', 'late_entries', 'errors',
                  'activity_callbacks_open', 'activity_outstanding', 'late_activity'):
        if row[field] != 0:
            raise ValueError('postmortem_' + field)
    if not 0 < row['entered'] == row['exited'] <= 2_000_000:
        raise ValueError('postmortem_api_reconciliation')
    if not 0 <= row['retained_names'] <= 65536 or row['sampler_disabled'] != int(sampled):
        raise ValueError('postmortem_sampler')
    if not sampled and row['retained_names']:
        raise ValueError('postmortem_unexpected_names')
    return {key: value for key, value in row.items() if key not in ('magic', 'bytes')}


class Ledger:
    """Mapping stays owned by the controller until its bounded worker is reaped."""
    def __init__(self):
        if os.name != 'nt':
            raise OSError('postmortem_windows_only')
        self.name = 'Local\\xvram-audit-' + secrets.token_hex(24)
        self.mapping = mmap.mmap(-1, FORMAT.size, tagname=self.name, access=mmap.ACCESS_WRITE)
        self.mapping[:] = FORMAT.pack(MAGIC, 1, FORMAT.size, *([0]*(len(FIELDS)-3)))

    def environment(self):
        return {'XVRAM_AUDIT_POSTMORTEM': self.name}

    def snapshot_after_reap(self, capture, *, sampled=False):
        # decode verifies reap BEFORE interpreting the counters as evidence.
        return decode(self.mapping[:], capture, sampled=sampled)

    def close(self):
        self.mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
