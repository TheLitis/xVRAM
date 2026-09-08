"""Synthetic, no-driver fixtures; never derive owned identity from an ETW row."""
import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from xvram import compat_audit_nsight_identity as audit
from xvram.compat_audit_owned_identity import PrivateIdentity


IDENTITY = PrivateIdentity(1234567, 130000000000000001, 130000000000000099,
                           10000000, 10, 20, 21, 40, 41, 50)


def row(provider, event, timestamp, **values):
    payload = dict.fromkeys(audit.VERSIONS[provider, event][2], '0')
    if provider == audit.DXG:
        for key in payload:
            if key.startswith('h') or key in ('pDxgAdapter', 'DxgProcess', 'VirtualGpu', 'ParentDxgContext', 'ContextHandle', 'ParentDxgHwQueue'):
                payload[key] = '0x0'
    payload.update(values)
    return dict(provider=provider, event=event, timestamp=timestamp, payload=payload)


def fixture():
    common = dict(ProcessID=str(IDENTITY.native_pid), CreateTime=str(IDENTITY.creation_filetime), ProcessSequenceNumber='987654321')
    result = [row(audit.PROCESS, 1, 100, **common),
              row(audit.PROCESS, 2, 900, **common, ExitTime=str(IDENTITY.exit_filetime), ImageName='private-name.exe\0')]
    for device in range(2):
        fields = dict(hProcessId=hex(IDENTITY.native_pid), hDevice=hex(0xa000+device), DxgProcess='0xbb000')
        result += [row(audit.DXG, 27, 200+device, **fields), row(audit.DXG, 28, 800+device, **dict(fields,DxgProcess='0x0'))]
    for context in range(3):
        fields = dict(hDevice=hex(0xa000+context%2), hContext=hex(0xcc000+context), ContextHandle='0xdd000')
        result += [row(audit.DXG, 30, 300+context, **fields), row(audit.DXG, 31, 700+context, **dict(fields, ContextHandle='0x0'))]
        for queue in range(4):
            fields = dict(hContext=hex(0xcc000+context), ParentDxgHwQueue=hex(0xee000+context*4+queue), hHwQueue='0x0')
            result += [row(audit.DXG, 422, 400+queue, **fields), row(audit.DXG, 423, 600+queue, **fields)]
    return result


def database(path, rows):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript('CREATE TABLE ETW_PROVIDERS(providerId INTEGER PRIMARY KEY,guid TEXT);'
                         'CREATE TABLE GENERIC_EVENT_TYPES(typeId INTEGER PRIMARY KEY,etwProviderId INTEGER,etwGuid TEXT,etwEventId INTEGER,etwVersion INTEGER);'
                         'CREATE TABLE ETW_EVENTS(typeId INTEGER,opcode INTEGER,timestamp INTEGER,rawTimestamp INTEGER,data TEXT);'
                         'CREATE TABLE META_DATA_EXPORT(name TEXT,value TEXT);')
        names = ('PRODUCT_NAME','PRODUCT_VERSION','SCHEMA_VERSION','PLATFORM','ARCH','PARAM_TIME_NORMALIZE','PARAM_TIME_SHIFT','CONFIG_TIME_SHIFT')
        values = ('NVIDIA Nsight Systems','2026.1.3.425','3.24.18','windows-desktop','x64','false','0','0')
        db.executemany('INSERT INTO META_DATA_EXPORT VALUES (?,?)', [('EXPORT_'+key,value) for key,value in zip(names,values)])
        db.executemany('INSERT INTO ETW_PROVIDERS VALUES (?,?)', [(0,audit.PROCESS),(1,audit.DXG)])
        for index, ((provider,event),(version,_,_)) in enumerate(audit.VERSIONS.items(),1):
            db.execute('INSERT INTO GENERIC_EVENT_TYPES VALUES (?,?,?,?,?)', (index,int(provider==audit.DXG),provider,event,version))
        types = {key:index for index,key in enumerate(audit.VERSIONS,1)}
        for item in rows:
            key = item['provider'], item['event']
            db.execute('INSERT INTO ETW_EVENTS VALUES (?,?,?,?,?)', (types[key],audit.VERSIONS[key][1],item['timestamp'],item['timestamp'],json.dumps(item['payload'])))


class NsightIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'fixture.sqlite'

    def analyze(self, rows=None, identity=IDENTITY):
        database(self.path, fixture() if rows is None else rows)
        return audit.analyze(self.path, identity)

    def test_exact_identity_graph_and_privacy(self):
        value = self.analyze()
        self.assertTrue(value['owned_process_exactly_bound'])
        self.assertEqual(value['observed_resource_counts'], dict(devices=2,contexts=3,queues=12))
        self.assertEqual(len(value['normalized_events']),36)
        self.assertFalse(value['common_clock_verified'])
        self.assertFalse(value['full_capture_lifetimes_validated'])
        self.assertFalse(any(value['proof'].values()))
        serialized = json.dumps(value)
        for forbidden in (str(IDENTITY.native_pid),str(IDENTITY.creation_filetime),str(IDENTITY.exit_filetime),
                          '987654321','0xcc000','private-name.exe','timestamp_ticks','hHwQueue','ContextHandle'):
            self.assertNotIn(forbidden, serialized)
        before = self.path.read_bytes()
        audit.analyze(self.path, IDENTITY)
        self.assertEqual(before, self.path.read_bytes())

    def test_preserves_negative_relative_timestamp_without_clock_conversion(self):
        rows = fixture()
        for item in rows:
            item['timestamp'] -= 1000
        result = self.analyze(rows)
        self.assertLess(result['normalized_events'][0]['timestamp_ns'],0)
        self.assertFalse(result['common_clock_verified'])

    def test_exact_decimal_text_not_float_integer_or_name(self):
        for value in (IDENTITY.native_pid, float(IDENTITY.native_pid), '01234567', '1.234567e6'):
            with self.subTest(value_type=type(value).__name__):
                rows = fixture(); rows[0]['payload']['ProcessID'] = value
                self.path.unlink(missing_ok=True)
                with self.assertRaises(ValueError): self.analyze(rows)

    def test_missing_duplicate_and_wrong_generation_fail(self):
        for mutation in ('missing','duplicate','create','exit','sequence','nonzero'):
            with self.subTest(mutation=mutation):
                rows = fixture()
                if mutation == 'missing': rows.pop(1)
                elif mutation == 'duplicate': rows.append(copy.deepcopy(rows[0]))
                else:
                    field = {'create':'CreateTime','exit':'ExitTime','sequence':'ProcessSequenceNumber','nonzero':'ExitCode'}[mutation]
                    rows[1]['payload'][field] = '1'
                self.path.unlink(missing_ok=True)
                with self.assertRaises(ValueError): self.analyze(rows)

    def test_resource_missing_stop_and_wrong_parent_fail(self):
        for mutation in ('missing','wrong_queue','wrong_device','unknown_context_parent','duplicate_stop'):
            with self.subTest(mutation=mutation):
                rows = fixture()
                target = next(item for item in rows if item['event']==423)
                if mutation=='missing': rows.remove(target)
                elif mutation=='wrong_queue': target['payload']['ParentDxgHwQueue']='0x12345'
                elif mutation=='wrong_device': next(item for item in rows if item['event']==31)['payload']['hDevice']='0x12345'
                elif mutation=='unknown_context_parent': next(item for item in rows if item['event']==30)['payload']['ParentDxgContext']='0x12345'
                else: rows.append(copy.deepcopy(target))
                self.path.unlink(missing_ok=True)
                with self.assertRaises(ValueError): self.analyze(rows)

    def test_wrong_provider_version_opcode_and_metadata(self):
        statements = ("UPDATE GENERIC_EVENT_TYPES SET etwGuid='unknown' WHERE etwEventId=1",
                      'UPDATE GENERIC_EVENT_TYPES SET etwVersion=3 WHERE etwEventId=1',
                      'UPDATE ETW_EVENTS SET opcode=9 WHERE typeId=1',
                      "UPDATE META_DATA_EXPORT SET value='true' WHERE name='EXPORT_PARAM_TIME_NORMALIZE'")
        for statement in statements:
            self.path.unlink(missing_ok=True); database(self.path,fixture())
            with closing(sqlite3.connect(self.path)) as db, db: db.execute(statement)
            with self.assertRaises(ValueError): audit.analyze(self.path,IDENTITY)

    def test_unknown_field_embedded_nul_and_duplicate_json_rejected(self):
        for mutation in ('field','nul','double_nul','json'):
            self.path.unlink(missing_ok=True); rows=fixture()
            if mutation=='field': rows[0]['payload']['raw_pointer']='0x12345'
            elif mutation=='nul': rows[1]['payload']['ImageName']='pri\0vate'
            elif mutation=='double_nul': rows[1]['payload']['ImageName']='private\0\0'
            database(self.path,rows)
            if mutation=='json':
                with closing(sqlite3.connect(self.path)) as db, db: db.execute('UPDATE ETW_EVENTS SET data=? WHERE typeId=1',('{"ProcessID":"1","ProcessID":"2"}',))
            with self.assertRaises(ValueError): audit.analyze(self.path,IDENTITY)

    def test_active_sidecar_rejected(self):
        database(self.path,fixture())
        sidecar = Path(str(self.path)+'-wal'); sidecar.touch()
        with self.assertRaisesRegex(ValueError,'live_sidecar'): audit.analyze(self.path,IDENTITY)

    def test_resource_identity_reuse_gets_new_generation(self):
        rows = fixture()
        start = copy.deepcopy(next(item for item in rows if item['event']==422))
        stop = copy.deepcopy(next(item for item in rows if item['event']==423))
        start['timestamp'],stop['timestamp'] = 620,640
        result = self.analyze(rows+[start,stop])
        queues = [item for item in result['normalized_events'] if item['kind']=='queue_start' and item['queue_id']==1]
        self.assertEqual([item['queue_generation'] for item in queues],[1,2])
        self.assertFalse(any(result['proof'].values()))

    def test_foreign_system_null_namespace_rundown_not_owned(self):
        rows=fixture()
        rows.extend((row(audit.DXG,29,-100),row(audit.DXG,32,-90),row(audit.DXG,424,-80)))
        result=self.analyze(rows)
        self.assertEqual(result['observed_resource_counts'],dict(devices=2,contexts=3,queues=12))
        self.assertFalse(result['full_capture_lifetimes_validated'])

    def test_selected_null_identity_is_not_accepted(self):
        for event,field in ((27,'hDevice'),(30,'hContext'),(422,'ParentDxgHwQueue')):
            self.path.unlink(missing_ok=True); rows=fixture()
            next(item for item in rows if item['event']==event)['payload'][field]='0x0'
            with self.assertRaises(ValueError): self.analyze(rows)

    def test_orphan_owned_device_stop_and_resource_aba_fail(self):
        for category in ('orphan','aba'):
            self.path.unlink(missing_ok=True); rows=fixture()
            if category=='orphan':
                extra=copy.deepcopy(next(item for item in rows if item['event']==28))
                extra['payload']['hDevice']='0x12345'
            else:
                extra=copy.deepcopy(next(item for item in rows if item['event']==422))
                extra['timestamp']=450
            with self.assertRaises(ValueError): self.analyze(rows+[extra])

    def test_descendant_is_explicit_unresolved_not_silently_ignored(self):
        rows=fixture()
        rows.append(row(audit.PROCESS,1,500,ProcessID='7654321',CreateTime='130000000000000010',
                        ProcessSequenceNumber='987654322',ParentProcessID=str(IDENTITY.native_pid),
                        ParentProcessSequenceNumber='987654321'))
        result=self.analyze(rows)
        self.assertEqual(result['descendant_processes_observed'],1)
        self.assertIn('owned_descendant_process_graph_not_normalized',result['unresolved'])
        self.assertFalse(result['full_capture_lifetimes_validated'])

    def test_private_type_required(self):
        database(self.path,fixture())
        with self.assertRaises(TypeError): audit.analyze(self.path,dict(native_pid=IDENTITY.native_pid))


if __name__ == '__main__':
    unittest.main()
