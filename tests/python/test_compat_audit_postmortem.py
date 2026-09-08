import unittest
from xvram.compat_audit_postmortem import FIELDS, FORMAT, MAGIC, decode


def data(**changes):
    row = dict.fromkeys(FIELDS, 0)
    row.update(magic=MAGIC, version=1, bytes=FORMAT.size, armed=1, sealed=1,
               finished=1, gpu_drained=1, activity_drained=1, census_closed=1,
               entered=42, exited=42)
    row.update(changes)
    return FORMAT.pack(*(row[key] for key in FIELDS))


def capture(**changes):
    return dict(dict(controller_reaped=True, process_tree_drained=True,
                     timed_out=False, exit_code=0, errors=[]), **changes)


class PostmortemTests(unittest.TestCase):
    def test_clean_native_and_sampled_reap(self):
        self.assertEqual(decode(data(), capture())['entered'], 42)
        self.assertEqual(decode(data(retained_names=39, sampler_disabled=1),
                                capture(), sampled=True)['retained_names'], 39)

    def test_live_or_failed_worker_never_proves_cutoff(self):
        for key, value in [('controller_reaped', False), ('process_tree_drained', False),
                           ('timed_out', True), ('exit_code', 27), ('exit_code', False),
                           ('errors', ['unknown'])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode(data(), capture(**{key: value}))

    def test_late_producer_survives_successful_file_footer(self):
        for key in ('late_entries', 'callbacks_open', 'apis_open', 'errors',
                    'activity_callbacks_open', 'activity_outstanding', 'late_activity'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode(data(**{key: 1}), capture())

    def test_partial_initialization_or_teardown(self):
        for key in ('armed', 'sealed', 'finished', 'gpu_drained', 'activity_drained', 'census_closed'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode(data(**{key: 0}), capture())

    def test_header_ranges_reconciliation_and_corruption(self):
        for changes in ({'magic': 1}, {'version': 2}, {'bytes': 0}, {'exited': 43},
                        {'entered': 0, 'exited': 0}, {'entered': 2000001, 'exited': 2000001},
                        {'retained_names': 1}, {'sampler_disabled': 1}, {'errors': -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                decode(data(**changes), capture())
        with self.assertRaises(ValueError):
            decode(data()[:-1], capture())
        with self.assertRaises(ValueError):
            decode(data(sampler_disabled=2), capture(), sampled=2)


if __name__ == '__main__':
    unittest.main()
