import os
import tempfile
import unittest

from leasebatch.errors import SpecError
from leasebatch.spec import load_spec_dir


def _write(directory, name, content):
    with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
        handle.write(content)


class SpecValidationTest(unittest.TestCase):
    def test_loads_sample_spec(self):
        spec = load_spec_dir(os.path.join(os.path.dirname(__file__),
                                          "..", "samples"))
        self.assertEqual([w.worker_id for w in spec.workers], ["w1", "w2"])
        self.assertEqual(len(spec.requests), 6)
        self.assertEqual(spec.lease_policy.global_max_active, 2)

    def _minimal(self, directory, *, workers=None, requests=None):
        workers = workers or (
            '[{"worker_id":"w1","host":"127.0.0.1","port":19201,'
            '"model_dim":2,"hidden_dim":2,"slot_cap":1}]'
        )
        requests = requests or (
            '{"request_id":"q1","arrival":1,"worker_id":"w1"}\n'
        )
        _write(directory, "workers.json", workers)
        _write(directory, "requests.ndjson", requests)
        _write(directory, "lease_policy.json",
               '{"lease_duration_ms":10,"global_max_active":1}')
        _write(directory, "merge_rules.json",
               '{"default_param_version":0}')

    def test_rejects_non_loopback_host_with_line_and_field(self):
        with tempfile.TemporaryDirectory() as directory:
            self._minimal(
                directory,
                workers='[{"worker_id":"w1","host":"10.0.0.1",'
                        '"port":19201,"model_dim":2,"hidden_dim":2,'
                        '"slot_cap":1}]',
            )
            with self.assertRaises(SpecError) as caught:
                load_spec_dir(directory)
            self.assertEqual(caught.exception.code, "E_SPEC")
            self.assertEqual(caught.exception.field, "host")
            self.assertEqual(caught.exception.line, 1)

    def test_rejects_duplicate_port_and_reports_first_offender(self):
        with tempfile.TemporaryDirectory() as directory:
            self._minimal(
                directory,
                workers=(
                    '[{"worker_id":"w1","host":"127.0.0.1","port":19201,'
                    '"model_dim":2,"hidden_dim":2,"slot_cap":1},'
                    '{"worker_id":"w2","host":"127.0.0.1","port":19201,'
                    '"model_dim":2,"hidden_dim":2,"slot_cap":1}]'
                ),
            )
            with self.assertRaises(SpecError) as caught:
                load_spec_dir(directory)
            self.assertEqual(caught.exception.field, "port")

    def test_rejects_negative_slot_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            self._minimal(
                directory,
                workers='[{"worker_id":"w1","host":"127.0.0.1","port":19201,'
                        '"model_dim":2,"hidden_dim":2,"slot_cap":-1}]',
            )
            with self.assertRaises(SpecError) as caught:
                load_spec_dir(directory)
            self.assertEqual(caught.exception.field, "slot_cap")

    def test_rejects_duplicate_request_with_ndjson_line(self):
        with tempfile.TemporaryDirectory() as directory:
            self._minimal(
                directory,
                requests=(
                    '{"request_id":"q1","arrival":1,"worker_id":"w1"}\n'
                    '{"request_id":"q1","arrival":2,"worker_id":"w1"}\n'
                ),
            )
            with self.assertRaises(SpecError) as caught:
                load_spec_dir(directory)
            self.assertEqual(caught.exception.field, "request_id")
            self.assertEqual(caught.exception.line, 2)

    def test_json_array_line_numbers_are_tracked(self):
        with tempfile.TemporaryDirectory() as directory:
            self._minimal(
                directory,
                workers=(
                    "[\n"
                    '  {"worker_id":"w1","host":"127.0.0.1","port":19201,'
                    '"model_dim":2,"hidden_dim":2,"slot_cap":1},\n'
                    '  {"worker_id":"w2","host":"127.0.0.1","port":70000,'
                    '"model_dim":2,"hidden_dim":2,"slot_cap":1}\n'
                    "]"
                ),
            )
            with self.assertRaises(SpecError) as caught:
                load_spec_dir(directory)
            self.assertEqual(caught.exception.field, "port")
            self.assertEqual(caught.exception.line, 3)


if __name__ == "__main__":
    unittest.main()
