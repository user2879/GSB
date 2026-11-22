import unittest

from dns_workbench.errors import FormatError, VersionError
from dns_workbench.records import (
    MAX_TTL,
    STRIP_TO_V1,
    V3_DEFAULTS,
    export_view,
    load_cache_record,
    load_cache_records,
    load_upstream_answer,
)


def v1_record() -> dict:
    return {"version": "v1", "name": "a.test", "qtype": 1, "value": "A 10.0.0.1", "ttl": 60}


class CacheRecordTest(unittest.TestCase):
    def test_three_versions_normalize(self) -> None:
        payloads = [
            v1_record(),
            {**v1_record(), "version": "v2", "rcode": 0, "aa": True, "source": "authoritative"},
            {**v1_record(), "version": "v3", "rcode": 0, "aa": False, "source": "upstream",
             "ecs_prefix": "192.0.2.0/24", "hits": 2, "seq": 7},
        ]
        entries = load_cache_records(payloads)
        self.assertEqual([entry.source_version for entry in entries], ["v1", "v2", "v3"])
        self.assertTrue(all(entry.content_digest for entry in entries))

    def test_unknown_version_is_e_ver(self) -> None:
        with self.assertRaises(VersionError) as ctx:
            load_cache_record({**v1_record(), "version": "v9"})
        self.assertEqual(ctx.exception.fields, ["version"])

    def test_missing_field_is_e_format(self) -> None:
        payload = v1_record()
        del payload["name"]
        with self.assertRaises(FormatError) as ctx:
            load_cache_record(payload)
        self.assertEqual(ctx.exception.fields, ["name"])

    def test_ttl_bounds(self) -> None:
        with self.assertRaises(FormatError):
            load_cache_record({**v1_record(), "ttl": -1})
        with self.assertRaises(FormatError):
            load_cache_record({**v1_record(), "ttl": MAX_TTL + 1})

    def test_sequence_regression_is_e_ver(self) -> None:
        high = {**v1_record(), "version": "v3", "rcode": 0, "aa": False, "source": "upstream",
                "ecs_prefix": None, "hits": 0, "seq": 10}
        low = dict(high, seq=9)
        with self.assertRaises(VersionError) as ctx:
            load_cache_records([high, low])
        self.assertEqual(ctx.exception.fields, ["seq"])

    def test_v1_downgrade_lists_stripped_fields(self) -> None:
        entry = load_cache_record({**v1_record(), "version": "v3", "rcode": 0, "aa": False,
                                   "source": "upstream", "ecs_prefix": None, "hits": 0, "seq": 1})
        view = export_view(entry, "v1", remaining_ttl=33)
        self.assertEqual(view.stripped_fields, tuple(sorted(STRIP_TO_V1)))
        self.assertNotIn("seq", view.record)
        self.assertEqual(view.record["ttl"], 33)

    def test_v3_upgrade_fills_defaults(self) -> None:
        entry = load_cache_record(v1_record())
        view = export_view(entry, "v3", remaining_ttl=33)
        self.assertEqual(view.record["ecs_prefix"], V3_DEFAULTS["ecs_prefix"])
        self.assertEqual(view.record["hits"], 0)
        self.assertEqual(view.record["seq"], 0)
        self.assertEqual(set(view.filled_fields), {"aa", "ecs_prefix", "hits", "rcode", "seq", "source"})

    def test_views_keep_identity_bytes_equal(self) -> None:
        entry = load_cache_record(v1_record())
        v1 = export_view(entry, "v1", remaining_ttl=21)
        v3 = export_view(entry, "v3", remaining_ttl=21)
        self.assertEqual(v1.record["name"], v3.record["name"])
        self.assertEqual(v1.record["qtype"], v3.record["qtype"])
        self.assertEqual(v1.record["ttl"], v3.record["ttl"])


class UpstreamAnswerTest(unittest.TestCase):
    def test_u1_u2(self) -> None:
        u1 = {"format": "u1", "txn_id": 7, "question": {"name": "a.test", "qtype": 1},
              "answers": [{"name": "a.test", "qtype": 1, "value": "A 1.1.1.1", "ttl": 30}]}
        first = load_upstream_answer(u1)
        self.assertEqual(first.format_version, "u1")
        self.assertFalse(first.edns)
        u2 = dict(u1, format="u2", edns=True, dnssec_ok=True)
        second = load_upstream_answer(u2)
        self.assertTrue(second.dnssec_ok)
        self.assertTrue(second.content_digest)

    def test_unknown_format(self) -> None:
        with self.assertRaises(VersionError):
            load_upstream_answer({"format": "u3", "txn_id": 1, "question": {}, "answers": []})

    def test_bad_txn_range(self) -> None:
        with self.assertRaises(FormatError):
            load_upstream_answer({"format": "u1", "txn_id": 70000, "question": {"name": "a", "qtype": 1}, "answers": []})


if __name__ == "__main__":
    unittest.main()
