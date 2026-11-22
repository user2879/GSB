import json
import tempfile
import unittest
from pathlib import Path

from dns_workbench.errors import ZoneError
from dns_workbench.wire import TYPE_A, TYPE_CNAME
from dns_workbench.zone import AuthorityResolver, parse_zone


ZONE = {
    "zone_version": 1,
    "origin": "example.test",
    "soa": {"mname": "ns1.example.test", "rname": "admin.example.test", "serial": 1,
            "refresh": 7200, "retry": 3600, "expire": 1209600, "minimum": 60},
    "records": [
        {"name": "www.example.test", "qtype": TYPE_CNAME, "ttl": 300, "value": "edge.example.test"},
        {"name": "edge.example.test", "qtype": TYPE_A, "ttl": 300, "value": "10.0.0.5", "rdata_hex": "0a000005"},
    ],
}


class ZoneTest(unittest.TestCase):
    def test_parse_and_cname_chain(self) -> None:
        zone = parse_zone(ZONE)
        resolver = AuthorityResolver(zone)
        result = resolver.resolve("www.example.test", TYPE_A)
        self.assertEqual(result.rcode, 0)
        self.assertEqual(result.cname_chain[0].value, "edge.example.test")
        self.assertEqual(result.answers[0].value, "10.0.0.5")

    def test_nxdomain_returns_soa(self) -> None:
        resolver = AuthorityResolver(parse_zone(ZONE))
        result = resolver.resolve("missing.example.test", TYPE_A)
        self.assertEqual(result.rcode, 3)
        self.assertIsNotNone(result.soa)

    def test_corrupt_zone_is_e_zone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{not-json", encoding="utf-8")
            from dns_workbench.zone import load_zone

            with self.assertRaises(ZoneError) as ctx:
                load_zone(path)
            self.assertEqual(ctx.exception.code, "E_ZONE")

    def test_structural_corruption(self) -> None:
        bad = json.loads(json.dumps(ZONE))
        bad["records"] = "nope"
        with self.assertRaises(ZoneError) as ctx:
            parse_zone(bad)
        self.assertIn("records", ctx.exception.fields)


if __name__ == "__main__":
    unittest.main()
