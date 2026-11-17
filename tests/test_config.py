from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from tunnelvpn.config import load_config, parse_config
from tunnelvpn.errors import ErrorCode, TunnelError

from tests.helpers import raw_config


class ConfigValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.raw, self.ports = raw_config(self.path, peer_behaviors=("normal",))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_config_roundtrip(self) -> None:
        config = parse_config(self.raw)
        self.assertEqual(config.route_version, "test-v1")
        self.assertEqual(len(config.routes), 1)
        self.assertEqual(config.routes[0].listen.host, "127.0.0.1")
        self.assertEqual(config.to_dict()["routes"][0]["id"], "alpha")

    def test_duplicate_tunnel_id_is_config_error(self) -> None:
        raw = copy.deepcopy(self.raw)
        duplicate = copy.deepcopy(raw["routes"][0])
        duplicate["listen"]["port"] = raw["routes"][0]["listen"]["port"] + 1
        duplicate["target"]["port"] = raw["routes"][0]["target"]["port"] + 1
        raw["peers"].append(
            {
                "id": "peer-extra",
                "host": "127.0.0.1",
                "port": duplicate["target"]["port"],
                "behavior": "normal",
            }
        )
        duplicate["peer_id"] = "peer-extra"
        raw["routes"].append(duplicate)
        with self.assertRaises(TunnelError) as caught:
            parse_config(raw)
        self.assertEqual(caught.exception.code, ErrorCode.E_CONFIG.value)
        self.assertEqual(caught.exception.field, "routes[1].id")

    def test_non_loopback_rejected(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["routes"][0]["listen"]["host"] = "0.0.0.0"
        with self.assertRaises(TunnelError) as caught:
            parse_config(raw)
        self.assertEqual(caught.exception.code, "E_CONFIG")
        self.assertEqual(caught.exception.field, "routes[0].listen.host")

    def test_duplicate_port_rejected(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["peers"][0]["port"] = raw["routes"][0]["listen"]["port"]
        raw["routes"][0]["target"]["port"] = raw["routes"][0]["listen"]["port"]
        with self.assertRaises(TunnelError) as caught:
            parse_config(raw)
        self.assertEqual(caught.exception.field, "routes[0].target.port")

    def test_mtu_range_and_bool_rejected(self) -> None:
        for value, field in ((127, "routes[0].mtu"), (9001, "routes[0].mtu"), (True, "routes[0].mtu")):
            raw = copy.deepcopy(self.raw)
            raw["routes"][0]["mtu"] = value
            with self.assertRaises(TunnelError) as caught:
                parse_config(raw)
            self.assertEqual(caught.exception.code, "E_CONFIG")
            self.assertEqual(caught.exception.field, field)

    def test_cipher_bandwidth_and_sessions(self) -> None:
        cases = [
            ("cipher", "ROT13", "routes[0].cipher"),
            ("bandwidth_kbps", 0, "routes[0].bandwidth_kbps"),
            ("max_sessions", "2", "routes[0].max_sessions"),
            ("max_sessions", False, "routes[0].max_sessions"),
        ]
        for key, value, field in cases:
            raw = copy.deepcopy(self.raw)
            raw["routes"][0][key] = value
            with self.assertRaises(TunnelError) as caught:
                parse_config(raw)
            self.assertEqual(caught.exception.field, field)

    def test_peer_target_mismatch_rejected(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["routes"][0]["target"]["port"] = self.ports["target"][0] + 1
        with self.assertRaises(TunnelError) as caught:
            parse_config(raw)
        self.assertEqual(caught.exception.field, "routes[0].target")

    def test_unreferenced_peer_cannot_reuse_listen_port(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["peers"].append(
            {
                "id": "peer-unused",
                "host": "127.0.0.1",
                "port": raw["routes"][0]["listen"]["port"],
                "behavior": "normal",
            }
        )
        with self.assertRaises(TunnelError) as caught:
            parse_config(raw)
        self.assertEqual(caught.exception.field, "routes[0].listen.port")

    def test_invalid_peer_behavior(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["peers"][0]["behavior"] = "surprise"
        with self.assertRaises(TunnelError) as caught:
            parse_config(raw)
        self.assertEqual(caught.exception.field, "peers[0].behavior")

    def test_invalid_last_route_rejects_whole_document(self) -> None:
        raw = copy.deepcopy(self.raw)
        extra = copy.deepcopy(raw["routes"][0])
        extra["id"] = "beta"
        extra["listen"]["port"] = extra["listen"]["port"] + 1
        extra["target"]["port"] = extra["target"]["port"] + 1
        raw["peers"].append(
            {
                "id": "peer-beta",
                "host": "127.0.0.1",
                "port": extra["target"]["port"],
                "behavior": "normal",
            }
        )
        extra["peer_id"] = "peer-beta"
        raw["routes"].append(extra)
        raw["routes"][1]["mtu"] = -1
        with self.assertRaises(TunnelError) as caught:
            parse_config(raw)
        self.assertEqual(caught.exception.field, "routes[1].mtu")
        self.assertFalse(hasattr(caught.exception, "partial_config"))

    def test_json_error_has_location(self) -> None:
        path = self.path / "bad.json"
        path.write_text('{"schema_version": 1,', encoding="utf-8")
        with self.assertRaises(TunnelError) as caught:
            load_config(path)
        self.assertEqual(caught.exception.code, "E_CONFIG")
        self.assertIn("$.line", caught.exception.field or "")


if __name__ == "__main__":
    unittest.main()
