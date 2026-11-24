from __future__ import annotations

import threading
import time
import unittest

from leasecache.digest import generate_artifact
from leasecache.errors import LeaseError
from leasecache.ledger import Ledger
from leasecache.spec import load_spec
from leasecache.util import mono_ms

from .helpers import free_ports, make_spec, test_tempdir, write_spec


class LedgerOwnershipTests(unittest.TestCase):
    def _setup(self):
        directory = test_tempdir()
        stages = [
            {
                "id": "s",
                "concurrency": 1,
                "artifacts": [
                    {"key": "a", "references": []},
                    {"key": "b", "references": []},
                ],
            }
        ]
        raw = make_spec(
            free_ports(2),
            stages=stages,
            seeds=[7, 9],
            capacities=[2, 2],
            limits=[2, 2],
        )
        spec = load_spec(write_spec(directory, raw))
        ledger = Ledger(directory / "ledger.sqlite")
        ledger.start_session(spec.content_sha256)
        ledger.sync_node_specs(spec)
        snapshot = {}
        for node in spec.nodes:
            snapshot[node.node_id] = {
                "digests": {
                    artifact.key: generate_artifact(node, artifact.key).digest
                    for artifact in spec.artifacts
                },
                "refs": {
                    artifact.key: list(artifact.references)
                    for artifact in spec.artifacts
                },
            }
        ledger.sync_digests(snapshot)
        return spec, ledger

    def _grant(self, spec, ledger, node_id: str, key: str, seq: int, lease_ms: int = 100):
        node = spec.node_by_id[node_id]
        digest = generate_artifact(node, key).digest
        refs = {key: list(spec.artifact_by_key[key].references)}
        source_nodes = sorted(
            candidate.node_id
            for candidate in spec.nodes
            if generate_artifact(candidate, key).digest == digest
        )
        lease_id = f"{node_id}:lease:test:{seq}"
        ledger.begin_pending(lease_id, node_id, key, seq, digest, refs[key])
        return lease_id, ledger.commit_grant(
            lease_id,
            node_id,
            [key],
            {key: digest},
            refs,
            mono_ms() + lease_ms,
            source_nodes=source_nodes,
        )

    def test_same_artifact_cannot_have_two_capacity_owners(self) -> None:
        spec, ledger = self._setup()
        try:
            self._grant(spec, ledger, "node-a", "a", 1)
            lease_id = "node-b:lease:test:2"
            node = spec.node_by_id["node-b"]
            digest = generate_artifact(node, "a").digest
            ledger.begin_pending(lease_id, "node-b", "a", 2, digest, [])
            with self.assertRaises(LeaseError) as caught:
                ledger.commit_grant(
                    lease_id,
                    "node-b",
                    ["a"],
                    {"a": digest},
                    {"a": []},
                    mono_ms() + 100,
                    source_nodes=["node-b"],
                )
            self.assertEqual(caught.exception.code, "E_LEASE_CLAIMED")
            self.assertEqual(ledger.nodes()[0]["active_leases"], 1)
        finally:
            ledger.close()

    def test_expiry_and_quota_release_race_has_one_winner(self) -> None:
        spec, ledger = self._setup()
        try:
            lease_id, _ = self._grant(spec, ledger, "node-a", "a", 1, lease_ms=20)
            time.sleep(0.04)
            results = []
            lock = threading.Lock()

            def expire() -> None:
                result = ledger.expire_lease(lease_id)
                with lock:
                    results.append(("expire", result))

            def release() -> None:
                result = ledger.release_lease(lease_id)
                with lock:
                    results.append(("release", result))

            threads = [
                threading.Thread(target=expire),
                threading.Thread(target=release),
                threading.Thread(target=expire),
                threading.Thread(target=release),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            winners = [item for item in results if item[1] is not None]
            self.assertEqual(len(winners), 1)
            terminal_events = [
                event
                for event in ledger.events()
                if event["kind"] in {"LEASE_EXPIRED", "QUOTA_RELEASED"}
                and event["lease_id"] == lease_id
            ]
            self.assertEqual(len(terminal_events), 1)
            node = ledger.nodes()[0]
            self.assertEqual(node["used_slots"], 0)
            self.assertEqual(node["active_leases"], 0)
            self.assertEqual(ledger.active_leases(), [])
            ledger.verify_content_digest()
        finally:
            ledger.close()

    def test_renewal_keeps_single_owner_and_extends_expiry(self) -> None:
        spec, ledger = self._setup()
        try:
            lease_id, granted = self._grant(
                spec, ledger, "node-a", "a", 1, lease_ms=100
            )
            before = ledger.lease(lease_id)
            renewed = ledger.renew_lease(lease_id, before["expires_mono_ms"] + 50)
            after = ledger.lease(lease_id)
            self.assertEqual(after["status"], "ACTIVE")
            self.assertEqual(after["expires_mono_ms"], renewed["expires_mono_ms"])
            self.assertEqual(len(ledger.active_leases()), 1)
            self.assertEqual(ledger.nodes()[0]["used_slots"], 1)
            renewal_events = [
                event for event in ledger.events() if event["kind"] == "LEASE_RENEWED"
            ]
            self.assertEqual(len(renewal_events), 1)
            ledger.verify_content_digest()
        finally:
            ledger.close()

    def test_digest_change_invalidates_active_lease_and_sources(self) -> None:
        spec, ledger = self._setup()
        try:
            lease_id, _ = self._grant(
                spec, ledger, "node-a", "a", 1, lease_ms=5000
            )
            self.assertEqual(ledger.lease(lease_id)["status"], "ACTIVE")
            snapshot = {}
            for node in spec.nodes:
                snapshot[node.node_id] = {
                    "digests": {
                        artifact.key: generate_artifact(node, artifact.key).digest
                        for artifact in spec.artifacts
                    },
                    "refs": {
                        artifact.key: list(artifact.references)
                        for artifact in spec.artifacts
                    },
                }
            snapshot["node-a"]["digests"]["a"] = "digest-changed"
            sync = ledger.sync_digests(snapshot)
            self.assertIn("a", sync["affected_keys"])
            self.assertIn("a", sync["canonical_changed_keys"])
            self.assertEqual(ledger.lease(lease_id)["status"], "INVALIDATED")
            self.assertEqual(ledger.nodes()[0]["active_leases"], 0)
            self.assertEqual(ledger.nodes()[0]["used_slots"], 0)
            self.assertEqual(
                [source for source in ledger.sources(valid_only=True) if source["artifact_key"] == "a"],
                [],
            )
        finally:
            ledger.close()


if __name__ == "__main__":
    unittest.main()
