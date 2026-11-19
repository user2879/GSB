from __future__ import annotations

import base64
import shutil
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from chunkdeps.client import WorkbenchClient
from chunkdeps.protocol import ProtocolClient
from chunkdeps.server import StorageService
from chunkdeps.util import b64encode, digest_bytes


def max_overlap(schedules: list[dict]) -> int:
    events = []
    for schedule in schedules:
        events.append((schedule["start_mono_ms"], 1))
        events.append((schedule["end_mono_ms"], -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak


class NetworkEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="chunkdeps-net-test-"))
        self.source = self.temp / "sample.bin"
        self.data = bytes((index * 19 + 3) % 256 for index in range(9000))
        self.source.write_bytes(self.data)
        self.service = StorageService(
            self.temp / "workbench",
            port=0,
            global_concurrency=2,
            session_concurrency=2,
        )
        self.thread = threading.Thread(target=self.service.serve_forever, daemon=True)
        self.thread.start()
        self.client = WorkbenchClient(port=self.service.port, timeout=60.0)

    def tearDown(self) -> None:
        try:
            self.service.shutdown()
        finally:
            self.thread.join(timeout=5)
            shutil.rmtree(self.temp, ignore_errors=True)

    def test_concurrent_upload_is_throttled_and_duplicate_chunk_converges(self) -> None:
        created = self.client.create_session(
            object_key="network/concurrent.bin",
            source=self.source,
            chunk_size=241,
            tags={"case": "network"},
        )
        uploaded = self.client.upload_session(
            session_id=created["session_id"], source=self.source, concurrency=8
        )
        schedules = uploaded["schedule"]
        self.assertEqual(len(schedules), created["chunk_count"])
        self.assertLessEqual(max_overlap(schedules), 2)
        self.assertGreaterEqual(max_overlap(schedules), 2)
        self.assertTrue(all(item["queue_ms"] >= 0 for item in schedules))
        self.assertTrue(all(item["end_mono_ms"] >= item["start_mono_ms"] for item in schedules))

        committed = self.client.request(
            "commit_session", {"session_id": created["session_id"]}
        )
        self.assertEqual(committed["version"]["content_digest"], digest_bytes(self.data))
        session = self.client.request(
            "get_session", {"session_id": created["session_id"]}
        )
        self.assertEqual(
            [chunk["chunk_index"] for chunk in session["chunks"]],
            list(range(created["chunk_count"])),
        )
        self.assertIn(created["session_id"], self.client.ping()["throttle"]["peak_by_session"])
        self.assertLessEqual(
            self.client.ping()["throttle"]["peak_by_session"][created["session_id"]], 2
        )

        duplicate = self.client.create_session(
            object_key="network/duplicate.bin",
            source=self.source,
            chunk_size=241,
        )
        payload = self.data[:241]
        params = {
            "session_id": duplicate["session_id"],
            "chunk_index": 0,
            "data_b64": b64encode(payload),
            "digest": digest_bytes(payload),
        }
        raw = ProtocolClient(port=self.service.port)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: raw.request("upload_chunk", params), range(2)))
        self.assertEqual(len(results), 2)
        detail = self.client.request(
            "get_session", {"session_id": duplicate["session_id"]}
        )
        self.assertEqual(len(detail["chunks"]), 1)
        self.assertEqual(detail["chunks"][0]["digest"], digest_bytes(payload))
        self.assertEqual(detail["bitmap"], "1" + "0" * (duplicate["chunk_count"] - 1))

    def test_local_json_scenario_and_cli_compatible_output(self) -> None:
        from chunkdeps.scenario import ScenarioRunner

        runner = ScenarioRunner(self.client, base_dir=self.temp)
        scenario = {
            "defaults": {"concurrency": 3},
            "steps": [
                {
                    "op": "upload_file",
                    "object_key": "scenario/item.bin",
                    "source": "sample.bin",
                    "chunk_size": 700,
                    "tags": {"runner": "json"},
                },
                {"op": "audit"},
                {"op": "views", "prefix": "scenario"},
            ],
        }
        result = runner.run(scenario)
        self.assertTrue(result["ok"])
        self.assertEqual(result["step_count"], 3)
        self.assertTrue(result["steps"][1]["result"]["passed"])
        self.assertEqual(result["steps"][2]["result"]["view"]["count"], 1)

    def test_client_resume_skips_registered_chunks_without_rewriting_them(self) -> None:
        created = self.client.create_session(
            object_key="network/resume.bin",
            source=self.source,
            chunk_size=500,
        )
        raw = ProtocolClient(port=self.service.port)
        for index in (0, 1):
            payload = self.data[index * 500 : (index + 1) * 500]
            raw.request(
                "upload_chunk",
                {
                    "session_id": created["session_id"],
                    "chunk_index": index,
                    "data_b64": b64encode(payload),
                    "digest": digest_bytes(payload),
                },
            )
        before = self.client.request(
            "get_session", {"session_id": created["session_id"]}
        )
        registered_before = {
            item["chunk_index"]: item for item in before["chunks"]
        }
        resumed = self.client.upload_session(
            session_id=created["session_id"],
            source=self.source,
            concurrency=4,
        )
        self.assertEqual(resumed["skipped_indices"], [0, 1])
        self.assertEqual(resumed["uploaded_indices"], list(range(2, created["chunk_count"])))
        after = self.client.request(
            "get_session", {"session_id": created["session_id"]}
        )
        registered_after = {
            item["chunk_index"]: item for item in after["chunks"]
        }
        for index in (0, 1):
            self.assertEqual(
                registered_before[index]["digest"], registered_after[index]["digest"]
            )
            self.assertEqual(
                registered_before[index]["registered_mono_ms"],
                registered_after[index]["registered_mono_ms"],
            )


if __name__ == "__main__":
    unittest.main()
