from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from dp_grad_sim.config import load_config
from dp_grad_sim.data import load_shards
from dp_grad_sim.engine import InjectedCrash, Simulator
from dp_grad_sim.errors import SimError
from dp_grad_sim.injection import Injection
from dp_grad_sim.samplegen import generate_sample
from dp_grad_sim.topology import load_topology
from dp_grad_sim.verify import verify_snapshot


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _inputs(self, name: str = "sample", *, topology: str = "ring", steps: int = 3):
        sample_dir = self.root / name
        generate_sample(
            sample_dir,
            workers=3,
            steps=steps,
            layers=2,
            hidden_dim=8,
            topology_type=topology,
            link_bandwidth_bytes=4096,
            window_budget_bytes=65536,
        )
        config = load_config(sample_dir / "config.yaml")
        shards = load_shards(sample_dir / "shards.jsonl", config.workers)
        topo = load_topology(sample_dir / "topology.json", config.workers)
        return sample_dir, config, shards, topo

    def test_resume_does_not_duplicate_completed_windows(self) -> None:
        _, config, shards, topo = self._inputs(steps=3)
        snapshot_dir = self.root / "snapshots"
        first = Simulator(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(snapshot_dir),
            max_windows=1,
        ).run()
        self.assertEqual(first["completed_steps"], 1)
        self.assertEqual(len(first["windows"]), 1)

        second = Simulator(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(snapshot_dir),
        ).run()
        self.assertEqual(second["completed_steps"], 3)
        self.assertEqual(len(second["windows"]), 3)
        self.assertEqual([window["window"] for window in second["windows"]], [0, 1, 2])
        self.assertEqual(second["error_count"], 0)
        self.assertTrue(second["parameters_equal_across_workers"])

        verification = verify_snapshot(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(snapshot_dir),
        )
        self.assertTrue(verification["ok"], verification)

    def test_fidelity_rotation_keeps_latest_generations(self) -> None:
        _, config, shards, topo = self._inputs(steps=3)
        report = Simulator(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(self.root / "snapshots"),
        ).run()
        self.assertEqual(
            report["retained_fidelity_ids"],
            ["fidelity_w000001_s000002", "fidelity_w000002_s000003"],
        )
        self.assertEqual(len(report["retention_events"]), 1)
        self.assertEqual(
            report["retention_events"][0]["replaced_gradient_id"],
            "fidelity_w000000_s000001",
        )

    def test_crash_after_communicated_phase_recovers_without_resend(self) -> None:
        _, config, shards, topo = self._inputs(steps=1)
        snapshot_dir = self.root / "snapshots"
        injection = Injection(raw={"crash_after_phase": "communicated"})
        with self.assertRaises(InjectedCrash):
            Simulator(
                config=config,
                shards=shards,
                topology=topo,
                snapshot_dir=str(snapshot_dir),
                injection=injection,
            ).run()

        resumed = Simulator(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(snapshot_dir),
        )
        self.assertEqual(resumed.state.phase, "communicated")
        report = resumed.run()
        self.assertEqual(report["completed_steps"], 1)
        self.assertEqual(len(report["windows"]), 1)
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(len(report["windows"][0]["send_records"]), 24)

    def test_nonfinite_gradient_is_reported_and_run_continues(self) -> None:
        _, config, shards, topo = self._inputs(steps=1)
        injection = Injection(
            raw={
                "nonfinite": [
                    {
                        "window": 0,
                        "worker_id": 1,
                        "tensor": "layer_0.weight",
                        "index": [0, 0],
                        "value": float("nan"),
                    }
                ]
            }
        )
        report = Simulator(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(self.root / "snapshots"),
            injection=injection,
        ).run()
        matching = [error for error in report["errors"] if error["code"] == "E_PARAM_NONFINITE"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["worker_id"], 1)
        self.assertEqual(matching[0]["tensor"], "layer_0.weight")
        self.assertEqual(report["completed_steps"], 1)

    def test_budget_exhaustion_records_window_link_and_sent_bytes(self) -> None:
        _, config, shards, topo = self._inputs(steps=1, topology="full")
        injection = Injection(raw={"link_budget_overrides": {"0->1": 1}})
        report = Simulator(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(self.root / "snapshots"),
            injection=injection,
        ).run()
        budget_errors = [
            error for error in report["errors"] if error["code"] == "E_COMM_BUDGET_EXHAUSTED"
        ]
        self.assertGreaterEqual(len(budget_errors), 1)
        first = budget_errors[0]
        self.assertEqual(first["window"], 0)
        self.assertEqual(first["link"], "0->1")
        self.assertIn("sent_bytes", first)
        self.assertEqual(first["sent_bytes"], 0)

    def test_single_link_failure_is_isolated(self) -> None:
        _, config, shards, topo = self._inputs(steps=1, topology="full")
        injection = Injection(
            raw={
                "communication_failures": [
                    {
                        "window": 0,
                        "origin_worker": 0,
                        "dst_worker": 1,
                        "tensor": "layer_0.weight",
                        "error_type": "InjectedSocketError",
                    }
                ]
            }
        )
        report = Simulator(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(self.root / "snapshots"),
            injection=injection,
        ).run()
        failures = [error for error in report["errors"] if error["code"] == "E_COMM_SEND_FAILED"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["exception_type"], "InjectedSocketError")
        self.assertEqual(failures[0]["worker_id"], 0)
        self.assertEqual(failures[0]["tensor"], "layer_0.weight")
        self.assertTrue(any(record["sent"] for record in report["windows"][0]["send_records"]))

    def test_config_or_topology_mismatch_refuses_resume(self) -> None:
        _, config, shards, topo = self._inputs(steps=1)
        snapshot_dir = self.root / "snapshots"
        Simulator(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(snapshot_dir),
        ).run()
        changed_topology = load_topology(self.root / "sample" / "topology.json", 3)
        changed_links = dict(changed_topology.links)
        from dp_grad_sim.topology import Link, Topology
        from dp_grad_sim.util import hash_mapping

        link_payload = []
        for key in sorted(changed_links):
            link = changed_links[key]
            replaced = Link(link.src, link.dst, link.bandwidth_bytes + 1)
            changed_links[key] = replaced
            link_payload.append(
                {"src": replaced.src, "dst": replaced.dst, "bandwidth_bytes": replaced.bandwidth_bytes}
            )
        changed = Topology(
            kind="ring",
            workers=3,
            links=changed_links,
            topology_hash=hash_mapping({"type": "ring", "workers": 3, "links": link_payload}),
        )
        with self.assertRaises(SimError) as context:
            Simulator(
                config=config,
                shards=shards,
                topology=changed,
                snapshot_dir=str(snapshot_dir),
            )
        self.assertEqual(context.exception.code, "E_CONFIG_MISMATCH")

    def test_concurrency_does_not_change_results(self) -> None:
        sample_dir, config, shards, topo = self._inputs(steps=2)
        reports = []
        for concurrency in (1, 4):
            raw = yaml.safe_load((sample_dir / "config.yaml").read_text(encoding="utf-8"))
            raw["concurrency"] = concurrency
            config_path = self.root / f"config-c{concurrency}.yaml"
            config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            variant = load_config(config_path)
            report = Simulator(
                config=variant,
                shards=shards,
                topology=topo,
                snapshot_dir=str(self.root / f"snapshots-c{concurrency}"),
            ).run()
            reports.append(report)
        self.assertEqual(reports[0]["parameter_hashes"], reports[1]["parameter_hashes"])
        for left, right in zip(reports[0]["windows"], reports[1]["windows"]):
            self.assertEqual(
                [record["stable_key"] for record in left["send_records"]],
                [record["stable_key"] for record in right["send_records"]],
            )
            self.assertEqual(left["decision"]["selected_candidate"], right["decision"]["selected_candidate"])
            self.assertEqual(
                [(item["worker_id"], item["tensor"]) for item in left["error_tracking"]],
                [(item["worker_id"], item["tensor"]) for item in right["error_tracking"]],
            )

    def test_multi_step_compression_window_resumes_on_window_boundary(self) -> None:
        sample_dir, config, shards, topo = self._inputs(steps=3)
        raw = yaml.safe_load((sample_dir / "config.yaml").read_text(encoding="utf-8"))
        raw["compression"]["window_steps"] = 2
        config_path = self.root / "window2.yaml"
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        config = load_config(config_path)
        snapshot_dir = self.root / "snapshots-window2"
        first = Simulator(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(snapshot_dir),
            max_windows=1,
        ).run()
        self.assertEqual(first["completed_steps"], 2)
        second = Simulator(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(snapshot_dir),
        ).run()
        self.assertEqual(second["completed_steps"], 3)
        self.assertEqual([window["window"] for window in second["windows"]], [0, 1])
        verification = verify_snapshot(
            config=config,
            shards=shards,
            topology=topo,
            snapshot_dir=str(snapshot_dir),
        )
        self.assertTrue(verification["ok"], verification)


if __name__ == "__main__":
    unittest.main()
