from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dp_grad_sim.config import load_config
from dp_grad_sim.data import load_shards
from dp_grad_sim.decision import decide_window
from dp_grad_sim.model import derive_window_gradients
from dp_grad_sim.samplegen import generate_sample
from dp_grad_sim.topology import load_topology


class DecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        generate_sample(
            self.root / "sample",
            workers=3,
            steps=2,
            layers=2,
            hidden_dim=8,
            topology_type="full",
            link_bandwidth_bytes=4096,
        )
        self.config = load_config(self.root / "sample" / "config.yaml")
        self.shards = load_shards(self.root / "sample" / "shards.jsonl", self.config.workers)
        self.topology = load_topology(
            self.root / "sample" / "topology.json", self.config.workers
        )
        self.gradients = derive_window_gradients(self.config, self.shards, 0, 1)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_decision_records_scores_budget_and_order(self) -> None:
        result = decide_window(
            config=self.config,
            topology=self.topology,
            window=0,
            gradients=self.gradients,
        )
        decision = result.decision
        self.assertEqual(decision["selected_candidate"]["bits"], 8)
        self.assertEqual(decision["aggregation_order"], "bytes_desc")
        self.assertGreater(decision["source_gradient_bytes"], 0)
        self.assertTrue(decision["fits_budget"])
        self.assertEqual(len(decision["candidate_scores"]), 6)
        for candidate in decision["candidate_scores"]:
            self.assertIn("score", candidate)
            self.assertIn("max_abs_error", candidate)
            self.assertIn("max_rel_error", candidate)
            self.assertIn("total_compressed_bytes", candidate)
            self.assertIn("tolerance_ok", candidate)
            self.assertIn("fits_budget", candidate)

        ordered = [item["tensor"] for item in decision["ordered_tensors"]]
        sizes = decision["selected_packet_bytes_by_tensor"]
        expected = sorted(ordered, key=lambda name: (-sizes[name], name))
        self.assertEqual(ordered, expected)
        self.assertEqual(decision["planned_send_count"], len(result.tasks))
        for task in result.tasks:
            self.assertEqual(len(task["stable_key"]), 5)
            self.assertIn("candidate_scores", task)
            self.assertIn("selection_reason", task)

        link_totals: dict[str, int] = {}
        for task in result.tasks:
            link = f"{task['src_holder']}->{task['dst_worker']}"
            link_totals[link] = link_totals.get(link, 0) + int(task["wire_bytes"])
        for link, total in link_totals.items():
            self.assertLessEqual(total, decision["link_budgets"][link])
        self.assertLessEqual(
            sum(link_totals.values()), self.config.compression.max_bytes_per_window
        )

    def test_bytes_desc_uses_tensor_name_as_tie_break(self) -> None:
        result = decide_window(
            config=self.config,
            topology=self.topology,
            window=0,
            gradients=self.gradients,
        )
        ordered = [item["tensor"] for item in result.decision["ordered_tensors"]]
        sizes = result.decision["selected_packet_bytes_by_tensor"]
        self.assertLess(ordered.index("layer_0.weight"), ordered.index("layer_0.bias"))
        self.assertEqual(
            [name for name in ordered if sizes[name] == sizes["layer_0.weight"]],
            ["layer_0.weight", "layer_1.weight"],
        )

    def test_no_fitting_candidate_selects_smallest_packet_width_and_flags_budget(self) -> None:
        overrides = {(link.src, link.dst): 1 for link in self.topology.link_list()}
        result = decide_window(
            config=self.config,
            topology=self.topology,
            window=0,
            gradients=self.gradients,
            budget_overrides=overrides,
        )
        self.assertFalse(result.decision["fits_budget"])
        self.assertEqual(result.decision["selected_candidate"]["bits"], 4)
        self.assertEqual(result.decision["bandwidth_concurrency_limit"], 1)


if __name__ == "__main__":
    unittest.main()

