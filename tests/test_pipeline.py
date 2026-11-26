from __future__ import annotations

import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from fusion_recall.config import ALLOWED_FRAC_BITS
from fusion_recall.diagnostics import diagnose_query_flip
from fusion_recall.engine import Engine, full_rebuild_engine, prepare_config
from fusion_recall.generator import build_config, build_vocab, make_document
from fusion_recall.hashing import canonical_hash
from fusion_recall.index import build_index
from fusion_recall.loading import validate_document
from fusion_recall.scoring import score_query
from fusion_recall.scheduler import begin_batch, new_scheduler_state, plan_schedule
from fusion_recall.snapshot_recovery_check import simulated_recovery
from fusion_recall.vocab import Vocabulary


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def config_vocab_fixture():
    vocab = Vocabulary(build_vocab())
    config = prepare_config(build_config(), vocab)
    return config, vocab


def make_valid_doc(config, vocab, doc_id, version, body_terms, view="original"):
    raw = {
        "doc_id": doc_id,
        "version": version,
        "views": {
            "original": {
                "language": "zh-CN",
                "fields": {"title": [], "body": []},
            },
            "translation": {
                "language": "en-US",
                "fields": {"title": [], "body": []},
            },
        },
    }
    raw["views"][view]["fields"]["body"] = list(body_terms)
    doc, failure = validate_document(raw, config, vocab)
    assert failure is None, failure
    return doc


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config, self.vocab = config_vocab_fixture()

    def parity_hash(self, engine):
        material = engine.parity_material()
        material.pop("command_history")
        return canonical_hash(material)

    def test_invalid_inputs_are_skipped(self):
        from fusion_recall.commands import parse_commands
        from fusion_recall.loading import load_documents, load_queries

        docs = load_documents(self.config, self.vocab, path=PROJECT_ROOT / "samples/documents.jsonl")
        queries = load_queries(self.config, self.vocab, path=PROJECT_ROOT / "samples/queries.jsonl")
        commands = parse_commands(self.config, self.vocab, path=PROJECT_ROOT / "samples/commands.jsonl")
        self.assertEqual(docs["loaded"], 12)
        self.assertEqual(queries["loaded"], 10)
        self.assertEqual(len(commands["commands"]), 14)
        self.assertTrue(all(item["code"] == "E_INPUT_INVALID" for item in docs["failures"]))
        self.assertGreaterEqual(len(docs["failures"]) + len(queries["failures"]) + len(commands["failures"]), 9)

    def test_fixed_point_width_and_tier_change_integers(self):
        doc = make_valid_doc(self.config, self.vocab, "D001", 1, ["C05"])
        index = build_index([doc], self.config, self.vocab)
        query = {"query_id": "Q1", "terms": ["C05"], "language_preference_rational": {
            "original": [1, 1], "translation": [1, 1]
        }, "query_fingerprint": "q1"}
        low = score_query(
            query,
            index,
            self.config,
            self.vocab,
            fixed_override={"frac_bits": 8, "quant_tier": "floor"},
            include_boundary=False,
        )
        high = score_query(
            query,
            index,
            self.config,
            self.vocab,
            fixed_override={"frac_bits": 16, "quant_tier": "nearest"},
            include_boundary=False,
        )
        self.assertNotEqual(low["scale"], high["scale"])
        low_sat = low["evidence"][0]["fields"][0]["saturation_q"]
        high_sat = high["evidence"][0]["fields"][0]["saturation_q"]
        self.assertNotEqual(low_sat, high_sat)
        self.assertNotEqual(low["rows"][0]["fusion_score_q"], high["rows"][0]["fusion_score_q"])

        ceil_result = score_query(
            query,
            index,
            self.config,
            self.vocab,
            fixed_override={"frac_bits": 8, "quant_tier": "ceil"},
            include_boundary=False,
        )
        floor_result = score_query(
            query,
            index,
            self.config,
            self.vocab,
            fixed_override={"frac_bits": 8, "quant_tier": "floor"},
            include_boundary=False,
        )
        self.assertNotEqual(
            floor_result["evidence"][0]["fields"][0]["saturation_q"],
            ceil_result["evidence"][0]["fields"][0]["saturation_q"],
        )

    def test_tf_quantization_flip_attribution(self):
        config = deepcopy(self.config)
        config["top_k"] = 1
        config["config_fingerprint"] = "test-topk1"
        docs = [
            make_valid_doc(config, self.vocab, "D001", 1, ["C05"] * 15),
            make_valid_doc(config, self.vocab, "D002", 1, ["C05"] * 16),
        ]
        config["saturation_k"] = 1
        index = build_index(docs, config, self.vocab)
        query = {
            "query_id": "QTF",
            "terms": ["C05"],
            "language_preference_rational": {name: [1, 1] for name in config["view_names"]},
            "query_fingerprint": "qtf",
        }
        result = diagnose_query_flip(
            query,
            index,
            config,
            self.vocab,
            profile_a={"frac_bits": 16, "quant_tier": "floor"},
            profile_b={"frac_bits": 8, "quant_tier": "floor"},
        )
        self.assertTrue(result["flipped"])
        self.assertTrue(result["tf_quantization_changes_hybrid_ranking"])
        self.assertEqual(result["primary_cause"], "tf_quantization")
        self.assertTrue(result["tie_resolution_decides_boundary"])

    def test_language_rounding_flip_attribution(self):
        config = deepcopy(self.config)
        config["top_k"] = 1
        config["language_weight_rational"]["original"] = [1001, 1000]
        config["language_weight_rational"]["translation"] = [1, 1]
        config["language_weights"] = {"original": "1001/1000", "translation": "1/1"}
        original_doc = make_valid_doc(config, self.vocab, "D002", 1, ["C06"], view="original")
        translation_doc = make_valid_doc(config, self.vocab, "D001", 1, ["C06"], view="translation")
        index = build_index([original_doc, translation_doc], config, self.vocab)
        query = {
            "query_id": "QLANG",
            "terms": ["C06"],
            "language_preference_rational": {name: [1, 1] for name in config["view_names"]},
            "query_fingerprint": "qlang",
        }
        result = diagnose_query_flip(
            query,
            index,
            config,
            self.vocab,
            profile_a={"frac_bits": 64, "quant_tier": "nearest"},
            profile_b={"frac_bits": 8, "quant_tier": "floor"},
        )
        self.assertTrue(result["flipped"], result)
        self.assertTrue(result["language_rounding_changes_hybrid_ranking"])
        self.assertEqual(result["primary_cause"], "language_weight_rounding")

    def test_budget_quota_and_pending_priority(self):
        state = new_scheduler_state(40, 40)
        begin_batch(state, 40)
        costs = {"Q1": 36, "Q2": 48, "Q3": 24, "Q4": 12}
        first = plan_schedule(state, list(costs), costs, query_quota_points=40)
        self.assertEqual([item["query_id"] for item in first["ready"]], ["Q1"])
        self.assertEqual([item["query_id"] for item in first["quota_failures"]], ["Q2"])
        self.assertEqual([item["query_id"] for item in first["pending"]], ["Q3", "Q4"])
        self.assertEqual(state["todo"], ["Q3", "Q4"])

        begin_batch(state, 24)
        second = plan_schedule(state, list(costs), costs, query_quota_points=40)
        self.assertEqual([item["query_id"] for item in second["ready"]], ["Q3"])
        self.assertEqual(state["todo"], ["Q4"])

    def test_snapshot_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = simulated_recovery(Path(tmp) / "recovery")
            self.assertTrue(result["recovered_to_v2"])
            self.assertTrue(result["incomplete_cleaned"])
            self.assertEqual(result["recovery_result"]["decision"], "promoted_prepared")

    def test_replacement_invalidates_only_referenced_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = [
                make_valid_doc(self.config, self.vocab, "D001", 1, ["C01"]),
                make_valid_doc(self.config, self.vocab, "D002", 1, ["C02"]),
            ]
            queries = [
                {
                    "query_id": "Q_C01",
                    "terms": ["C01"],
                    "query_fingerprint": "q_c01",
                    "language_preference_rational": {
                        name: [1, 1] for name in self.config["view_names"]
                    },
                },
                {
                    "query_id": "Q_C02",
                    "terms": ["C02"],
                    "query_fingerprint": "q_c02",
                    "language_preference_rational": {
                        name: [1, 1] for name in self.config["view_names"]
                    },
                },
            ]
            engine = Engine(
                deepcopy(self.config), self.vocab, queries, root=Path(tmp) / "ws"
            )
            engine.cold_build(docs)
            engine.run_batch(unlimited=True)
            replacement = make_valid_doc(self.config, self.vocab, "D001", 2, ["C01"])
            engine.apply_command(
                {"op": "replace_document", "document": replacement, "line": 1}
            )
            report = engine.run_batch(unlimited=True)
            self.assertEqual(report["last_run"]["fusion_recomputed"], ["Q_C01"])
            self.assertEqual(report["last_run"]["fusion_reused"], ["Q_C02"])
            self.assertIn("original:C02", report["last_run"]["buckets_reused"])
            self.assertTrue(
                any(
                    bucket in report["last_run"]["buckets_recomputed"]
                    for bucket in ("original:C01", "translation:C01")
                )
            )

    def test_random_command_sequences_match_full_rebuild_and_concurrency(self):
        rng = random.Random(20261127)
        for case in range(8):
            config = deepcopy(self.config)
            initial_docs = [make_document(number) for number in range(1, 7)]
            normalized_docs = []
            seen: set[str] = set()
            for raw in initial_docs:
                doc, failure = validate_document(raw, config, self.vocab, seen_ids=seen)
                self.assertIsNone(failure, failure)
                normalized_docs.append(doc)

            incremental = Engine(config, self.vocab, [])
            incremental.cold_build(normalized_docs)
            next_number = 20 + case * 20
            for _ in range(12):
                op = rng.choice(
                    ["add", "replace", "delete", "lang", "field", "sat", "bits", "tier"]
                )
                if op == "add":
                    next_number += 1
                    raw = make_document(next_number, version=1, seed=1000 + next_number)
                    command = {"op": "add_document", "document": self.normalize_raw(raw, config), "line": next_number}
                elif op == "replace" and incremental.documents:
                    doc_id = rng.choice(sorted(incremental.documents))
                    number = int(doc_id[1:])
                    raw = make_document(number, version=rng.randint(2, 9), seed=5000 + rng.randint(1, 10_000))
                    command = {"op": "replace_document", "document": self.normalize_raw(raw, config), "line": number}
                elif op == "delete" and incremental.documents:
                    doc_id = rng.choice(sorted(incremental.documents))
                    command = {"op": "delete_document", "doc_id": doc_id, "line": rng.randint(1, 9999)}
                elif op == "lang":
                    command = {
                        "op": "set_language_weight",
                        "view": rng.choice(config["view_names"]),
                        "weight": f"{rng.randint(8, 15)}/10",
                        "line": rng.randint(1, 9999),
                    }
                elif op == "field":
                    command = {
                        "op": "set_field_weight",
                        "field": rng.choice(config["field_order"]),
                        "weight": rng.randint(1, 7),
                        "line": rng.randint(1, 9999),
                    }
                elif op == "sat":
                    command = {"op": "set_saturation_k", "value": rng.randint(1, 8), "line": 1}
                elif op == "bits":
                    command = {"op": "set_frac_bits", "value": rng.choice(ALLOWED_FRAC_BITS), "line": 1}
                else:
                    command = {
                        "op": "set_quant_tier",
                        "value": rng.choice(["floor", "nearest", "ceil"]),
                        "line": 1,
                    }
                incremental.apply_command(command)

            query = {
                "query_id": "QRAND",
                "terms": ["C01", "C05", "C08"],
                "language_preference_rational": {name: [1, 1] for name in config["view_names"]},
                "language_preference": {name: "1/1" for name in config["view_names"]},
                "query_fingerprint": "qrand",
            }
            incremental.queries = [query]
            incremental.query_by_id = {"QRAND": query}
            serial_report = incremental.run_batch(unlimited=True, workers=1)
            serial_hash = self.parity_hash(incremental)

            full_config = deepcopy(incremental.config)
            full = full_rebuild_engine(
                full_config,
                self.vocab,
                [deepcopy(query)],
                list(incremental.documents.values()),
                global_version=incremental.global_version,
                workers=1,
            )
            full_hash = self.parity_hash(full)
            self.assertEqual(serial_hash, full_hash)

            concurrent = Engine(deepcopy(config), self.vocab, [deepcopy(query)])
            concurrent.global_version = incremental.global_version
            concurrent.documents = deepcopy(incremental.documents)
            concurrent.index = build_index(list(concurrent.documents.values()), concurrent.config, self.vocab)
            concurrent.run_batch(unlimited=True, workers=2)
            self.assertEqual(self.parity_hash(concurrent), serial_hash)

    def normalize_raw(self, raw, config):
        doc, failure = validate_document(raw, config, self.vocab)
        assert failure is None
        return doc


if __name__ == "__main__":
    unittest.main()
