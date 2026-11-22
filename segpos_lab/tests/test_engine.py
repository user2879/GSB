import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.context import ROOT
from segpos.engine import stable_task_key
from segpos.errors import E_INTERACT_BUDGET_PENDING
from segpos.loader import (
    load_charset,
    load_commands,
    load_config,
    load_prob_table,
)
from segpos.runner import _materialize_commands, build_engine, prepare_bundle

BASE = ROOT / "samples" / "base"


def make_mini_bundle(tmp_dir, **config_overrides):
    """3 句迷你样本，用于预算/终止类睡眠测试，控制用例时长。"""
    mini = Path(tmp_dir) / "mini"
    mini.mkdir(parents=True, exist_ok=True)
    charset = load_charset(BASE / "charset.txt")
    (mini / "charset.txt").write_text("".join(sorted(charset)), encoding="utf-8")
    (mini / "sentences.txt").write_text(
        "T01\t天地人和。\nT02\t山水风云。\n", encoding="utf-8"
    )
    table, _ = load_prob_table(BASE / "prob_table.json")
    (mini / "prob_table.json").write_text((BASE / "prob_table.json").read_text(encoding="utf-8"), encoding="utf-8")
    (mini / "lexicon.txt").write_text(
        "天地 100 n\n人和 80 n\n山水 60 n\n风云 40 n\n日月 30 n\n星辰 20 n\n",
        encoding="utf-8",
    )
    config, _ = load_config(BASE / "config.json")
    config_data = config.__dict__.copy()
    config_data.update(config_overrides)
    (mini / "config.json").write_text(json.dumps(config_data, ensure_ascii=False), encoding="utf-8")
    (mini / "commands.json").write_text(
        json.dumps(
            [
                {"type": "adjust_freq", "word": "天地", "freq": 250},
                {"type": "remove_entry", "word": "风云"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return prepare_bundle(mini)[0]



def bundle_with(overrides=None):
    bundle, _ = prepare_bundle(BASE, overrides)
    return bundle


def comparable(report: dict) -> dict:
    """剔除时序/路径类字段，保留需要逐字段一致的业务产物。"""
    sentences = {}
    for sid, part in report["sentences"].items():
        sentences[sid] = {
            "segment": part["segment"],
            "oov": part["oov"],
            "tag": part["tag"],
            "segment_meta": part["segment_meta"],
            "oov_meta": part["oov_meta"],
            "tag_meta": {
                k: v
                for k, v in part["tag_meta"].items()
                if k not in ("marginal_log_sum",)
            },
        }
    return {
        "sentences": sentences,
        "low_confidence": report["low_confidence"],
        "failures": report["failures"],
        "total_sentences": report["total_sentences"],
        "success_count": report["success_count"],
        "segment_token_total": report["segment_token_total"],
        "oov_token_total": report["oov_token_total"],
    }


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _final_state_bundle(self):
        bundle = bundle_with()
        commands, command_failures = load_commands(BASE / "commands.json")
        materialized, resolve_failures = _materialize_commands(commands, bundle)
        engine = build_engine(bundle, Path(self.tmp) / "snap-full")
        for command in materialized:
            engine.apply_command(command)
        return engine, materialized, command_failures + resolve_failures

    def test_incremental_matches_full_from_scratch(self):
        engine_final, commands, extra_failures = self._final_state_bundle()
        engine_final.run_due(set(engine_final.states))
        full_report = engine_final.build_report(len(commands))

        bundle = bundle_with()
        engine_inc = build_engine(bundle, Path(self.tmp) / "snap-inc")
        engine_inc.failures.extend(extra_failures)
        responses = engine_inc.run_command_stream(commands)
        inc_report = engine_inc.build_report(len(commands))

        self.assertEqual(comparable(full_report), comparable(inc_report))
        self.assertTrue(all(r["response"]["code"] == "OK" for r in responses))
        self.assertEqual(
            [t["text"] for t in inc_report["sentences"][sorted(inc_report["sentences"])[0]]["tag"]],
            [t["text"] for t in full_report["sentences"][sorted(full_report["sentences"])[0]]["tag"]],
        )

    def test_chunked_command_streams_match(self):
        engine_final, commands, extra = self._final_state_bundle()
        engine_final.run_due(set(engine_final.states))
        full = comparable(engine_final.build_report(len(commands)))

        reports = []
        for chunk_size in (1, 2, 4):
            bundle = bundle_with()
            engine = build_engine(bundle, Path(self.tmp) / f"snap-chunk-{chunk_size}")
            engine.failures.extend(extra)
            for start in range(0, len(commands), chunk_size):
                engine.run_command_stream(commands[start : start + chunk_size])
            reports.append(comparable(engine.build_report(len(commands))))
        for report in reports:
            self.assertEqual(report, full)

    def test_precise_invalidation_only_affects_referencing_sentences(self):
        bundle = bundle_with()
        engine = build_engine(bundle, Path(self.tmp) / "snap-inv")
        engine.run_due(set(engine.states))
        target_word = next(iter(engine.lexicon.entries))
        referencing = engine._sentences_referencing(target_word)
        affected = engine.apply_command(
            {"type": "adjust_freq", "word": target_word, "freq": 777, "_index": 1}
        )
        self.assertEqual(affected, referencing)
        before = dict(engine.reuse_counts)
        engine.global_version += 1
        response = engine.run_due(affected)
        self.assertEqual(sorted(response["affected_sentences"]), sorted(referencing))
        for sid in engine.states:
            state = engine.states[sid]
            if sid not in referencing:
                self.assertTrue(all(state.stages_valid[s] for s in ("segment", "oov", "tag")))
        self.assertGreaterEqual(engine.reuse_counts["segment"], before["segment"])

    def test_prob_table_replacement_invalidates_all(self):
        bundle = bundle_with()
        engine = build_engine(bundle, Path(self.tmp) / "snap-table")
        engine.run_due(set(engine.states))
        commands, _ = load_commands(BASE / "commands.json")
        materialized, _ = _materialize_commands(commands, bundle)
        replace_cmd = next(cmd for cmd in materialized if cmd["type"] == "replace_prob_table")
        affected = engine.apply_command(replace_cmd)
        self.assertEqual(affected, set(engine.states))


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_budget_pending_then_drain_keeps_final_clean(self):
        bundle = make_mini_bundle(
            self.tmp,
            first_response_budget_ms=8.0,
            simulated_task_delay_ms=4.0,
            snapshot_commit_interval_commands=100,
        )
        engine = build_engine(bundle, Path(self.tmp) / "snap-budget")
        response = engine.run_due(set(engine.states))
        self.assertEqual(response["code"], E_INTERACT_BUDGET_PENDING)
        self.assertTrue(response["pending_sentences"])
        self.assertLess(response["completed_sentences"], response["total_sentences"])
        self.assertTrue(engine.snapshot.load_manifest() is None or engine.pending)

        follow_up = engine.run_due(set())
        while follow_up["code"] == E_INTERACT_BUDGET_PENDING:
            follow_up = engine.run_due(set())
        self.assertEqual(follow_up["code"], "OK")
        self.assertEqual(follow_up["completed_sentences"], follow_up["total_sentences"])
        self.assertEqual(engine.pending, [])
        report = engine.build_report(0)
        self.assertEqual(report["success_count"], report["total_sentences"])

    def test_pending_does_not_pollute_snapshot_manifest(self):
        bundle = make_mini_bundle(
            self.tmp,
            first_response_budget_ms=3.0,
            simulated_task_delay_ms=5.0,
            snapshot_commit_interval_commands=1,
        )
        engine = build_engine(bundle, Path(self.tmp) / "snap-manifest")
        cmd = {"type": "adjust_freq", "word": next(iter(engine.lexicon.entries)), "freq": 123, "_index": 1}
        engine.apply_command(cmd)
        engine.global_version += 1
        response = engine.run_due({cmd["word"]} and engine._sentences_referencing(cmd["word"]))
        engine.commit_snapshot(response["code"] == E_INTERACT_BUDGET_PENDING)
        manifest = engine.snapshot.load_manifest()
        self.assertEqual(manifest["pending_sentences"], response["pending_sentences"])
        self.assertTrue(manifest["budget_consumed"])


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_threadpool_matches_single_thread_order(self):
        reports = {}
        for concurrency in (1, 2, 4, 8):
            bundle = bundle_with({"concurrency": concurrency})
            engine = build_engine(bundle, Path(self.tmp) / f"snap-c{concurrency}")
            engine.run_due(set(engine.states))
            reports[concurrency] = comparable(engine.build_report(0))
        for concurrency in (2, 4, 8):
            self.assertEqual(reports[concurrency], reports[1])

    def test_stable_task_key(self):
        raw = [(f"S{i:03d}", stage) for i in (3, 1, 2) for stage in ("tag", "segment")]
        keys = [stable_task_key(sid, stage) for sid, stage in raw]
        ordered = [stable_task_key(sid, stage) for sid, stage in sorted(raw, key=lambda item: stable_task_key(item[0], item[1]))]
        self.assertEqual(ordered, sorted(keys))
        self.assertTrue(all(isinstance(key, str) and "::" in key for key in keys))


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_restore_uses_only_completed_snapshot(self):
        bundle = bundle_with()
        engine = build_engine(bundle, Path(self.tmp) / "snap-restore")
        engine.run_due(set(engine.states))
        engine.global_version = 1
        engine.commit_snapshot(False)

        restored = build_engine(bundle_with(), Path(self.tmp) / "snap-restore")
        self.assertEqual(restored.recovery_version, 1)
        for sid, state in restored.states.items():
            self.assertTrue(all(state.stages_valid[s] for s in ("segment", "oov", "tag")))
        report = restored.build_report(0)
        self.assertEqual(report["recovery_version"], 1)
        self.assertEqual(report["success_count"], len(restored.states))

    def test_forced_kill_recovery_in_subprocess(self):
        mini_bundle = make_mini_bundle(
            self.tmp,
            first_response_budget_ms=100.0,
            simulated_task_delay_ms=4.0,
            snapshot_commit_interval_commands=1,
        )
        sample_dir = str(mini_bundle["base"])
        snap_dir = str(Path(self.tmp) / "snap-kill")
        report_path = str(Path(self.tmp) / "kill-report.json")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        code = f"""
from segpos.runner import prepare_bundle, build_engine, _materialize_commands
from segpos.loader import (
    load_charset,
    load_commands,
    load_config,
    load_prob_table,
)
bundle, _ = prepare_bundle({sample_dir!r}, {{"first_response_budget_ms": 100.0, "simulated_task_delay_ms": 4.0, "snapshot_commit_interval_commands": 1}})
commands, _ = load_commands(bundle['base'] / 'commands.json')
materialized, _ = _materialize_commands(commands, bundle)
engine = build_engine(bundle, {snap_dir!r})
engine.run_due(set(engine.states))
engine.global_version = 1
engine.commit_snapshot(False)
for idx, command in enumerate(materialized):
    engine.apply_command(command)
    engine.global_version += 1
    engine.run_due(set(engine.states))
    engine.commands_processed += 1
    engine.commit_snapshot(True)
    if idx == 0:
        import os
        os.kill(os.getpid(), 9)
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertNotEqual(proc.returncode, 0)
        bundle = make_mini_bundle(
            self.tmp,
            first_response_budget_ms=100.0,
            simulated_task_delay_ms=4.0,
            snapshot_commit_interval_commands=1,
        )
        engine = build_engine(bundle, Path(self.tmp) / "snap-kill")
        self.assertIsNotNone(engine.recovery_version)
        manifest = engine.snapshot.load_manifest()
        self.assertEqual(manifest["global_version"], engine.recovery_version)
        engine.run_due(set(engine.states))
        report = engine.build_report(0)
        self.assertEqual(report["success_count"], report["total_sentences"])
        Path(report_path).write_text(json.dumps({"ok": True}), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
