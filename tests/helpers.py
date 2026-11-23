from __future__ import annotations

import tempfile
from pathlib import Path

from near_dup.config import load_config
from near_dup.plan import load_edit_plan
from near_dup.sample_data import generate_samples
from near_dup.text_store import load_text_jsonl, records_by_id


class SampleData:
    def __init__(self) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="near-dup-tests-")
        self.root = Path(self._temp.name)
        generate_samples(str(self.root / "samples"))
        config, failures = load_config(str(self.root / "samples" / "config.yaml"))
        if config is None or failures:
            raise AssertionError([failure.to_dict() for failure in failures])
        records, text_failures = load_text_jsonl(str(self.root / "samples" / "texts.jsonl"), config)
        if text_failures:
            raise AssertionError([failure.to_dict() for failure in text_failures])
        self.config = config
        self.records = records
        self.record_map = records_by_id(records)
        self.plans, plan_failures = load_edit_plan(str(self.root / "samples" / "edit_plan.json"), config)
        if plan_failures:
            raise AssertionError([failure.to_dict() for failure in plan_failures])

    def fresh_snapshot(self, name: str) -> str:
        return str(self.root / "runs" / name)

    def close(self) -> None:
        self._temp.cleanup()

