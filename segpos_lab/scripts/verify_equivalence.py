"""验收辅助：任意命令序列后，增量结果与最终状态全量结果逐字段比对。"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segpos.loader import load_commands  # noqa: E402
from segpos.runner import _materialize_commands, build_engine, prepare_bundle  # noqa: E402

DROP_TAG_KEYS = ("marginal_log_sum",)


def comparable(report: dict) -> dict:
    sentences = {}
    for sid, part in report["sentences"].items():
        sentences[sid] = {
            "segment": part["segment"],
            "oov": part["oov"],
            "tag": part["tag"],
            "segment_meta": part["segment_meta"],
            "oov_meta": part["oov_meta"],
            "tag_meta": {k: v for k, v in part["tag_meta"].items() if k not in DROP_TAG_KEYS},
        }
    return {
        "sentences": sentences,
        "low_confidence": report["low_confidence"],
        "failures": report["failures"],
        "totals": {
            "total": report["total_sentences"],
            "success": report["success_count"],
            "segment_tokens": report["segment_token_total"],
            "oov_tokens": report["oov_token_total"],
        },
    }


def main(sample_dir: str) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="segpos-verify-"))
    try:
        bundle = prepare_bundle(sample_dir)[0]
        commands, command_failures = load_commands(bundle["base"] / "commands.json")
        materialized, resolve_failures = _materialize_commands(commands, bundle)

        full_bundle = prepare_bundle(sample_dir)[0]
        full_engine = build_engine(full_bundle, tmp / "full")
        for command in materialized:
            full_engine.apply_command(command)
        full_engine.run_due(set(full_engine.states))
        full_report = full_engine.build_report(len(materialized))

        for chunk_size in (1, 2, 1000):
            inc_bundle = prepare_bundle(sample_dir)[0]
            engine = build_engine(inc_bundle, tmp / f"inc-{chunk_size}")
            engine.failures.extend(command_failures)
            engine.failures.extend(resolve_failures)
            for start in range(0, len(materialized), chunk_size):
                engine.run_command_stream(materialized[start : start + chunk_size])
            inc_report = engine.build_report(len(materialized))
            if comparable(inc_report) != comparable(full_report):
                print(f"MISMATCH at chunk_size={chunk_size}")
                return 1
            print(f"chunk_size={chunk_size}: identical")
        print("EQUIVALENCE_OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "samples/base"))
