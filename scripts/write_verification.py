from __future__ import annotations

import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    normal = load(root / "artifacts/report.json")
    rerun = load(root / "artifacts/report_rerun.json")
    sequence = load(root / "artifacts/report_sequence.json")
    injected = load(root / "artifacts/report_injected.json")
    pytest_text = (root / "artifacts/pytest.txt").read_text(encoding="utf-8")
    pytest_tokens = pytest_text.strip().replace("\n", " ").split()
    passed = (
        int(pytest_tokens[pytest_tokens.index("passed") - 1])
        if "passed" in pytest_tokens
        else None
    )
    verification = {
        "pytest": {
            "summary": pytest_text.strip(),
            "passed": passed,
        },
        "normal_run": {
            "status": normal["status"],
            "format_versions": normal["format_versions"],
            "render_signature_sha256": normal["render_signature_sha256"],
            "cache_summary": normal["cache_summary"],
            "concurrency_comparison": normal["concurrency_comparison"],
            "conversion_loss_count": len(normal["conversion_degradation_losses"]),
            "clip_pcm_sha256": {
                clip["clip_id"]: clip["output_pcm_sha256"] for clip in normal["clips"]
            },
        },
        "warm_rerun": {
            "cache_summary": rerun["cache_summary"],
            "render_signature_sha256": rerun["render_signature_sha256"],
            "same_signature_as_first_run": (
                rerun["render_signature_sha256"]
                == normal["render_signature_sha256"]
            ),
        },
        "edit_sequence": {
            "all_edits_match_full": sequence["edit_sequence"]["all_edits_match_full"],
            "all_event_orders_match_serial": sequence["edit_sequence"][
                "all_event_orders_match_serial"
            ],
            "all_failure_orders_match_serial": sequence["edit_sequence"][
                "all_failure_orders_match_serial"
            ],
            "edit_count": sequence["edit_sequence"]["edit_count"],
            "edit_summaries": [
                {
                    "edit_id": item["edit_id"],
                    "matches_full": item["comparison"]["matches"],
                    "reuse_count": item["cache_summary"]["reuse_count"],
                    "recompute_count": item["cache_summary"]["recompute_count"],
                }
                for item in sequence["edit_sequence"]["edits"]
            ],
        },
        "injections": {
            "failure_count": len(injected["failures"]),
            "all_hit": all(item["hit"] for item in injected["injection_summary"]),
            "all_expected_codes_match": all(
                item["expected_code_matched"] for item in injected["injection_summary"]
            ),
            "items": [
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "observed_codes": item["observed_codes"],
                    "hit": item["hit"],
                }
                for item in injected["injection_summary"]
            ],
        },
        "snapshot_recovery": {
            "normal_recovery_point": normal["recovery"].get("point"),
            "normal_recovery_stage_versions": normal["recovery"].get("stage_versions"),
            "sequence_recovery_used": sequence["recovery"]["used"],
            "sequence_recovery_version": sequence["recovery"]["snapshot_version"],
        },
    }
    path = root / "artifacts/verification.json"
    path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "wrote", "path": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
