"""Reproducible acceptance scenario covering additive and mutating edits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine import run_diagnostics
from .fingerprint import sha256_json, write_json_atomic
from .synth import WordProfile, generate_demo, synthesize_word_audio


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _append_labels(path: Path, labels: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for label in labels:
            handle.write(json.dumps(label, ensure_ascii=False, sort_keys=True) + "\n")


def _add_word(root: Path, config: dict[str, Any], profile: WordProfile, *, seed: int) -> None:
    template_entries = []
    for index in (1, 2):
        template_id = f"T{index:02d}"
        path = root / "templates" / profile.command_word / f"{template_id}.wav"
        synthesize_word_audio(path, profile, seed=seed, variant_key=f"added-template-{index}")
        template_entries.append(
            {"template_id": template_id, "path": f"templates/{profile.command_word}/{template_id}.wav"}
        )
    config["words"].append({"command_word": profile.command_word, "templates": template_entries})
    config["words"].sort(key=lambda item: item["command_word"])
    new_labels = []
    for index in (1, 2):
        clip_id = f"pos_{profile.command_word.lower()}_{index:02d}"
        path = root / "clips" / f"{clip_id}.wav"
        synthesize_word_audio(path, profile, seed=seed, variant_key=f"added-clip-{index}")
        new_labels.append(
            {
                "clip_id": clip_id,
                "path": f"clips/{path.name}",
                "true_command": profile.command_word,
                "confusion_with": profile.confusion_with,
                "kind": "positive",
                "seed": seed,
                "variant_index": index,
            }
        )
    _append_labels(root / "labels.jsonl", new_labels)
    annotations = config.setdefault("annotations", {}).setdefault("confusion_pairs", [])
    if profile.confusion_with:
        annotations.append([profile.command_word, profile.confusion_with])
        annotations.sort()


def _remove_template(root: Path, config: dict[str, Any], command_word: str, template_id: str) -> None:
    for word in config["words"]:
        if word["command_word"] != command_word:
            continue
        kept = []
        for template in word["templates"]:
            if template["template_id"] == template_id:
                (root / template["path"]).unlink(missing_ok=True)
            else:
                kept.append(template)
        word["templates"] = kept
        return
    raise KeyError(f"template not found: {command_word}:{template_id}")


def _replace_template(root: Path, profile: WordProfile, command_word: str, template_id: str, seed: int) -> None:
    path = root / "templates" / command_word / f"{template_id}.wav"
    synthesize_word_audio(path, profile, seed=seed, variant_key=f"replacement-{command_word}-{template_id}-7")


def run_acceptance_scenario(root: Path, *, seed: int = 20261126) -> dict[str, Any]:
    root = root.resolve()
    corpus = root / "corpus"
    generate_demo(corpus, seed=seed)
    incremental_state = root / "state-incremental"
    full_state = root / "state-full"
    config_path = corpus / "config.json"

    initial_result, initial_report = run_diagnostics(
        config_path,
        incremental_state,
        result_path=root / "result-initial.json",
        report_path=root / "report-initial.json",
        mode="auto",
    )
    edit_steps: list[dict[str, Any]] = []

    config = _load(config_path)
    added_profile = WordProfile("RA", 1, 410.0, 1830.0, 2670.0, 0.18, "LIN")
    _add_word(corpus, config, added_profile, seed=seed)
    _save(config_path, config)
    result, report = run_diagnostics(config_path, incremental_state)
    edit_steps.append(
        {
            "operation": "add_word",
            "command_word": "RA",
            "canonical_result_sha256": sha256_json(result),
            "reuse_hit_count": report["cache"]["reuse_hit_count"],
        }
    )

    ma_profile = next(
        profile for profile in (WordProfile("MA", 1, 745.0, 1110.0, 2450.0, 0.025, "NA"),)
    )
    _replace_template(corpus, ma_profile, "MA", "T02", seed)
    result, report = run_diagnostics(config_path, incremental_state)
    edit_steps.append(
        {
            "operation": "replace_template",
            "template_key": "MA:T02",
            "canonical_result_sha256": sha256_json(result),
            "reuse_hit_count": report["cache"]["reuse_hit_count"],
        }
    )

    config = _load(config_path)
    config["registration"]["template_min"] = 1
    _save(config_path, config)
    _remove_template(corpus, config, "NA", "T03")
    _save(config_path, config)
    result, report = run_diagnostics(config_path, incremental_state)
    _remove_template(corpus, config, "NA", "T02")
    _save(config_path, config)
    result, report = run_diagnostics(config_path, incremental_state)
    na_representative = next(
        item for item in result["representatives"] if item["command_word"] == "NA"
    )
    edit_steps.append(
        {
            "operation": "delete_templates_to_degenerate",
            "command_word": "NA",
            "degenerate": na_representative["degenerate"],
            "skip_cross_word_merge": na_representative["skip_cross_word_merge"],
            "canonical_result_sha256": sha256_json(result),
            "reuse_hit_count": report["cache"]["reuse_hit_count"],
        }
    )

    config = _load(config_path)
    config["quantization"]["profile"] = "q16"
    config["quantization"]["fractional_bits"] = 6
    config["thresholds"]["reject_permille"] = 1000
    config["thresholds"]["confusion_candidate_permille"] = 850
    config["thresholds"]["cluster_merge_permille"] = 999
    config["registration"]["representative_rule"] = "most_frames"
    config["registration"]["comparison_budget"] = 300000
    _save(config_path, config)
    incremental_result, incremental_report = run_diagnostics(
        config_path,
        incremental_state,
        result_path=root / "result-incremental.json",
        report_path=root / "report-incremental.json",
        mode="auto",
    )
    edit_steps.append(
        {
            "operation": "quantization_thresholds_and_representative_rule",
            "profile": "q16",
            "fractional_bits": 6,
            "reject_permille": 1000,
            "confusion_candidate_permille": 850,
            "cluster_merge_permille": 999,
            "representative_rule": "most_frames",
            "canonical_result_sha256": sha256_json(incremental_result),
            "reuse_hit_count": incremental_report["cache"]["reuse_hit_count"],
        }
    )

    full_result, full_report = run_diagnostics(
        config_path,
        full_state,
        result_path=root / "result-full.json",
        report_path=root / "report-full.json",
        mode="full",
    )
    exact_equal = incremental_result == full_result
    summary = {
        "seed": seed,
        "edit_steps": edit_steps,
        "final_incremental_result_sha256": sha256_json(incremental_result),
        "final_full_result_sha256": sha256_json(full_result),
        "canonical_results_exactly_equal": exact_equal,
        "incremental_reuse_hit_count": incremental_report["cache"]["reuse_hit_count"],
        "incremental_computed_count": incremental_report["cache"]["computed_count"],
        "full_computed_count": full_report["cache"]["computed_count"],
        "failure_count_incremental": len(incremental_result["failures"]),
        "failure_count_full": len(full_result["failures"]),
        "degenerate_words": [
            item["command_word"] for item in incremental_result["representatives"] if item["degenerate"]
        ],
        "cluster_count": len(incremental_result["clusters"]["clusters"]),
        "confusion_candidate_count": len(incremental_result["confusion_candidates"]),
        "conclusion_count": len(incremental_result["conclusions"]),
        "rejected_conclusion_count": sum(
            1 for item in incremental_result["conclusions"] if item["rejected"]
        ),
        "matrix_consistency": incremental_result["confusion_matrix"]["consistency"],
        "budget_pending_count": len(incremental_result["pending_queue"]),
        "initial_failure_count": len(initial_result["failures"]),
        "initial_conclusion_count": len(initial_result["conclusions"]),
    }
    write_json_atomic(root / "scenario-report.json", summary)
    if not exact_equal:
        raise AssertionError("incremental and full canonical results differ")
    return summary
