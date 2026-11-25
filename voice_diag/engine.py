"""End-to-end deterministic/incremental diagnostic pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .arbitration import arbitrate_confusion_pairs
from .cache import ContentCache, summarize_cache_events
from .clustering import build_clusters, pair_key, select_representatives
from .config import RegistrationConfig, load_registration_config
from .corpus import load_clip_labels
from .errors import DiagnosticFailure
from .features import extract_acoustic_template
from .fingerprint import sha256_json, write_json_atomic
from .matrix import build_confusion_matrix
from .quantization import quantize_acoustic, quantization_cache_key
from .recognition import recognition_cache_key, recognize_clip, reference_comparison
from .scheduler import ComparisonTask, schedule_tasks
from .similarity import compute_similarity_base, materialize_similarity, similarity_base_key
from .snapshot import SnapshotStore
from .wavio import read_pcm16_wav, scan_wav_files

RESULT_SCHEMA = 1


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _acoustic_cache_key(
    config: RegistrationConfig,
    *,
    template_key: str,
    relative_path: str,
    content_sha256: str,
) -> str:
    return "acoustic:" + sha256_json(
        {
            "template_key": template_key,
            "path": relative_path,
            "content_sha256": content_sha256,
            "feature_fingerprint": config.feature_fingerprint,
            "frame_length": config.features["frame_length"],
            "frame_shift": config.features["frame_shift"],
            "dimension": config.features["dimension"],
            "projection": config.projection,
        }
    )


def _representative_stage_key(
    quantized: dict[str, dict[str, Any]],
    config: RegistrationConfig,
    similarities: dict[str, dict[str, Any]],
) -> str:
    words: dict[str, list[dict[str, Any]]] = {}
    for template in quantized.values():
        words.setdefault(template["command_word"], []).append(template)
    member_keys = {}
    for word in sorted(words):
        members = sorted(words[word], key=lambda item: item["template_key"])
        member_keys[word] = "representative:" + sha256_json(
            {
                "command_word": word,
                "rule": config.representative_rule,
                "members": [
                    {
                        "key": member["template_key"],
                        "integer_fingerprint": member["integer_fingerprint"],
                        "effective_frames": member.get("effective_frames", {}).get("total", 0),
                    }
                    for member in members
                ],
            }
        )
    return "representatives-stage:" + sha256_json(
        {
            "representatives": member_keys,
            "rule": config.representative_rule,
            "similarities": [
                {
                    "pair_key": key,
                    "similarity_permille": similarities[key]["similarity_permille"]
                    if key in similarities
                    else None,
                }
                for key in sorted(
                    pair_key(first, second)
                    for index, first in enumerate(sorted(quantized))
                    for second in sorted(quantized)[index + 1 :]
                )
            ],
        }
    )


def _pending_for_tasks(tasks: list[ComparisonTask], remaining_budget: int) -> list[dict[str, Any]]:
    return [
        {
            "code": "E_BUDGET_PENDING",
            "stable_sort_key": task.stable_key,
            "stage": task.stage,
            "command_word": task.command_word,
            "template_id": task.template_id,
            "comparison_points": task.cost,
            "references": list(task.references),
            "remaining_budget": max(0, remaining_budget),
        }
        for task in sorted(tasks, key=lambda item: item.stable_key)
    ]


class DiagnosticEngine:
    def __init__(self, config_path: Path | str, state_dir: Path | str):
        self.config_path = Path(config_path)
        self.state_dir = Path(state_dir)

    def run(self, *, mode: str = "auto") -> tuple[dict[str, Any], dict[str, Any]]:
        if mode not in ("auto", "full"):
            raise ValueError("mode must be auto or full")
        failures: list[DiagnosticFailure] = []
        config = load_registration_config(self.config_path, failures)
        snapshot_store = SnapshotStore(self.state_dir)
        if config is None:
            result = self._invalid_config_result(failures)
            report = {
                "run_mode": mode,
                "cache": summarize_cache_events([]),
                "cache_events": [],
                "snapshot": None,
                "failure_count": len(failures),
            }
            return result, report

        cache = ContentCache(self.state_dir / "cache", read_enabled=mode != "full")
        current = snapshot_store.current()
        if mode == "full":
            cache.clear()
        elif current and current.get("structural_config_hash") != config.structural_fingerprint:
            cache.clear()

        parsed_audio = {}
        all_wavs = scan_wav_files(config.workspace)
        for wav_path in all_wavs:
            relative = _relative(wav_path, config.workspace)
            try:
                parsed_audio[relative] = read_pcm16_wav(wav_path, expected_rate=16000)
            except Exception as exc:
                failure = getattr(exc, "failure", None)
                if failure is not None:
                    failures.append(
                        DiagnosticFailure(failure.code, relative, failure.line, failure.field, failure.message)
                    )
                else:
                    failures.append(
                        DiagnosticFailure("E_AUDIO_FORMAT", relative, None, "data", f"unreadable WAV: {exc}")
                    )

        template_specs = config.template_map()
        acoustic_templates: dict[str, dict[str, Any]] = {}
        for template_key in sorted(template_specs):
            spec = template_specs[template_key]
            relative = _relative(spec["path"], config.workspace)
            audio = parsed_audio.get(relative)
            if audio is None:
                if not spec["path"].is_file():
                    failures.append(
                        DiagnosticFailure(
                            "E_AUDIO_FORMAT", relative, None, "path", "referenced template file is missing"
                        )
                    )
                continue
            key = _acoustic_cache_key(
                config,
                template_key=template_key,
                relative_path=relative,
                content_sha256=audio.content_sha256,
            )
            acoustic = cache.get("template-acoustic", key, references=[template_key])
            if acoustic is None:
                acoustic = extract_acoustic_template(
                    audio,
                    config,
                    command_word=spec["command_word"],
                    template_id=spec["template_id"],
                    relative_path=relative,
                )
                acoustic["cache_key"] = key
                cache.put("template-acoustic", key, acoustic, references=[template_key])
            else:
                acoustic = dict(acoustic)
                acoustic["cache_key"] = key
            acoustic_templates[template_key] = acoustic

        quantized: dict[str, dict[str, Any]] = {}
        for template_key in sorted(acoustic_templates):
            acoustic = acoustic_templates[template_key]
            quant_key = quantization_cache_key(acoustic, config.profile, config.fractional_bits)
            quant = cache.get("quantized-template", quant_key, references=[template_key])
            if quant is None:
                quant = quantize_acoustic(acoustic, config.profile, config.fractional_bits)
                cache.put("quantized-template", quant_key, quant, references=[template_key])
            else:
                quant = dict(quant)
                quant["cache_key"] = quant_key
            quantized[template_key] = quant

        clips, labels = load_clip_labels(config, failures)
        valid_clips: list[dict[str, Any]] = []
        for clip_path in clips:
            relative = _relative(clip_path, config.workspace)
            audio = parsed_audio.get(relative)
            label = labels.get(relative)
            if audio is None or label is None:
                continue
            key = _acoustic_cache_key(
                config,
                template_key=f'clip:{label["clip_id"]}',
                relative_path=relative,
                content_sha256=audio.content_sha256,
            )
            acoustic = cache.get("clip-acoustic", key, references=[label["clip_id"]])
            if acoustic is None:
                acoustic = extract_acoustic_template(
                    audio,
                    config,
                    command_word=label["true_command"],
                    template_id=label["clip_id"],
                    relative_path=relative,
                )
                acoustic["cache_key"] = key
                cache.put("clip-acoustic", key, acoustic, references=[label["clip_id"]])
            else:
                acoustic = dict(acoustic)
                acoustic["cache_key"] = key
            quant_key = quantization_cache_key(acoustic, config.profile, config.fractional_bits)
            clip_quant = cache.get("clip-quantized", quant_key, references=[label["clip_id"]])
            if clip_quant is None:
                clip_quant = quantize_acoustic(acoustic, config.profile, config.fractional_bits)
                cache.put("clip-quantized", quant_key, clip_quant, references=[label["clip_id"]])
            else:
                clip_quant = dict(clip_quant)
                clip_quant["cache_key"] = quant_key
            valid_clips.append(
                {
                    "path": clip_path,
                    "relative_path": relative,
                    "label": label,
                    "acoustic": acoustic,
                    "quantized": clip_quant,
                }
            )

        template_keys = sorted(quantized)
        pair_tasks: list[ComparisonTask] = []
        pair_payloads: dict[str, dict[str, Any]] = {}
        for first_index, first_key in enumerate(template_keys):
            for second_key in template_keys[first_index + 1 :]:
                first = quantized[first_key]
                second = quantized[second_key]
                base_key = similarity_base_key(first, second)
                stable_key = f"01-similarity|{pair_key(first_key, second_key)}"
                pair_payloads[stable_key] = {
                    "first": first,
                    "second": second,
                    "cache_key": base_key,
                    "references": [first_key, second_key],
                }
                pair_tasks.append(
                    ComparisonTask(
                        stable_key=stable_key,
                        stage="similarity",
                        command_word=first["command_word"],
                        template_id=first["template_id"],
                        cost=int(config.features["dimension"]),
                        references=(first_key, second_key),
                        payload=pair_payloads[stable_key],
                    )
                )

        def pair_worker(task: ComparisonTask) -> dict[str, Any]:
            payload = pair_payloads[task.stable_key]
            cached = cache.get("similarity", payload["cache_key"], references=payload["references"])
            if cached is not None:
                return cached
            base = compute_similarity_base(payload["first"], payload["second"])
            return cache.put("similarity", payload["cache_key"], base, references=payload["references"])

        pair_results, pair_pending, pair_budget = schedule_tasks(
            pair_tasks,
            pair_worker,
            budget=config.budget,
            max_workers=config.max_workers,
        )
        phase_reports = [
            {
                "phase": "similarity",
                "budget_input_points": config.budget,
                "consumed_points": pair_budget["consumed_points"],
                "scheduled_task_count": pair_budget["scheduled_task_count"],
                "pending_task_count": pair_budget["pending_task_count"],
                "max_workers_used": pair_budget["max_workers_used"],
            }
        ]
        dashboard_peak = [
            {
                "phase": "similarity",
                "peak_concurrency": pair_budget["peak_concurrency"],
            }
        ]
        similarities: dict[str, dict[str, Any]] = {}
        for stable_key in sorted(pair_results):
            base = pair_results[stable_key]
            materialized = materialize_similarity(base, config.confusion_threshold)
            similarities[materialized["pair_key"]] = materialized
        similarities = {key: similarities[key] for key in sorted(similarities)}

        representative_stage_key = _representative_stage_key(quantized, config, similarities)
        cached_rep_stage = cache.get(
            "representatives", representative_stage_key, references=template_keys
        )
        if cached_rep_stage is not None:
            representatives = cached_rep_stage["representatives"]
            representative_stage_key = cached_rep_stage["cache_key"]
        else:
            representatives, _ = select_representatives(quantized, config, similarities)
            member_index: dict[str, list[str]] = {}
            for template in quantized.values():
                member_index.setdefault(template["command_word"], []).append(template["template_key"])
            for word in sorted(representatives):
                member_keys = sorted(member_index.get(word, []))
                comparisons_complete = True
                for first_index, first_key in enumerate(member_keys):
                    for second_key in member_keys[first_index + 1 :]:
                        if pair_key(first_key, second_key) not in similarities:
                            comparisons_complete = False
                cached_word = cache.get(
                    "representative-word",
                    representatives[word]["cache_key"],
                    references=member_keys,
                )
                if cached_word is not None and comparisons_complete:
                    representatives[word] = cached_word
                else:
                    cache.put(
                        "representative-word",
                        representatives[word]["cache_key"],
                        representatives[word],
                        references=member_keys,
                    )
            cached_rep_stage = {
                "cache_key": representative_stage_key,
                "representatives": representatives,
            }
            cache.put(
                "representatives",
                representative_stage_key,
                cached_rep_stage,
                references=template_keys,
            )

        representative_templates = {
            word: quantized[representative["template_key"]]
            for word, representative in representatives.items()
            if representative["template_key"] in quantized
        }
        clusters, cluster_key = build_clusters(quantized, representatives, similarities, config)
        cached_clusters = cache.get("clusters", cluster_key, references=template_keys)
        if cached_clusters is not None:
            clusters = cached_clusters
        else:
            cache.put("clusters", cluster_key, clusters, references=template_keys)
        updated_traces = []
        for trace in clusters["comparison_trace"]:
            cached_trace = cache.get(
                "cluster-trace", trace["cache_key"], references=[trace["command_word"]]
            )
            if cached_trace is not None:
                updated_traces.append(cached_trace)
            else:
                cache.put(
                    "cluster-trace", trace["cache_key"], trace, references=[trace["command_word"]]
                )
                updated_traces.append(trace)
        clusters["comparison_trace"] = updated_traces
        updated_clusters = []
        for cluster in clusters["clusters"]:
            cached_cluster = cache.get(
                "cluster", cluster["cache_key"], references=cluster["references"]
            )
            if cached_cluster is not None:
                updated_clusters.append(cached_cluster)
            else:
                cache.put("cluster", cluster["cache_key"], cluster, references=cluster["references"])
                updated_clusters.append(cluster)
        clusters["clusters"] = updated_clusters
        updated_assignments = []
        for assignment in clusters["assignments"]:
            cached_assignment = cache.get(
                "cluster-assignment",
                assignment["cache_key"],
                references=[assignment["command_word"]],
            )
            if cached_assignment is not None:
                updated_assignments.append(cached_assignment)
            else:
                cache.put(
                    "cluster-assignment",
                    assignment["cache_key"],
                    assignment,
                    references=[assignment["command_word"]],
                )
                updated_assignments.append(assignment)
        clusters["assignments"] = updated_assignments
        cluster_assignments = {
            item["command_word"]: item["cluster_id"] for item in clusters["assignments"]
        }

        recognition_tasks: list[ComparisonTask] = []
        recognition_payloads: dict[str, dict[str, Any]] = {}
        for valid in valid_clips:
            label = valid["label"]
            clip_quant = valid["quantized"]
            expected_key = recognition_cache_key(
                clip_quant["integer_fingerprint"],
                representatives,
                representative_templates,
                reject_threshold_permille=config.reject_threshold,
                overlap_candidates=config.overlap_candidates,
                fractional_bits=config.fractional_bits,
                label=label,
            )
            stable_key = f'02-recognition|{label["true_command"]}|{label["clip_id"]}'
            recognition_payloads[stable_key] = {
                "clip": clip_quant,
                "label": label,
                "cache_key": expected_key,
                "references": [label["clip_id"], *[rep["template_key"] for rep in representatives.values()]],
            }
            recognition_tasks.append(
                ComparisonTask(
                    stable_key=stable_key,
                    stage="recognition",
                    command_word=label["true_command"],
                    template_id=label["clip_id"],
                    cost=int(config.features["dimension"]),
                    references=(label["clip_id"],),
                    payload=recognition_payloads[stable_key],
                )
            )

        def recognition_worker(task: ComparisonTask) -> dict[str, Any]:
            payload = recognition_payloads[task.stable_key]
            cached = cache.get("recognition", payload["cache_key"], references=payload["references"])
            if cached is not None:
                return cached
            result = recognize_clip(
                payload["clip"],
                payload["label"],
                representatives,
                representative_templates,
                reject_threshold_permille=config.reject_threshold,
                overlap_candidates=config.overlap_candidates,
                fractional_bits=config.fractional_bits,
            )
            return cache.put("recognition", payload["cache_key"], result, references=payload["references"])

        if pair_pending:
            remaining_budget = max(0, config.budget - pair_budget["consumed_points"])
            recognition_results = {}
            recognition_pending = _pending_for_tasks(recognition_tasks, remaining_budget)
            recognition_budget = {
                "consumed_points": 0,
                "scheduled_task_count": 0,
                "pending_task_count": len(recognition_pending),
                "max_workers_used": min(config.max_workers, 64),
                "peak_concurrency": 0,
            }
        else:
            remaining_budget = max(0, config.budget - pair_budget["consumed_points"])
            recognition_results, recognition_pending, recognition_budget = schedule_tasks(
                recognition_tasks,
                recognition_worker,
                budget=remaining_budget,
                max_workers=config.max_workers,
            )
        phase_reports.append(
            {
                "phase": "recognition",
                "budget_input_points": remaining_budget,
                "consumed_points": recognition_budget["consumed_points"],
                "scheduled_task_count": recognition_budget["scheduled_task_count"],
                "pending_task_count": recognition_budget["pending_task_count"],
                "max_workers_used": recognition_budget["max_workers_used"],
            }
        )
        dashboard_peak.append(
            {"phase": "recognition", "peak_concurrency": recognition_budget["peak_concurrency"]}
        )
        conclusions = sorted(
            recognition_results.values(), key=lambda item: (item["path"], item["clip_id"])
        )

        boundary_conclusions = [item for item in conclusions if item["boundary"]]
        high_references: dict[str, dict[str, Any]] = {}
        reference_tasks: list[ComparisonTask] = []
        reference_payloads: dict[str, dict[str, Any]] = {}
        high_representative_templates: dict[str, dict[str, Any]] = {}
        for word in sorted(representative_templates):
            low_template = representative_templates[word]
            acoustic = acoustic_templates[low_template["template_key"]]
            high_key = quantization_cache_key(acoustic, config.reference_profile, config.fractional_bits)
            high_quant = cache.get(
                "reference-quantized-template", high_key, references=[low_template["template_key"]]
            )
            if high_quant is None:
                high_quant = quantize_acoustic(acoustic, config.reference_profile, config.fractional_bits)
                cache.put(
                    "reference-quantized-template",
                    high_key,
                    high_quant,
                    references=[low_template["template_key"]],
                )
            else:
                high_quant = dict(high_quant)
                high_quant["cache_key"] = high_key
            high_representative_templates[word] = high_quant

        valid_by_clip_id = {item["label"]["clip_id"]: item for item in valid_clips}
        for low_result in boundary_conclusions:
            clip_id = low_result["clip_id"]
            valid = valid_by_clip_id.get(clip_id)
            if valid is None:
                continue
            acoustic = valid["acoustic"]
            high_clip_key = quantization_cache_key(
                acoustic, config.reference_profile, config.fractional_bits
            )
            high_clip = cache.get("reference-clip-quantized", high_clip_key, references=[clip_id])
            if high_clip is None:
                high_clip = quantize_acoustic(acoustic, config.reference_profile, config.fractional_bits)
                cache.put("reference-clip-quantized", high_clip_key, high_clip, references=[clip_id])
            else:
                high_clip = dict(high_clip)
                high_clip["cache_key"] = high_clip_key
            expected_key = recognition_cache_key(
                high_clip["integer_fingerprint"],
                representatives,
                high_representative_templates,
                reject_threshold_permille=config.reject_threshold,
                overlap_candidates=config.overlap_candidates,
                fractional_bits=config.fractional_bits,
                label=valid["label"],
            )
            stable_key = f'03-reference|{low_result["true_command"]}|{clip_id}'
            reference_payloads[stable_key] = {
                "clip": high_clip,
                "label": valid["label"],
                "cache_key": expected_key,
                "low_result": low_result,
                "references": [clip_id, *[rep["template_key"] for rep in representatives.values()]],
            }
            reference_tasks.append(
                ComparisonTask(
                    stable_key=stable_key,
                    stage="reference-recognition",
                    command_word=low_result["true_command"],
                    template_id=clip_id,
                    cost=int(config.features["dimension"]),
                    references=(clip_id,),
                    payload=reference_payloads[stable_key],
                )
            )

        def reference_worker(task: ComparisonTask) -> dict[str, Any]:
            payload = reference_payloads[task.stable_key]
            high_result = cache.get(
                "reference-recognition", payload["cache_key"], references=payload["references"]
            )
            if high_result is None:
                high_result = recognize_clip(
                    payload["clip"],
                    payload["label"],
                    representatives,
                    high_representative_templates,
                    reject_threshold_permille=config.reject_threshold,
                    overlap_candidates=config.overlap_candidates,
                    fractional_bits=config.fractional_bits,
                )
                cache.put(
                    "reference-recognition",
                    payload["cache_key"],
                    high_result,
                    references=payload["references"],
                )
            comparison = reference_comparison(
                payload["low_result"],
                high_result,
                low_profile=config.profile,
                high_profile=config.reference_profile,
            )
            comparison["topology_source"] = "same_low_width_representative_template_keys"
            comparison["clip_id"] = payload["label"]["clip_id"]
            return {"high_result": high_result, "comparison": comparison}

        if recognition_pending:
            remaining_budget = max(
                0,
                config.budget
                - pair_budget["consumed_points"]
                - recognition_budget["consumed_points"],
            )
            reference_results = {}
            reference_pending = _pending_for_tasks(reference_tasks, remaining_budget)
            reference_budget = {
                "consumed_points": 0,
                "scheduled_task_count": 0,
                "pending_task_count": len(reference_pending),
                "max_workers_used": min(config.max_workers, 64),
                "peak_concurrency": 0,
            }
        else:
            remaining_budget = max(
                0,
                config.budget
                - pair_budget["consumed_points"]
                - recognition_budget["consumed_points"],
            )
            reference_results, reference_pending, reference_budget = schedule_tasks(
                reference_tasks,
                reference_worker,
                budget=remaining_budget,
                max_workers=config.max_workers,
            )
        phase_reports.append(
            {
                "phase": "reference-recognition",
                "budget_input_points": remaining_budget,
                "consumed_points": reference_budget["consumed_points"],
                "scheduled_task_count": reference_budget["scheduled_task_count"],
                "pending_task_count": reference_budget["pending_task_count"],
                "max_workers_used": reference_budget["max_workers_used"],
            }
        )
        dashboard_peak.append(
            {"phase": "reference-recognition", "peak_concurrency": reference_budget["peak_concurrency"]}
        )
        for stable_key in sorted(reference_results):
            result = reference_results[stable_key]
            comparison = result["comparison"]
            high_references[comparison["clip_id"]] = comparison

        confusion_candidates = [
            similarity for similarity in similarities.values() if similarity["candidate"]
        ]
        arbitration_key_input = {
            "candidates": [
                candidate["candidate_cache_key"] for candidate in confusion_candidates
            ],
            "representatives": {
                word: {
                    "template_key": representative["template_key"],
                    "composite_score_permille": representative["composite_score_permille"],
                    "rule": representative["representative_rule"],
                }
                for word, representative in sorted(representatives.items())
            },
            "cluster_assignments": cluster_assignments,
            "cluster_threshold": config.cluster_threshold,
        }
        arbitration_key = "arbitration-stage:" + sha256_json(arbitration_key_input)
        cached_arbitration = cache.get(
            "confusion-arbitration", arbitration_key, references=[item["pair_key"] for item in confusion_candidates]
        )
        if cached_arbitration is None:
            arbitration = arbitrate_confusion_pairs(
                confusion_candidates,
                representatives,
                cluster_assignments,
                cluster_threshold_permille=config.cluster_threshold,
            )
            arbitration_stage = {"cache_key": arbitration_key, "results": arbitration}
            cache.put(
                "confusion-arbitration",
                arbitration_key,
                arbitration_stage,
                references=[item["pair_key"] for item in confusion_candidates],
            )
        else:
            arbitration_stage = cached_arbitration
            arbitration = arbitration_stage["results"]
        updated_arbitration = []
        for item in arbitration:
            cached_pair = cache.get(
                "confusion-arbitration-pair", item["cache_key"], references=item["references"]
            )
            if cached_pair is not None:
                updated_arbitration.append(cached_pair)
            else:
                cache.put(
                    "confusion-arbitration-pair",
                    item["cache_key"],
                    item,
                    references=item["references"],
                )
                updated_arbitration.append(item)
        arbitration = updated_arbitration
        arbitration_stage["results"] = arbitration

        matrix = build_confusion_matrix(
            config,
            conclusions,
            high_references,
            expected_clip_count=len(clips),
            cluster_assignments=cluster_assignments,
        )
        cached_matrix = cache.get(
            "confusion-matrix", matrix["cache_key"], references=[item["cache_key"] for item in conclusions]
        )
        if cached_matrix is not None:
            matrix = cached_matrix
        else:
            cache.put(
                "confusion-matrix",
                matrix["cache_key"],
                matrix,
                references=[item["cache_key"] for item in conclusions],
            )
        updated_cells = []
        for cell in matrix["cells"]:
            cached_cell = cache.get("confusion-matrix-cell", cell["cache_key"], references=cell["references"])
            if cached_cell is not None:
                updated_cells.append(cached_cell)
            else:
                cache.put("confusion-matrix-cell", cell["cache_key"], cell, references=cell["references"])
                updated_cells.append(cell)
        matrix["cells"] = updated_cells

        pending_queue = sorted(
            pair_pending + recognition_pending + reference_pending,
            key=lambda item: item["stable_sort_key"],
        )
        for pending in pending_queue:
            failures.append(
                DiagnosticFailure(
                    "E_BUDGET_PENDING",
                    pending["stable_sort_key"],
                    None,
                    "comparison_budget",
                    f'{pending["stage"]} task retained with {pending["comparison_points"]} required points',
                )
            )
        consumed = (
            pair_budget["consumed_points"]
            + recognition_budget["consumed_points"]
            + reference_budget["consumed_points"]
        )
        budget = {
            "budget_points": config.budget,
            "consumed_points": consumed,
            "remaining_points": max(0, config.budget - consumed),
            "pending_task_count": len(pending_queue),
            "stable_task_order_preserved": True,
            "phases": phase_reports,
        }

        input_digest = sha256_json(
            {
                "config": config.config_fingerprint,
                "audio": [
                    {"path": relative, "content_sha256": parsed_audio[relative].content_sha256}
                    for relative in sorted(parsed_audio)
                ],
                "labels_sha256": _file_sha256(config.labels_path),
            }
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "config": {
                "version": config.version,
                "config_fingerprint": config.config_fingerprint,
                "structural_config_hash": config.structural_fingerprint,
                "feature_fingerprint": config.feature_fingerprint,
                "runtime_fingerprint": config.runtime_fingerprint,
                "sample_rate": config.audio["sample_rate"],
                "channels": config.audio["channels"],
                "bit_depth": config.audio["bit_depth"],
                "frame_length": config.features["frame_length"],
                "frame_shift": config.features["frame_shift"],
                "feature_dimension": config.features["dimension"],
                "profile": config.profile,
                "fractional_bits": config.fractional_bits,
                "reference_profile": config.reference_profile,
                "reject_threshold_permille": config.reject_threshold,
                "confusion_candidate_threshold_permille": config.confusion_threshold,
                "cluster_merge_threshold_permille": config.cluster_threshold,
                "representative_rule": config.representative_rule,
                "max_workers": config.max_workers,
                "comparison_budget": config.budget,
            },
            "corpus": {
                "audio_file_count": len(all_wavs),
                "valid_audio_file_count": len(parsed_audio),
                "template_count": len(acoustic_templates),
                "clip_count": len(clips),
                "valid_labeled_clip_count": len(valid_clips),
                "input_digest": input_digest,
            },
            "failures": [failure.as_dict() for failure in failures],
            "templates": [
                {
                    "template_key": key,
                    "acoustic": acoustic_templates[key],
                    "quantized": quantized[key],
                }
                for key in sorted(acoustic_templates)
            ],
            "similarities": list(similarities.values()),
            "confusion_candidates": confusion_candidates,
            "representatives": [representatives[word] for word in sorted(representatives)],
            "clusters": clusters,
            "conclusions": conclusions,
            "confusion_arbitration": arbitration,
            "confusion_matrix": matrix,
            "budget": budget,
            "pending_queue": pending_queue,
            "stage_cache_keys": {
                "representatives": representative_stage_key,
                "clusters": clusters["cache_key"],
                "confusion_arbitration": arbitration_key,
                "confusion_matrix": matrix["cache_key"],
            },
        }

        cache_events = sorted(
            cache.events, key=lambda event: (event["stage"], event["cache_key"], event["status"])
        )
        for sequence, event in enumerate(cache_events, 1):
            event["sequence"] = sequence
        cache_summary = summarize_cache_events(cache_events)
        snapshot_metadata = {
            "config_version": config.version,
            "structural_config_hash": config.structural_fingerprint,
            "runtime_config_hash": config.runtime_fingerprint,
            "config_fingerprint": config.config_fingerprint,
            "input_digest": input_digest,
            "run_mode": mode,
        }
        stages = {
            "audio_manifest": {
                "files": [
                    {
                        "path": relative,
                        "content_sha256": parsed_audio[relative].content_sha256,
                        "sample_count": parsed_audio[relative].sample_count,
                    }
                    for relative in sorted(parsed_audio)
                ]
            },
            "template_acoustic": [
                acoustic_templates[key] for key in sorted(acoustic_templates)
            ],
            "quantized_templates": [quantized[key] for key in sorted(quantized)],
            "similarities": list(similarities.values()),
            "clusters": clusters,
            "conclusions": conclusions,
            "confusion_arbitration": arbitration,
            "confusion_matrix": matrix,
        }
        completed_cursors = {
            "last_completed_stage": "confusion_matrix",
            "completed_command_words": sorted(representatives),
            "pending_stage_cursor": pending_queue[0]["stable_sort_key"] if pending_queue else None,
        }
        run_report = {
            "run_mode": mode,
            "cache": cache_summary,
            "cache_events": cache_events,
            "phase_peak_concurrency": dashboard_peak,
            "snapshot": None,
            "failure_count": len(failures),
            "stderr_marker": "E_BUDGET_PENDING" if pending_queue else None,
        }
        pointer = snapshot_store.commit(
            result=result,
            run_report=run_report,
            stages=stages,
            pending_queue=pending_queue,
            completed_cursors=completed_cursors,
            metadata=snapshot_metadata,
        )
        run_report["snapshot"] = {
            "state_version": pointer["state_version"],
            "snapshot_dir": pointer["snapshot_dir"],
            "result_file": pointer["result_file"],
            "report_file": pointer["report_file"],
        }
        return result, run_report

    def _invalid_config_result(self, failures: list[DiagnosticFailure]) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA,
            "config": {"config_path": _relative(self.config_path, self.config_path.parent)},
            "corpus": {},
            "failures": [failure.as_dict() for failure in failures],
            "templates": [],
            "similarities": [],
            "confusion_candidates": [],
            "representatives": [],
            "clusters": {"clusters": [], "assignments": [], "comparison_trace": []},
            "conclusions": [],
            "confusion_arbitration": [],
            "confusion_matrix": {"cells": [], "totals": {}, "consistency": {"consistent": False}},
            "budget": {},
            "pending_queue": [],
            "stage_cache_keys": {},
        }


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def run_diagnostics(
    config_path: Path | str,
    state_dir: Path | str,
    *,
    result_path: Path | str | None = None,
    report_path: Path | str | None = None,
    mode: str = "auto",
) -> tuple[dict[str, Any], dict[str, Any]]:
    engine = DiagnosticEngine(config_path, state_dir)
    result, report = engine.run(mode=mode)
    if result_path is not None:
        write_json_atomic(Path(result_path), result)
    if report_path is not None:
        write_json_atomic(Path(report_path), report)
    return result, report
