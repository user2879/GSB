from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .config import AdjudicationConfig, AnalysisConfig
from .util import json_hash, round_half_away_from_zero


def _posterior_map(value: Mapping[Any, Any]) -> dict[int, int]:
    result: dict[int, int] = {}
    for key, probability in value.items():
        result[int(key)] = int(probability)
    return {code: result[code] for code in sorted(result)}


def _winning_posterior(posterior: Mapping[int, int]) -> tuple[int | None, int, int | None, int]:
    ordered = sorted(((int(code), int(prob)) for code, prob in posterior.items()), key=lambda item: (-item[1], item[0]))
    if not ordered:
        return None, 0, None, 0
    best_code, best_probability = ordered[0]
    if len(ordered) > 1:
        return best_code, best_probability, ordered[1][0], best_probability - ordered[1][1]
    return best_code, best_probability, None, best_probability


def build_window_attribution(
    *,
    window_id: str,
    window_index: int,
    sample_start: int,
    sample_end: int,
    frame_indices: Iterable[int],
    sources: Iterable[Mapping[str, Any]],
    frame_count: int,
    frame_length: int,
    frame_shift: int,
    adjudication: AdjudicationConfig,
) -> dict[str, Any]:
    """Create one deterministic, cacheable window attribution contribution.

    Ownership weight is the literal product of posterior margin, source trust,
    and the number of frames in which that source overlaps the window.  Keeping
    all three factors in the ledger makes a later reviewer able to reconstruct
    every overlap decision.
    """

    source_list = sorted((dict(source) for source in sources), key=lambda item: str(item["interval_id"]))
    contributions: list[dict[str, Any]] = []
    for frame_index in sorted(int(item) for item in frame_indices):
        if frame_index < 0 or frame_index >= frame_count:
            continue
        center = frame_index * frame_shift + frame_length // 2
        active_sources = [
            source
            for source in source_list
            if int(source["interval_start_sample"]) <= center < int(source["interval_end_sample"])
        ]
        for source in active_sources:
            margin = max(0, int(source.get("margin_ppm", 0)))
            trust = max(0, int(source.get("trust_ppm", 0)))
            overlap_frames = max(0, int(source.get("overlap_frames", 0)))
            weight = margin * trust * overlap_frames
            contributions.append(
                {
                    "frame_index": frame_index,
                    "frame_center_sample": center,
                    "interval_id": str(source["interval_id"]),
                    "speaker_id": str(source["speaker_id"]),
                    "interval_start_sample": int(source["interval_start_sample"]),
                    "interval_end_sample": int(source["interval_end_sample"]),
                    "posterior_ppm": _posterior_map(source["posterior_ppm"]),
                    "margin_ppm": margin,
                    "best_code": source.get("best_code"),
                    "runner_up_code": source.get("runner_up_code"),
                    "trust_ppm": trust,
                    "overlap_frames": overlap_frames,
                    "speech_ppm": int(source.get("speech_ppm", 0)),
                    "feature_digest": str(source.get("feature_digest", "")),
                    "weight": weight,
                }
            )
    contributions.sort(key=lambda item: (item["frame_index"], item["interval_id"]))
    payload = {
        "window_id": window_id,
        "window_index": window_index,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "frame_indices": [int(item) for item in frame_indices],
        "sources": [
            {
                key: source[key]
                for key in (
                    "interval_id",
                    "speaker_id",
                    "interval_start_sample",
                    "interval_end_sample",
                    "overlap_frames",
                    "trust_ppm",
                    "speech_ppm",
                    "margin_ppm",
                    "feature_digest",
                )
                if key in source
            }
            for source in source_list
        ],
        "contributions": contributions,
        "adjudication_parameters": adjudication.as_parameters(),
    }
    payload["attribution_digest"] = json_hash(payload)
    return payload


def adjudicate_frames(
    *,
    attribution_payloads: Iterable[Mapping[str, Any]],
    frame_metrics: Iterable[Mapping[str, Any]],
    candidate_mask: Iterable[bool],
    analysis: AnalysisConfig,
    adjudication: AdjudicationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_list = list(frame_metrics)
    mask_list = list(candidate_mask)
    by_frame: dict[int, list[Mapping[str, Any]]] = {}
    for payload in sorted(
        attribution_payloads,
        key=lambda item: (int(item.get("window_index", 0)), str(item.get("window_id", ""))),
    ):
        for contribution in payload.get("contributions", []):
            frame_index = int(contribution["frame_index"])
            by_frame.setdefault(frame_index, []).append(contribution)
    frames: list[dict[str, Any]] = []
    overlap_frame_indices: list[int] = []
    boundary_frame_indices: list[int] = []
    contested_frame_count = 0
    quant_half = max(analysis.posterior_quantum_ppm // 2, adjudication.tie_half_step_ppm)
    for metric in metric_list:
        frame_index = int(metric["frame_index"])
        rows = sorted(
            by_frame.get(frame_index, []),
            key=lambda item: (str(item["interval_id"]), int(item["weight"])),
        )
        interval_weights: dict[str, int] = {}
        interval_info: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            interval_id = str(row["interval_id"])
            interval_weights[interval_id] = interval_weights.get(interval_id, 0) + int(row["weight"])
            interval_info[interval_id] = row
        active_intervals = sorted(interval_weights)
        if len(active_intervals) > 1:
            overlap_frame_indices.append(frame_index)
        ownership_ppm: dict[str, int] = {}
        total_interval_weight = sum(interval_weights.values())
        for interval_id in active_intervals:
            if total_interval_weight > 0:
                ownership_ppm[interval_id] = round_half_away_from_zero(
                    interval_weights[interval_id] * 1_000_000.0 / total_interval_weight
                )
            else:
                ownership_ppm[interval_id] = round_half_away_from_zero(1_000_000.0 / len(active_intervals))
        winner_interval: str | None = None
        boundary = False
        interval_arbitration = "no_source"
        ownership_margin = 0
        if active_intervals:
            ranked_intervals = sorted(
                active_intervals,
                key=lambda interval_id: (
                    -interval_weights[interval_id],
                    int(interval_info[interval_id]["interval_start_sample"]),
                    str(interval_info[interval_id]["speaker_id"]),
                    interval_id,
                ),
            )
            winner_interval = ranked_intervals[0]
            if len(ranked_intervals) > 1:
                top_weight = interval_weights[ranked_intervals[0]]
                second_weight = interval_weights[ranked_intervals[1]]
                normalized_margin = abs(ownership_ppm[ranked_intervals[0]] - ownership_ppm[ranked_intervals[1]])
                ownership_margin = normalized_margin
                all_equal_zero_weight = total_interval_weight == 0
                if normalized_margin <= adjudication.tie_half_step_ppm or all_equal_zero_weight:
                    candidates = [
                        interval_id
                        for interval_id in active_intervals
                        if all_equal_zero_weight
                        or abs(top_weight - interval_weights[interval_id]) * 1_000_000
                        <= max(total_interval_weight, 1) * adjudication.tie_half_step_ppm
                    ]
                    winner_interval = sorted(
                        candidates,
                        key=lambda interval_id: (
                            int(interval_info[interval_id]["interval_start_sample"]),
                            str(interval_info[interval_id]["speaker_id"]),
                            interval_id,
                        ),
                    )[0]
                    boundary = True
                    interval_arbitration = "interval_tie_start_then_speaker"
                else:
                    interval_arbitration = "higher_margin_x_trust_x_overlap"
            else:
                interval_arbitration = "single_interval"

        posterior_sums: dict[int, int] = {}
        speech_weighted = 0
        speech_weight = 0
        winner_rows = [row for row in rows if str(row["interval_id"]) == winner_interval]
        for row in winner_rows:
            weight = int(row["weight"])
            for code, probability in _posterior_map(row["posterior_ppm"]).items():
                posterior_sums[code] = posterior_sums.get(code, 0) + probability * weight
        total_posterior_weight = sum(int(row["weight"]) for row in winner_rows)
        if total_posterior_weight > 0 and posterior_sums:
            posterior_ppm = {
                code: round_half_away_from_zero(posterior_sums[code] / total_posterior_weight)
                for code in sorted(posterior_sums)
            }
            remainder = 1_000_000 - sum(posterior_ppm.values())
            best_code_for_remainder = max(sorted(posterior_ppm), key=lambda code: posterior_ppm[code])
            posterior_ppm[best_code_for_remainder] += remainder
        elif winner_rows:
            posterior_ppm = _posterior_map(winner_rows[0]["posterior_ppm"])
        else:
            posterior_ppm = {}
        command_code, command_probability, runner_up_command, command_margin = _winning_posterior(posterior_ppm)
        command_arbitration = "uncontested"
        if len(winner_rows) > 1:
            command_arbitration = "weighted_window_posterior"
        other_best: list[tuple[str, int, int]] = []
        for interval_id in active_intervals:
            if interval_id == winner_interval:
                continue
            best_for_other = _winning_posterior(_posterior_map(interval_info[interval_id]["posterior_ppm"]))[0]
            if best_for_other is not None:
                other_best.append((interval_id, best_for_other, interval_weights[interval_id]))
        contested = any(other_code != command_code for _, other_code, _ in other_best)
        if contested:
            contested_frame_count += 1
            command_arbitration = "higher_interval_ownership" if not boundary else interval_arbitration
        if runner_up_command is not None and command_margin <= quant_half:
            command_candidates = [
                code for code, probability in posterior_ppm.items() if command_probability - probability <= quant_half
            ]
            command_code = min(command_candidates)
            command_probability = posterior_ppm[command_code]
            command_arbitration = "command_tie_code_ascending"
        for row in rows:
            weight = int(row["weight"])
            speech_weighted += int(row["speech_ppm"]) * max(0, weight)
            speech_weight += max(0, weight)
        if speech_weight > 0:
            speech_ppm = round_half_away_from_zero(speech_weighted / speech_weight)
        elif rows:
            speech_ppm = round_half_away_from_zero(sum(int(row["speech_ppm"]) for row in rows) / len(rows))
        else:
            speech_ppm = 0
        candidate_active = bool(mask_list[frame_index]) if frame_index < len(mask_list) else True
        active = bool(winner_interval is not None and candidate_active and speech_ppm >= adjudication.frame_endpoint_threshold_ppm)
        if boundary:
            boundary_frame_indices.append(frame_index)
        runner_up_interval = None
        runner_up_command_probability = 0
        if other_best:
            sorted_other = sorted(other_best, key=lambda item: (-item[2], item[0]))
            runner_up_interval, runner_up_command, _ = sorted_other[0]
            runner_up_command_probability = posterior_ppm.get(runner_up_command, 0)
        elif runner_up_command is not None:
            runner_up_command_probability = posterior_ppm.get(runner_up_command, 0)
        frame_decision = {
            "frame_index": frame_index,
            "sample_start": int(metric.get("start_sample", frame_index * analysis.frame_shift)),
            "sample_end": int(metric.get("end_sample", frame_index * analysis.frame_shift + analysis.frame_length)),
            "frame_center_sample": frame_index * analysis.frame_shift + analysis.frame_length // 2,
            "active": active,
            "candidate_active": candidate_active,
            "speech_ppm": speech_ppm,
            "winning_interval_id": winner_interval,
            "winning_speaker_id": str(interval_info[winner_interval]["speaker_id"]) if winner_interval else None,
            "active_intervals": active_intervals,
            "overlap": len(active_intervals) > 1,
            "ownership_ppm": {key: ownership_ppm[key] for key in sorted(ownership_ppm)},
            "ownership_margin_ppm": ownership_margin,
            "command_code": command_code if active else None,
            "command_probability_ppm": command_probability if active else 0,
            "posterior_ppm": {str(code): value for code, value in sorted(posterior_ppm.items())} if active else {},
            "runner_up_interval_id": runner_up_interval,
            "runner_up_command_code": runner_up_command,
            "runner_up_command_probability_ppm": runner_up_command_probability if active else 0,
            "contested": bool(contested and active),
            "boundary": boundary,
            "interval_arbitration": interval_arbitration,
            "command_arbitration": command_arbitration if active else "inactive",
        }
        frames.append(frame_decision)
    summary = {
        "overlap_frame_count": len(overlap_frame_indices),
        "overlap_frame_indices": overlap_frame_indices,
        "boundary_frame_indices": boundary_frame_indices,
        "contested_frame_count": contested_frame_count,
        "frame_count": len(frames),
        "active_frame_count": sum(1 for frame in frames if frame["active"]),
    }
    return frames, summary


def _group_frame_commands(
    frames: Iterable[Mapping[str, Any]],
    *,
    min_command_frames: int,
    analysis: AnalysisConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    active_frames = [dict(frame) for frame in frames if frame.get("active")]
    groups: list[list[dict[str, Any]]] = []
    for frame in active_frames:
        key = (frame.get("winning_interval_id"), frame.get("command_code"))
        if not groups:
            groups.append([frame])
            continue
        previous = groups[-1][-1]
        previous_key = (previous.get("winning_interval_id"), previous.get("command_code"))
        if key == previous_key and int(frame["frame_index"]) == int(previous["frame_index"]) + 1:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    accepted: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for group in groups:
        first = group[0]
        last = group[-1]
        contested = sum(1 for item in group if item["contested"])
        arbitration_reasons: dict[str, int] = {}
        for item in group:
            reason = str(item["command_arbitration"])
            arbitration_reasons[reason] = arbitration_reasons.get(reason, 0) + 1
        segment = {
            "start_frame": int(first["frame_index"]),
            "end_frame": int(last["frame_index"]),
            "frame_count": len(group),
            "start_sample": int(first.get("sample_start", first.get("start_sample", 0))),
            "end_sample": int(last.get("sample_end", last.get("end_sample", 0))),
            "interval_id": first.get("winning_interval_id"),
            "speaker_id": first.get("winning_speaker_id"),
            "command_code": first.get("command_code"),
            "mean_speech_ppm": round_half_away_from_zero(
                sum(int(item["speech_ppm"]) for item in group) / len(group)
            ),
            "mean_command_probability_ppm": round_half_away_from_zero(
                sum(int(item["command_probability_ppm"]) for item in group) / len(group)
            ),
            "contested_frame_count": contested,
            "arbitration_reasons": {key: arbitration_reasons[key] for key in sorted(arbitration_reasons)},
            "boundary_frames": [int(item["frame_index"]) for item in group if item["boundary"]],
        }
        if len(group) >= min_command_frames:
            segment["status"] = "accepted"
            accepted.append(segment)
        else:
            segment["status"] = "suppressed_too_short"
            suppressed.append(segment)
    return accepted, suppressed


def merge_command_conclusions(
    frames: Iterable[Mapping[str, Any]],
    *,
    analysis: AnalysisConfig,
    adjudication: AdjudicationConfig,
) -> dict[str, Any]:
    minimum_frames = max(
        1,
        -(-adjudication.min_command_ms * 16 // analysis.frame_shift),
    )
    accepted, suppressed = _group_frame_commands(
        frames,
        min_command_frames=minimum_frames,
        analysis=analysis,
    )
    return {
        "minimum_command_frames": minimum_frames,
        "accepted": accepted,
        "suppressed": suppressed,
        "accepted_count": len(accepted),
        "suppressed_count": len(suppressed),
    }


def aggregate_interval_evidence(
    *,
    sources: Iterable[Mapping[str, Any]],
    analysis: AnalysisConfig,
) -> list[dict[str, Any]]:
    """Aggregate cached per-window source evidence for interval receipts."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for source in sources:
        grouped.setdefault(str(source["interval_id"]), []).append(source)
    result: list[dict[str, Any]] = []
    quant_half = max(1, analysis.posterior_quantum_ppm // 2)
    for interval_id in sorted(grouped):
        rows = sorted(
            grouped[interval_id],
            key=lambda item: (int(item.get("window_index", 0)), str(item.get("window_id", ""))),
        )
        posterior_sums: dict[int, int] = {}
        speech_sum = 0
        speech_weight = 0
        total_weight = 0
        speaker_id = str(rows[0]["speaker_id"])
        window_ids: list[str] = []
        for row in rows:
            weight = max(1, int(row.get("overlap_samples", 0))) * max(1, int(row.get("trust_ppm", 1)))
            total_weight += weight
            speech_sum += int(row.get("speech_ppm", 0)) * weight
            speech_weight += weight
            window_ids.append(str(row.get("window_id", "")))
            for code, probability in _posterior_map(row["posterior_ppm"]).items():
                posterior_sums[code] = posterior_sums.get(code, 0) + probability * weight
        if total_weight > 0 and posterior_sums:
            posterior_ppm = {
                code: round_half_away_from_zero(posterior_sums[code] / total_weight)
                for code in sorted(posterior_sums)
            }
            remainder = 1_000_000 - sum(posterior_ppm.values())
            best_code_for_remainder = max(sorted(posterior_ppm), key=lambda code: posterior_ppm[code])
            posterior_ppm[best_code_for_remainder] += remainder
        else:
            posterior_ppm = {}
        best, best_probability, runner_up, margin = _winning_posterior(posterior_ppm)
        arbitration = "highest_interval_aggregate_posterior"
        if runner_up is not None and margin <= quant_half:
            best = min(
                code for code, probability in posterior_ppm.items() if best_probability - probability <= quant_half
            )
            best_probability = posterior_ppm[best]
            arbitration = "posterior_tie_code_ascending"
        endpoint_confidence_ppm = round_half_away_from_zero(speech_sum / speech_weight) if speech_weight else 0
        result.append(
            {
                "interval_id": interval_id,
                "speaker_id": speaker_id,
                "endpoint_confidence_ppm": endpoint_confidence_ppm,
                "endpoint_confidence": endpoint_confidence_ppm / 1_000_000.0,
                "command_code": best,
                "command_probability_ppm": best_probability,
                "command_margin_ppm": margin,
                "runner_up_command_code": runner_up,
                "command_arbitration": arbitration,
                "posterior_ppm": {str(code): value for code, value in sorted(posterior_ppm.items())},
                "window_ids": sorted(window_ids),
                "window_count": len(rows),
            }
        )
    return result


def frame_digest(frames: Iterable[Mapping[str, Any]], commands: Mapping[str, Any], intervals: Iterable[Mapping[str, Any]]) -> str:
    payload = {
        "frames": [
            {
                key: frame.get(key)
                for key in (
                    "frame_index",
                    "active",
                    "speech_ppm",
                    "winning_interval_id",
                    "command_code",
                    "command_probability_ppm",
                    "posterior_ppm",
                    "runner_up_interval_id",
                    "runner_up_command_code",
                    "boundary",
                    "interval_arbitration",
                    "command_arbitration",
                )
            }
            for frame in frames
        ],
        "commands": commands,
        "intervals": [
            {
                key: interval.get(key)
                for key in (
                    "interval_id",
                    "speaker_id",
                    "endpoint_confidence_ppm",
                    "command_code",
                    "command_probability_ppm",
                    "posterior_ppm",
                )
            }
            for interval in intervals
        ],
    }
    return json_hash(payload)
