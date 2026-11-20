from __future__ import annotations

import math
import platform
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .decoder import decode_grid, decode_grid_reusing_prefix
from .errors import (
    AppError,
    E_COMMAND_INVALID,
    E_DECODE_PARAM_INVALID,
    E_INPUT_INVALID,
    E_INTERNAL,
    failure,
    sort_failures,
)
from .fingerprint import content_hash, round_score
from .grid import (
    build_grid,
    build_shard,
    effective_config_for_session,
    grid_fingerprint,
    ownership_fingerprint,
    resolve_ownership,
    stable_fingerprint,
    summarize_grid,
    tail_fingerprint,
)
from .models import (
    Dataset,
    DecodeConfig,
    GridFrame,
    OwnedFrame,
    SessionArtifacts,
    SessionDataset,
    ShardBuild,
    ShardRecord,
    Vocabulary,
)
from .slots import fill_slots


SUPPORTED_INJECTIONS = {
    "non_finite_score",
    "frame_count_mismatch",
    "overlap_not_less_than_shard",
    "zero_lookahead",
    "invalid_slot_bounds",
}


@dataclass
class _ShardJob:
    session_id: str
    record: ShardRecord
    rows: Dict[int, Tuple[float, ...]]
    config: DecodeConfig
    injection_id: Optional[str] = None
    injected_non_finite: bool = False
    injected_frame_delta: int = 0

    @property
    def sort_key(self) -> Tuple[str, int, int, int]:
        return (self.session_id, self.record.shard_seq, self.record.start_frame, self.record.revision)


@dataclass
class _SessionPreparation:
    session_id: str
    session: SessionDataset
    config: Optional[DecodeConfig]
    effective_overlap: int
    failures: List[Dict[str, Any]] = field(default_factory=list)
    builds: List[ShardBuild] = field(default_factory=list)
    skip_decode: bool = False


@dataclass
class _SessionPrelim:
    session_id: str
    session: SessionDataset
    config: DecodeConfig
    effective_overlap: int
    builds: List[ShardBuild]
    owned: List[OwnedFrame]
    conflicts: List[Dict[str, Any]]
    grid: List[GridFrame]
    failures: List[Dict[str, Any]]
    ownership_fp: str
    grid_fp: str


def _failure_key(item: Mapping[str, Any]) -> str:
    return content_hash(
        {
            "business_code": item.get("business_code"),
            "stage": item.get("stage"),
            "session_id": item.get("session_id"),
            "shard_seq": item.get("shard_seq"),
            "message": item.get("message"),
            "injection_id": item.get("injection_id"),
            "details": item.get("details") or {},
        }
    )


def dedupe_failures(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for item in items:
        copied = dict(item)
        unique.setdefault(_failure_key(copied), copied)
    return sort_failures(list(unique.values()))


def _injection_items(injections: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not injections:
        return []
    raw_items = injections.get("items", [])
    if not isinstance(raw_items, list):
        return []
    return [dict(item) for item in raw_items if isinstance(item, dict)]


def _new_injection_tracker(items: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    tracker: Dict[str, Dict[str, Any]] = {}
    for item in sorted(items, key=lambda value: str(value.get("id", ""))):
        injection_id = str(item.get("id", ""))
        tracker[injection_id] = {
            "id": injection_id,
            "type": item.get("type"),
            "target_session_id": item.get("session_id"),
            "target_shard_seq": item.get("shard_seq"),
            "configured_business_code": item.get("business_code"),
            "hit_count": 0,
            "observed_business_code": None,
            "stage": None,
            "failure_message": None,
        }
    return tracker


def _mark_injection_hit(
    tracker: Dict[str, Dict[str, Any]],
    injection_id: Optional[str],
    failure_item: Mapping[str, Any],
) -> None:
    if not injection_id or injection_id not in tracker:
        return
    item = tracker[injection_id]
    item["hit_count"] = int(item["hit_count"]) + 1
    if item["observed_business_code"] is None:
        item["observed_business_code"] = failure_item.get("business_code")
        item["stage"] = failure_item.get("stage")
        item["failure_message"] = failure_item.get("message")


def _config_hash(
    config: DecodeConfig,
    overrides: Mapping[str, Any],
    injection_items: Sequence[Mapping[str, Any]],
) -> str:
    return content_hash(
        {
            "config": config.as_dict(),
            "overrides": dict(overrides),
            "injections": [
                {
                    "id": item.get("id"),
                    "type": item.get("type"),
                    "session_id": item.get("session_id"),
                    "shard_seq": item.get("shard_seq"),
                    "params": item.get("params", {}),
                }
                for item in sorted(injection_items, key=lambda value: str(value.get("id", "")))
            ],
        }
    )


def _session_sequence_failure(session: SessionDataset) -> Optional[Dict[str, Any]]:
    seqs = sorted(session.shards)
    if not seqs:
        return failure(
            E_INPUT_INVALID,
            "session_validation",
            "session has no shard rows",
            session_id=session.session_id,
        )
    expected = list(range(seqs[-1] + 1))
    if seqs != expected:
        return failure(
            E_INPUT_INVALID,
            "session_validation",
            "shard sequence must start at zero and contain no gaps",
            session_id=session.session_id,
            details={"expected": expected, "actual": seqs},
        )
    return None


def _run_shard_job(job: _ShardJob, vocab: Vocabulary) -> ShardBuild:
    try:
        return build_shard(
            job.session_id,
            job.record,
            job.rows,
            job.config,
            vocab,
            injection_id=job.injection_id,
            injected_non_finite=job.injected_non_finite,
            injected_frame_delta=job.injected_frame_delta,
        )
    except Exception as exc:  # isolation boundary for one shard task
        return ShardBuild(
            session_id=job.session_id,
            shard_seq=job.record.shard_seq,
            revision=job.record.revision,
            start_frame=job.record.start_frame,
            end_frame=job.record.end_frame,
            frames=[],
            failure=failure(
                E_INTERNAL,
                "shard_worker",
                f"unexpected shard worker failure: {type(exc).__name__}: {exc}",
                session_id=job.session_id,
                shard_seq=job.record.shard_seq,
                injection_id=job.injection_id,
            ),
        )


def _known_bad_shards(initial_failures: Sequence[Mapping[str, Any]]) -> Set[Tuple[str, int]]:
    bad: Set[Tuple[str, int]] = set()
    for item in initial_failures:
        session_id = item.get("session_id")
        shard_seq = item.get("shard_seq")
        if isinstance(session_id, str) and shard_seq is not None:
            try:
                bad.add((session_id, int(shard_seq)))
            except (TypeError, ValueError):
                continue
    return bad


def _injection_maps(items: Sequence[Mapping[str, Any]]) -> Tuple[
    Dict[Tuple[str, int], Dict[str, Any]],
    Dict[Tuple[str, int], Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
    List[Dict[str, Any]],
]:
    non_finite: Dict[Tuple[str, int], Dict[str, Any]] = {}
    mismatch: Dict[Tuple[str, int], Dict[str, Any]] = {}
    config_overrides: Dict[str, List[Dict[str, Any]]] = {}
    invalid_targets: List[Dict[str, Any]] = []
    for item in sorted(items, key=lambda value: str(value.get("id", ""))):
        injection_id = str(item.get("id", ""))
        injection_type = item.get("type")
        session_id = item.get("session_id")
        shard_seq = item.get("shard_seq")
        if injection_type not in SUPPORTED_INJECTIONS:
            invalid_targets.append(
                failure(
                    E_INPUT_INVALID,
                    "injection_load",
                    f"unsupported injection type: {injection_type}",
                    session_id=session_id if isinstance(session_id, str) else None,
                    injection_id=injection_id,
                )
            )
            continue
        if not isinstance(session_id, str) or not session_id:
            invalid_targets.append(
                failure(
                    E_INPUT_INVALID,
                    "injection_load",
                    "injection requires a target session_id",
                    injection_id=injection_id,
                )
            )
            continue
        if injection_type in ("non_finite_score", "frame_count_mismatch"):
            if shard_seq is None:
                invalid_targets.append(
                    failure(
                        E_INPUT_INVALID,
                        "injection_load",
                        f"{injection_type} requires a target shard_seq",
                        session_id=session_id,
                        injection_id=injection_id,
                    )
                )
                continue
            try:
                key = (session_id, int(shard_seq))
            except (TypeError, ValueError):
                invalid_targets.append(
                    failure(
                        E_INPUT_INVALID,
                        "injection_load",
                        f"{injection_type} shard_seq must be an integer",
                        session_id=session_id,
                        injection_id=injection_id,
                    )
                )
                continue
            if injection_type == "non_finite_score":
                non_finite[key] = item
            else:
                mismatch[key] = item
        else:
            config_overrides.setdefault(session_id, []).append(item)
    return non_finite, mismatch, config_overrides, invalid_targets


def _apply_config_injections(
    base_config: DecodeConfig,
    session: SessionDataset,
    items: Sequence[Mapping[str, Any]],
    shortest_shard_frames: int,
) -> Tuple[DecodeConfig, List[Dict[str, Any]]]:
    config = base_config
    failures: List[Dict[str, Any]] = []
    for item in items:
        injection_id = str(item.get("id", ""))
        injection_type = item.get("type")
        overrides: Dict[str, Any] = {}
        params = item.get("params") if isinstance(item.get("params"), dict) else {}
        if injection_type == "overlap_not_less_than_shard":
            if "overlap_frames" in item:
                overrides["overlap_frames"] = int(item["overlap_frames"])
            elif "overlap_frames" in params:
                overrides["overlap_frames"] = int(params["overlap_frames"])
            else:
                overrides["overlap_frames"] = max(0, shortest_shard_frames)
        elif injection_type == "zero_lookahead":
            overrides.update({"strategy": "lookahead", "lookahead_frames": 0})
        elif injection_type == "invalid_slot_bounds":
            overrides.update({"slot_min_frames": 0, "slot_max_frames": 0})
        try:
            config = config.with_overrides(overrides)
        except AppError as exc:
            exc.injection_id = injection_id
            failures.append(exc.as_failure())
            return config, failures
    return config, failures


def _decoded_payload(
    decode: Any,
    vocab: Vocabulary,
) -> Dict[str, Any]:
    if decode is None:
        return {
            "stable_prefix": {
                "length": 0,
                "start_frame": None,
                "end_frame": None,
                "token_ids": [],
                "text": "",
                "subword_text": "",
            },
            "variable_tail": {
                "length": 0,
                "start_frame": None,
                "end_frame": None,
                "frame_interval": [],
                "token_ids": [],
                "text": "",
            },
            "final_instruction": {
                "token_ids": [],
                "tokens": [],
                "text": "",
                "subword_text": "",
            },
            "frame_decisions": [],
            "greedy_decisions": [],
        }
    decisions = decode.decisions
    stable = decisions[: decode.stable_length]
    tail = decisions[decode.stable_length :]
    stable_ids = [item.token_id for item in stable]
    tail_ids = [item.token_id for item in tail]
    all_ids = [item.token_id for item in decisions]
    stable_start = stable[0].global_frame if stable else None
    stable_end = stable[-1].global_frame if stable else None
    tail_start = tail[0].global_frame if tail else None
    tail_end = tail[-1].global_frame if tail else None
    return {
        "stable_prefix": {
            "length": decode.stable_length,
            "start_frame": stable_start,
            "end_frame": stable_end,
            "token_ids": stable_ids,
            "text": vocab.render(stable_ids),
            "subword_text": vocab.subword_text(stable_ids),
        },
        "variable_tail": {
            "length": len(tail),
            "start_frame": tail_start,
            "end_frame": tail_end,
            "frame_interval": [] if tail_start is None else [tail_start, tail_end],
            "token_ids": tail_ids,
            "text": vocab.render(tail_ids),
        },
        "final_instruction": {
            "token_ids": all_ids,
            "tokens": [vocab.token(token_id) for token_id in all_ids],
            "text": vocab.render(all_ids),
            "subword_text": vocab.subword_text(all_ids),
        },
        "frame_decisions": [item.as_dict() for item in decisions],
        "greedy_decisions": [item.as_dict() for item in decode.greedy_decisions],
    }


def _derive_affected(
    previous: Optional[SessionArtifacts],
    artifacts: SessionArtifacts,
) -> List[Dict[str, Any]]:
    if previous is None:
        if artifacts.frame_count == 0:
            return []
        return [
            {
                "start_frame": artifacts.grid[0].global_frame,
                "end_frame": artifacts.grid[-1].global_frame,
                "reason": "full_decode",
            }
        ]

    old_frames = {item.global_frame: (item.owner_shard, item.owner_revision, item.candidates) for item in previous.grid}
    new_frames = {item.global_frame: (item.owner_shard, item.owner_revision, item.candidates) for item in artifacts.grid}
    changed = sorted(
        frame
        for frame in set(old_frames) | set(new_frames)
        if old_frames.get(frame) != new_frames.get(frame)
    )
    if changed:
        return [{"start_frame": changed[0], "end_frame": changed[-1], "reason": "grid_content_changed"}]
    if previous.fingerprints.get("tail_fp") != artifacts.fingerprints.get("tail_fp"):
        start = artifacts.decode.stable_length if artifacts.decode is not None else 0
        if artifacts.frame_count == 0:
            return []
        start_frame = artifacts.grid[min(start, artifacts.frame_count - 1)].global_frame
        return [
            {
                "start_frame": start_frame,
                "end_frame": artifacts.grid[-1].global_frame,
                "reason": "tail_parameters_changed",
            }
        ]
    return []


def _cache_actions(previous: Optional[SessionArtifacts], current: SessionArtifacts) -> Dict[str, Any]:
    keys = ("ownership_fp", "grid_fp", "stable_fp", "tail_fp")
    if previous is None:
        return {
            "ownership": "invalidate",
            "grid": "invalidate",
            "stable_prefix": "invalidate",
            "tail_decode": "invalidate",
            "keys": {key: current.fingerprints.get(key) for key in keys},
        }

    def frame_reuse(old_frames: Sequence[Any], new_frames: Sequence[Any], key_fn: Any) -> Tuple[str, int, int]:
        old_map = {key_fn(item): item for item in old_frames}
        new_map = {key_fn(item): item for item in new_frames}
        common = set(old_map) & set(new_map)
        reused = sum(1 for key in common if old_map[key] == new_map[key])
        changed = len((set(old_map) | set(new_map)) - {key for key in common if old_map[key] == new_map[key]})
        if reused == len(old_map) == len(new_map) and not changed:
            return "reuse", reused, 0
        if reused > 0:
            return "partial_invalidate", reused, changed
        return "invalidate", reused, changed

    ownership_action, ownership_reused, ownership_changed = frame_reuse(
        previous.owned_frames,
        current.owned_frames,
        lambda item: item.global_frame,
    )
    grid_action, grid_reused, grid_changed = frame_reuse(
        previous.grid,
        current.grid,
        lambda item: item.global_frame,
    )

    previous_stable = previous.decode.stable_token_ids if previous.decode is not None else ()
    current_stable = current.decode.stable_token_ids if current.decode is not None else ()
    previous_frames = (
        tuple(item.global_frame for item in previous.decode.decisions[: previous.decode.stable_length])
        if previous.decode is not None
        else ()
    )
    current_frames = (
        tuple(item.global_frame for item in current.decode.decisions[: current.decode.stable_length])
        if current.decode is not None
        else ()
    )
    if previous.fingerprints.get("stable_fp") == current.fingerprints.get("stable_fp"):
        stable_action = "reuse"
    elif len(previous_stable) <= len(current_stable) and current_stable[: len(previous_stable)] == previous_stable and current_frames[: len(previous_frames)] == previous_frames:
        stable_action = "reuse_with_extension" if len(current_stable) > len(previous_stable) else "reuse"
    elif len(current_stable) < len(previous_stable) and previous_stable[: len(current_stable)] == current_stable and previous_frames[: len(current_frames)] == current_frames:
        stable_action = "reuse_partial"
    else:
        stable_action = "invalidate"

    actions = {
        "ownership": ownership_action,
        "ownership_reused_frames": ownership_reused,
        "ownership_invalidated_frames": ownership_changed,
        "grid": grid_action,
        "grid_reused_frames": grid_reused,
        "grid_invalidated_frames": grid_changed,
        "stable_prefix": stable_action,
        "stable_prefix_reused_frames": min(len(previous_stable), len(current_stable)) if stable_action != "invalidate" else 0,
        "stable_prefix_length_before": len(previous_stable),
        "stable_prefix_length_after": len(current_stable),
        "tail_decode": "reuse" if previous.fingerprints.get("tail_fp") == current.fingerprints.get("tail_fp") else "invalidate",
        "keys": {key: current.fingerprints.get(key) for key in keys},
        "previous_keys": {key: previous.fingerprints.get(key) for key in keys},
    }
    return actions


def _semantic_session(report_session: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "session_id": report_session.get("session_id"),
        "status": report_session.get("status"),
        "frame_count": report_session.get("frame_count"),
        "stable_prefix": report_session.get("stable_prefix"),
        "variable_tail": report_session.get("variable_tail"),
        "final_instruction": report_session.get("final_instruction"),
        "frame_decisions": report_session.get("frame_decisions"),
        "greedy_decisions": report_session.get("greedy_decisions"),
        "slots": {"filled": (report_session.get("slots") or {}).get("filled", [])},
        "ownership_conflicts": report_session.get("ownership_conflicts", []),
        "ownership_frames": report_session.get("ownership_frames", []),
        "overlap_audit": report_session.get("overlap_audit", []),
        "smoothed_frames": report_session.get("smoothed_frames", []),
        "failures": report_session.get("failures", []),
    }


def _overlap_audit(builds: Sequence[ShardBuild], configured_overlap: int) -> List[Dict[str, Any]]:
    successful = sorted(
        [item for item in builds if item.failure is None],
        key=lambda item: item.shard_seq,
    )
    result: List[Dict[str, Any]] = []
    for left_index, left in enumerate(successful):
        for right in successful[left_index + 1 :]:
            start = max(left.start_frame, right.start_frame)
            end = min(left.end_frame, right.end_frame)
            if end < start:
                continue
            result.append(
                {
                    "left_shard_seq": left.shard_seq,
                    "right_shard_seq": right.shard_seq,
                    "start_frame": start,
                    "end_frame": end,
                    "actual_overlap_frames": end - start + 1,
                    "configured_overlap_frames": configured_overlap,
                    "authoritative_shard_seq": right.shard_seq if (right.shard_seq, right.revision) > (left.shard_seq, left.revision) else left.shard_seq,
                }
            )
    return result


def semantic_payload(report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sessions": [_semantic_session(item) for item in report.get("sessions", [])],
        "failures": report.get("failures", []),
    }


def _environment() -> Dict[str, Any]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
    }


def full_decode(
    dataset: Dataset,
    config: DecodeConfig,
    *,
    overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    injections: Optional[Mapping[str, Any]] = None,
    version: int = 0,
    initial_failures: Optional[Sequence[Mapping[str, Any]]] = None,
    previous_artifacts: Optional[Mapping[str, SessionArtifacts]] = None,
    affected_by_session: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    snapshot_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute the canonical full result for the current dataset snapshot.

    Incremental commands use this function with cached previous artifacts; the
    semantic payload is independent of edit history by construction.
    """

    overrides = overrides or {}
    injection_items = _injection_items(injections)
    tracker = _new_injection_tracker(injection_items)
    base_failures = [dict(item) for item in (initial_failures or [])]
    config_hash = _config_hash(config, overrides, injection_items)
    known_bad = _known_bad_shards(base_failures)
    non_finite_map, mismatch_map, config_injection_map, invalid_injection_targets = _injection_maps(injection_items)
    for item in invalid_injection_targets:
        _mark_injection_hit(tracker, item.get("injection_id"), item)
    base_failures.extend(invalid_injection_targets)

    preparations: Dict[str, _SessionPreparation] = {}
    jobs: List[_ShardJob] = []

    for session_id in dataset.sorted_session_ids():
        session = dataset.sessions[session_id]
        session_failures = [
            dict(item) for item in base_failures if item.get("session_id") == session_id
        ]
        if not session.input_valid:
            preparations[session_id] = _SessionPreparation(
                session_id=session_id,
                session=session,
                config=None,
                effective_overlap=config.overlap_frames,
                failures=session_failures,
                skip_decode=True,
            )
            continue

        seq_failure = _session_sequence_failure(session)
        if seq_failure is not None:
            session_failures.append(seq_failure)
            preparations[session_id] = _SessionPreparation(
                session_id=session_id,
                session=session,
                config=None,
                effective_overlap=config.overlap_frames,
                failures=session_failures,
                skip_decode=True,
            )
            continue

        arrived = session.ordered_shards()
        shortest = min((record.frame_count for record in arrived), default=0)
        try:
            effective = effective_config_for_session(config, overrides.get(session_id))
        except AppError as exc:
            exc.session_id = session_id
            session_failures.append(exc.as_failure())
            preparations[session_id] = _SessionPreparation(
                session_id=session_id,
                session=session,
                config=None,
                effective_overlap=config.overlap_frames,
                failures=session_failures,
                skip_decode=True,
            )
            continue

        injection_configs = config_injection_map.get(session_id, [])
        effective, injection_config_failures = _apply_config_injections(
            effective,
            session,
            injection_configs,
            shortest,
        )
        for item in injection_config_failures:
            item["session_id"] = session_id
            _mark_injection_hit(tracker, item.get("injection_id"), item)
            session_failures.append(item)
        if injection_config_failures:
            preparations[session_id] = _SessionPreparation(
                session_id=session_id,
                session=session,
                config=None,
                effective_overlap=effective.overlap_frames,
                failures=session_failures,
                skip_decode=True,
            )
            continue

        try:
            effective.validate_for_session(shortest)
        except AppError as exc:
            exc.session_id = session_id
            # Associate a session-level config injection when there is exactly
            # one candidate, providing a direct proof of fault injection.
            if len(injection_configs) == 1:
                exc.injection_id = str(injection_configs[0].get("id", ""))
                _mark_injection_hit(tracker, exc.injection_id, exc.as_failure())
            session_failures.append(exc.as_failure())
            preparations[session_id] = _SessionPreparation(
                session_id=session_id,
                session=session,
                config=None,
                effective_overlap=effective.overlap_frames,
                failures=session_failures,
                skip_decode=True,
            )
            continue

        prep = _SessionPreparation(
            session_id=session_id,
            session=session,
            config=effective,
            effective_overlap=effective.overlap_frames,
            failures=session_failures,
        )
        preparations[session_id] = prep

        for record in arrived:
            key = (session_id, record.shard_seq)
            if key in known_bad:
                continue
            nf_item = non_finite_map.get(key)
            mismatch_item = mismatch_map.get(key)
            injection_id: Optional[str] = None
            if nf_item is not None and mismatch_item is not None:
                injection_id = min(str(nf_item.get("id", "")), str(mismatch_item.get("id", "")))
            elif nf_item is not None:
                injection_id = str(nf_item.get("id", ""))
            elif mismatch_item is not None:
                injection_id = str(mismatch_item.get("id", ""))
            delta = 0
            if mismatch_item is not None:
                try:
                    delta = int(mismatch_item.get("delta", 1))
                except (TypeError, ValueError):
                    delta = 1
            jobs.append(
                _ShardJob(
                    session_id=session_id,
                    record=record,
                    rows=session.scores.get(record.shard_seq, {}),
                    config=effective,
                    injection_id=injection_id,
                    injected_non_finite=nf_item is not None,
                    injected_frame_delta=delta,
                )
            )

    # Validate injection targets that point to missing sessions/shards.
    all_target_items = list(non_finite_map.items()) + list(mismatch_map.items())
    all_target_items.sort(key=lambda pair: (pair[0][0], pair[0][1], str(pair[1].get("id", ""))))
    for (sid, seq), item in all_target_items:
        target = dataset.sessions.get(sid)
        if target is None or seq not in target.shards:
            issue = failure(
                E_INPUT_INVALID,
                "injection_load",
                "injection target shard does not exist",
                session_id=sid,
                shard_seq=seq,
                injection_id=str(item.get("id", "")),
            )
            _mark_injection_hit(tracker, issue.get("injection_id"), issue)
            base_failures.append(issue)
            preparations.setdefault(
                sid,
                _SessionPreparation(
                    session_id=sid,
                    session=target or SessionDataset(session_id=sid, input_valid=False),
                    config=None,
                    effective_overlap=config.overlap_frames,
                    failures=[],
                    skip_decode=True,
                ),
            ).failures.append(issue)

    jobs.sort(key=lambda item: item.sort_key)
    builds_by_session: Dict[str, List[ShardBuild]] = {sid: [] for sid in preparations}
    workers = max(1, int(config.thread_pool_concurrency))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shard-grid") as executor:
        results: List[ShardBuild] = []
        futures = [executor.submit(_run_shard_job, job, dataset.vocab) for job in jobs]
        for job, future in zip(jobs, futures):
            result = future.result()
            results.append(result)
            injections_for_key = []
            if (job.session_id, job.record.shard_seq) in non_finite_map:
                injections_for_key.append(non_finite_map[(job.session_id, job.record.shard_seq)])
            if (job.session_id, job.record.shard_seq) in mismatch_map:
                injections_for_key.append(mismatch_map[(job.session_id, job.record.shard_seq)])
            for injection_item in injections_for_key:
                injection_id = str(injection_item.get("id", ""))
                if result.failure is not None and result.failure.get("injection_id") == injection_id:
                    _mark_injection_hit(tracker, injection_id, result.failure)
                else:
                    proxy = dict(result.failure or {})
                    proxy.setdefault("business_code", E_INPUT_INVALID)
                    proxy.setdefault("stage", "shard_validation")
                    proxy.setdefault("message", "injection applied to shard task")
                    _mark_injection_hit(tracker, injection_id, proxy)

        # Commit interval snapshots after deterministic sorted batches.  These
        # are intentionally minimal cursor manifests; the final snapshot below
        # contains all stage artifacts.
        if snapshot_manager is not None:
            interval = max(1, int(config.snapshot_interval_shards))
            for batch_start in range(0, len(results), interval):
                batch = results[batch_start : batch_start + interval]
                cursor: Dict[str, Any] = {}
                for build in batch:
                    entry = cursor.setdefault(
                        build.session_id,
                        {"completed_frames": 0, "last_global_frame": None},
                    )
                    if build.failure is None and build.frames:
                        entry["completed_frames"] += len(build.frames)
                        entry["last_global_frame"] = build.end_frame
                snapshot_manager.commit_stage(
                    stage=f"shard_batch_{batch_start // interval:04d}",
                    expected_version=version,
                    config_hash=config_hash,
                    completed_frame_cursor=cursor,
                    payload={
                        "kind": "shard_batch",
                        "batch_index": batch_start // interval,
                        "shards": [
                            {
                                "session_id": build.session_id,
                                "shard_seq": build.shard_seq,
                                "start_frame": build.start_frame,
                                "end_frame": build.end_frame,
                                "status": "failed" if build.failure is not None else "ok",
                            }
                            for build in batch
                        ],
                    },
                )

        for build in results:
            builds_by_session.setdefault(build.session_id, []).append(build)
            if build.failure is not None:
                preparations[build.session_id].failures.append(build.failure)

        prelims: List[_SessionPrelim] = []
        for session_id in sorted(preparations):
            prep = preparations[session_id]
            if prep.config is None or prep.skip_decode:
                continue
            builds = sorted(builds_by_session.get(session_id, []), key=lambda item: item.sort_key)
            owned, conflicts = resolve_ownership(session_id, builds)
            grid = build_grid(owned, prep.config, dataset.vocab)
            ownership_fp = ownership_fingerprint(session_id, builds)
            grid_fp = grid_fingerprint(
                session_id,
                ownership_fp,
                dataset.vocab,
                prep.config,
                owned,
                grid,
            )
            prelims.append(
                _SessionPrelim(
                    session_id=session_id,
                    session=prep.session,
                    config=prep.config,
                    effective_overlap=prep.effective_overlap,
                    builds=builds,
                    owned=owned,
                    conflicts=conflicts,
                    grid=grid,
                    failures=prep.failures,
                    ownership_fp=ownership_fp,
                    grid_fp=grid_fp,
                )
            )

        def decode_session(prelim: _SessionPrelim) -> Any:
            previous = (previous_artifacts or {}).get(prelim.session_id)
            if (
                previous is not None
                and previous.decode is not None
                and previous.fingerprints.get("grid_fp") == prelim.grid_fp
                and previous.config.min_stable_frames == prelim.config.min_stable_frames
                and previous.config.stable_margin == prelim.config.stable_margin
            ):
                return decode_grid_reusing_prefix(
                    prelim.grid,
                    dataset.vocab,
                    prelim.config,
                    previous.decode,
                )
            return decode_grid(prelim.grid, dataset.vocab, prelim.config)

        decode_futures = [executor.submit(decode_session, prelim) for prelim in prelims]
        decode_results = [future.result() for future in decode_futures]

    session_reports: List[Dict[str, Any]] = []
    artifacts_by_session: Dict[str, SessionArtifacts] = {}
    all_failures = list(base_failures)
    decoded_by_session = {prelim.session_id: decoded for prelim, decoded in zip(prelims, decode_results)}

    for session_id in sorted(preparations):
        prep = preparations[session_id]
        decoded = decoded_by_session.get(session_id)
        prelim = next((item for item in prelims if item.session_id == session_id), None)
        if prelim is None:
            failures = dedupe_failures(prep.failures)
            artifacts = SessionArtifacts(
                session_id=session_id,
                config=prep.config or config,
                owned_frames=[],
                grid=[],
                decode=None,
                failures=failures,
                effective_overlap=prep.effective_overlap,
                fingerprints={
                    "vocab_fp": dataset.vocab.source_fingerprint,
                    "ownership_fp": content_hash({"session_id": session_id, "missing": True}),
                    "grid_fp": content_hash({"session_id": session_id, "empty_grid": True}),
                    "stable_fp": content_hash({"session_id": session_id, "empty_stable": True}),
                    "tail_fp": content_hash({"session_id": session_id, "empty_tail": True}),
                    "strategy_fp": content_hash((prep.config or config).strategy_fingerprint_payload()),
                },
                cache_actions={},
                edit_affected=[],
            )
            artifacts.cache_actions = _cache_actions(
                (previous_artifacts or {}).get(session_id), artifacts
            )
            artifacts.edit_affected = list(
                (affected_by_session or {}).get(session_id, _derive_affected(
                    (previous_artifacts or {}).get(session_id), artifacts
                ))
            )
            report_session = {
                "session_id": session_id,
                "status": "failed" if failures else "pending",
                "effective_config": (prep.config or config).as_dict(),
                "effective_overlap_frames": prep.effective_overlap,
                "frame_count": 0,
                "frames": summarize_grid([]),
                "smoothed_frames": [],
                "stable_prefix": _decoded_payload(None, dataset.vocab)["stable_prefix"],
                "variable_tail": _decoded_payload(None, dataset.vocab)["variable_tail"],
                "final_instruction": _decoded_payload(None, dataset.vocab)["final_instruction"],
                "frame_decisions": [],
                "greedy_decisions": [],
                "slots": {"filled": [], "unfilled": []},
                "stable_slots": [],
                "ownership_conflicts": [],
                "ownership_frames": [],
                "overlap_audit": [],
                "failures": failures,
                "fingerprints": artifacts.fingerprints,
                "cache_actions": artifacts.cache_actions,
                "affected_intervals": artifacts.edit_affected,
                "strategy": (prep.config or config).strategy,
                "lookahead_frames": (prep.config or config).lookahead_frames,
                "shards": [],
            }
            artifacts_by_session[session_id] = artifacts
            session_reports.append(report_session)
            all_failures.extend(failures)
            continue

        assert decoded is not None
        failures = dedupe_failures(prelim.failures)
        stable_decisions = decoded.decisions[: decoded.stable_length]
        tail_decisions = decoded.decisions[decoded.stable_length :]
        stable_fp = stable_fingerprint(prelim.grid_fp, dataset.vocab, prelim.config, stable_decisions)
        tail_fp = tail_fingerprint(stable_fp, prelim.grid_fp, prelim.config, tail_decisions)
        fingerprints = {
            "vocab_fp": dataset.vocab.source_fingerprint,
            "ownership_fp": prelim.ownership_fp,
            "grid_fp": prelim.grid_fp,
            "stable_fp": stable_fp,
            "tail_fp": tail_fp,
            "strategy_fp": content_hash(prelim.config.strategy_fingerprint_payload()),
        }
        artifacts = SessionArtifacts(
            session_id=session_id,
            config=prelim.config,
            owned_frames=prelim.owned,
            grid=prelim.grid,
            decode=decoded,
            failures=failures,
            effective_overlap=prelim.effective_overlap,
            fingerprints=fingerprints,
        )
        previous = (previous_artifacts or {}).get(session_id)
        artifacts.cache_actions = _cache_actions(previous, artifacts)
        default_affected = _derive_affected(previous, artifacts)
        artifacts.edit_affected = list(
            (affected_by_session or {}).get(session_id, default_affected)
        )
        payload = _decoded_payload(decoded, dataset.vocab)
        slots = fill_slots(dataset.vocab, decoded.decisions, prelim.config)
        stable_end = payload["stable_prefix"]["end_frame"]
        stable_slots = [
            item
            for item in slots["filled"]
            if stable_end is not None and item["end_frame"] <= stable_end
        ]
        status = "partial" if failures else "ok"
        report_session = {
            "session_id": session_id,
            "status": status,
            "effective_config": prelim.config.as_dict(),
            "effective_overlap_frames": prelim.effective_overlap,
            "frame_count": len(prelim.grid),
            "frames": summarize_grid(prelim.grid),
            "smoothed_frames": [item.global_frame for item in prelim.grid if item.smoothed],
            "stable_prefix": payload["stable_prefix"],
            "variable_tail": payload["variable_tail"],
            "final_instruction": payload["final_instruction"],
            "frame_decisions": payload["frame_decisions"],
            "greedy_decisions": payload["greedy_decisions"],
            "slots": slots,
            "stable_slots": stable_slots,
            "ownership_conflicts": prelim.conflicts,
            "ownership_frames": [item.as_dict() for item in prelim.owned],
            "overlap_audit": _overlap_audit(prelim.builds, prelim.effective_overlap),
            "failures": failures,
            "fingerprints": fingerprints,
            "cache_actions": artifacts.cache_actions,
            "affected_intervals": artifacts.edit_affected,
            "strategy": prelim.config.strategy,
            "lookahead_frames": prelim.config.lookahead_frames,
            "strategy_agreement_prefix": decoded.strategy_agreement_prefix,
            "decode_mode": decoded.decode_mode,
            "decode_reused_prefix_frames": decoded.reused_prefix_frames,
            "shards": [
                {
                    "shard_seq": build.shard_seq,
                    "revision": build.revision,
                    "start_frame": build.start_frame,
                    "end_frame": build.end_frame,
                    "status": "failed" if build.failure is not None else "ok",
                }
                for build in prelim.builds
            ],
        }
        artifacts_by_session[session_id] = artifacts
        session_reports.append(report_session)
        all_failures.extend(failures)

    all_failures = dedupe_failures(all_failures)
    injection_report_items = list(tracker.values())
    injection_report_items.sort(key=lambda item: item["id"])
    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "application": "streamparse",
        "global_version": version,
        "config_hash": config_hash,
        "environment": _environment(),
        "base_config": config.as_dict(),
        "session_overrides": {key: dict(value) for key, value in sorted(overrides.items())},
        "thread_pool": {"max_workers": workers, "deterministic_sort_key": "session_id,shard_seq,start_frame,revision"},
        "sessions": session_reports,
        "session_order": [item["session_id"] for item in session_reports],
        "failures": all_failures,
        "overlap_ownership": [
            conflict
            for item in session_reports
            for conflict in item.get("ownership_conflicts", [])
        ],
        "injection_report": {
            "items": injection_report_items,
            "unhit_ids": [item["id"] for item in injection_report_items if item["hit_count"] <= 0],
        },
        "artifacts": {
            session_id: {
                "fingerprints": artifact.fingerprints,
                "cache_actions": artifact.cache_actions,
                "affected_intervals": artifact.edit_affected,
                "effective_overlap_frames": artifact.effective_overlap,
            }
            for session_id, artifact in sorted(artifacts_by_session.items())
        },
    }

    if snapshot_manager is not None:
        cursor = {
            item["session_id"]: {
                "completed_frames": item["frame_count"],
                "last_global_frame": (item["frames"].get("last_frame") if item.get("frames") else None),
            }
            for item in session_reports
        }
        snapshot_manager.commit_stage(
            stage="final",
            expected_version=version,
            config_hash=config_hash,
            completed_frame_cursor=cursor,
            payload=report,
            extra_files={
                "ownership.json": {
                    "sessions": {
                        item["session_id"]: item["ownership_frames"] for item in session_reports
                    }
                },
                "grid.json": {
                    "sessions": {
                        item["session_id"]: item["frames"] for item in session_reports
                    }
                },
                "decode.json": {
                    "sessions": {
                        item["session_id"]: {
                            "stable_prefix": item["stable_prefix"],
                            "variable_tail": item["variable_tail"],
                            "final_instruction": item["final_instruction"],
                        }
                        for item in session_reports
                    }
                },
                "failures.json": {"failures": all_failures},
            },
        )
        report["snapshot"] = snapshot_manager.report()

    report["_artifacts_by_session"] = artifacts_by_session
    return report
