"""End-to-end deterministic workbench pipeline."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .audio import audio_dependency_hash, evaluate_audio
from .closure import (
    assert_closure_match,
    build_context,
    closure_report,
    compute_frame_closure,
    compute_segment_closure,
    segment_deps,
)
from .constants import (
    DEFAULT_INJECTION,
    ENCODING_PARAMS,
    FRAME_COUNT,
    INJECTION_SCHEMA,
    METRIC_ENCODING_VERSION,
    METRIC_POLICY,
    PROJECT_SCHEMA,
    SHOTS,
    VERSION,
)
from .equivalence import compare_carriers, compare_model_to_actual
from .errors import BusinessError, as_error
from .formats import (
    CarrierModel,
    parse_legacy_json,
    parse_metric_binary,
    parse_project_json,
    write_metric_binary,
)
from .ledger import LedgerStore
from .quality import (
    metric_ledger_digest,
    record_content_hash,
    summarize_segments,
    validate_shots,
)
from .util import (
    atomic_write_canonical_json,
    atomic_write_json,
    canonical_json_bytes,
    hash_object,
    json_load,
    relative_posix,
    resolve_workspace_path,
    sha256_file,
)
from .video_eval import compute_actual_frames, export_preview


CARRIER_FILES = {
    "project_json": "carriers/project_v2.json",
    "metric_binary": "carriers/metrics_v2.emf",
    "legacy_json": "carriers/project_legacy_v1.json",
    "metric_binary_v1": "carriers/metrics_v1.emf",
}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_truth(workspace: Path) -> dict[str, Any]:
    truth_path = workspace / "assets" / "truth.json"
    truth = json_load(truth_path)
    if not isinstance(truth, dict):
        raise BusinessError("E_SCHEMA", "truth manifest must be an object")
    if truth.get("frame_count") != FRAME_COUNT:
        raise BusinessError("E_SCHEMA_VERSION", "truth frame count does not match the workbench contract")
    return truth


def verify_assets(workspace: Path, truth: dict[str, Any]) -> dict[str, Any]:
    declared = truth.get("file_hashes", {})
    checks = (
        ("source_video_sha256", truth["video"]["source_file"]),
        ("encoded_video_sha256", truth["video"]["encoded_file"]),
        ("audio_wav_sha256", truth["audio"]["wav_file"]),
        ("audio_reference_sha256", truth["audio"]["reference_file"]),
    )
    result: dict[str, Any] = {"checks": [], "ok": True}
    for hash_key, declared_path in checks:
        path = resolve_workspace_path(workspace, declared_path)
        actual = sha256_file(path)
        expected = declared.get(hash_key)
        ok = expected == actual
        result["checks"].append(
            {
                "name": hash_key,
                "path": relative_posix(path, workspace),
                "declared": expected,
                "actual": actual,
                "ok": ok,
            }
        )
        if not ok:
            result["ok"] = False
    truth_path = workspace / "assets" / "truth.json"
    truth_hash = sha256_file(truth_path)
    result["truth_sha256"] = truth_hash
    result["checks"].append(
        {
            "name": "truth_sha256",
            "path": relative_posix(truth_path, workspace),
            "declared": None,
            "actual": truth_hash,
            "ok": True,
        }
    )
    if not result["ok"]:
        raise BusinessError("E_ASSET", "one or more asset hashes do not match truth", {"checks": result["checks"]})
    return result


def carrier_content(workspace: Path, truth: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": truth["asset_id"],
        "content_hash": truth["content_hash"],
        "video_path": truth["video"]["encoded_file"],
        "source_video_path": truth["video"]["source_file"],
        "audio_path": truth["audio"]["wav_file"],
        "reference_audio_path": truth["audio"]["reference_file"],
        "truth_path": relative_posix(workspace / "assets" / "truth.json", workspace),
        "video_sha256": truth["file_hashes"]["encoded_video_sha256"],
        "source_video_sha256": truth["file_hashes"]["source_video_sha256"],
        "audio_sha256": truth["file_hashes"]["audio_wav_sha256"],
        "reference_audio_sha256": truth["file_hashes"]["audio_reference_sha256"],
        "truth_sha256": sha256_file(workspace / "assets" / "truth.json"),
    }


def _declared_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declared: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: int(item["frame_index"])):
        declared.append(
            {
                "frame_index": int(record["frame_index"]),
                "shot_id": int(record["shot_id"]),
                "timestamp_ms": record["timestamp_ms"],
                "psnr_db": record["psnr_db"],
                "ssim": record["ssim"],
                "chroma_psnr_db": record.get("chroma_psnr_db"),
                "flags": int(record.get("flags", 0)),
            }
        )
    return declared


def build_carriers(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    truth = load_truth(workspace)
    verify_assets(workspace, truth)
    shots = validate_shots(copy.deepcopy(truth["shots"]), FRAME_COUNT)
    policy = copy.deepcopy(METRIC_POLICY)
    encoding = copy.deepcopy(ENCODING_PARAMS)
    actual = compute_actual_frames(
        workspace,
        truth,
        shots,
        policy,
        set(range(FRAME_COUNT)),
        copy.deepcopy(DEFAULT_INJECTION),
    )
    if any(record.get("missing") for record in actual.values()) or len(actual) != FRAME_COUNT:
        raise BusinessError("E_DECODE", "carrier construction could not decode all fixture frames")
    for record in actual.values():
        record.setdefault("flags", 0)
    records = [actual[index] for index in range(FRAME_COUNT)]
    segments = summarize_segments(records, shots, policy)
    content = carrier_content(workspace, truth)
    carriers_dir = workspace / "carriers"
    carriers_dir.mkdir(parents=True, exist_ok=True)

    project_document = {
        "schema_version": PROJECT_SCHEMA,
        "project_id": f"{truth['asset_id']}-project",
        "content": content,
        "encoding_params": encoding,
        "shots": shots,
        "metric_policy": policy,
        "frames": _declared_records(records),
        "segments": segments,
    }
    project_path = workspace / CARRIER_FILES["project_json"]
    atomic_write_canonical_json(project_path, project_document)

    legacy_document = {
        "schema_version": "eval-project/1.0",
        "project_id": f"{truth['asset_id']}-legacy",
        "content": content,
        "encoding_params": encoding,
        "frames": [
            {
                "frame_index": frame["frame_index"],
                "timestamp_ms": frame["timestamp_ms"],
                "psnr_db": frame["psnr_db"],
                "ssim": frame["ssim"],
                "chroma_psnr_db": frame["chroma_psnr_db"],
                "flags": frame["flags"],
            }
            for frame in _declared_records(records)
        ],
    }
    legacy_path = workspace / CARRIER_FILES["legacy_json"]
    atomic_write_canonical_json(legacy_path, legacy_document)

    binary_meta = {
        "content": content,
        "encoding_params": encoding,
        "shots": shots,
        "metric_policy": policy,
    }
    binary_v2_info = write_metric_binary(
        workspace / CARRIER_FILES["metric_binary"],
        binary_meta,
        _declared_records(records),
        metric_version=2,
        endian="little",
        index_width=2,
    )
    binary_v1_info = write_metric_binary(
        workspace / CARRIER_FILES["metric_binary_v1"],
        binary_meta,
        _declared_records(records),
        metric_version=1,
        endian="little",
        index_width=2,
    )
    files = {
        name: {
            "path": relative_posix(workspace / relative, workspace),
            "bytes": (workspace / relative).stat().st_size,
            "sha256": sha256_file(workspace / relative),
        }
        for name, relative in CARRIER_FILES.items()
    }
    manifest = {
        "schema_version": "carrier-manifest/1.0",
        "content_hash": truth["content_hash"],
        "frame_count": FRAME_COUNT,
        "shot_count": len(shots),
        "files": files,
        "binary_v2": binary_v2_info,
        "binary_v1": binary_v1_info,
        "metric_digest": metric_ledger_digest(records, segments, policy),
    }
    manifest["manifest_hash"] = hash_object({key: value for key, value in manifest.items() if key != "manifest_hash"})
    atomic_write_canonical_json(carriers_dir / "manifest.json", manifest)
    return manifest


def load_injection(path: Path | None) -> dict[str, Any]:
    injection = copy.deepcopy(DEFAULT_INJECTION)
    if path is not None:
        raw = json_load(path)
        if not isinstance(raw, dict):
            raise BusinessError("E_INJECTION", "injection configuration must be an object")
        if raw.get("schema_version", INJECTION_SCHEMA) != INJECTION_SCHEMA:
            raise BusinessError(
                "E_SCHEMA_VERSION",
                "unsupported injection schema version",
                {"actual": raw.get("schema_version"), "expected": INJECTION_SCHEMA},
            )
        injection.update(raw)
    if injection.get("schema_version") != INJECTION_SCHEMA:
        raise BusinessError("E_SCHEMA_VERSION", "injection schema version is unsupported")
    dropped: list[int] = []
    for value in injection.get("dropped_frames", []):
        frame_index = int(value)
        if not 0 <= frame_index < FRAME_COUNT:
            raise BusinessError("E_INJECTION", "dropped frame index is out of range", {"frame_index": frame_index})
        dropped.append(frame_index)
    injection["dropped_frames"] = sorted(set(dropped))
    jumps: dict[str, float] = {}
    if not isinstance(injection.get("timestamp_jumps", {}), dict):
        raise BusinessError("E_INJECTION", "timestamp_jumps must be an object")
    for key, value in injection.get("timestamp_jumps", {}).items():
        frame_index = int(key)
        if not 0 <= frame_index < FRAME_COUNT:
            raise BusinessError("E_INJECTION", "timestamp jump frame is out of range", {"frame_index": frame_index})
        jumps[str(frame_index)] = float(value)
    injection["timestamp_jumps"] = jumps
    valid_binary_faults = {None, "endian", "index_width", "metric_version", "field_bitmap"}
    if injection.get("binary_fault") not in valid_binary_faults:
        raise BusinessError("E_INJECTION", "unknown binary fault", {"binary_fault": injection.get("binary_fault")})
    injection["audio_sample_offset"] = int(injection.get("audio_sample_offset", 0))
    injection["audio_gain"] = float(injection.get("audio_gain", 1.0))
    injection["audio_tolerance_ms"] = float(injection.get("audio_tolerance_ms", 1.0))
    injection["timestamp_tolerance_ms"] = float(injection.get("timestamp_tolerance_ms", 1.0))
    if injection["audio_tolerance_ms"] <= 0 or injection["timestamp_tolerance_ms"] <= 0:
        raise BusinessError("E_INJECTION", "tolerance values must be positive")
    if injection["audio_gain"] <= 0:
        raise BusinessError("E_INJECTION", "audio gain must be positive")
    injection["force_closure_mismatch"] = bool(injection.get("force_closure_mismatch", False))
    injection["config_hash"] = hash_object({key: value for key, value in injection.items() if key != "config_hash"})
    return injection


def load_mutation(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"present": False, "encoding_params_patch": {}, "metric_policy_patch": {}, "shots": None}
    raw = json_load(path)
    if not isinstance(raw, dict):
        raise BusinessError("E_INJECTION", "mutation file must be an object")
    encoding_patch = raw.get("encoding_params", {})
    policy_patch = raw.get("metric_policy", {})
    shots = raw.get("shots")
    if not isinstance(encoding_patch, dict) or not isinstance(policy_patch, dict):
        raise BusinessError("E_INJECTION", "mutation encoding_params and metric_policy must be objects")
    if shots is not None and not isinstance(shots, list):
        raise BusinessError("E_INJECTION", "mutation shots must be an array")
    mutation = {
        "present": True,
        "encoding_params_patch": encoding_patch,
        "metric_policy_patch": policy_patch,
        "shots": shots,
    }
    mutation["mutation_hash"] = hash_object(mutation)
    return mutation


def _apply_mutation(
    mutation: dict[str, Any],
    base_encoding: dict[str, Any],
    base_policy: dict[str, Any],
    base_shots: list[dict],
) -> tuple[dict[str, Any], dict[str, Any], list[dict]]:
    encoding = _deep_merge(base_encoding, mutation["encoding_params_patch"])
    policy = _deep_merge(base_policy, mutation["metric_policy_patch"])
    shots_raw = mutation["shots"] if mutation["shots"] is not None else base_shots
    shots = validate_shots(copy.deepcopy(shots_raw), FRAME_COUNT)
    if list(policy.get("pixel_range", [])) != [0, 255]:
        raise BusinessError("E_SCHEMA_VERSION", "mutated metric policy has an unsupported pixel range")
    return encoding, policy, shots


def load_carriers(
    workspace: Path,
    truth: dict[str, Any],
    injection: dict[str, Any],
) -> tuple[dict[str, CarrierModel], dict[str, dict[str, Any]], dict[str, Any]]:
    models: dict[str, CarrierModel] = {}
    errors: dict[str, dict[str, Any]] = {}
    files: dict[str, Any] = {}
    loaders = {
        "project_json": lambda path: parse_project_json(path),
        "metric_binary": lambda path: parse_metric_binary(path, injection.get("binary_fault")),
        "legacy_json": lambda path: parse_legacy_json(path, workspace, truth),
    }
    for carrier_type, relative in (
        ("project_json", CARRIER_FILES["project_json"]),
        ("metric_binary", CARRIER_FILES["metric_binary"]),
        ("legacy_json", CARRIER_FILES["legacy_json"]),
    ):
        path = workspace / relative
        if not path.exists():
            errors[carrier_type] = BusinessError("E_IO", "carrier file is missing", {"path": relative}).to_dict()
            continue
        files[carrier_type] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        try:
            models[carrier_type] = loaders[carrier_type](path)
        except BusinessError as exc:
            errors[carrier_type] = exc.to_dict()
        except Exception as exc:  # isolate one carrier from the rest
            errors[carrier_type] = as_error(exc).to_dict()
    return models, errors, files


def _missing_code_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [record for record in records if record.get("missing") or record.get("psnr_db") is None]
    reasons: dict[str, int] = {}
    for record in missing:
        reason = str(record.get("reason") or "E_DECODE")
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "missing_count": len(missing),
        "missing_frames": [int(record["frame_index"]) for record in missing],
        "reasons": reasons,
    }


def _collect_business_errors(
    carrier_errors: dict[str, dict[str, Any]],
    equivalence: dict[str, Any],
    audio: dict[str, Any],
    missing: dict[str, Any],
    closure_error: dict[str, Any] | None,
    claim_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for carrier_type, error in carrier_errors.items():
        errors.append({"scope": f"carrier:{carrier_type}", **error})
    for reason, count in missing.get("reasons", {}).items():
        errors.append(
            {
                "scope": "actual_frames",
                "code": reason.split(":", 1)[0],
                "message": "one or more decoded frames are unavailable",
                "details": {"reason": reason, "count": count},
            }
        )
    if audio.get("error"):
        errors.append({"scope": "audio", **audio["error"]})
    if equivalence["overall"].get("business_code"):
        errors.append(
            {
                "scope": "equivalence",
                "code": equivalence["overall"]["business_code"],
                "message": "carrier conclusions are not equivalent",
                "details": {
                    "first_mismatch": equivalence.get("first_mismatch"),
                    "pairs": equivalence["pairs"],
                },
            }
        )
    for report in claim_reports:
        if report["status"] == "mismatch":
            errors.append(
                {
                    "scope": f"carrier_claim:{report['carrier']}",
                    "code": "E_EQUIV_MISMATCH",
                    "message": "carrier-declared metrics differ from decoded actual metrics",
                    "details": {"first_mismatch": report["first_mismatch"]},
                }
            )
    if closure_error:
        errors.append({"scope": "closure", **closure_error})
    unique: dict[str, dict[str, Any]] = {}
    for error in errors:
        key = f"{error.get('scope')}:{error.get('code')}:{json.dumps(error.get('details', {}), sort_keys=True, ensure_ascii=False)}"
        unique[key] = error
    return list(unique.values())


def evaluate_workspace(
    workspace: Path,
    injection_path: Path | None = None,
    mutation_path: Path | None = None,
    crash_point: str | None = None,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    truth = load_truth(workspace)
    assets = verify_assets(workspace, truth)
    injection = load_injection(injection_path)
    mutation = load_mutation(mutation_path)
    store = LedgerStore(workspace)
    prior = store.load_latest_committed()
    run = store.begin(expected_revision=expected_revision)

    models, carrier_errors, carrier_files = load_carriers(workspace, truth, injection)
    base_encoding = copy.deepcopy(ENCODING_PARAMS)
    base_policy = copy.deepcopy(METRIC_POLICY)
    base_shots = validate_shots(copy.deepcopy(truth["shots"]), FRAME_COUNT)
    encoding, policy, shots = _apply_mutation(mutation, base_encoding, base_policy, base_shots)
    content = carrier_content(workspace, truth)
    content_hash = str(content["content_hash"])
    context = build_context(content_hash, encoding, policy, shots, injection, FRAME_COUNT)

    prior_frames = None
    prior_segments = None
    prior_audio = None
    prior_meta = None
    if prior is not None:
        prior_frames = prior["shards"].get("actual_frames.json", {}).get("records")
        prior_segments = prior["shards"].get("actual_segments.json", {}).get("segments")
        prior_audio = prior["shards"].get("audio.json", {}).get("audio")
        prior_meta = prior["shards"].get("meta.json")
        if prior_meta is not None:
            prior_meta = copy.deepcopy(prior_meta)
            prior_meta["revision"] = prior["manifest"]["revision"]

    expected_frames = compute_frame_closure(prior_frames, context)
    computed = compute_actual_frames(
        workspace,
        truth,
        shots,
        policy,
        set(expected_frames),
        injection,
    )
    frame_by_index: dict[int, dict[str, Any]] = {}
    if prior_frames is not None:
        for frame in prior_frames:
            frame_by_index[int(frame["frame_index"])] = copy.deepcopy(frame)
    for frame_index, record in computed.items():
        record.setdefault("flags", 0)
        record["deps_hash"] = context["frame_deps"][frame_index]
        record["record_hash"] = record_content_hash(record)
        frame_by_index[frame_index] = record
    missing = {"missing_count": FRAME_COUNT, "missing_frames": list(range(FRAME_COUNT)), "reasons": {"E_DECODE": FRAME_COUNT}}
    if frame_by_index:
        missing = _missing_code_summary([frame_by_index[index] for index in sorted(frame_by_index)])
        if len(frame_by_index) != FRAME_COUNT:
            absent = [index for index in range(FRAME_COUNT) if index not in frame_by_index]
            missing["missing_count"] += len(absent)
            missing["missing_frames"].extend(absent)
            missing["missing_frames"] = sorted(set(missing["missing_frames"]))
            missing["reasons"]["E_DECODE"] = missing["reasons"].get("E_DECODE", 0) + len(absent)
    frames = [frame_by_index[index] for index in range(FRAME_COUNT) if index in frame_by_index]
    candidate_segment_deps = segment_deps(shots, frames, context)
    expected_segments = compute_segment_closure(prior_segments, shots, frames, context)
    new_segment_summaries = summarize_segments(frames, shots, policy, shot_ids=set(expected_segments))
    new_segment_by_id = {int(segment["shot_id"]): segment for segment in new_segment_summaries}
    segment_by_id: dict[int, dict[str, Any]] = {}
    if prior_segments is not None:
        prior_ids = {int(segment["shot_id"]) for segment in prior_segments}
        expected_set = set(expected_segments)
        for segment in prior_segments:
            shot_id = int(segment["shot_id"])
            if shot_id in expected_set:
                continue
            segment_by_id[shot_id] = copy.deepcopy(segment)
    for shot_id, segment in new_segment_by_id.items():
        segment["deps_hash"] = candidate_segment_deps[shot_id]
        segment_by_id[shot_id] = segment
    segments = [segment_by_id[shot_id] for shot_id in sorted(segment_by_id)]
    actual_recomputed_frames = sorted(expected_frames)
    actual_recomputed_segments = sorted(expected_segments)
    closure_error: dict[str, Any] | None = None
    force_mismatch = bool(injection.get("force_closure_mismatch", False))
    try:
        assert_closure_match(
            expected_frames,
            actual_recomputed_frames,
            expected_segments,
            actual_recomputed_segments,
            force_mismatch=force_mismatch,
        )
    except BusinessError as exc:
        closure_error = exc.to_dict()
    if force_mismatch:
        actual_recomputed_frames = sorted(set(actual_recomputed_frames) | {-1})
    closure = closure_report(
        prior_meta,
        context,
        expected_frames,
        actual_recomputed_frames,
        expected_segments,
        actual_recomputed_segments,
    )

    source_audio_sha = str(content["audio_sha256"])
    audio_dep = audio_dependency_hash(source_audio_sha, shots, injection)
    if prior_audio is not None and prior_audio.get("deps_hash") == audio_dep:
        audio = copy.deepcopy(prior_audio)
        audio_recomputed = False
    else:
        audio = evaluate_audio(
            resolve_workspace_path(workspace, truth["audio"]["wav_file"]),
            resolve_workspace_path(workspace, truth["audio"]["reference_file"]),
            shots,
            injection,
            source_audio_sha,
        )
        audio_recomputed = True

    equivalence = compare_carriers(models, carrier_errors)
    claim_reports: list[dict[str, Any]] = []
    if mutation["present"]:
        claim_reports = [
            {
                "carrier": carrier_type,
                "status": "not_run",
                "reason": "canonical inputs were mutated; carrier claims intentionally describe the base snapshot",
                "first_mismatch": None,
            }
            for carrier_type in ("project_json", "metric_binary", "legacy_json")
        ]
    else:
        for carrier_type in ("project_json", "metric_binary", "legacy_json"):
            if carrier_type in models:
                claim_reports.append(compare_model_to_actual(models[carrier_type], frames, segments))

    business_errors = _collect_business_errors(
        carrier_errors,
        equivalence,
        audio,
        missing,
        closure_error,
        claim_reports,
    )
    status = "ok" if not business_errors else "issues"
    metric_digest = metric_ledger_digest(frames, segments, policy)
    meta = {
        "schema_version": "ledger-meta/1.0",
        "tool_version": VERSION,
        "run_id": run["run_id"],
        "revision": run["revision"],
        "content_hash": content_hash,
        "context": {
            "content_hash": content_hash,
            "encoding_hash": context["encoding_hash"],
            "policy_hash": context["policy_hash"],
            "shots_hash": context["shots_hash"],
            "video_fault_hash": context["video_fault_hash"],
        },
        "frame_count": FRAME_COUNT,
        "shot_count": len(shots),
        "encoding_params": encoding,
        "metric_policy": policy,
        "shots": shots,
        "mutation": mutation,
        "injection_hash": injection["config_hash"],
        "metric_encoding_version": METRIC_ENCODING_VERSION,
        "dependencies": {
            "frames": "content+encoding+metric_policy+shot_assignment+per_frame_video_fault",
            "segments": "content+encoding+metric_policy+shot+member_frame_hashes",
            "audio": "audio_hash+shots+audio_injection+timebase",
        },
    }
    actual_frames_shard = {
        "schema_version": "actual-frame-metrics/1.0",
        "content_hash": content_hash,
        "policy_hash": context["policy_hash"],
        "metric_encoding_version": METRIC_ENCODING_VERSION,
        "records": frames,
        "missing": missing,
        "digest": metric_digest,
    }
    actual_segments_shard = {
        "schema_version": "actual-segment-metrics/1.0",
        "content_hash": content_hash,
        "policy_hash": context["policy_hash"],
        "segments": segments,
        "digest": hash_object(segments),
    }
    carriers_shard = {
        "schema_version": "carrier-load-report/1.0",
        "files": carrier_files,
        "models": {name: model.summary() for name, model in models.items()},
        "errors": carrier_errors,
        "claim_comparisons": claim_reports,
    }
    injections_shard = {
        "schema_version": "injection-ledger/1.0",
        "config": injection,
        "config_hash": injection["config_hash"],
        "serialized_canonical_utf8": canonical_json_bytes(injection).decode("utf-8"),
        "replay_command": "python -m evalworkbench.cli evaluate --workspace <workspace> --inject-file <recorded-config.json>",
    }
    run_summary = {
        "schema_version": "run-summary/1.0",
        "status": status,
        "content_hash": content_hash,
        "metric_ledger_digest": metric_digest,
        "equivalence": equivalence["overall"],
        "audio_status": audio["overall"]["status"],
        "closure": {
            "reason": closure["reason"],
            "frames": closure["frames"],
            "segments": closure["segments"],
            "match": closure["match"],
        },
        "business_errors": business_errors,
        "missing_units": missing,
        "audio_recomputed": audio_recomputed,
        "carrier_claims": claim_reports,
    }

    store.write_shard("meta.json", meta)
    if crash_point == "after_meta":
        os._exit(91)
    store.write_shard("actual_frames.json", actual_frames_shard)
    store.write_shard("actual_segments.json", actual_segments_shard)
    store.write_shard("audio.json", {"schema_version": "audio-ledger/1.0", "audio": audio})
    store.write_shard("equivalence.json", equivalence)
    store.write_shard("closure.json", closure)
    store.write_shard("carriers.json", carriers_shard)
    store.write_shard("injections.json", injections_shard)
    store.write_shard("assets.json", assets)
    store.write_shard("run_summary.json", run_summary)
    if crash_point == "before_manifest":
        os._exit(92)
    manifest = store.commit(
        content_hash=content_hash,
        metric_ledger_digest=metric_digest,
        expected_revision=expected_revision,
        extra={
            "evaluation_status": status,
            "business_codes": sorted({error["code"] for error in business_errors}),
            "closure_match": closure["match"],
            "equivalence_ok": equivalence["overall"]["equivalent"],
        },
    )
    if crash_point == "after_manifest":
        os._exit(93)
    closed = store.mark_closed(manifest)
    return {
        "schema_version": "evaluation-result/1.0",
        "workspace": str(workspace),
        "ledger": {
            "revision": manifest["revision"],
            "run_id": manifest["run_id"],
            "manifest_hash": manifest["manifest_hash"],
            "directory": str(store.run_dir),
            "closed": closed["complete"],
        },
        "status": status,
        "content_hash": content_hash,
        "metric_ledger_digest": metric_digest,
        "frame_count": FRAME_COUNT,
        "shot_count": len(shots),
        "segment_summaries": segments,
        "audio": audio,
        "equivalence": equivalence,
        "closure": closure,
        "business_errors": business_errors,
        "carrier_claims": claim_reports,
        "missing_units": missing,
    }


def recover_workspace(workspace: Path) -> dict[str, Any]:
    return LedgerStore(workspace.resolve()).recover()


def preview_workspace(workspace: Path, frame_index: int, output: Path) -> dict[str, Any]:
    truth = load_truth(workspace.resolve())
    verify_assets(workspace.resolve(), truth)
    return export_preview(workspace.resolve(), truth, frame_index, output)
