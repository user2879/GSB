from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .cache import Artifact, ArtifactStore
from .config import (
    clip_interval,
    config_fingerprint,
    ms_to_samples,
    track_interval,
    validate_global,
    validate_track_settings,
)
from .conversion import CONVERTER_VERSION, convert_plane
from .errors import BusinessError, E_MIX_CONFIG, E_TRACK_FORMAT, MixConfigError
from .fingerprint import (
    array_fingerprint,
    canonical_json,
    digest_file,
    digest_json,
    digest_tree,
    state_key,
)
from .formats import package_files, parse_package
from .injections import (
    Injection,
    apply_parse_faults,
    config_injections,
    gain_injection,
    injections_for_clip,
)
from .mixing import (
    MIX_ALGORITHM_VERSION,
    FragmentPlan,
    GainArtifact,
    full_component_matrix,
    plan_fragments,
    quantize_gain,
    render_fragment,
    weighted_contributions,
)
from .models import (
    DEFAULT_TRACKS,
    CacheEvent,
    TrackIssue,
    TrackPlane,
)
from .pcm import (
    int16_vector_sha256,
    i16_to_bytes,
    read_wav,
    sha256_bytes,
    write_wav,
)
from .snapshots import SnapshotManager

PARSER_VERSION = "formats-v1"
GAIN_VERSION = "q23-gain-v1"
SCHEMA_VERSION = 1


@dataclass
class RunResult:
    report: dict[str, Any] | None
    stopped: bool
    exit_code: int
    workdir: Path


class StopController:
    def __init__(self) -> None:
        self.requested = False
        self.received_signal: int | None = None
        self._previous: dict[int, Any] = {}

    def handler(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.received_signal = signum

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self.handler)
            except (ValueError, OSError):
                pass

    def restore(self) -> None:
        for signum, previous in self._previous.items():
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError):
                pass
        self._previous.clear()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def _issue_from_error(
    error: BusinessError,
    *,
    clip_id: str,
    stage: str,
    fallback_track: str | None = None,
    fallback_injection: str | None = None,
    fragment_index: int | None = None,
) -> TrackIssue:
    return TrackIssue(
        clip_id=clip_id,
        track=error.track or fallback_track or "<clip>",
        stage=stage,
        code=error.code,
        message=error.message,
        injection_id=error.injection_id or fallback_injection,
        fragment_index=fragment_index,
    )


def _plane_arrays(plane: TrackPlane) -> dict[str, np.ndarray]:
    return {f"track__{name}": plane.tracks[name] for name in plane.track_names}


def _plane_from_artifact(artifact: Artifact) -> TrackPlane:
    metadata = artifact.meta["plane"]
    tracks = {
        key[len("track__") :]: np.asarray(value, dtype=np.int32)
        for key, value in artifact.arrays.items()
        if key.startswith("track__")
    }
    issues = [TrackIssue(**item) for item in metadata.get("issues", [])]
    return TrackPlane(
        clip_id=metadata["clip_id"],
        format_version=metadata["format_version"],
        sample_rate=int(metadata["sample_rate"]),
        bit_depth=int(metadata["bit_depth"]),
        frames=int(metadata["frames"]),
        channel_map={str(k): int(v) for k, v in metadata["channel_map"].items()},
        tracks=tracks,
        track_frames={
            str(k): int(v) for k, v in metadata["track_frames"].items()
        },
        source_metadata=metadata.get("source_metadata", {}),
        missing_tracks=list(metadata.get("missing_tracks", [])),
        issues=issues,
        package_path=metadata.get("package_path", ""),
    )


def _track_fingerprint(plane: TrackPlane, track: str) -> str:
    return array_fingerprint(
        {track: plane.tracks[track]},
        {
            "format_version": plane.format_version,
            "sample_rate": plane.sample_rate,
            "frames": plane.frames,
            "channel": plane.channel_map.get(track),
        },
    )


def clone_plane(plane: TrackPlane) -> TrackPlane:
    return TrackPlane(
        clip_id=plane.clip_id,
        format_version=plane.format_version,
        sample_rate=plane.sample_rate,
        bit_depth=plane.bit_depth,
        frames=plane.frames,
        channel_map=dict(plane.channel_map),
        tracks={name: values.copy() for name, values in plane.tracks.items()},
        track_frames=dict(plane.track_frames),
        source_metadata=json.loads(canonical_json(plane.source_metadata)),
        missing_tracks=list(plane.missing_tracks),
        issues=[
            TrackIssue(**issue.as_dict()) for issue in plane.issues
        ],
        package_path=plane.package_path,
    )


def render_signature(report: dict[str, Any], workdir: str | Path) -> dict[str, Any]:
    clips = []
    for clip in sorted(report.get("clips", []), key=lambda item: item["clip_id"]):
        fragments = []
        for fragment in clip.get("fragments", []):
            fragments.append(
                {
                    "fragment_index": fragment["fragment_index"],
                    "sample_start": fragment["sample_start"],
                    "sample_end": fragment["sample_end"],
                    "output_start": fragment["output_start"],
                    "output_end": fragment["output_end"],
                    "frames": fragment["frames"],
                    "tracks": fragment["tracks"],
                    "peak": fragment["peak"],
                    "pcm_sha256": fragment["pcm_sha256"],
                    "component_sha256": fragment["component_sha256"],
                    "component_sums": fragment["component_sums"],
                    "first_frame": fragment["first_frame"],
                    "last_frame": fragment["last_frame"],
                }
            )
        clips.append(
            {
                "clip_id": clip["clip_id"],
                "status": clip.get("status"),
                "format_version": clip.get("format_version"),
                "track_names": clip.get("track_names"),
                "output_frames": clip.get("output_frames"),
                "output_pcm_sha256": clip.get("output_pcm_sha256"),
                "fragments": fragments,
            }
        )
    return {
        "clips": clips,
        "failures": [
            {
                "clip_id": item["clip_id"],
                "track": item["track"],
                "stage": item["stage"],
                "code": item["code"],
                "fragment_index": item["fragment_index"],
            }
            for item in sorted(
                report.get("failures", []),
                key=lambda item: (
                    item["clip_id"],
                    item["track"],
                    item["stage"],
                    -1 if item["fragment_index"] is None else item["fragment_index"],
                    item["code"],
                ),
            )
        ],
    }


def compare_signatures(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_digest = digest_json(left)
    right_digest = digest_json(right)
    return {
        "matches": left_digest == right_digest,
        "incremental_signature": left_digest,
        "full_signature": right_digest,
        "fields_compared": [
            "mix PCM bytes",
            "fragment sample ranges",
            "fragment peaks",
            "per-frame weighted component digests",
            "per-frame first/last weighted summaries",
            "track ordering",
            "failure ordering",
        ],
    }


class RemixEngine:
    def __init__(
        self,
        *,
        samples_root: str | Path,
        config: dict[str, Any],
        workdir: str | Path,
        injections: Iterable[Injection] = (),
        force_workers: int | None = None,
        stop_controller: StopController | None = None,
        quiet: bool = False,
    ) -> None:
        self.samples_root = Path(samples_root).resolve()
        self.config = config
        self.workdir = Path(workdir).resolve()
        self.injections = sorted(list(injections), key=lambda item: item.id)
        self.force_workers = force_workers
        self.stop = stop_controller or StopController()
        self.quiet = quiet
        self.config_hash = config_fingerprint(config)
        self.sample_manifest = self._load_sample_manifest()
        self.workers = max(1, int(force_workers or config["thread_pool_size"]))
        self.events: list[CacheEvent] = []
        self.failures: list[TrackIssue] = []
        self.new_keys: dict[str, set[str]] = {
            "parse": set(),
            "convert": set(),
            "gain": set(),
            "mix": set(),
        }
        self.reused = 0
        self.recomputed = 0
        self.completed_fragments: set[str] = set()
        self._executor: ThreadPoolExecutor | None = None
        self._start_snapshot_version = 0

        cache_root = self.workdir / "cache" / "artifacts"
        snapshot_root = self.workdir / "snapshots"
        probe = ArtifactStore(cache_root)
        self.snapshots = SnapshotManager(
            snapshot_root, artifact_exists=lambda key: probe.path_for(key).is_file()
        )
        self._start_snapshot_version = self.snapshots.state.global_version
        trusted = set(self.snapshots.state.artifacts) if self.snapshots.recovery["used"] else set()
        self.cache = ArtifactStore(cache_root, trusted_keys=trusted)
        if (
            self.snapshots.state.config_hash
            and self.snapshots.state.config_hash == self.config_hash
        ):
            self.completed_fragments.update(self.snapshots.state.completed_fragments)

    def _load_sample_manifest(self) -> dict[str, Any]:
        path = self.samples_root / "manifest.json"
        if not path.is_file():
            return {"clips": []}
        return json.loads(path.read_text(encoding="utf-8"))

    @property
    def progress_path(self) -> Path:
        return self.workdir / "progress.json"

    def _format_versions(self) -> list[str]:
        return sorted({clip["source_format"] for clip in self.config["clips"]})

    def _write_progress(
        self,
        *,
        stage: str,
        stopped: bool = False,
        stage_progress: dict[str, Any] | None = None,
    ) -> None:
        _atomic_json(
            self.progress_path,
            {
                "schema_version": SCHEMA_VERSION,
                "global_version": self.snapshots.state.global_version,
                "stage": stage,
                "config_hash": self.config_hash,
                "format_versions": self._format_versions(),
                "completed_fragments": sorted(self.completed_fragments),
                "stage_versions": self.snapshots.state.stage_versions,
                "stage_progress": stage_progress or {},
                "stopped": stopped,
                "received_signal": self.stop.received_signal,
            },
        )

    def _commit(
        self,
        stage: str,
        *,
        stage_progress: dict[str, Any] | None = None,
        new_keys: Iterable[str] | None = None,
        completed_fragments: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        manifest = self.snapshots.commit(
            stage=stage,
            config_hash=self.config_hash,
            format_versions=self._format_versions(),
            completed_fragments=(
                self.completed_fragments
                if completed_fragments is None
                else completed_fragments
            ),
            new_artifact_keys=new_keys or (),
            stage_progress=stage_progress,
            snapshot_interval=int(self.config["snapshot_interval_clips"]),
        )
        self.cache.trust(set(manifest.get("artifacts", [])))
        self._write_progress(stage=stage, stage_progress=stage_progress)
        return manifest

    def _maybe_stop_or_kill(self, stage: str) -> None:
        delay = os.environ.get("RMXW_STAGE_DELAY_SECONDS")
        if delay:
            try:
                time.sleep(max(0.0, float(delay)))
            except ValueError:
                pass
        env_stop = os.environ.get("RMXW_GRACEFUL_STOP_AFTER_STAGE")
        if env_stop and env_stop == stage:
            self.stop.requested = True
        env_kill = os.environ.get("RMXW_FORCE_KILL_AFTER_STAGE")
        if env_kill and env_kill == stage:
            os._exit(137)

    def _parallel(
        self,
        items: list[tuple[tuple, Callable[[], Any]]],
    ) -> list[Any]:
        if self.workers == 1:
            return [function() for _key, function in items]
        assert self._executor is not None
        futures = [self._executor.submit(function) for _key, function in items]
        return [future.result() for future in futures]

    def _add_failure(self, issue: TrackIssue) -> None:
        self.failures.append(issue)

    def _record_event(
        self,
        *,
        stage: str,
        clip_id: str,
        track: str | None,
        fragment_index: int | None,
        cache_key: str,
        status: str,
        injection_id: str | None = None,
    ) -> None:
        if status == "reuse":
            self.reused += 1
        else:
            self.recomputed += 1
        self.events.append(
            CacheEvent(
                stage=stage,
                clip_id=clip_id,
                track=track,
                fragment_index=fragment_index,
                cache_key=cache_key,
                status=status,
                injection_id=injection_id,
            )
        )

    def _clip_by_id(self) -> dict[str, dict[str, Any]]:
        return {str(clip["clip_id"]): clip for clip in self.config["clips"]}

    def _resolve_sample_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.samples_root / path

    def _parse_clip(
        self,
        clip: dict[str, Any],
        clip_injections: list[Injection],
    ) -> tuple[TrackPlane | None, TrackIssue | None]:
        clip_id = str(clip["clip_id"])
        format_version = str(clip["source_format"])
        source = self._resolve_sample_path(clip["source_package"])
        overrides_raw = self.config.get("track_overrides", {}).get(clip_id, {})
        overrides = {
            str(track): self._resolve_sample_path(path)
            for track, path in sorted(overrides_raw.items())
        }
        parse_injections = [item for item in clip_injections if item.stage == "parse"]
        try:
            prepared, applied = apply_parse_faults(
                source,
                format_version,
                parse_injections,
                self.workdir / "injections" / clip_id,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issue = TrackIssue(
                clip_id=clip_id,
                track=parse_injections[0].track if parse_injections else "<clip>",
                stage="parse",
                code=E_TRACK_FORMAT,
                message=f"fault preparation failed: {exc}",
                injection_id=parse_injections[0].id if parse_injections else None,
            )
            return None, issue
        source_digest = digest_tree(prepared)
        override_digests = {
            track: digest_file(path) for track, path in overrides.items() if path.is_file()
        }
        params = {
            "format_version": format_version,
            "expected_tracks": sorted(clip["tracks"]),
            "overrides": override_digests,
            "injection_ids": [item.id for item in applied],
        }
        key = state_key("parse", source_digest, params, PARSER_VERSION)
        artifact = self.cache.get(key)
        if artifact is not None:
            try:
                plane = _plane_from_artifact(artifact)
                for issue in plane.issues:
                    self._add_failure(issue)
                self._record_event(
                    stage="parse",
                    clip_id=clip_id,
                    track=None,
                    fragment_index=None,
                    cache_key=key,
                    status="reuse",
                    injection_id=applied[0].id if applied else None,
                )
                return plane, None
            except (KeyError, TypeError, ValueError):
                artifact = None
        injection_by_track = {
            item.track: item for item in applied if item.track is not None
        }
        primary = applied[0] if applied else None
        try:
            plane = parse_package(
                clip_id,
                prepared,
                format_version,
                expected_tracks=clip["tracks"],
                overrides=overrides,
                injection_id=primary.id if primary else None,
                injection_track=primary.track if primary else None,
            )
            for issue in plane.issues:
                injection = injection_by_track.get(issue.track)
                if injection and not issue.injection_id:
                    issue.injection_id = injection.id
            meta = {"plane": plane.as_metadata(), "content_fingerprint": plane.content_fingerprint()}
            self.cache.put(key, meta=meta, arrays=_plane_arrays(plane))
            self.new_keys["parse"].add(key)
            self._record_event(
                stage="parse",
                clip_id=clip_id,
                track=None,
                fragment_index=None,
                cache_key=key,
                status="recompute",
                injection_id=primary.id if primary else None,
            )
            for issue in plane.issues:
                self._add_failure(issue)
            return plane, None
        except BusinessError as error:
            target_injection = primary
            if error.track and error.track in injection_by_track:
                target_injection = injection_by_track[error.track]
            issue = _issue_from_error(
                error,
                clip_id=clip_id,
                stage="parse",
                fallback_track=target_injection.track if target_injection else None,
                fallback_injection=target_injection.id if target_injection else None,
            )
            self._record_event(
                stage="parse",
                clip_id=clip_id,
                track=None,
                fragment_index=None,
                cache_key=key,
                status="recompute-failed",
                injection_id=issue.injection_id,
            )
            return None, issue
        except (OSError, ValueError) as error:
            issue = TrackIssue(
                clip_id=clip_id,
                track=primary.track if primary and primary.track else "<clip>",
                stage="parse",
                code=E_TRACK_FORMAT,
                message=str(error),
                injection_id=primary.id if primary else None,
            )
            self._record_event(
                stage="parse",
                clip_id=clip_id,
                track=None,
                fragment_index=None,
                cache_key=key,
                status="recompute-failed",
                injection_id=issue.injection_id,
            )
            return None, issue

    def _run_parse_stage(
        self,
        clips: list[dict[str, Any]],
    ) -> tuple[dict[str, TrackPlane], set[str]]:
        tasks: list[tuple[tuple, Callable[[], Any]]] = []
        for clip in sorted(clips, key=lambda item: item["clip_id"]):
            clip_id = str(clip["clip_id"])
            clip_injections = injections_for_clip(self.injections, clip_id)
            tasks.append(
                (
                    (clip_id,),
                    lambda clip=clip, clip_injections=clip_injections: self._parse_clip(
                        clip, clip_injections
                    ),
                )
            )
        results = self._parallel(tasks)
        planes: dict[str, TrackPlane] = {}
        fatal: set[str] = set()
        for (clip_id,), (plane, issue) in zip(
            [key for key, _ in tasks], results
        ):
            if issue is not None:
                self._add_failure(issue)
                fatal.add(clip_id)
            if plane is not None:
                planes[clip_id] = plane
        progress = {
            "clips_completed": sorted(planes),
            "clips_failed": sorted(fatal),
            "tracks": {
                clip_id: sorted(plane.tracks)
                for clip_id, plane in sorted(planes.items())
            },
        }
        self._commit(
            "parse",
            stage_progress=progress,
            new_keys=self.new_keys["parse"],
        )
        self._maybe_stop_or_kill("parse")
        return planes, fatal

    def _run_canonical_conversion_stage(
        self,
        clips: list[dict[str, Any]],
        planes: dict[str, TrackPlane],
        fatal_parse: set[str],
    ) -> tuple[dict[str, TrackPlane], dict[str, str]]:
        converged: dict[str, TrackPlane] = {}
        fingerprints: dict[str, str] = {}
        tasks: list[tuple[tuple, Callable[[], Any]]] = []
        for clip in sorted(clips, key=lambda item: item["clip_id"]):
            clip_id = str(clip["clip_id"])
            plane = planes.get(clip_id)
            if plane is None or clip_id in fatal_parse:
                continue

            def task(plane: TrackPlane = plane, clip_id: str = clip_id):
                key = state_key(
                    "convert",
                    plane.content_fingerprint(),
                    {
                        "source_format": plane.format_version,
                        "target_format": "plane-v1",
                        "channel_map": plane.channel_map,
                    },
                    CONVERTER_VERSION,
                )
                artifact = self.cache.get(key)
                if artifact is not None:
                    fingerprint = str(artifact.meta["conversion_fingerprint"])
                    status = "reuse"
                else:
                    fingerprint = digest_json(
                        {
                            "upstream": plane.content_fingerprint(),
                            "conversion_key": key,
                            "target_format": "plane-v1",
                        }
                    )
                    self.cache.put(
                        key,
                        meta={
                            "conversion_fingerprint": fingerprint,
                            "source_format": plane.format_version,
                            "target_format": "plane-v1",
                            "plane": plane.as_metadata(),
                            "losses": [],
                        },
                    )
                    self.new_keys["convert"].add(key)
                    status = "recompute"
                event = CacheEvent(
                    stage="convert",
                    clip_id=clip_id,
                    track=None,
                    fragment_index=None,
                    cache_key=key,
                    status=status,
                )
                return clip_id, clone_plane(plane), fingerprint, event

            tasks.append(((clip_id,), task))
        results = self._parallel(tasks)
        for clip_id, plane, fingerprint, event in sorted(results, key=lambda item: item[0]):
            converged[clip_id] = plane
            fingerprints[clip_id] = fingerprint
            self._record_event(
                stage=event.stage,
                clip_id=event.clip_id,
                track=event.track,
                fragment_index=event.fragment_index,
                cache_key=event.cache_key,
                status=event.status,
            )
        progress = {"clips_completed": sorted(converged), "target_format": "plane-v1"}
        self._commit(
            "convert",
            stage_progress=progress,
            new_keys=self.new_keys["convert"],
        )
        return converged, fingerprints

    def _actual_conversions(self) -> list[dict[str, Any]]:
        specs = self.config.get("conversions", [])
        if not specs:
            return []
        tasks: list[tuple[tuple, Callable[[], Any]]] = []
        for spec in specs:
            conversion_id = str(spec["id"])
            source_package = self._resolve_sample_path(spec["source_package"])
            source_format = str(spec["source_format"])
            target_format = str(spec["target_format"])

            def task(
                spec: dict[str, Any] = spec,
                conversion_id: str = conversion_id,
                source_package: Path = source_package,
                source_format: str = source_format,
                target_format: str = target_format,
            ):
                destination = self.workdir / "conversions" / conversion_id
                try:
                    source_digest = digest_tree(source_package)
                except OSError as exc:
                    issue = TrackIssue(
                        clip_id=str(spec.get("clip_id", conversion_id)),
                        track="<clip>",
                        stage="convert",
                        code=E_TRACK_FORMAT,
                        message=f"conversion source unreadable: {exc}",
                    )
                    return conversion_id, None, issue, None
                key = state_key(
                    "convert",
                    source_digest,
                    {
                        "source_format": source_format,
                        "target_format": target_format,
                    },
                    CONVERTER_VERSION,
                )
                artifact = self.cache.get(key)
                report_path = destination / "conversion_report.json"
                if artifact is not None and report_path.is_file():
                    event = CacheEvent(
                        stage="convert",
                        clip_id=str(spec.get("clip_id", conversion_id)),
                        track=None,
                        fragment_index=None,
                        cache_key=key,
                        status="reuse",
                    )
                    report = dict(artifact.meta["conversion_report"])
                    report["destination"] = str(destination)
                    return conversion_id, report, None, event
                try:
                    plane = parse_package(
                        str(spec.get("clip_id", conversion_id)),
                        source_package,
                        source_format,
                        expected_tracks=DEFAULT_TRACKS,
                    )
                    report = convert_plane(
                        plane,
                        target_format,
                        destination,
                        source_package=source_package,
                    )
                    self.cache.put(key, meta={"conversion_report": report})
                    self.new_keys["convert"].add(key)
                    event = CacheEvent(
                        stage="convert",
                        clip_id=str(spec.get("clip_id", conversion_id)),
                        track=None,
                        fragment_index=None,
                        cache_key=key,
                        status="recompute",
                    )
                    return conversion_id, report, None, event
                except BusinessError as error:
                    issue = _issue_from_error(
                        error,
                        clip_id=str(spec.get("clip_id", conversion_id)),
                        stage="convert",
                    )
                    return conversion_id, None, issue, None
            tasks.append(((conversion_id,), task))
        results = self._parallel(tasks)
        reports = []
        for conversion_id, report, issue, event in sorted(
            results, key=lambda item: item[0]
        ):
            if event is not None:
                self._record_event(
                    stage=event.stage,
                    clip_id=event.clip_id,
                    track=event.track,
                    fragment_index=event.fragment_index,
                    cache_key=event.cache_key,
                    status=event.status,
                )
            if issue is not None:
                self._add_failure(issue)
            if report is not None:
                reports.append(report)
        return reports

    def _run_conversion_stage(
        self,
        clips: list[dict[str, Any]],
        planes: dict[str, TrackPlane],
        fatal_parse: set[str],
    ) -> tuple[dict[str, TrackPlane], dict[str, str], list[dict[str, Any]]]:
        converged, fingerprints = self._run_canonical_conversion_stage(
            clips, planes, fatal_parse
        )
        conversions = self._actual_conversions()
        progress = {
            "clips_completed": sorted(converged),
            "interchange_conversions": sorted(item["source_clip_id"] + ":" + item["source_format_version"] + "->" + item["target_format_version"] for item in conversions),
        }
        self._commit(
            "convert",
            stage_progress=progress,
            new_keys=self.new_keys["convert"],
        )
        self._maybe_stop_or_kill("convert")
        return converged, fingerprints, conversions

    def _run_gain_stage(
        self,
        clips: list[dict[str, Any]],
        planes: dict[str, TrackPlane],
        conversion_fingerprints: dict[str, str],
        fatal_parse: set[str],
        global_mix_blocked: bool,
    ) -> tuple[dict[str, dict[str, GainArtifact]], dict[str, dict[str, tuple[int, int]]], set[str]]:
        gains: dict[str, dict[str, GainArtifact]] = {}
        trims: dict[str, dict[str, tuple[int, int]]] = {}
        blocked: set[str] = set()
        if global_mix_blocked:
            self._commit(
                "gain",
                stage_progress={"skipped": True, "reason": "global E_MIX_CONFIG"},
            )
            self._maybe_stop_or_kill("gain")
            return gains, trims, blocked
        tasks: list[tuple[tuple, Callable[[], Any]]] = []
        quantization = int(self.config["gain_quantization"])
        for clip in sorted(clips, key=lambda item: item["clip_id"]):
            clip_id = str(clip["clip_id"])
            plane = planes.get(clip_id)
            if plane is None or clip_id in fatal_parse:
                continue
            valid_tracks: dict[str, tuple[int, bool, int, int]] = {}
            for track in sorted(clip["tracks"]):
                settings = clip["tracks"][track]
                injection = gain_injection(self.injections, clip_id, track)
                effective_settings = settings
                if injection is not None:
                    effective_settings = {
                        **settings,
                        "gain": injection.value
                        if injection.kind in {"gain_nonfinite", "gain_negative"}
                        else settings.get("gain"),
                    }
                try:
                    gain_q, mute, start_ms, end_ms = validate_track_settings(
                        {**clip, "_gain_quantization": quantization},
                        track,
                        effective_settings,
                    )
                    start_sample = ms_to_samples(start_ms, plane.sample_rate)
                    end_sample = (
                        plane.frames
                        if end_ms is None
                        else ms_to_samples(end_ms, plane.sample_rate)
                    )
                    if start_sample < 0 or start_sample >= end_sample or start_sample >= plane.frames:
                        raise MixConfigError(
                            f"{clip_id}.{track} trim starts outside the source",
                            track=track,
                        )
                    if end_sample > plane.frames:
                        raise MixConfigError(
                            f"{clip_id}.{track} trim end exceeds source frames",
                            track=track,
                        )
                    valid_tracks[track] = (gain_q, mute, start_sample, end_sample)
                except MixConfigError as error:
                    issue = _issue_from_error(
                        error,
                        clip_id=clip_id,
                        stage="gain",
                        fallback_track=track,
                        fallback_injection=injection.id if injection else None,
                    )
                    self._add_failure(issue)
                    blocked.add(clip_id)
                    continue
                track_fp = _track_fingerprint(plane, track)
                key = state_key(
                    "gain",
                    track_fp,
                    {
                        "track": track,
                        "gain_parameter": effective_settings.get("gain"),
                        "gain_q": gain_q,
                        "quantization": quantization,
                        "mute": mute,
                        "format_version": plane.format_version,
                        "channel": plane.channel_map.get(track),
                    },
                    GAIN_VERSION,
                )

                def task(
                    clip_id: str = clip_id,
                    track: str = track,
                    key: str = key,
                    plane: TrackPlane = plane,
                    settings: dict[str, Any] = settings,
                    gain_q: int = gain_q,
                    mute: bool = mute,
                ):
                    artifact = self.cache.get(key)
                    if artifact is not None and "weights" in artifact.arrays:
                        weights = np.asarray(artifact.arrays["weights"], dtype=np.int64)
                        status = "reuse"
                    else:
                        weights = weighted_contributions(
                            plane.tracks[track], gain_q, mute
                        )
                        self.cache.put(
                            key,
                            meta={
                                "track": track,
                                "gain_q": gain_q,
                                "gain_parameter": float(effective_settings["gain"]),
                                "mute": mute,
                                "quantization": quantization,
                                "format_version": plane.format_version,
                            },
                            arrays={"weights": weights},
                        )
                        self.new_keys["gain"].add(key)
                        status = "recompute"
                    event = CacheEvent(
                        stage="gain",
                        clip_id=clip_id,
                        track=track,
                        fragment_index=None,
                        cache_key=key,
                        status=status,
                    )
                    return (
                        GainArtifact(
                            track=track,
                            gain_q=gain_q,
                            mute=mute,
                            weights=weights,
                            gain_parameter=float(effective_settings["gain"]),
                        ),
                        event,
                    )

                tasks.append(((clip_id, track), task))
            trims[clip_id] = {
                track: (values[2], values[3]) for track, values in valid_tracks.items()
            }
            if valid_tracks and all(values[1] for values in valid_tracks.values()):
                self._add_failure(
                    TrackIssue(
                        clip_id=clip_id,
                        track="<all>",
                        stage="gain",
                        code=E_MIX_CONFIG,
                        message="all configured tracks are muted",
                    )
                )
                blocked.add(clip_id)
        results = self._parallel(tasks)
        for (clip_id, _track), (gain_artifact, event) in zip(
            [key for key, _ in tasks], results
        ):
            gains.setdefault(clip_id, {})[gain_artifact.track] = gain_artifact
            self._record_event(
                stage=event.stage,
                clip_id=event.clip_id,
                track=event.track,
                fragment_index=event.fragment_index,
                cache_key=event.cache_key,
                status=event.status,
            )
        progress = {
            "clips_completed": sorted(gains),
            "clips_blocked": sorted(blocked),
            "tracks": {clip_id: sorted(items) for clip_id, items in sorted(gains.items())},
        }
        self._commit(
            "gain",
            stage_progress=progress,
            new_keys=self.new_keys["gain"],
        )
        self._maybe_stop_or_kill("gain")
        return gains, trims, blocked

    def _fragment_key(
        self,
        *,
        clip: dict[str, Any],
        plan: FragmentPlan,
        gains: dict[str, GainArtifact],
        trims: dict[str, tuple[int, int]],
        quantization: int,
    ) -> tuple[str, str]:
        refs = [
            {
                "track": track,
                "gain_key": _gain_cache_key(gains[track]),
                "trim": list(trims[track]),
            }
            for track in plan.tracks
        ]
        upstream = digest_json(
            {
                "refs": refs,
                "format_version": clip["source_format"],
            }
        )
        parameters = {
            "clip_id": plan.clip_id,
            "fragment_index": plan.fragment_index,
            "sample_start": plan.sample_start,
            "sample_end": plan.sample_end,
            "output_start": plan.output_start,
            "output_end": plan.output_end,
            "quantization": quantization,
            "format_version": clip["source_format"],
            "refs": refs,
        }
        return state_key("mix", upstream, parameters, MIX_ALGORITHM_VERSION), upstream

    def _run_render_stage(
        self,
        clips: list[dict[str, Any]],
        planes: dict[str, TrackPlane],
        gains: dict[str, dict[str, GainArtifact]],
        trims: dict[str, dict[str, tuple[int, int]]],
        blocked: set[str],
        global_mix_blocked: bool,
        fatal_parse: set[str],
    ) -> dict[str, dict[str, Any]]:
        render_reports: dict[str, dict[str, Any]] = {}
        if global_mix_blocked:
            for clip in clips:
                stale = self.workdir / "mixes" / f"{clip['clip_id']}.wav"
                if stale.exists():
                    stale.unlink()
            self._commit(
                "render",
                stage_progress={"skipped": True, "reason": "global E_MIX_CONFIG"},
            )
            self._maybe_stop_or_kill("render")
            return render_reports
        quantization = int(self.config["gain_quantization"])
        frame_length = int(self.config["render_frame_length"])
        interval = int(self.config["snapshot_interval_clips"])
        all_tasks: list[tuple[str, FragmentPlan, Callable[[], Any]]] = []
        clip_context: dict[str, dict[str, Any]] = {}
        for clip in sorted(clips, key=lambda item: item["clip_id"]):
            clip_id = str(clip["clip_id"])
            plane = planes.get(clip_id)
            if (
                plane is None
                or clip_id in fatal_parse
                or clip_id in blocked
                or clip_id not in gains
            ):
                output_path = self.workdir / "mixes" / f"{clip_id}.wav"
                if output_path.exists():
                    output_path.unlink()
                continue
            try:
                clip_start, clip_end = clip_interval(
                    clip, plane.frames, plane.sample_rate
                )
                muted = {
                    track
                    for track, artifact in gains[clip_id].items()
                    if artifact.mute
                }
                plans = plan_fragments(
                    clip_id=clip_id,
                    frames=plane.frames,
                    render_frame_length=frame_length,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    track_trims=trims[clip_id],
                    muted=muted,
                )
            except (MixConfigError, KeyError) as error:
                issue = _issue_from_error(error, clip_id=clip_id, stage="mix")
                self._add_failure(issue)
                continue
            clip_context[clip_id] = {
                "clip": clip,
                "plane": plane,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "plans": plans,
            }
            for plan in plans:
                key, upstream = self._fragment_key(
                    clip=clip,
                    plan=plan,
                    gains=gains[clip_id],
                    trims=trims[clip_id],
                    quantization=quantization,
                )

                def task(
                    clip_id: str = clip_id,
                    plan: FragmentPlan = plan,
                    key: str = key,
                    upstream: str = upstream,
                ):
                    artifact = self.cache.get(key)
                    if artifact is not None and "pcm" in artifact.arrays and "weighted" in artifact.arrays:
                        pcm = np.asarray(artifact.arrays["pcm"], dtype=np.int16)
                        weighted = np.asarray(artifact.arrays["weighted"], dtype=np.int64)
                        mixed_q23 = np.asarray(
                            artifact.arrays.get(
                                "mixed_q23", np.zeros(pcm.size, dtype=np.int64)
                            ),
                            dtype=np.int64,
                        )
                        meta = dict(artifact.meta["fragment"])
                        status = "reuse"
                    else:
                        rendered = render_fragment(
                            plan,
                            gains[clip_id],
                            quantization=quantization,
                            all_tracks=gains[clip_id],
                        )
                        pcm = rendered["pcm"]
                        weighted = rendered["weighted"]
                        mixed_q23 = rendered["mixed_q23"]
                        meta = rendered["meta"]
                        self.cache.put(
                            key,
                            meta={"fragment": meta, "upstream": upstream},
                            arrays={
                                "pcm": pcm,
                                "weighted": weighted,
                                "mixed_q23": mixed_q23,
                            },
                        )
                        self.new_keys["mix"].add(key)
                        status = "recompute"
                    event = CacheEvent(
                        stage="mix",
                        clip_id=clip_id,
                        track=None,
                        fragment_index=plan.fragment_index,
                        cache_key=key,
                        status=status,
                    )
                    return {
                        "pcm": pcm,
                        "weighted": weighted,
                        "mixed_q23": mixed_q23,
                        "meta": meta,
                        "event": event,
                    }

                all_tasks.append((f"{clip_id}:{plan.fragment_index:08d}", plan, task))
        all_tasks.sort(key=lambda item: item[0])
        batch_completed = 0
        for batch_start in range(0, len(all_tasks), interval):
            batch = all_tasks[batch_start : batch_start + interval]
            results = self._parallel(
                [((stable_key, plan.fragment_index), function) for stable_key, plan, function in batch]
            )
            for (stable_key, plan, _function), result in zip(batch, results):
                event = result["event"]
                self._record_event(
                    stage=event.stage,
                    clip_id=event.clip_id,
                    track=event.track,
                    fragment_index=event.fragment_index,
                    cache_key=event.cache_key,
                    status=event.status,
                )
                self.completed_fragments.add(
                    f"{event.clip_id}:{event.fragment_index:08d}"
                )
                render_reports.setdefault(event.clip_id, {}).setdefault(
                    "fragments", []
                ).append(result)
                batch_completed += 1
            progress = {
                "fragments_completed": batch_completed,
                "batches_completed": batch_start // interval + 1,
                "snapshot_interval_fragments": interval,
            }
            self._commit(
                "render",
                stage_progress=progress,
                new_keys=self.new_keys["mix"],
                completed_fragments=self.completed_fragments,
            )

        output_reports: dict[str, dict[str, Any]] = {}
        for clip_id, context in sorted(clip_context.items()):
            clip = context["clip"]
            plane = context["plane"]
            clip_start = context["clip_start"]
            clip_end = context["clip_end"]
            fragment_results = render_reports.get(clip_id, {}).get("fragments", [])
            fragment_results.sort(key=lambda item: item["meta"]["fragment_index"])
            pcm = (
                np.concatenate([item["pcm"] for item in fragment_results])
                if fragment_results
                else np.zeros(0, dtype=np.int16)
            )
            expected_frames = clip_end - clip_start
            if pcm.size != expected_frames:
                issue = TrackIssue(
                    clip_id=clip_id,
                    track="<clip>",
                    stage="mix",
                    code=E_MIX_CONFIG,
                    message=(
                        f"rendered {pcm.size} frames but trim range requires {expected_frames}"
                    ),
                )
                self._add_failure(issue)
                continue
            output_path = self.workdir / "mixes" / f"{clip_id}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
            write_wav(
                temp,
                sample_rate=plane.sample_rate,
                channels=1,
                sample_width=2,
                raw_pcm=i16_to_bytes(pcm),
            )
            os.replace(temp, output_path)
            component_matrix, mixed_q23, matrix_tracks = full_component_matrix(
                fragment_results,
                gains[clip_id],
                clip_start=clip_start,
                clip_end=clip_end,
            )
            component_path = self.workdir / "components" / f"{clip_id}.npz"
            component_path.parent.mkdir(parents=True, exist_ok=True)
            component_temp = component_path.with_name(
                component_path.name + f".tmp.{os.getpid()}"
            )
            with component_temp.open("wb") as handle:
                np.savez(
                    handle,
                    weighted=component_matrix,
                    mixed_q23=mixed_q23,
                    track_names=np.asarray(matrix_tracks),
                    clip_id=np.asarray([clip_id]),
                )
            os.replace(component_temp, component_path)
            fragment_reports = []
            for item in fragment_results:
                fragment_reports.append(dict(item["meta"]))
            output_reports[clip_id] = {
                "clip_id": clip_id,
                "status": "rendered",
                "format_version": plane.format_version,
                "source_package": str(clip["source_package"]),
                "source_frame_count": plane.frames,
                "sample_rate": plane.sample_rate,
                "bit_depth": plane.bit_depth,
                "channel_map": plane.channel_map,
                "track_names": plane.track_names,
                "track_frames": plane.track_frames,
                "missing_tracks": plane.missing_tracks,
                "trim_start_sample": clip_start,
                "trim_end_sample": clip_end,
                "output_frames": int(pcm.size),
                "output_wav": str(output_path),
                "output_file_sha256": digest_file(output_path),
                "output_pcm_sha256": int16_vector_sha256(pcm),
                "peak": int(np.max(np.abs(pcm.astype(np.int32)))) if pcm.size else 0,
                "fragments": fragment_reports,
                "weighted_components": {
                    "path": str(component_path),
                    "sha256": digest_file(component_path),
                    "track_order": matrix_tracks,
                    "frames": int(mixed_q23.size),
                    "mixed_q23_sha256": sha256_bytes(mixed_q23.astype("<i8").tobytes()),
                },
                "parse_issues": [issue.as_dict() for issue in plane.issues],
            }
        self._commit(
            "render",
            stage_progress={
                "fragments_completed": len(self.events),
                "outputs": sorted(output_reports),
            },
            new_keys=self.new_keys["mix"],
            completed_fragments=self.completed_fragments,
        )
        self._maybe_stop_or_kill("render")
        return output_reports

    def _config_injection_failures(self) -> tuple[list[TrackIssue], set[str], bool]:
        issues: list[TrackIssue] = []
        blocked_clips: set[str] = set()
        global_blocked = False
        for injection in config_injections(self.injections):
            if injection.kind == "quantization_out_of_range":
                if injection.clip_id:
                    blocked_clips.add(injection.clip_id)
                else:
                    global_blocked = True
                validation_errors = validate_global(
                    {
                        **self.config,
                        "gain_quantization": injection.value,
                    }
                )
                message = (
                    validation_errors[0]["message"]
                    if validation_errors
                    else f"injected {injection.kind}"
                )
            else:
                message = f"injected {injection.kind}"
            issues.append(
                TrackIssue(
                    clip_id=injection.clip_id or "<global>",
                    track=injection.track or "<global>",
                    stage="config",
                    code=E_MIX_CONFIG,
                    message=message,
                    injection_id=injection.id,
                )
            )
        return issues, blocked_clips, global_blocked

    def _injection_summary(self) -> list[dict[str, Any]]:
        summary = []
        for injection in self.injections:
            matches = [
                failure for failure in self.failures if failure.injection_id == injection.id
            ]
            codes = sorted({failure.code for failure in matches})
            summary.append(
                {
                    **injection.as_dict(),
                    "hit": bool(matches),
                    "hit_count": len(matches),
                    "observed_codes": codes,
                    "expected_code_matched": (
                        injection.expected_code in codes
                        if injection.expected_code and codes
                        else None
                    ),
                    "failures": [failure.as_dict() for failure in matches],
                }
            )
        return sorted(summary, key=lambda item: item["id"])

    def execute(self) -> RunResult:
        self.workdir.mkdir(parents=True, exist_ok=True)
        global_errors = validate_global(self.config)
        config_issues, config_blocked_clips, config_global_block = (
            self._config_injection_failures()
        )
        global_mix_blocked = bool(global_errors) or config_global_block
        for error in global_errors:
            self._add_failure(
                TrackIssue(
                    clip_id=error["clip_id"],
                    track=error["track"] or "<global>",
                    stage="config",
                    code=error["code"],
                    message=error["message"],
                )
            )
        for issue in config_issues:
            self._add_failure(issue)
        clips = list(self.config["clips"])
        self.stop.install()
        self._executor = (
            ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="remix")
            if self.workers > 1
            else None
        )
        stopped = False
        try:
            planes, fatal_parse = self._run_parse_stage(clips)
            if self.stop.requested:
                stopped = True
                self._write_progress(stage="parse", stopped=True)
                return RunResult(None, True, 0, self.workdir)
            planes, conversion_fps, conversions = self._run_conversion_stage(
                clips, planes, fatal_parse
            )
            if self.stop.requested:
                stopped = True
                self._write_progress(stage="convert", stopped=True)
                return RunResult(None, True, 0, self.workdir)
            gains, trims, blocked = self._run_gain_stage(
                clips,
                planes,
                conversion_fps,
                fatal_parse,
                global_mix_blocked,
            )
            blocked.update(config_blocked_clips)
            if self.stop.requested:
                stopped = True
                self._write_progress(stage="gain", stopped=True)
                return RunResult(None, True, 0, self.workdir)
            outputs = self._run_render_stage(
                clips,
                planes,
                gains,
                trims,
                blocked,
                global_mix_blocked,
                fatal_parse,
            )
            if self.stop.requested:
                stopped = True
                self._write_progress(stage="render", stopped=True)
                return RunResult(None, True, 0, self.workdir)
            report = self._build_report(
                clips=clips,
                planes=planes,
                gains=gains,
                outputs=outputs,
                conversions=conversions,
                fatal_parse=fatal_parse,
                blocked=blocked,
                global_mix_blocked=global_mix_blocked,
            )
            return RunResult(report, False, 0, self.workdir)
        finally:
            self.stop.restore()
            if self._executor is not None:
                self._executor.shutdown(wait=True)

    def _build_report(
        self,
        *,
        clips: list[dict[str, Any]],
        planes: dict[str, TrackPlane],
        gains: dict[str, dict[str, GainArtifact]],
        outputs: dict[str, dict[str, Any]],
        conversions: list[dict[str, Any]],
        fatal_parse: set[str],
        blocked: set[str],
        global_mix_blocked: bool,
    ) -> dict[str, Any]:
        clip_reports = []
        for clip in sorted(clips, key=lambda item: item["clip_id"]):
            clip_id = str(clip["clip_id"])
            if clip_id in outputs:
                clip_reports.append(outputs[clip_id])
                continue
            plane = planes.get(clip_id)
            failure_codes = sorted(
                {
                    failure.code
                    for failure in self.failures
                    if failure.clip_id == clip_id
                }
            )
            clip_reports.append(
                {
                    "clip_id": clip_id,
                    "status": "failed",
                    "format_version": (
                        plane.format_version if plane else clip["source_format"]
                    ),
                    "source_package": str(clip["source_package"]),
                    "source_frame_count": plane.frames if plane else None,
                    "track_names": plane.track_names if plane else sorted(clip["tracks"]),
                    "output_frames": 0,
                    "output_wav": None,
                    "output_pcm_sha256": None,
                    "failure_codes": failure_codes,
                    "fragments": [],
                }
            )
        failures = sorted(self.failures, key=lambda item: item.stable_key)
        events = sorted(self.events, key=lambda item: item.stable_key)
        reuse = [event for event in events if event.status == "reuse"]
        recompute = [event for event in events if event.status != "reuse"]
        loss_items = []
        for conversion in conversions:
            for loss in conversion.get("losses", []):
                loss_items.append(
                    {
                        "conversion_id": conversion["source_clip_id"]
                        + ":"
                        + conversion["source_format_version"]
                        + "->"
                        + conversion["target_format_version"],
                        **loss,
                    }
                )
        loss_items.sort(key=lambda item: (item["conversion_id"], item["field"]))
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "complete_with_failures"
                if failures
                else "complete"
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "samples_root": str(self.samples_root),
            "workdir": str(self.workdir),
            "config_hash": self.config_hash,
            "sample_manifest_seed": self.sample_manifest.get("seed"),
            "format_versions": self._format_versions(),
            "thread_pool_size": self.workers,
            "snapshot_interval_clips": int(self.config["snapshot_interval_clips"]),
            "snapshot_interval_fragments": int(
                self.config["snapshot_interval_clips"]
            ),
            "clips": clip_reports,
            "failures": [failure.as_dict() for failure in failures],
            "failure_order": [
                failure.stable_key_text
                for failure in failures
            ],
            "cache_reuse": [event.as_dict() for event in reuse],
            "cache_recompute": [event.as_dict() for event in recompute],
            "cache_reuse_order": [
                event.stable_key_text for event in reuse
            ],
            "cache_recompute_order": [
                event.stable_key_text for event in recompute
            ],
            "cache_event_order": [
                event.stable_key_text
                for event in events
            ],
            "cache_summary": {
                "reuse_count": len(reuse),
                "recompute_count": len(recompute),
                "events": len(events),
                "artifact_commits_this_run": sum(len(keys) for keys in self.new_keys.values()),
            },
            "conversions": conversions,
            "conversion_degradation_losses": loss_items,
            "injection_summary": self._injection_summary(),
            "recovery": {
                **self.snapshots.recovery,
                "snapshot_on_start": self._start_snapshot_version,
            },
            "snapshots": {
                **self.snapshots.status(),
                "completed_fragments": sorted(self.completed_fragments),
            },
            "global_mix_blocked": global_mix_blocked,
            "blocked_clips": sorted(blocked),
            "fatal_parse_clips": sorted(fatal_parse),
        }
        report["render_signature_sha256"] = digest_json(render_signature(report, self.workdir))
        return report


def _gain_cache_key(artifact: GainArtifact) -> str:
    """The gain artifact carries its identity in metadata; derive a stable token."""
    return digest_json(
        {
            "track": artifact.track,
            "gain_q": artifact.gain_q,
            "mute": artifact.mute,
            "weight_sha256": sha256_bytes(
                np.asarray(artifact.weights, dtype="<i8").tobytes(order="C")
            ),
        }
    )
