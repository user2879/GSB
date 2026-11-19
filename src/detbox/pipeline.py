"""End-to-end staged pipeline with incremental cache reuse and recovery."""
from __future__ import annotations

from pathlib import Path

from . import fingerprint as fp
from .config import DetectionParams, MatchConfig, Injection
from .detect import detect_frame, load_frame_image
from .errors import DetboxError
from .exchange import (
    parse_v1_text,
    parse_v2_document,
    serialize_v1,
    serialize_v2_document,
    v2_to_v1,
)
from .injection import (
    apply_detect_injections,
    apply_match_injections,
    apply_serialize_injections,
    injections_for,
)
from .match import match_sequence
from .models import (
    FORMAT_V1,
    FORMAT_V2,
    FailureRecord,
    FrameCandidates,
)
from .store import SnapshotStore, SNAPSHOT_VERSION, STAGE_DETECT, STAGE_MATCH, STAGE_SERIALIZE
from .util import (
    atomic_write_json,
    atomic_write_text,
    content_hash,
    read_json,
)


class Pipeline:
    def __init__(
        self,
        dataset_root: str | Path,
        workdir: str | Path,
        fmt: str,
        detection_params: DetectionParams,
        match_config: MatchConfig,
        classes,
        width: int,
        height: int,
        injections: list[Injection] | None = None,
        overrides: dict | None = None,
        strict_resume: bool = False,
    ):
        self.dataset_root = Path(dataset_root).resolve()
        self.store = SnapshotStore(workdir)
        self.fmt = fmt
        self.params = detection_params
        self.match_config = match_config
        self.classes = classes
        self.width = width
        self.height = height
        self.injections = injections or []
        self.overrides = overrides or {}
        self.strict_resume = strict_resume
        self.failures: list[FailureRecord] = []
        self.hits: list[dict] = []
        self.reused: list[str] = []
        self.invalidated: list[str] = []
        self.recomputed: list[str] = []
        self.evictions: list[dict] = []
        self.signal_state = {"shutdown_requested": False}
        SnapshotStore.install_signal_handlers(self.signal_state)

        self.detect_fp = fp.detection_params_fingerprint(self.params.to_dict())
        self.match_fp = fp.match_config_fingerprint(self.match_config.to_dict())
        self.format_fp = fp.format_version_fingerprint(self.fmt)
        self.global_hash = fp.global_config_hash(
            str(self.dataset_root),
            self.params.to_dict(),
            self.match_config.to_dict(),
            self.classes.to_dict(),
            width,
            height,
        )

    # --------------------------------------------------------- enumeration
    def enumerate_frames(self) -> list[tuple[str, int, Path]]:
        result = []
        image_root = self.dataset_root / "images"
        for seq_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
            for image_path in sorted(seq_dir.glob("frame_*.png")):
                frame_no = int(image_path.stem.split("_")[1])
                result.append((seq_dir.name, frame_no, image_path))
        return result

    def _record_failure(
        self,
        sequence: str,
        frame: int,
        stage: str,
        exc: Exception,
        injection_id: str | None = None,
    ) -> None:
        code = getattr(exc, "code", "E_STAGE_FAILED")
        self.failures.append(
            FailureRecord(
                sequence=sequence,
                frame=frame,
                stage=stage,
                exception_type=type(exc).__name__,
                code=code,
                message=str(exc),
                injection_id=injection_id,
            )
        )

    def _hit(self, injection, code: str) -> None:
        self.hits.append(
            {
                "injection_id": injection.injection_id,
                "sequence": injection.sequence,
                "frame": injection.frame,
                "stage": injection.stage,
                "fault_type": injection.fault_type,
                "code": code,
            }
        )

    # --------------------------------------------------------- detect stage
    def run_detect(self) -> dict[str, dict[int, FrameCandidates]]:
        self.store.verify_resume(
            SNAPSHOT_VERSION, self.global_hash, strict=self.strict_resume
        )
        frames: dict[str, dict[int, FrameCandidates]] = {}
        cache_index: dict[str, str] = {}

        for sequence, frame_no, image_path in self.enumerate_frames():
            if self.signal_state["shutdown_requested"]:
                break
            stage_inj = injections_for(self.injections, sequence, frame_no, STAGE_DETECT)
            override = self.overrides.get("frames", {}).get(f"{sequence}:{frame_no}")
            frame_content_fp = fp.frame_content_fingerprint(image_path)
            key = fp.detect_cache_key(
                frame_content_fp,
                self.detect_fp,
                self.match_fp,
            )
            cache_index[f"{sequence}:{frame_no}"] = key
            try:
                if stage_inj:
                    apply_detect_injections(stage_inj, image_path, self.params)
                cached = self.store.cache_get(key)
                if cached is not None and not override:
                    frame = FrameCandidates.from_dict(cached["frame"])
                    self.reused.append(f"{STAGE_DETECT}:{sequence}:{frame_no}")
                else:
                    if cached is None:
                        self.invalidated.append(f"{STAGE_DETECT}:{sequence}:{frame_no}")
                    frame = self._detect_one(image_path, sequence, frame_no, override)
                    payload = {
                        "cache_key": key,
                        "frame": frame.to_dict(),
                        "frame_fingerprint": fp.frame_candidate_fingerprint(frame.to_dict()),
                    }
                    self.store.cache_put(key, payload)
                    self.recomputed.append(f"{STAGE_DETECT}:{sequence}:{frame_no}")
                frames.setdefault(sequence, {})[frame_no] = frame
            except DetboxError as exc:
                injection_id = stage_inj[0].injection_id if stage_inj else None
                if stage_inj:
                    self._hit(stage_inj[0], exc.code)
                self._record_failure(sequence, frame_no, STAGE_DETECT, exc, injection_id)

        def commit(staging: Path) -> None:
            index = {
                "format": self.fmt,
                "entries": cache_index,
                "failures": [f.to_dict() for f in self.failures if f.stage == STAGE_DETECT],
            }
            atomic_write_json(staging / "detect_index.json", index)

        self.store.commit_stage(STAGE_DETECT, commit)
        return frames

    def _detect_one(self, image_path, sequence, frame_no, override) -> FrameCandidates:
        image = load_frame_image(image_path)
        candidates, events = detect_frame(
            image, self.params, sequence, frame_no
        )
        for event in events:
            self.evictions.append(event.to_dict())
        frame = FrameCandidates(
            sequence=sequence,
            frame=frame_no,
            width=self.width,
            height=self.height,
            candidates=candidates,
            source_format=FORMAT_V2,
        )
        if override:
            frame = self._apply_candidate_override(frame, override)
        frame = self._to_exchange_format(frame)
        return frame

    def _apply_candidate_override(self, frame: FrameCandidates, override: dict) -> FrameCandidates:
        # Single-frame candidate replacement entering through the exchange
        # parser, so format faults surface as E_CAND_FORMAT.
        if override.get("format") == FORMAT_V1:
            parsed, bad = parse_v1_text(
                override["text"], self.classes, self.width, self.height
            )
            if bad:
                raise DetboxError(bad[0]["message"], code=bad[0]["code"])
            frame.candidates = parsed[0].candidates if parsed else []
        else:
            document = {"sequences": [{"sequence": frame.sequence,
                                       "frames": [{"frame": frame.frame,
                                                   "candidates": override.get("candidates", [])}]}]}
            parsed, bad = parse_v2_document(
                document, self.classes, self.width, self.height
            )
            if bad:
                raise DetboxError(bad[0]["message"], code=bad[0]["code"])
            frame.candidates = parsed[0].candidates if parsed else []
        return frame

    def _to_exchange_format(self, frame: FrameCandidates) -> FrameCandidates:
        if self.fmt == FORMAT_V2:
            return frame
        downgraded, sidecar = v2_to_v1([frame], self.classes)
        sidecar_path = self.store.root / "sidecar" / f"{frame.sequence}_{frame.frame:04d}.json"
        atomic_write_json(sidecar_path, sidecar)
        return downgraded[0]

    def _current_sidecar_fp(self) -> str:
        sidecar_dir = self.store.root / "sidecar"
        if not sidecar_dir.exists():
            return fp.sidecar_fingerprint(None)
        payload = {}
        for path in sorted(sidecar_dir.glob("*.json")):
            payload[path.name] = read_json(path)
        return fp.sidecar_fingerprint(payload if payload else None)

    # ---------------------------------------------------------- match stage
    def run_match(
        self, frames: dict[str, dict[int, FrameCandidates]]
    ) -> dict[str, list]:
        tracks_by_seq: dict[str, list] = {}
        track_index: dict[str, str] = {}
        frame_fps = {
            seq: {
                fno: fp.frame_candidate_fingerprint(frame.to_dict())
                for fno, frame in seq_frames.items()
            }
            for seq, seq_frames in frames.items()
        }
        sidecar_fp = self._current_sidecar_fp()

        for sequence in sorted(frames):
            seq_frames = [frames[sequence][fno] for fno in sorted(frames[sequence])]
            try:
                stage_inj = [
                    item
                    for fno in frames[sequence]
                    for item in injections_for(self.injections, sequence, fno, STAGE_MATCH)
                ]
                if stage_inj:
                    hit, exc = apply_match_injections(stage_inj, seq_frames)
                    if exc is not None:
                        self.hits.append(hit.to_dict())
                        raise exc
                ordered_fps = [frame_fps[sequence][f.frame] for f in seq_frames]
                key = fp.track_cache_key(
                    sequence,
                    ordered_fps,
                    self.detect_fp,
                    self.match_fp,
                    self.format_fp,
                    sidecar_fp,
                )
                track_index[sequence] = key
                cached = self.store.cache_get(key)
                if cached is not None:
                    from .models import Track

                    tracks = [Track.from_dict(item) for item in cached["tracks"]]
                    self.reused.append(f"{STAGE_MATCH}:{sequence}")
                else:
                    self.invalidated.append(f"{STAGE_MATCH}:{sequence}")
                    tracks, events = match_sequence(
                        sequence,
                        seq_frames,
                        self.match_config,
                        self.params.pool_capacity,
                    )
                    for event in events:
                        self.evictions.append(event.to_dict())
                    self.store.cache_put(
                        key,
                        {
                            "cache_key": key,
                            "tracks": [track.to_dict() for track in tracks],
                        },
                    )
                    self.recomputed.append(f"{STAGE_MATCH}:{sequence}")
                tracks_by_seq[sequence] = tracks
            except DetboxError as exc:
                injection_id = stage_inj[0].injection_id if stage_inj else None
                frame_no = stage_inj[0].frame if stage_inj else -1
                self._record_failure(
                    sequence, frame_no, STAGE_MATCH, exc, injection_id
                )

        def commit(staging: Path) -> None:
            atomic_write_json(
                staging / "match_index.json",
                {
                    "entries": track_index,
                    "failures": [f.to_dict() for f in self.failures if f.stage == STAGE_MATCH],
                },
            )

        self.store.commit_stage(STAGE_MATCH, commit)
        return tracks_by_seq

    # ------------------------------------------------------ serialize stage
    def run_serialize(
        self,
        frames: dict[str, dict[int, FrameCandidates]],
        tracks_by_seq: dict[str, list],
    ) -> dict:
        all_frames = [
            frame
            for sequence in sorted(frames)
            for fno in sorted(frames[sequence])
            for frame in [frames[sequence][fno]]
        ]
        track_fps = [
            content_hash({"kind": "track", "track": track.to_dict()})
            for tracks in tracks_by_seq.values()
            for track in tracks
        ]
        sidecar_fp = self._current_sidecar_fp()
        ser_key = fp.serialization_cache_key(
            track_fps, self.detect_fp, self.match_fp, self.format_fp, sidecar_fp
        )
        cached = self.store.cache_get(ser_key)
        if cached is not None and self.store.stage_exists(STAGE_SERIALIZE):
            self.reused.append(STAGE_SERIALIZE)
        else:
            self.invalidated.append(STAGE_SERIALIZE)
            self._serialize_isolated(all_frames, ser_key)
            self.store.cache_put(
                ser_key,
                {"cache_key": ser_key, "format": self.fmt, "artifact": "serialized"},
            )
            self.recomputed.append(STAGE_SERIALIZE)

        def commit(staging: Path) -> None:
            atomic_write_json(
                staging / "serialize_index.json",
                {"cache_key": ser_key, "format": self.fmt},
            )

        self.store.commit_stage(STAGE_SERIALIZE, commit)
        output_path = self.store.stage_path(STAGE_SERIALIZE) / (
            "candidates.txt" if self.fmt == FORMAT_V1 else "candidates.json"
        )
        return {"cache_key": ser_key, "output": str(output_path), "reused": cached is not None}

    def _serialize_isolated(self, all_frames: list[FrameCandidates], ser_key: str) -> None:
        # Per-frame injection isolation: one faulty frame is recorded and the
        # remaining frames still serialize; failed frames never enter output.
        good_frames: list[FrameCandidates] = []
        for frame in all_frames:
            stage_inj = injections_for(
                self.injections, frame.sequence, frame.frame, STAGE_SERIALIZE
            )
            frame_failed = False
            for injection in stage_inj:
                if not frame.candidates:
                    exc = DetboxError(
                        "no candidate available for injected serialization fault",
                        code="E_CAND_FORMAT",
                    )
                else:
                    hit, exc = apply_serialize_injections(
                        [injection], frame.candidates[0], frame.sequence, frame.frame
                    )
                    self.hits.append(hit.to_dict())
                self._record_failure(
                    frame.sequence,
                    frame.frame,
                    STAGE_SERIALIZE,
                    exc,
                    injection.injection_id,
                )
                frame_failed = True
            if not frame_failed:
                good_frames.append(frame)

        def build(staging: Path) -> None:
            if self.fmt == FORMAT_V1:
                atomic_write_text(staging / "candidates.txt", serialize_v1(good_frames))
            else:
                atomic_write_json(
                    staging / "candidates.json",
                    serialize_v2_document(good_frames, self.classes),
                )
            atomic_write_json(
                staging / "serialize_index.json",
                {"cache_key": ser_key, "format": self.fmt},
            )

        self.store.commit_stage(STAGE_SERIALIZE, build)

    # -------------------------------------------------------------- run all
    def run(self) -> dict:
        frames = self.run_detect()
        tracks_by_seq = self.run_match(frames)
        serialize_info = self.run_serialize(frames, tracks_by_seq)

        manifest = {
            "snapshot_version": SNAPSHOT_VERSION,
            "global_version": SNAPSHOT_VERSION,
            "config_hash": self.global_hash,
            "detection_params_fingerprint": self.detect_fp,
            "match_config_fingerprint": self.match_fp,
            "format_version_fingerprint": self.format_fp,
            "sidecar_fingerprint": self._current_sidecar_fp(),
            "stage_versions": {
                STAGE_DETECT: content_hash(
                    read_json(self.store.stage_path(STAGE_DETECT) / "detect_index.json")
                ),
                STAGE_MATCH: content_hash(
                    read_json(self.store.stage_path(STAGE_MATCH) / "match_index.json")
                ),
                STAGE_SERIALIZE: content_hash(
                    read_json(
                        self.store.stage_path(STAGE_SERIALIZE) / "serialize_index.json"
                    )
                ),
            },
            "format": self.fmt,
            "shutdown_requested": self.signal_state["shutdown_requested"],
        }
        self.store.write_manifest(manifest)
        return {
            "manifest": manifest,
            "frames": frames,
            "tracks": tracks_by_seq,
            "serialize": serialize_info,
            "failures": [item.to_dict() for item in self.failures],
            "injection_hits": self.hits,
            "reused": list(self.reused),
            "invalidated": list(self.invalidated),
            "recomputed": list(self.recomputed),
            "evictions": self.evictions,
            "exit_code": 0,
        }
