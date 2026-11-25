"""Workload specification loading and strict validation.

The specification is intentionally small, integer-only, and free of ambient
defaults that could make two machines disagree.  Validation reports the first
field in a documented traversal order; the source line comes from the
position-aware JSON parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_sha256
from .errors import DiagnosticError
from .jsonpos import PositionedJSON, parse_positioned_json


SUPPORTED_SAMPLE_RATE = 16_000
SUPPORTED_CHANNELS = 1
SUPPORTED_SAMPLE_WIDTH_BITS = 16


def _is_int(value: Any) -> bool:
    return type(value) is int


def _fmt_path(path: tuple[Any, ...]) -> str:
    out = ""
    for part in path:
        if isinstance(part, int):
            out += f"[{part}]"
        elif out:
            out += f".{part}"
        else:
            out = str(part)
    return out or "$"


@dataclass(frozen=True)
class QuantizationSpec:
    scheme: str
    fractional_bits: int

    def to_dict(self) -> dict[str, Any]:
        return {"scheme": self.scheme, "fractional_bits": self.fractional_bits}


@dataclass(frozen=True)
class AudioSpec:
    duration_ms: int
    sample_rate: int
    channels: int
    sample_width_bits: int
    speaker_count: int
    pause_min_ms: int
    pause_max_ms: int
    overlap_bps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bits": self.sample_width_bits,
            "speaker_count": self.speaker_count,
            "pause_ms": [self.pause_min_ms, self.pause_max_ms],
            "overlap_bps": self.overlap_bps,
        }


@dataclass(frozen=True)
class WindowSpec:
    window_ms: int
    hop_ms: int
    overlap_bps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_ms": self.window_ms,
            "hop_ms": self.hop_ms,
            "overlap_bps": self.overlap_bps,
        }


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    port: int
    model_dim: int
    hidden_dim: int
    seed: int
    parameter_version: int
    quantization: QuantizationSpec
    max_concurrent: int
    max_worker_leases: int
    lease_ms: int
    source_index: int
    source_line: int

    def to_dict(self, include_source: bool = False) -> dict[str, Any]:
        result = {
            "worker_id": self.worker_id,
            "host": self.host,
            "port": self.port,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "parameter_version": self.parameter_version,
            "quantization": self.quantization.to_dict(),
            "max_concurrent": self.max_concurrent,
            "max_worker_leases": self.max_worker_leases,
            "lease_ms": self.lease_ms,
        }
        if include_source:
            result["_source_index"] = self.source_index
            result["_source_line"] = self.source_line
        return result


@dataclass(frozen=True)
class RequestSpec:
    batch_id: str
    worker_id: str
    window_ids: tuple[str, ...]
    request_index: int
    source_line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "window_ids": list(self.window_ids),
        }


@dataclass(frozen=True)
class LimitsSpec:
    global_batch_leases: int
    metric_scale: int
    batch_score_error_bound_units: int
    worker_watermark_error_bound_units: int
    handshake_timeout_ms: int
    io_timeout_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_batch_leases": self.global_batch_leases,
            "metric_scale": self.metric_scale,
            "batch_score_error_bound_units": self.batch_score_error_bound_units,
            "worker_watermark_error_bound_units": self.worker_watermark_error_bound_units,
            "handshake_timeout_ms": self.handshake_timeout_ms,
            "io_timeout_ms": self.io_timeout_ms,
        }


@dataclass(frozen=True)
class WorkloadSpec:
    spec_version: int
    session_version: int
    seed: int
    audio: AudioSpec
    window: WindowSpec
    batch_size: int
    quantization: QuantizationSpec
    workers: tuple[WorkerSpec, ...]
    requests: tuple[RequestSpec, ...]
    limits: LimitsSpec
    source_name: str = "<memory>"
    source_text: str = field(default="", repr=False, compare=False)
    _positions: PositionedJSON | None = field(default=None, repr=False, compare=False)

    def line(self, path: tuple[Any, ...] | str) -> int:
        if isinstance(path, str):
            path = tuple(path.split("."))
        if self._positions is None:
            return 1
        current = tuple(path)
        while current:
            if current in self._positions.locations:
                return self._positions.locations[current]
            current = current[:-1]
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "session_version": self.session_version,
            "seed": self.seed,
            "audio": self.audio.to_dict(),
            "window": self.window.to_dict(),
            "batch": {"batch_size": self.batch_size},
            "quantization": self.quantization.to_dict(),
            "workers": [worker.to_dict() for worker in self.workers],
            "requests": [request.to_dict() for request in self.requests],
            "limits": self.limits.to_dict(),
        }

    @property
    def spec_digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def window_samples(self) -> int:
        return self.window.window_ms * self.audio.sample_rate // 1000

    @property
    def hop_samples(self) -> int:
        return self.window.hop_ms * self.audio.sample_rate // 1000

    @property
    def total_samples(self) -> int:
        return self.audio.duration_ms * self.audio.sample_rate // 1000

    @property
    def window_starts_samples(self) -> tuple[int, ...]:
        final = self.total_samples - self.window_samples
        return tuple(range(0, final + 1, self.hop_samples))

    @property
    def window_ids(self) -> tuple[str, ...]:
        return tuple(f"W{index + 1:04d}" for index in range(len(self.window_starts_samples)))

    def worker_by_id(self, worker_id: str) -> WorkerSpec:
        for worker in self.workers:
            if worker.worker_id == worker_id:
                return worker
        raise DiagnosticError(
            "E_SPEC",
            f"unknown worker_id {worker_id!r}",
            field="worker_id",
            line=1,
        )

    def request_by_batch(self, batch_id: str) -> RequestSpec:
        for request in self.requests:
            if request.batch_id == batch_id:
                return request
        raise DiagnosticError(
            "E_SPEC",
            f"unknown batch_id {batch_id!r}",
            field="batch_id",
            line=1,
        )

    def request_by_index(self, request_index: int) -> RequestSpec:
        return self.requests[request_index]

    def parameter_identity(self, worker: WorkerSpec) -> dict[str, Any]:
        return {
            "worker_id": worker.worker_id,
            "parameter_version": worker.parameter_version,
            "seed": worker.seed,
            "model_dim": worker.model_dim,
            "hidden_dim": worker.hidden_dim,
            "quantization": worker.quantization.to_dict(),
            "speaker_count": self.audio.speaker_count,
        }

    def batch_order_key(self, batch_id: str) -> tuple[str, int]:
        request = self.request_by_batch(batch_id)
        return (request.worker_id, request.request_index)


class _Validator:
    def __init__(self, positioned: PositionedJSON, source_name: str, source_text: str):
        self.pos = positioned
        self.data = positioned.value
        self.source_name = source_name
        self.source_text = source_text

    def line(self, path: tuple[Any, ...]) -> int:
        current = tuple(path)
        while current:
            if current in self.pos.locations:
                return self.pos.locations[current]
            current = current[:-1]
        return 1

    def fail(self, code: str, path: tuple[Any, ...], message: str, **details: Any) -> None:
        raise DiagnosticError(
            code,
            message,
            field=_fmt_path(path),
            line=self.line(path),
            details={"source": self.source_name, **details},
        )

    def expect_object(self, value: Any, path: tuple[Any, ...]) -> dict[str, Any]:
        if not isinstance(value, dict):
            self.fail("E_SPEC", path, "expected a JSON object")
        return value

    def integer(
        self,
        obj: dict[str, Any],
        key: str,
        path: tuple[Any, ...],
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        child = path + (key,)
        if key not in obj:
            self.fail("E_SPEC", child, "required integer field is missing")
        value = obj[key]
        if not _is_int(value):
            self.fail("E_SPEC", child, "expected an integer (booleans and floats are rejected)")
        if minimum is not None and value < minimum:
            self.fail("E_SPEC", child, f"must be >= {minimum}", value=value)
        if maximum is not None and value > maximum:
            self.fail("E_SPEC", child, f"must be <= {maximum}", value=value)
        return value

    def string(self, obj: dict[str, Any], key: str, path: tuple[Any, ...]) -> str:
        child = path + (key,)
        if key not in obj:
            self.fail("E_SPEC", child, "required string field is missing")
        value = obj[key]
        if not isinstance(value, str) or not value.strip():
            self.fail("E_SPEC", child, "expected a non-empty string")
        if value != value.strip():
            self.fail("E_SPEC", child, "leading/trailing whitespace is not allowed")
        return value

    def quantization(self, obj: dict[str, Any], path: tuple[Any, ...]) -> QuantizationSpec:
        quant_obj = self.expect_object(obj, path)
        scheme = self.string(quant_obj, "scheme", path)
        fractional_bits = self.integer(quant_obj, "fractional_bits", path, minimum=1, maximum=30)
        return QuantizationSpec(scheme=scheme, fractional_bits=fractional_bits)

    def validate(self) -> WorkloadSpec:
        root = self.expect_object(self.data, ())
        spec_version = self.integer(root, "spec_version", (), minimum=1) if "spec_version" in root else 1
        if spec_version != 1:
            self.fail("E_SPEC", ("spec_version",), "unsupported spec_version", value=spec_version)
        session_version = self.integer(root, "session_version", (), minimum=1) if "session_version" in root else 1
        seed = self.integer(root, "seed", (), minimum=0, maximum=2**63 - 1)

        audio_obj = self.expect_object(root.get("audio"), ("audio",))
        duration_ms = self.integer(audio_obj, "duration_ms", ("audio",), minimum=1, maximum=600_000)
        sample_rate = self.integer(audio_obj, "sample_rate", ("audio",))
        if sample_rate != SUPPORTED_SAMPLE_RATE:
            self.fail("E_SPEC", ("audio", "sample_rate"), "sample_rate must be 16000")
        channels = self.integer(audio_obj, "channels", ("audio",))
        if channels != SUPPORTED_CHANNELS:
            self.fail("E_SPEC", ("audio", "channels"), "channels must be 1")
        sample_width_bits = self.integer(audio_obj, "sample_width_bits", ("audio",))
        if sample_width_bits != SUPPORTED_SAMPLE_WIDTH_BITS:
            self.fail("E_SPEC", ("audio", "sample_width_bits"), "sample_width_bits must be 16")
        speaker_count = (
            self.integer(audio_obj, "speaker_count", ("audio",), minimum=2, maximum=3)
            if "speaker_count" in audio_obj
            else 3
        )
        pause = audio_obj.get("pause_ms")
        if pause is None:
            pause = [70, 120]
        if not isinstance(pause, list) or len(pause) != 2:
            self.fail("E_SPEC", ("audio", "pause_ms"), "pause_ms must be [minimum, maximum]")
        pause_min = self.integer({"minimum": pause[0]}, "minimum", ("audio", "pause_ms"), minimum=0)
        pause_max = self.integer({"maximum": pause[1]}, "maximum", ("audio", "pause_ms"), minimum=0)
        if pause_min > pause_max:
            self.fail("E_SPEC", ("audio", "pause_ms"), "pause minimum must not exceed maximum")
        audio_overlap_bps = (
            self.integer(audio_obj, "overlap_bps", ("audio",), minimum=0, maximum=5000)
            if "overlap_bps" in audio_obj
            else 1200
        )

        window_obj = self.expect_object(root.get("window"), ("window",))
        window_ms = self.integer(window_obj, "window_ms", ("window",), minimum=1)
        hop_ms = self.integer(window_obj, "hop_ms", ("window",), minimum=1)
        if hop_ms >= window_ms:
            self.fail("E_SPEC", ("window", "hop_ms"), "hop_ms must be strictly smaller than window_ms")
        if (window_ms * sample_rate) % 1000 or (hop_ms * sample_rate) % 1000:
            self.fail("E_SPEC", ("window", "hop_ms"), "window/hop milliseconds must map to whole samples")
        approximate_overlap = (window_ms - hop_ms) * 10_000 // window_ms
        overlap_bps = (
            self.integer(window_obj, "overlap_bps", ("window",), minimum=0, maximum=9999)
            if "overlap_bps" in window_obj
            else approximate_overlap
        )
        if overlap_bps != approximate_overlap:
            self.fail(
                "E_SPEC",
                ("window", "overlap_bps"),
                "overlap_bps is inconsistent with window_ms/hop_ms",
                expected=approximate_overlap,
                value=overlap_bps,
            )
        total_samples = duration_ms * sample_rate // 1000
        window_samples = window_ms * sample_rate // 1000
        hop_samples = hop_ms * sample_rate // 1000
        if total_samples < window_samples:
            self.fail("E_SPEC", ("audio", "duration_ms"), "duration is shorter than one acoustic window")
        window_count = (total_samples - window_samples) // hop_samples + 1
        if window_count > 100_000:
            self.fail("E_SPEC", ("window", "window_ms"), "window count exceeds supported maximum")

        batch_obj = self.expect_object(root.get("batch"), ("batch",))
        batch_size = self.integer(batch_obj, "batch_size", ("batch",), minimum=1, maximum=10_000)
        quantization = self.quantization(self.expect_object(root.get("quantization"), ("quantization",)), ("quantization",))

        workers_obj = root.get("workers")
        if not isinstance(workers_obj, list) or not workers_obj:
            self.fail("E_SPEC", ("workers",), "workers must be a non-empty array")
        if len(workers_obj) > 64:
            self.fail("E_SPEC", ("workers",), "at most 64 workers are supported")
        workers: list[WorkerSpec] = []
        worker_ids: set[str] = set()
        ports: dict[int, str] = {}
        for index, raw_worker in enumerate(workers_obj):
            path = ("workers", index)
            worker_obj = self.expect_object(raw_worker, path)
            worker_id = self.string(worker_obj, "worker_id", path)
            if worker_id in worker_ids:
                self.fail("E_SPEC", path + ("worker_id",), "worker_id must be globally unique", value=worker_id)
            worker_ids.add(worker_id)
            host = self.string(worker_obj, "host", path)
            if host != "127.0.0.1":
                self.fail("E_SPEC", path + ("host",), "host must be 127.0.0.1 (loopback only)")
            port = self.integer(worker_obj, "port", path, minimum=1024, maximum=65535)
            if port in ports:
                self.fail("E_SPEC", path + ("port",), f"port is already used by worker {ports[port]!r}", value=port)
            ports[port] = worker_id
            model_dim = self.integer(worker_obj, "model_dim", path, minimum=1, maximum=4096)
            hidden_dim = self.integer(worker_obj, "hidden_dim", path, minimum=1, maximum=4096)
            worker_seed = self.integer(worker_obj, "seed", path, minimum=0, maximum=2**63 - 1)
            parameter_version = self.integer(worker_obj, "parameter_version", path, minimum=1)
            if "quantization" in worker_obj:
                worker_quantization = self.quantization(
                    self.expect_object(worker_obj["quantization"], path + ("quantization",)),
                    path + ("quantization",),
                )
            else:
                worker_quantization = quantization
            max_concurrent = self.integer(worker_obj, "max_concurrent", path, minimum=1, maximum=64)
            max_worker_leases = self.integer(worker_obj, "max_worker_leases", path, minimum=1, maximum=64)
            if max_worker_leases < max_concurrent:
                self.fail(
                    "E_SPEC",
                    path + ("max_worker_leases",),
                    "max_worker_leases must be at least max_concurrent",
                )
            lease_ms = self.integer(worker_obj, "lease_ms", path, minimum=1, maximum=3_600_000)
            workers.append(
                WorkerSpec(
                    worker_id=worker_id,
                    host=host,
                    port=port,
                    model_dim=model_dim,
                    hidden_dim=hidden_dim,
                    seed=worker_seed,
                    parameter_version=parameter_version,
                    quantization=worker_quantization,
                    max_concurrent=max_concurrent,
                    max_worker_leases=max_worker_leases,
                    lease_ms=lease_ms,
                    source_index=index,
                    source_line=self.line(path),
                )
            )

        limits_obj = self.expect_object(root.get("limits", {}), ("limits",))
        defaults = {
            "global_batch_leases": 4,
            "metric_scale": 10_000,
            "batch_score_error_bound_units": 1,
            "worker_watermark_error_bound_units": 1,
            "handshake_timeout_ms": 3_000,
            "io_timeout_ms": 5_000,
        }
        limit_values: dict[str, int] = {}
        for key, default in defaults.items():
            source_obj = limits_obj if key in limits_obj else {"_default": default}
            source_key = key if key in limits_obj else "_default"
            minimum = 1
            maximum = 3_600_000 if key.endswith("_ms") else 1_000_000
            limit_values[key] = self.integer(source_obj, source_key, ("limits",), minimum=minimum, maximum=maximum)
            if source_key == "_default":
                limit_values[key] = default
        limits = LimitsSpec(**limit_values)

        expected_window_ids = tuple(f"W{index + 1:04d}" for index in range(window_count))
        known_window_ids = set(expected_window_ids)
        requests_obj = root.get("requests")
        if not isinstance(requests_obj, list) or not requests_obj:
            self.fail("E_SPEC", ("requests",), "requests must be a non-empty array")
        requests: list[RequestSpec] = []
        batch_ids: set[str] = set()
        for index, raw_request in enumerate(requests_obj):
            path = ("requests", index)
            request_obj = self.expect_object(raw_request, path)
            batch_id = self.string(request_obj, "batch_id", path)
            if batch_id in batch_ids:
                self.fail("E_SPEC", path + ("batch_id",), "batch_id must be globally unique", value=batch_id)
            batch_ids.add(batch_id)
            worker_id = self.string(request_obj, "worker_id", path)
            if worker_id not in worker_ids:
                self.fail("E_SPEC", path + ("worker_id",), "request references an unknown worker", value=worker_id)
            window_ids_raw = request_obj.get("window_ids")
            if not isinstance(window_ids_raw, list) or not window_ids_raw:
                self.fail("E_SPEC", path + ("window_ids",), "window_ids must be a non-empty array")
            if len(window_ids_raw) > batch_size:
                self.fail(
                    "E_SPEC",
                    path + ("window_ids",),
                    "request contains more windows than batch_size",
                    batch_size=batch_size,
                    count=len(window_ids_raw),
                )
            seen_in_batch: set[str] = set()
            window_ids: list[str] = []
            for window_index, window_id in enumerate(window_ids_raw):
                window_path = path + ("window_ids", window_index)
                if not isinstance(window_id, str) or not window_id:
                    self.fail("E_SPEC", window_path, "window id must be a non-empty string")
                if window_id in seen_in_batch:
                    self.fail("E_SPEC", window_path, "duplicate window id inside a batch request", value=window_id)
                if window_id not in known_window_ids:
                    self.fail("E_SPEC", window_path, "window id is outside the deterministic window range", value=window_id)
                seen_in_batch.add(window_id)
                window_ids.append(window_id)
            requests.append(
                RequestSpec(
                    batch_id=batch_id,
                    worker_id=worker_id,
                    window_ids=tuple(window_ids),
                    request_index=index,
                    source_line=self.line(path),
                )
            )

        spec = WorkloadSpec(
            spec_version=spec_version,
            session_version=session_version,
            seed=seed,
            audio=AudioSpec(
                duration_ms=duration_ms,
                sample_rate=sample_rate,
                channels=channels,
                sample_width_bits=sample_width_bits,
                speaker_count=speaker_count,
                pause_min_ms=pause_min,
                pause_max_ms=pause_max,
                overlap_bps=audio_overlap_bps,
            ),
            window=WindowSpec(window_ms=window_ms, hop_ms=hop_ms, overlap_bps=overlap_bps),
            batch_size=batch_size,
            quantization=quantization,
            workers=tuple(workers),
            requests=tuple(requests),
            limits=limits,
            source_name=self.source_name,
            source_text=self.source_text,
            _positions=self.pos,
        )
        return spec


def load_workload_data(
    data: Any | None = None,
    *,
    source_text: str | None = None,
    source_name: str = "<memory>",
    positioned: PositionedJSON | None = None,
) -> WorkloadSpec:
    if positioned is None:
        if source_text is None:
            import json

            source_text = json.dumps(data, ensure_ascii=False, indent=2)
        try:
            positioned = parse_positioned_json(source_text)
        except Exception as exc:
            line = getattr(exc, "lineno", 1)
            raise DiagnosticError(
                "E_SPEC",
                f"invalid JSON: {exc}",
                field="$",
                line=line,
                details={"source": source_name},
            ) from exc
    return _Validator(positioned, source_name, source_text or "").validate()


def load_workload(path: str | Path) -> WorkloadSpec:
    source_path = Path(path)
    try:
        text = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DiagnosticError(
            "E_SPEC",
            f"cannot read workload file: {exc}",
            field="file",
            line=1,
            details={"source": str(source_path)},
        ) from exc
    return load_workload_data(source_text=text, source_name=str(source_path))


def dump_workload(spec: WorkloadSpec) -> str:
    import json

    return json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def load_many(paths: Iterable[str | Path]) -> list[WorkloadSpec]:
    """Load in the caller's order without leaking that order into audio/window identity."""
    return [load_workload(path) for path in paths]
