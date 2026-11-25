from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from .errors import BusinessError, ConfigValidationError
from .util import fingerprint, is_finite_number, is_int


ALLOWED_FAMILY_LIMITS = {2, 3, 4, 5, 6, 8}
ALLOWED_FRACTIONAL_BITS = {8, 10, 12, 14, 16}
ALLOWED_QUANTIZATION_UNITS = {1, 2, 4, 8}
ALLOWED_KERNEL_BANDWIDTHS = {
    "0.10": 0.10,
    "0.15": 0.15,
    "0.20": 0.20,
    "0.30": 0.30,
    "0.40": 0.40,
}
ALLOWED_DISPERSIONS = {"population_std"}
ALLOWED_EDIT_OPS = {
    "append_batch",
    "replace_window",
    "delete_segment",
    "set_family_count",
    "set_kernel_bandwidth",
    "set_fractional_bits",
    "set_quantization_units",
    "noop",
}
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclass(frozen=True)
class PeriodicTerm:
    amplitude: float
    frequency_hz: float
    phase: float = 0.0


@dataclass(frozen=True)
class NonFiniteInjection:
    index: int
    value: float


@dataclass(frozen=True)
class WaveformChannelSpec:
    channel_id: str
    samples: int
    bias: float
    drift_per_sample: float
    periodic: tuple[PeriodicTerm, ...]
    noise_sigma: float
    missing_segments: tuple[tuple[int, int], ...]
    non_finite_samples: tuple[NonFiniteInjection, ...]


@dataclass(frozen=True)
class WaveformDeviceSpec:
    device_id: str
    channels: tuple[WaveformChannelSpec, ...]


@dataclass(frozen=True)
class WaveformSpec:
    sample_rate_hz: float
    devices: tuple[WaveformDeviceSpec, ...]


@dataclass(frozen=True)
class LeaseDeclaration:
    worker_id: str
    ttl_ms: int
    restore_generation: int


@dataclass(frozen=True)
class Band:
    low_hz: float
    high_hz: float


@dataclass(frozen=True)
class FixedPointConfig:
    fractional_bits: int
    quantization_units: int


@dataclass(frozen=True)
class EffectiveConfig:
    schema_version: int
    seed: int
    config_path: Path
    waveform_path: Path
    edits_path: Path | None
    lease_path: Path
    window_length: int
    window_shift: int
    bands: tuple[Band, ...]
    dispersion: str
    families: int
    family_limit: int
    max_iterations: int
    fixed_point: FixedPointConfig
    bandwidth_label: str
    bandwidth: float
    threshold_permille: int
    max_workers: int
    state_dir: Path
    report_path: Path | None
    waveform: WaveformSpec
    lease: LeaseDeclaration
    edits: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def fingerprinted_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "window": {"length": self.window_length, "shift": self.window_shift},
            "features": {
                "bands_hz": [[band.low_hz, band.high_hz] for band in self.bands],
            },
            "standardization": {"dispersion": self.dispersion},
            "clustering": {
                "families": self.families,
                "family_limit": self.family_limit,
                "max_iterations": self.max_iterations,
                "fixed_point": asdict(self.fixed_point),
            },
            "kde": {
                "bandwidth": self.bandwidth,
                "bandwidth_label": self.bandwidth_label,
                "threshold_permille": self.threshold_permille,
            },
            "runtime": {"max_workers": self.max_workers},
            "waveform": waveform_payload(self.waveform),
            "lease_declaration": asdict(self.lease),
        }

    @property
    def config_hash(self) -> str:
        return fingerprint(self.fingerprinted_payload())

    def feature_parameter_fingerprint(self) -> str:
        return fingerprint(
            {
                "window_length": self.window_length,
                "window_shift": self.window_shift,
                "bands": [[band.low_hz, band.high_hz] for band in self.bands],
            }
        )

    def standardization_fingerprint(self) -> str:
        return fingerprint({"dispersion": self.dispersion})

    def quantization_fingerprint(self) -> str:
        return fingerprint(asdict(self.fixed_point))

    def model_fingerprint(self) -> str:
        return fingerprint(
            {
                "families": self.families,
                "max_iterations": self.max_iterations,
                "fixed_point": asdict(self.fixed_point),
            }
        )

    def kde_fingerprint(self) -> str:
        return fingerprint(
            {
                "bandwidth": self.bandwidth,
                "threshold_permille": self.threshold_permille,
                "fixed_point": asdict(self.fixed_point),
            }
        )


def waveform_payload(spec: WaveformSpec) -> dict[str, Any]:
    return {
        "sample_rate_hz": spec.sample_rate_hz,
        "devices": [
            {
                "id": device.device_id,
                "channels": [
                    {
                        "id": channel.channel_id,
                        "samples": channel.samples,
                        "bias": channel.bias,
                        "drift_per_sample": channel.drift_per_sample,
                        "periodic": [asdict(term) for term in channel.periodic],
                        "noise_sigma": channel.noise_sigma,
                        "missing_segments": [list(segment) for segment in channel.missing_segments],
                        "non_finite_samples": [
                            asdict(injection) for injection in channel.non_finite_samples
                        ],
                    }
                    for channel in device.channels
                ],
            }
            for device in spec.devices
        ],
    }


def _issue(path: str, message: str) -> BusinessError:
    return BusinessError(
        code="E_CONFIG_INVALID",
        path=path,
        message=message,
        stage="config",
    )


def _load_yaml(path: Path, issues: list[BusinessError], path_name: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.load(handle, Loader=UniqueKeyLoader)
    except FileNotFoundError:
        issues.append(_issue(path_name, f"file not found: {path}"))
    except yaml.YAMLError as exc:
        issues.append(_issue(path_name, f"invalid YAML: {exc}"))
    except OSError as exc:
        issues.append(_issue(path_name, f"cannot read file: {exc}"))
    return None


def _require_mapping(
    value: Any, path: str, issues: list[BusinessError]
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        issues.append(_issue(path, "must be a mapping"))
        return None
    return value


def _require_int(
    value: Any,
    path: str,
    issues: list[BusinessError],
    *,
    minimum: int | None = None,
    allowed: Iterable[int] | None = None,
) -> int | None:
    if not is_int(value):
        issues.append(_issue(path, "must be an integer"))
        return None
    if minimum is not None and value < minimum:
        issues.append(_issue(path, f"must be >= {minimum}"))
        return None
    if allowed is not None and value not in set(allowed):
        issues.append(_issue(path, f"must be one of {sorted(set(allowed))}"))
        return None
    return int(value)


def _require_number(
    value: Any,
    path: str,
    issues: list[BusinessError],
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float | None:
    if not is_finite_number(value):
        issues.append(_issue(path, "must be a finite number"))
        return None
    number = float(value)
    if positive and number <= 0:
        issues.append(_issue(path, "must be > 0"))
        return None
    if nonnegative and number < 0:
        issues.append(_issue(path, "must be >= 0"))
        return None
    return number


def _require_string(
    value: Any, path: str, issues: list[BusinessError], *, identifier: bool = False
) -> str | None:
    if not isinstance(value, str) or not value:
        issues.append(_issue(path, "must be a non-empty string"))
        return None
    if identifier and not IDENTIFIER_RE.fullmatch(value):
        issues.append(_issue(path, "must match [A-Za-z0-9_.-]+"))
        return None
    return value


def _parse_waveform(
    raw: Any,
    root_path: str,
    issues: list[BusinessError],
    window_length: int | None,
) -> WaveformSpec | None:
    data = _require_mapping(raw, root_path, issues)
    if data is None:
        return None
    sample_rate = _require_number(
        data.get("sample_rate_hz"), f"{root_path}.sample_rate_hz", issues, positive=True
    )
    devices_raw = data.get("devices")
    if not isinstance(devices_raw, list) or not devices_raw:
        issues.append(_issue(f"{root_path}.devices", "must be a non-empty list"))
        return None

    devices: list[WaveformDeviceSpec] = []
    seen_devices: set[str] = set()
    for device_index, device_raw in enumerate(devices_raw):
        device_path = f"{root_path}.devices[{device_index}]"
        device_map = _require_mapping(device_raw, device_path, issues)
        if device_map is None:
            continue
        device_id = _require_string(
            device_map.get("id"), f"{device_path}.id", issues, identifier=True
        )
        if device_id is not None:
            if device_id in seen_devices:
                issues.append(
                    _issue(f"{device_path}.id", f"duplicate device id: {device_id}")
                )
            seen_devices.add(device_id)
        channels_raw = device_map.get("channels")
        if not isinstance(channels_raw, list) or not channels_raw:
            issues.append(_issue(f"{device_path}.channels", "must be a non-empty list"))
            continue
        channels: list[WaveformChannelSpec] = []
        seen_channels: set[str] = set()
        for channel_index, channel_raw in enumerate(channels_raw):
            channel_path = f"{device_path}.channels[{channel_index}]"
            channel_map = _require_mapping(channel_raw, channel_path, issues)
            if channel_map is None:
                continue
            channel_id = _require_string(
                channel_map.get("id"), f"{channel_path}.id", issues, identifier=True
            )
            if channel_id is not None:
                if channel_id in seen_channels:
                    issues.append(
                        _issue(
                            f"{channel_path}.id",
                            f"duplicate channel id in device: {channel_id}",
                        )
                    )
                seen_channels.add(channel_id)
            samples = _require_int(
                channel_map.get("samples"), f"{channel_path}.samples", issues, minimum=1
            )
            if samples is not None and window_length is not None and samples < window_length:
                issues.append(
                    _issue(
                        f"{channel_path}.samples",
                        f"must be >= window length ({window_length})",
                    )
                )
            bias = _require_number(
                channel_map.get("bias", 0.0), f"{channel_path}.bias", issues
            )
            drift = _require_number(
                channel_map.get("drift_per_sample", 0.0),
                f"{channel_path}.drift_per_sample",
                issues,
            )
            sigma = _require_number(
                channel_map.get("noise_sigma", 0.0),
                f"{channel_path}.noise_sigma",
                issues,
                nonnegative=True,
            )
            periodic_raw = channel_map.get("periodic", [])
            if not isinstance(periodic_raw, list):
                issues.append(_issue(f"{channel_path}.periodic", "must be a list"))
                periodic_raw = []
            periodic: list[PeriodicTerm] = []
            for term_index, term_raw in enumerate(periodic_raw):
                term_path = f"{channel_path}.periodic[{term_index}]"
                term_map = _require_mapping(term_raw, term_path, issues)
                if term_map is None:
                    continue
                amplitude = _require_number(
                    term_map.get("amplitude"),
                    f"{term_path}.amplitude",
                    issues,
                )
                frequency = _require_number(
                    term_map.get("frequency_hz"),
                    f"{term_path}.frequency_hz",
                    issues,
                    positive=True,
                )
                phase = _require_number(
                    term_map.get("phase", 0.0), f"{term_path}.phase", issues
                )
                if amplitude is not None and frequency is not None and phase is not None:
                    periodic.append(PeriodicTerm(amplitude, frequency, phase))
            missing_raw = channel_map.get("missing_segments", [])
            if not isinstance(missing_raw, list):
                issues.append(_issue(f"{channel_path}.missing_segments", "must be a list"))
                missing_raw = []
            missing: list[tuple[int, int]] = []
            for segment_index, segment_raw in enumerate(missing_raw):
                segment_path = f"{channel_path}.missing_segments[{segment_index}]"
                if not isinstance(segment_raw, list) or len(segment_raw) != 2:
                    issues.append(_issue(segment_path, "must be [start, end)"))
                    continue
                start = _require_int(segment_raw[0], f"{segment_path}[0]", issues, minimum=0)
                end = _require_int(segment_raw[1], f"{segment_path}[1]", issues, minimum=0)
                if start is None or end is None:
                    continue
                if end <= start:
                    issues.append(_issue(segment_path, "end must be greater than start"))
                elif samples is not None and end > samples:
                    issues.append(_issue(segment_path, "end exceeds sample count"))
                else:
                    missing.append((start, end))
            nonfinite_raw = channel_map.get("non_finite_samples", [])
            if not isinstance(nonfinite_raw, list):
                issues.append(
                    _issue(f"{channel_path}.non_finite_samples", "must be a list")
                )
                nonfinite_raw = []
            nonfinite: list[NonFiniteInjection] = []
            for injection_index, injection_raw in enumerate(nonfinite_raw):
                injection_path = f"{channel_path}.non_finite_samples[{injection_index}]"
                injection_map = _require_mapping(injection_raw, injection_path, issues)
                if injection_map is None:
                    continue
                sample_index = _require_int(
                    injection_map.get("index"),
                    f"{injection_path}.index",
                    issues,
                    minimum=0,
                )
                value = injection_map.get("value")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    issues.append(_issue(f"{injection_path}.value", "must be numeric"))
                    continue
                if sample_index is not None and samples is not None:
                    if sample_index >= samples:
                        issues.append(
                            _issue(f"{injection_path}.index", "exceeds sample count")
                        )
                    else:
                        nonfinite.append(NonFiniteInjection(sample_index, float(value)))
            if (
                channel_id is not None
                and samples is not None
                and bias is not None
                and drift is not None
                and sigma is not None
            ):
                channels.append(
                    WaveformChannelSpec(
                        channel_id=channel_id,
                        samples=samples,
                        bias=bias,
                        drift_per_sample=drift,
                        periodic=tuple(periodic),
                        noise_sigma=sigma,
                        missing_segments=tuple(missing),
                        non_finite_samples=tuple(nonfinite),
                    )
                )
        if device_id is not None and channels:
            devices.append(WaveformDeviceSpec(device_id, tuple(channels)))
    if sample_rate is None or not devices:
        return None
    return WaveformSpec(sample_rate, tuple(devices))


def _parse_lease(
    raw: Any, root_path: str, issues: list[BusinessError]
) -> LeaseDeclaration | None:
    data = _require_mapping(raw, root_path, issues)
    if data is None:
        return None
    worker_id = _require_string(
        data.get("worker_id"), f"{root_path}.worker_id", issues, identifier=True
    )
    ttl_ms = _require_int(
        data.get("ttl_ms"), f"{root_path}.ttl_ms", issues, minimum=1
    )
    restore_generation = _require_int(
        data.get("restore_generation", 0),
        f"{root_path}.restore_generation",
        issues,
        minimum=0,
    )
    if worker_id is None or ttl_ms is None or restore_generation is None:
        return None
    return LeaseDeclaration(worker_id, ttl_ms, restore_generation)


def _parse_edits(raw: Any, root_path: str, issues: list[BusinessError]) -> tuple[dict[str, Any], ...]:
    data = _require_mapping(raw, root_path, issues)
    if data is None:
        return ()
    commands_raw = data.get("commands", [])
    if not isinstance(commands_raw, list):
        issues.append(_issue(f"{root_path}.commands", "must be a list"))
        return ()
    commands: list[dict[str, Any]] = []
    for index, command_raw in enumerate(commands_raw):
        command_path = f"{root_path}.commands[{index}]"
        command_map = _require_mapping(command_raw, command_path, issues)
        if command_map is None:
            continue
        op = _require_string(command_map.get("op"), f"{command_path}.op", issues)
        if op is not None and op not in ALLOWED_EDIT_OPS:
            issues.append(
                _issue(
                    f"{command_path}.op",
                    f"must be one of {sorted(ALLOWED_EDIT_OPS)}",
                )
            )
            continue
        if op == "append_batch":
            _require_string(
                command_map.get("device_id"),
                f"{command_path}.device_id",
                issues,
                identifier=True,
            )
            _require_string(
                command_map.get("channel_id"),
                f"{command_path}.channel_id",
                issues,
                identifier=True,
            )
            _require_int(
                command_map.get("count"),
                f"{command_path}.count",
                issues,
                minimum=1,
            )
            _require_int(
                command_map.get("seed"), f"{command_path}.seed", issues, minimum=0
            )
        elif op == "replace_window":
            _require_string(
                command_map.get("window_id"),
                f"{command_path}.window_id",
                issues,
            )
            _require_int(
                command_map.get("seed"), f"{command_path}.seed", issues, minimum=0
            )
        elif op == "delete_segment":
            _require_string(
                command_map.get("device_id"),
                f"{command_path}.device_id",
                issues,
                identifier=True,
            )
            _require_string(
                command_map.get("channel_id"),
                f"{command_path}.channel_id",
                issues,
                identifier=True,
            )
            _require_int(
                command_map.get("start"),
                f"{command_path}.start",
                issues,
                minimum=0,
            )
            _require_int(
                command_map.get("length"),
                f"{command_path}.length",
                issues,
                minimum=1,
            )
        elif op == "set_family_count":
            _require_int(
                command_map.get("value"),
                f"{command_path}.value",
                issues,
                minimum=1,
            )
        elif op == "set_kernel_bandwidth":
            value = _require_string(
                command_map.get("value"), f"{command_path}.value", issues
            )
            if value is not None and value not in ALLOWED_KERNEL_BANDWIDTHS:
                issues.append(
                    _issue(
                        f"{command_path}.value",
                        f"kernel bandwidth must be one of {sorted(ALLOWED_KERNEL_BANDWIDTHS)}",
                    )
                )
        elif op == "set_fractional_bits":
            _require_int(
                command_map.get("value"),
                f"{command_path}.value",
                issues,
                allowed=ALLOWED_FRACTIONAL_BITS,
            )
        elif op == "set_quantization_units":
            _require_int(
                command_map.get("value"),
                f"{command_path}.value",
                issues,
                allowed=ALLOWED_QUANTIZATION_UNITS,
            )
        commands.append(copy.deepcopy(command_map))
    return tuple(commands)


def load_config(path: str | Path) -> EffectiveConfig:
    config_path = Path(path).resolve()
    issues: list[BusinessError] = []
    raw = _load_yaml(config_path, issues, "config")
    data = _require_mapping(raw, "$", issues) if raw is not None else None
    if data is None:
        raise ConfigValidationError(issues)

    schema_version = _require_int(
        data.get("schema_version", 1), "$.schema_version", issues, minimum=1
    )
    seed = _require_int(data.get("seed"), "$.seed", issues, minimum=0)
    window_map = _require_mapping(data.get("window"), "$.window", issues) or {}
    window_length = _require_int(
        window_map.get("length"), "$.window.length", issues, minimum=1
    )
    window_shift = _require_int(
        window_map.get("shift"), "$.window.shift", issues, minimum=1
    )
    if window_length is not None and window_shift is not None and window_shift > window_length:
        issues.append(_issue("$.window.shift", "must be <= window.length"))

    features_map = _require_mapping(data.get("features"), "$.features", issues) or {}
    bands_raw = features_map.get("bands_hz")
    if not isinstance(bands_raw, list) or not bands_raw:
        issues.append(_issue("$.features.bands_hz", "must be a non-empty list"))
        bands_raw = []
    bands: list[Band] = []
    for index, band_raw in enumerate(bands_raw):
        band_path = f"$.features.bands_hz[{index}]"
        if not isinstance(band_raw, list) or len(band_raw) != 2:
            issues.append(_issue(band_path, "must be [low_hz, high_hz]"))
            continue
        low = _require_number(band_raw[0], f"{band_path}[0]", issues, nonnegative=True)
        high = _require_number(band_raw[1], f"{band_path}[1]", issues, nonnegative=True)
        if low is None or high is None:
            continue
        if high <= low:
            issues.append(_issue(band_path, "high_hz must be greater than low_hz"))
        else:
            bands.append(Band(low, high))
    for index in range(1, len(bands)):
        if bands[index].low_hz < bands[index - 1].high_hz:
            issues.append(
                _issue(
                    f"$.features.bands_hz[{index}]",
                    "bands must be sorted and non-overlapping",
                )
            )

    standardization_map = (
        _require_mapping(data.get("standardization"), "$.standardization", issues) or {}
    )
    dispersion = _require_string(
        standardization_map.get("dispersion", "population_std"),
        "$.standardization.dispersion",
        issues,
    )
    if dispersion is not None and dispersion not in ALLOWED_DISPERSIONS:
        issues.append(
            _issue(
                "$.standardization.dispersion",
                f"must be one of {sorted(ALLOWED_DISPERSIONS)}",
            )
        )

    clustering_map = (
        _require_mapping(data.get("clustering"), "$.clustering", issues) or {}
    )
    families = _require_int(
        clustering_map.get("families"), "$.clustering.families", issues, minimum=1
    )
    family_limit = _require_int(
        clustering_map.get("family_limit"),
        "$.clustering.family_limit",
        issues,
        allowed=ALLOWED_FAMILY_LIMITS,
    )
    max_iterations = _require_int(
        clustering_map.get("max_iterations"),
        "$.clustering.max_iterations",
        issues,
        minimum=1,
    )
    if families is not None and family_limit is not None and families > family_limit:
        issues.append(
            _issue(
                "$.clustering.families",
                "must be <= $.clustering.family_limit",
            )
        )
    fixed_map = (
        _require_mapping(clustering_map.get("fixed_point"), "$.clustering.fixed_point", issues)
        or {}
    )
    fractional_bits = _require_int(
        fixed_map.get("fractional_bits"),
        "$.clustering.fixed_point.fractional_bits",
        issues,
        allowed=ALLOWED_FRACTIONAL_BITS,
    )
    quantization_units = _require_int(
        fixed_map.get("quantization_units"),
        "$.clustering.fixed_point.quantization_units",
        issues,
        allowed=ALLOWED_QUANTIZATION_UNITS,
    )
    if (
        fractional_bits is not None
        and quantization_units is not None
        and quantization_units > (1 << fractional_bits)
    ):
        issues.append(
            _issue(
                "$.clustering.fixed_point.quantization_units",
                "must not exceed the fixed-point scale",
            )
        )

    kde_map = _require_mapping(data.get("kde"), "$.kde", issues) or {}
    bandwidth_label = _require_string(
        kde_map.get("bandwidth"), "$.kde.bandwidth", issues
    )
    bandwidth: float | None = None
    if bandwidth_label is not None:
        bandwidth = ALLOWED_KERNEL_BANDWIDTHS.get(bandwidth_label)
        if bandwidth is None:
            issues.append(
                _issue(
                    "$.kde.bandwidth",
                    f"must be one of {sorted(ALLOWED_KERNEL_BANDWIDTHS)}",
                )
            )
    threshold_permille = _require_int(
        kde_map.get("threshold_permille"),
        "$.kde.threshold_permille",
        issues,
        minimum=0,
    )
    if threshold_permille is not None and threshold_permille > 1000:
        issues.append(_issue("$.kde.threshold_permille", "must be <= 1000"))

    runtime_map = _require_mapping(data.get("runtime"), "$.runtime", issues) or {}
    max_workers = _require_int(
        runtime_map.get("max_workers"),
        "$.runtime.max_workers",
        issues,
        minimum=1,
    )
    state_dir_text = _require_string(
        runtime_map.get("state_dir", "state"), "$.runtime.state_dir", issues
    )
    report_path_text = runtime_map.get("report_path")
    if report_path_text is not None:
        report_path_text = _require_string(
            report_path_text, "$.runtime.report_path", issues
        )

    waveform_path_text = _require_string(
        data.get("waveform_path"), "$.waveform_path", issues
    )
    lease_path_text = _require_string(
        data.get("lease_declaration_path"), "$.lease_declaration_path", issues
    )
    edits_path_text = data.get("edits_path")
    if edits_path_text is not None:
        edits_path_text = _require_string(edits_path_text, "$.edits_path", issues)

    base = config_path.parent
    waveform_path = (base / waveform_path_text).resolve() if waveform_path_text else base
    lease_path = (base / lease_path_text).resolve() if lease_path_text else base
    edits_path = (base / edits_path_text).resolve() if edits_path_text else None

    waveform_raw = (
        _load_yaml(waveform_path, issues, "waveform") if waveform_path_text else None
    )
    waveform = (
        _parse_waveform(waveform_raw, "waveform", issues, window_length)
        if waveform_raw is not None
        else None
    )
    lease_raw = _load_yaml(lease_path, issues, "lease") if lease_path_text else None
    lease = _parse_lease(lease_raw, "lease", issues) if lease_raw is not None else None
    edits: tuple[dict[str, Any], ...] = ()
    if edits_path_text and edits_path is not None:
        edits_raw = _load_yaml(edits_path, issues, "edits")
        if edits_raw is not None:
            edits = _parse_edits(edits_raw, "edits", issues)

    if waveform is not None:
        nyquist = waveform.sample_rate_hz / 2.0
        for index, band in enumerate(bands):
            if band.high_hz > nyquist:
                issues.append(
                    _issue(
                        f"$.features.bands_hz[{index}][1]",
                        f"must be <= Nyquist frequency ({nyquist})",
                    )
                )

    if issues:
        raise ConfigValidationError(sorted(issues, key=lambda item: item.sort_key))

    assert schema_version is not None
    assert seed is not None
    assert window_length is not None
    assert window_shift is not None
    assert dispersion is not None
    assert families is not None
    assert family_limit is not None
    assert max_iterations is not None
    assert fractional_bits is not None
    assert quantization_units is not None
    assert bandwidth is not None
    assert bandwidth_label is not None
    assert threshold_permille is not None
    assert max_workers is not None
    assert state_dir_text is not None
    assert waveform is not None
    assert lease is not None

    state_dir = (base / state_dir_text).resolve()
    report_path = (base / report_path_text).resolve() if report_path_text else None

    return EffectiveConfig(
        schema_version=schema_version,
        seed=seed,
        config_path=config_path,
        waveform_path=waveform_path,
        edits_path=edits_path,
        lease_path=lease_path,
        window_length=window_length,
        window_shift=window_shift,
        bands=tuple(bands),
        dispersion=dispersion,
        families=families,
        family_limit=family_limit,
        max_iterations=max_iterations,
        fixed_point=FixedPointConfig(fractional_bits, quantization_units),
        bandwidth_label=bandwidth_label,
        bandwidth=bandwidth,
        threshold_permille=threshold_permille,
        max_workers=max_workers,
        state_dir=state_dir,
        report_path=report_path,
        waveform=waveform,
        lease=lease,
        edits=edits,
    )


def override_family_count(config: EffectiveConfig, value: int, path: str) -> EffectiveConfig:
    if not is_int(value) or value < 1:
        raise ConfigValidationError([_issue(path, "must be a positive integer")])
    if value > config.family_limit:
        raise ConfigValidationError(
            [_issue(path, f"must be <= family_limit ({config.family_limit})")]
        )
    return replace(config, families=value)


def override_bandwidth(config: EffectiveConfig, value: str, path: str) -> EffectiveConfig:
    if value not in ALLOWED_KERNEL_BANDWIDTHS:
        raise ConfigValidationError(
            [_issue(path, f"must be one of {sorted(ALLOWED_KERNEL_BANDWIDTHS)}")]
        )
    return replace(
        config,
        bandwidth_label=value,
        bandwidth=ALLOWED_KERNEL_BANDWIDTHS[value],
    )


def override_fractional_bits(
    config: EffectiveConfig, value: int, path: str
) -> EffectiveConfig:
    if value not in ALLOWED_FRACTIONAL_BITS:
        raise ConfigValidationError(
            [_issue(path, f"must be one of {sorted(ALLOWED_FRACTIONAL_BITS)}")]
        )
    return replace(
        config,
        fixed_point=replace(config.fixed_point, fractional_bits=value),
    )


def override_quantization_units(
    config: EffectiveConfig, value: int, path: str
) -> EffectiveConfig:
    if value not in ALLOWED_QUANTIZATION_UNITS:
        raise ConfigValidationError(
            [_issue(path, f"must be one of {sorted(ALLOWED_QUANTIZATION_UNITS)}")]
        )
    if value > (1 << config.fixed_point.fractional_bits):
        raise ConfigValidationError(
            [_issue(path, "must not exceed the fixed-point scale")]
        )
    return replace(
        config,
        fixed_point=replace(config.fixed_point, quantization_units=value),
    )

