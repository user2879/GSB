from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SpecError
from .util import FORMAT_VERSION, canonical_json, content_digest


LOOPBACK_HOST = "127.0.0.1"


def _field(parent: str, name: str | int) -> str:
    if isinstance(name, int):
        return f"{parent}[{name}]"
    return f"{parent}.{name}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(path, "must be a JSON object")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SpecError(path, "must be a JSON array")
    return value


def _require_string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise SpecError(path, "must be a string")
    if nonempty and not value:
        raise SpecError(path, "must not be empty")
    return value


def _require_int(value: Any, path: str) -> int:
    if not _is_int(value):
        raise SpecError(path, "must be an integer")
    return value


def _reject_unknown_fields(
    value: dict[str, Any], allowed: set[str], path: str
) -> None:
    for key in value:
        if key not in allowed:
            raise SpecError(_field(path, key), "unknown field")


@dataclass(frozen=True)
class ModelSpec:
    version: str
    model_dim: int
    hidden_dim: int
    seed: int
    host: str
    port: int

    def canonical(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "host": self.host,
            "port": self.port,
        }


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    start_request: int
    end_request: int
    gray_weight_bp: int

    def canonical(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "start_request": self.start_request,
            "end_request": self.end_request,
            "gray_weight_bp": self.gray_weight_bp,
        }


@dataclass(frozen=True)
class RouteSpec:
    rule_id: str
    match_prefix: str
    stable_version: str
    canary_version: str
    gray_weight_bp: int
    batch_size: int
    stages: tuple[StageSpec, ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "match_prefix": self.match_prefix,
            "stable_version": self.stable_version,
            "canary_version": self.canary_version,
            "gray_weight_bp": self.gray_weight_bp,
            "batch_size": self.batch_size,
            "stages": [stage.canonical() for stage in self.stages],
        }

    def stage_for(self, request_seq: int) -> StageSpec | None:
        for stage in self.stages:
            if stage.start_request <= request_seq <= stage.end_request:
                return stage
        return None


@dataclass(frozen=True)
class GatewaySpec:
    format_version: int
    host: str
    port: int
    models: tuple[ModelSpec, ...]
    routes: tuple[RouteSpec, ...]
    rebuild_latency_budget_ms: int
    query_latency_budget_ms: int
    stream_threshold: int
    source_path: str | None = None

    @property
    def models_by_version(self) -> dict[str, ModelSpec]:
        return {model.version: model for model in self.models}

    @property
    def routes_by_id(self) -> dict[str, RouteSpec]:
        return {route.rule_id: route for route in self.routes}

    @property
    def stage_count(self) -> int:
        return sum(len(route.stages) for route in self.routes)

    @property
    def canonical_digest(self) -> str:
        return content_digest(self.canonical())

    def canonical(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "gateway": {"host": self.host, "port": self.port},
            "models": [model.canonical() for model in self.models],
            "routes": [route.canonical() for route in self.routes],
            "rebuild_latency_budget_ms": self.rebuild_latency_budget_ms,
            "query_latency_budget_ms": self.query_latency_budget_ms,
            "stream_threshold": self.stream_threshold,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.canonical())

    def model(self, version: str) -> ModelSpec:
        try:
            return self.models_by_version[version]
        except KeyError as exc:
            raise SpecError("$.routes", f"unknown model version {version!r}") from exc

    def route_for_feature(self, feature: str) -> RouteSpec:
        for route in self.routes:
            if feature.startswith(route.match_prefix):
                return route
        raise SpecError("$.request.feature", "no route prefix matched")


def _parse_stage(raw: Any, path: str) -> StageSpec:
    obj = _require_object(raw, path)
    _reject_unknown_fields(
        obj,
        {"stage_id", "start_request", "end_request", "gray_weight_bp", "weights"},
        path,
    )
    stage_id = _require_string(obj.get("stage_id"), _field(path, "stage_id"))
    start = _require_int(obj.get("start_request"), _field(path, "start_request"))
    end = _require_int(obj.get("end_request"), _field(path, "end_request"))
    weight = _require_int(obj.get("gray_weight_bp"), _field(path, "gray_weight_bp"))
    if start < 0:
        raise SpecError(_field(path, "start_request"), "must be >= 0")
    if end < start:
        raise SpecError(_field(path, "end_request"), "must be >= start_request")
    if weight < 0 or weight > 10000:
        raise SpecError(_field(path, "gray_weight_bp"), "must be in [0, 10000]")
    return StageSpec(stage_id, start, end, weight)


def _validate_stage_sequence(stages: list[StageSpec], path: str) -> None:
    if not stages:
        raise SpecError(path, "at least one stage is required")
    seen_ids: set[str] = set()
    for index, stage in enumerate(stages):
        if stage.stage_id in seen_ids:
            raise SpecError(
                _field(_field(path, index), "stage_id"),
                f"duplicate stage_id {stage.stage_id!r}",
            )
        seen_ids.add(stage.stage_id)
        if index == 0:
            continue
        previous = stages[index - 1]
        if stage.start_request != previous.end_request + 1:
            raise SpecError(
                _field(_field(path, index), "start_request"),
                f"must equal previous end_request + 1 ({previous.end_request + 1})",
            )
        if stage.gray_weight_bp < previous.gray_weight_bp:
            raise SpecError(
                _field(_field(path, index), "gray_weight_bp"),
                "gray weight must not decrease between contiguous stages",
            )


def _stage_weights_for_route(
    stage_obj: dict[str, Any],
    route_id: str,
    path: str,
    fallback: int,
) -> int:
    weights = stage_obj.get("weights")
    if weights is None:
        return fallback
    weights_obj = _require_object(weights, _field(path, "weights"))
    if route_id not in weights_obj:
        return fallback
    weight = _require_int(
        weights_obj[route_id], _field(_field(path, "weights"), route_id)
    )
    if weight < 0 or weight > 10000:
        raise SpecError(
            _field(_field(path, "weights"), route_id), "must be in [0, 10000]"
        )
    return weight


def parse_spec(raw: Any, *, source_path: str | None = None) -> GatewaySpec:
    root = _require_object(raw, "$")
    _reject_unknown_fields(
        root,
        {
            "format_version",
            "gateway",
            "models",
            "routes",
            "stages",
            "rebuild_latency_budget_ms",
            "query_latency_budget_ms",
            "stream_threshold",
        },
        "$",
    )

    format_version = _require_int(
        root.get("format_version"), "$.format_version"
    )
    if format_version != FORMAT_VERSION:
        raise SpecError(
            "$.format_version", f"unsupported format_version {format_version}"
        )

    gateway = _require_object(root.get("gateway"), "$.gateway")
    _reject_unknown_fields(gateway, {"host", "port"}, "$.gateway")
    host = _require_string(gateway.get("host"), "$.gateway.host")
    if host != LOOPBACK_HOST:
        raise SpecError("$.gateway.host", "must be the literal 127.0.0.1")
    gateway_port = _require_int(gateway.get("port"), "$.gateway.port")
    if gateway_port < 1 or gateway_port > 65535:
        raise SpecError("$.gateway.port", "must be in [1, 65535]")

    models_raw = _require_list(root.get("models"), "$.models")
    if not models_raw:
        raise SpecError("$.models", "at least one model is required")
    models: list[ModelSpec] = []
    model_ids: set[str] = set()
    used_ports: dict[int, str] = {gateway_port: "$.gateway.port"}
    for index, item in enumerate(models_raw):
        path = f"$.models[{index}]"
        obj = _require_object(item, path)
        _reject_unknown_fields(
            obj, {"version", "model_dim", "hidden_dim", "seed", "host", "port"}, path
        )
        version = _require_string(obj.get("version"), _field(path, "version"))
        if version in model_ids:
            raise SpecError(_field(path, "version"), f"duplicate model version {version!r}")
        model_ids.add(version)
        model_dim = _require_int(obj.get("model_dim"), _field(path, "model_dim"))
        hidden_dim = _require_int(obj.get("hidden_dim"), _field(path, "hidden_dim"))
        seed = _require_int(obj.get("seed"), _field(path, "seed"))
        model_host = _require_string(
            obj.get("host", LOOPBACK_HOST), _field(path, "host")
        )
        if model_host != LOOPBACK_HOST:
            raise SpecError(_field(path, "host"), "must be the literal 127.0.0.1")
        port = _require_int(obj.get("port"), _field(path, "port"))
        if model_dim <= 0:
            raise SpecError(_field(path, "model_dim"), "must be a positive integer")
        if hidden_dim <= 0:
            raise SpecError(_field(path, "hidden_dim"), "must be a positive integer")
        if port < 1 or port > 65535:
            raise SpecError(_field(path, "port"), "must be in [1, 65535]")
        if port in used_ports:
            raise SpecError(
                _field(path, "port"),
                f"port {port} is already used by {used_ports[port]}",
            )
        used_ports[port] = _field(path, "port")
        models.append(
            ModelSpec(
                version=version,
                model_dim=model_dim,
                hidden_dim=hidden_dim,
                seed=seed,
                host=model_host,
                port=port,
            )
        )

    routes_raw = _require_list(root.get("routes"), "$.routes")
    if not routes_raw:
        raise SpecError("$.routes", "at least one route is required")
    global_stages: list[StageSpec] = []
    global_stage_objects: list[dict[str, Any]] = []
    if "stages" in root:
        stages_raw = _require_list(root.get("stages"), "$.stages")
        global_stage_objects = [
            _require_object(item, f"$.stages[{index}]")
            for index, item in enumerate(stages_raw)
        ]
        global_stages = [
            _parse_stage(item, f"$.stages[{index}]")
            for index, item in enumerate(global_stage_objects)
        ]
        _validate_stage_sequence(global_stages, "$.stages")

    routes: list[RouteSpec] = []
    rule_ids: set[str] = set()
    prefixes: set[str] = set()
    for index, item in enumerate(routes_raw):
        path = f"$.routes[{index}]"
        obj = _require_object(item, path)
        _reject_unknown_fields(
            obj,
            {
                "rule_id",
                "match_prefix",
                "stable_version",
                "canary_version",
                "gray_weight_bp",
                "batch_size",
                "stages",
            },
            path,
        )
        rule_id = _require_string(obj.get("rule_id"), _field(path, "rule_id"))
        if rule_id in rule_ids:
            raise SpecError(_field(path, "rule_id"), f"duplicate rule_id {rule_id!r}")
        rule_ids.add(rule_id)
        prefix = _require_string(obj.get("match_prefix"), _field(path, "match_prefix"))
        if prefix in prefixes:
            raise SpecError(
                _field(path, "match_prefix"), f"duplicate match_prefix {prefix!r}"
            )
        prefixes.add(prefix)
        stable_version = _require_string(
            obj.get("stable_version"), _field(path, "stable_version")
        )
        canary_version = _require_string(
            obj.get("canary_version"), _field(path, "canary_version")
        )
        if stable_version not in model_ids:
            raise SpecError(
                _field(path, "stable_version"),
                f"unknown model version {stable_version!r}",
            )
        if canary_version not in model_ids:
            raise SpecError(
                _field(path, "canary_version"),
                f"unknown model version {canary_version!r}",
            )
        if stable_version == canary_version:
            raise SpecError(
                _field(path, "canary_version"),
                "stable_version and canary_version must differ",
            )
        route_weight = _require_int(
            obj.get("gray_weight_bp"), _field(path, "gray_weight_bp")
        )
        if route_weight < 0 or route_weight > 10000:
            raise SpecError(_field(path, "gray_weight_bp"), "must be in [0, 10000]")
        batch_size = _require_int(obj.get("batch_size"), _field(path, "batch_size"))
        if batch_size <= 0:
            raise SpecError(_field(path, "batch_size"), "must be a positive integer")

        if "stages" in obj:
            route_stage_raw = _require_list(obj.get("stages"), _field(path, "stages"))
            route_stages = [
                _parse_stage(stage, f"{path}.stages[{stage_index}]")
                for stage_index, stage in enumerate(route_stage_raw)
            ]
        elif global_stages:
            route_stages = []
            for stage_index, stage in enumerate(global_stages):
                route_stages.append(
                    StageSpec(
                        stage.stage_id,
                        stage.start_request,
                        stage.end_request,
                        _stage_weights_for_route(
                            global_stage_objects[stage_index],
                            rule_id,
                            f"$.stages[{stage_index}]",
                            route_weight,
                        ),
                    )
                )
        else:
            raise SpecError(
                _field(path, "stages"),
                "missing route stages and no global stages table was provided",
            )
        _validate_stage_sequence(route_stages, f"{path}.stages")
        routes.append(
            RouteSpec(
                rule_id=rule_id,
                match_prefix=prefix,
                stable_version=stable_version,
                canary_version=canary_version,
                gray_weight_bp=route_weight,
                batch_size=batch_size,
                stages=tuple(route_stages),
            )
        )

    rebuild_budget = _require_int(
        root.get("rebuild_latency_budget_ms"), "$.rebuild_latency_budget_ms"
    )
    query_budget = _require_int(
        root.get("query_latency_budget_ms"), "$.query_latency_budget_ms"
    )
    stream_threshold = _require_int(
        root.get("stream_threshold"), "$.stream_threshold"
    )
    if rebuild_budget <= 0:
        raise SpecError(
            "$.rebuild_latency_budget_ms", "must be a positive integer"
        )
    if query_budget <= 0:
        raise SpecError("$.query_latency_budget_ms", "must be a positive integer")
    if stream_threshold <= 0:
        raise SpecError("$.stream_threshold", "must be a positive integer")

    return GatewaySpec(
        format_version=format_version,
        host=host,
        port=gateway_port,
        models=tuple(models),
        routes=tuple(routes),
        rebuild_latency_budget_ms=rebuild_budget,
        query_latency_budget_ms=query_budget,
        stream_threshold=stream_threshold,
        source_path=source_path,
    )


def load_spec(path: str | Path) -> GatewaySpec:
    spec_path = Path(path)
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError("$", f"cannot read spec file: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError("$", f"invalid JSON: {exc.msg} at line {exc.lineno}") from exc
    return parse_spec(raw, source_path=str(spec_path.resolve()))

