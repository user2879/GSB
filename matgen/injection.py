"""Explicit anomaly injection with per-item isolation."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import E_INJECTION_CONFIG, InjectedFailure, Issue, make_issue


@dataclass(frozen=True)
class Injection:
    injection_id: str
    stage: str
    action: str
    expected_code: str
    target: str = ""
    target_layer: int | None = None
    target_material: str = ""
    delay_ms: int = 0


@dataclass
class InjectionEvent:
    injection_id: str
    expected_code: str
    observed_codes: list[str] = field(default_factory=list)
    triggered: int = 0
    matched: bool = False


def load_injection_config(path: str | Path | None) -> tuple[list[Injection], list[Issue]]:
    if path is None:
        return [], []
    source = str(path)
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [], [make_issue(E_INJECTION_CONFIG, f"cannot load injection config: {exc}", source=source)]
    if not isinstance(raw, dict) or not isinstance(raw.get("injections"), list):
        return [], [make_issue(E_INJECTION_CONFIG, "injections must be a list", source=source)]
    injections: list[Injection] = []
    issues: list[Issue] = []
    for index, item in enumerate(raw["injections"]):
        if not isinstance(item, dict):
            issues.append(make_issue(E_INJECTION_CONFIG, "injection must be an object", source=source, item=str(index)))
            continue
        injection_id = item.get("id")
        stage = item.get("stage")
        action = item.get("action")
        expected = item.get("expected_code")
        if not isinstance(injection_id, str) or not injection_id or not isinstance(stage, str):
            issues.append(make_issue(E_INJECTION_CONFIG, "injection id and stage are required", source=source, item=str(index)))
            continue
        if not isinstance(expected, str) or not expected.startswith("E_"):
            issues.append(make_issue(E_INJECTION_CONFIG, "expected_code must be an E_xxx code", source=source, item=injection_id))
            continue
        target_layer = item.get("target_layer")
        if target_layer is not None and (isinstance(target_layer, bool) or not isinstance(target_layer, int)):
            issues.append(make_issue(E_INJECTION_CONFIG, "target_layer must be an integer", source=source, item=injection_id))
            continue
        delay_ms = item.get("delay_ms", 0)
        if isinstance(delay_ms, bool) or not isinstance(delay_ms, int) or delay_ms < 0:
            issues.append(make_issue(E_INJECTION_CONFIG, "delay_ms must be a non-negative integer", source=source, item=injection_id))
            continue
        injections.append(
            Injection(
                injection_id=injection_id,
                stage=stage,
                action=str(action),
                expected_code=expected,
                target=str(item.get("target", "")),
                target_layer=target_layer,
                target_material=str(item.get("target_material", "")),
                delay_ms=delay_ms,
            )
        )
    return injections, issues


class InjectionRegistry:
    def __init__(self, injections: list[Injection] | None = None, *, apply_delays: bool = True) -> None:
        self.injections = list(injections or [])
        self.apply_delays = apply_delays
        self._lock = threading.Lock()
        self._events = {
            injection.injection_id: InjectionEvent(
                injection_id=injection.injection_id,
                expected_code=injection.expected_code,
            )
            for injection in self.injections
        }
        self._layer_delay_done: set[int] = set()
        self._layer_raise_done: set[int] = set()
        self._material_raise_done: set[str] = set()

    def reset_build(self) -> None:
        with self._lock:
            self._layer_delay_done.clear()
            self._layer_raise_done.clear()
            self._material_raise_done.clear()

    def _record(self, injection: Injection, code: str) -> None:
        event = self._events[injection.injection_id]
        event.triggered += 1
        if code not in event.observed_codes:
            event.observed_codes.append(code)
        event.matched = injection.expected_code in event.observed_codes

    def record_observed(self, code: str, *, target: str = "", stage: str = "") -> None:
        with self._lock:
            for injection in self.injections:
                target_match = not injection.target or injection.target == target
                stage_match = not injection.stage or injection.stage == stage
                if target_match and stage_match and injection.action.startswith("load_"):
                    self._record(injection, code)

    def record_code_for_action(self, code: str, *, action: str, stage: str = "") -> None:
        with self._lock:
            for injection in self.injections:
                if injection.action != action:
                    continue
                if stage and injection.stage != stage:
                    continue
                self._record(injection, code)

    def before_layer(self, layer_index: int) -> None:
        for injection in self.injections:
            if injection.stage != "layer_noise" or injection.target_layer != layer_index:
                continue
            if injection.action == "delay_ms":
                if not self.apply_delays:
                    continue
                with self._lock:
                    first = layer_index not in self._layer_delay_done
                    self._layer_delay_done.add(layer_index)
                if first and injection.delay_ms:
                    time.sleep(injection.delay_ms / 1000.0)
                with self._lock:
                    self._events[injection.injection_id].triggered += 1
            elif injection.action in {"raise", "fail"}:
                with self._lock:
                    first = layer_index not in self._layer_raise_done
                    self._layer_raise_done.add(layer_index)
                    if first:
                        self._record(injection, injection.expected_code)
                raise InjectedFailure(
                    injection.expected_code,
                    f"injected failure at noise layer {layer_index}",
                    injection.injection_id,
                )

    def before_material(self, material_id: str) -> None:
        for injection in self.injections:
            if injection.stage != "material_mix":
                continue
            if injection.target_material and injection.target_material != material_id:
                continue
            if injection.action in {"raise", "fail"}:
                with self._lock:
                    first = material_id not in self._material_raise_done
                    self._material_raise_done.add(material_id)
                    if first:
                        self._record(injection, injection.expected_code)
                raise InjectedFailure(
                    injection.expected_code,
                    f"injected failure at material {material_id}",
                    injection.injection_id,
                )

    def record_event(self, injection: Injection, code: str) -> None:
        with self._lock:
            self._record(injection, code)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": event.injection_id,
                    "expected_code": event.expected_code,
                    "observed_codes": list(event.observed_codes),
                    "triggered": event.triggered,
                    "matched": event.matched,
                }
                for _, event in sorted(self._events.items())
            ]

    def report(self) -> dict[str, Any]:
        events = self.events()
        return {
            "injection_count": len(self.injections),
            "events": events,
            "all_triggered_codes_matched": all(
                event["matched"] for event in events if event["triggered"] > 0
            )
            if events
            else True,
            "untouched_injections": [event["id"] for event in events if event["triggered"] == 0],
        }
