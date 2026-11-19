from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import SimError


@dataclass
class Injection:
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Injection":
        injection_path = Path(path)
        try:
            with injection_path.open("r", encoding="utf-8") as handle:
                value = yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            raise SimError("E_INJECTION_INVALID", f"injection file not found: {injection_path}") from exc
        except yaml.YAMLError as exc:
            raise SimError("E_INJECTION_INVALID", f"invalid injection YAML: {exc}") from exc
        if not isinstance(value, dict):
            raise SimError("E_INJECTION_INVALID", "injection configuration must be a mapping")
        return cls(raw=value)

    def apply_nonfinite(
        self,
        gradients: dict[int, dict[str, np.ndarray]],
        window: int,
        window_start: int,
        window_end: int,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        entries = self.raw.get("nonfinite", []) or []
        if not isinstance(entries, list):
            raise SimError("E_INJECTION_INVALID", "nonfinite injection must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise SimError("E_INJECTION_INVALID", "nonfinite injection entries must be mappings")
            entry_window = entry.get("window")
            entry_step = entry.get("step")
            if entry_window is not None and int(entry_window) != window:
                continue
            if entry_step is not None and not window_start <= int(entry_step) < window_end:
                continue
            worker_id = int(entry.get("worker_id", entry.get("worker", 0)))
            tensor = str(entry.get("tensor", ""))
            if worker_id not in gradients or tensor not in gradients[worker_id]:
                continue
            array = gradients[worker_id][tensor]
            index = entry.get("index")
            value = float(entry.get("value", "nan"))
            if index is None:
                array[...] = np.float32(value)
            else:
                if not isinstance(index, list):
                    raise SimError("E_INJECTION_INVALID", "nonfinite index must be a list")
                array[tuple(int(item) for item in index)] = np.float32(value)
            actions.append(
                {
                    "window": window,
                    "worker_id": worker_id,
                    "tensor": tensor,
                    "index": index,
                    "value": value,
                    "injected": "nonfinite",
                }
            )
        return actions

    def communication_failure(self, task: dict[str, Any], window: int) -> str | None:
        entries = self.raw.get("communication_failures", self.raw.get("comm_failures", [])) or []
        if not isinstance(entries, list):
            raise SimError("E_INJECTION_INVALID", "communication_failures must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise SimError("E_INJECTION_INVALID", "communication failure entries must be mappings")
            if "window" in entry and int(entry["window"]) != window:
                continue
            if "origin_worker" in entry and int(entry["origin_worker"]) != int(task["origin_worker"]):
                continue
            if "src_holder" in entry and int(entry["src_holder"]) != int(task["src_holder"]):
                continue
            if "dst_worker" in entry and int(entry["dst_worker"]) != int(task["dst_worker"]):
                continue
            if "tensor" in entry and str(entry["tensor"]) != str(task["tensor"]):
                continue
            if "route_round" in entry and int(entry["route_round"]) != int(task["route_round"]):
                continue
            return str(entry.get("error_type", "InjectedCommunicationError"))
        return None

    def budget_overrides(self) -> dict[tuple[int, int], int]:
        raw_value = self.raw.get("link_budget_overrides", {}) or {}
        result: dict[tuple[int, int], int] = {}
        if isinstance(raw_value, dict):
            for key, value in raw_value.items():
                if isinstance(key, str) and "->" in key:
                    src_text, dst_text = key.split("->", 1)
                    result[(int(src_text), int(dst_text))] = int(value)
                elif isinstance(key, list) and len(key) == 2:
                    result[(int(key[0]), int(key[1]))] = int(value)
                else:
                    raise SimError("E_INJECTION_INVALID", f"invalid budget override key {key!r}")
        elif isinstance(raw_value, list):
            for entry in raw_value:
                if not isinstance(entry, dict):
                    raise SimError("E_INJECTION_INVALID", "budget override entries must be mappings")
                result[(int(entry["src"]), int(entry["dst"]))] = int(
                    entry.get("budget_bytes", entry.get("bandwidth_bytes"))
                )
        else:
            raise SimError("E_INJECTION_INVALID", "link_budget_overrides must be a mapping or list")
        return result

    def should_crash_after_phase(self, phase: str, window: int) -> bool:
        value = self.raw.get("crash_after_phase")
        if value is None:
            return False
        entries = value if isinstance(value, list) else [value]
        for entry in entries:
            if isinstance(entry, str) and entry == phase:
                return True
            if isinstance(entry, dict) and str(entry.get("phase")) == phase:
                if "window" not in entry or int(entry["window"]) == window:
                    return True
        return False

    def send_delay_seconds(self) -> float:
        value = self.raw.get("send_delay_ms", 0)
        try:
            seconds = float(value) / 1000.0
        except (TypeError, ValueError) as exc:
            raise SimError("E_INJECTION_INVALID", "send_delay_ms must be numeric") from exc
        if seconds < 0:
            raise SimError("E_INJECTION_INVALID", "send_delay_ms must be non-negative")
        return seconds
