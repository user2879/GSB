"""Incremental preview engine and optional Tkinter window.

The headless engine is always available and is the single source feeding both
the optional GUI and the E_PREVIEW guard.  Tkinter is imported lazily so the
platform runs on headless machines (including CI/WSL without python3-tk).
"""
from __future__ import annotations

from typing import Any

from .convergence import build_authoritative_preview, verify_preview
from .errors import BusinessError, ErrorCode
from .models import ManifestVersion
from .fixedpoint import Grid


class PreviewEngine:
    """Holds the last authoritative preview and an editable UI cache."""

    def __init__(self) -> None:
        self.authoritative: dict[str, Any] = {}
        self.ui_cache: dict[str, Any] = {}
        self.last_version: int | None = None

    def refresh(self, version: ManifestVersion, plan: dict[str, Any],
                grid: Grid) -> dict[str, Any]:
        preview = build_authoritative_preview(version, plan, grid)
        self.authoritative = preview
        self.ui_cache = _deep_copy(preview)
        self.last_version = version.version
        return preview

    def simulate_ui_drift(self, node_id: str, shard_index: int,
                         field: str, corrupted_value: Any) -> None:
        """Test/demo helper: corrupt only the UI cache, never the authority."""
        self.ui_cache[node_id]["shards"][str(shard_index)]["derived_state"][field] = corrupted_value

    def verify_cache(self, version: ManifestVersion, plan: dict[str, Any],
                     grid: Grid) -> dict[str, Any]:
        return verify_preview(version, plan, self.ui_cache, grid)

    def diff_summary(self) -> list[dict[str, Any]]:
        diffs: list[dict[str, Any]] = []
        for node_id in sorted(self.authoritative):
            for shard in sorted(self.authoritative[node_id]["shards"]):
                expected = self.authoritative[node_id]["shards"][shard]
                actual = self.ui_cache.get(node_id, {}).get("shards", {}).get(shard, {})
                if expected != actual:
                    diffs.append({"node_id": node_id, "shard": shard,
                                  "expected": expected, "actual": actual})
        return diffs


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _deep_copy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(value) for value in obj]
    return obj


def tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401
        return True
    except Exception:
        return False


class PreviewWindow:
    """Optional Tkinter window; construction fails fast with a clear error."""

    def __init__(self, engine: PreviewEngine, title: str = "configmesh preview") -> None:
        if not tkinter_available():
            raise BusinessError(
                ErrorCode.E_INTERNAL,
                "tkinter unavailable; use headless preview output instead",
                [{"hint": "install python3-tk or run on the Windows official Python installer"}],
            )
        import tkinter as tk
        self.engine = engine
        self.root = tk.Tk()
        self.root.title(title)
        self.text = tk.Text(self.root, width=100, height=28)
        self.text.pack()
        self.render()

    def render(self) -> None:
        self.text.delete("1.0", "end")
        for node_id in sorted(self.engine.ui_cache):
            self.text.insert("end", f"node={node_id}\n")
            for shard, payload in sorted(self.engine.ui_cache[node_id]["shards"].items()):
                self.text.insert(
                    "end",
                    f"  shard {shard}: derived={payload['derived_state']} "
                    f"digest={payload['aggregate_digest'][:12]}...\n",
                )

    def run(self) -> None:
        self.root.mainloop()
