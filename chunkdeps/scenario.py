"""Local JSON scenario runner for repeatable upload demonstrations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .client import WorkbenchClient
from .errors import E_INPUT, BusinessError


class ScenarioRunner:
    def __init__(
        self,
        client: WorkbenchClient,
        *,
        base_dir: str | Path | None = None,
    ) -> None:
        self.client = client
        self.base_dir = Path(base_dir).resolve() if base_dir else Path.cwd()
        self.variables: dict[str, Any] = {}

    def run(self, scenario: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(scenario, dict):
            raise BusinessError(E_INPUT, "scenario must be a JSON object")
        steps = scenario.get("steps")
        if not isinstance(steps, list):
            raise BusinessError(E_INPUT, "scenario.steps must be a JSON array")
        defaults = scenario.get("defaults", {})
        if not isinstance(defaults, dict):
            raise BusinessError(E_INPUT, "scenario.defaults must be a JSON object")
        initial_vars = scenario.get("variables", {})
        if not isinstance(initial_vars, dict):
            raise BusinessError(E_INPUT, "scenario.variables must be a JSON object")
        self.variables.update(initial_vars)
        completed: list[dict[str, Any]] = []
        for position, raw_step in enumerate(steps, start=1):
            if not isinstance(raw_step, dict):
                raise BusinessError(E_INPUT, "scenario step must be a JSON object", {"step": position})
            step = self._resolve(raw_step)
            merged = {**defaults, **step}
            op = merged.get("op")
            if not isinstance(op, str) or not op:
                raise BusinessError(E_INPUT, "scenario step is missing op", {"step": position})
            try:
                result = self._execute(op, merged)
            except BusinessError as exc:
                exc.details.setdefault("scenario_step", position)
                exc.details.setdefault("scenario_op", op)
                raise
            alias = merged.get("as")
            if alias:
                if not isinstance(alias, str):
                    raise BusinessError(E_INPUT, "scenario step alias must be a string")
                self.variables[alias] = result
            completed.append(
                {
                    "step": position,
                    "op": op,
                    "ok": True,
                    "result": result,
                }
            )
        return {
            "ok": True,
            "step_count": len(completed),
            "variables": self.variables,
            "steps": completed,
        }

    def _resolve(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("$"):
            reference = value[1:]
            if reference not in self.variables:
                raise BusinessError(
                    E_INPUT,
                    "scenario variable is not defined",
                    {"variable": reference},
                )
            return self.variables[reference]
        if isinstance(value, list):
            return [self._resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve(item) for key, item in value.items()}
        return value

    def _source(self, value: Any) -> Path:
        if not isinstance(value, str) or not value:
            raise BusinessError(E_INPUT, "scenario source must be a non-empty string")
        path = Path(value)
        if not path.is_absolute():
            path = self.base_dir / path
        return path

    def _remember_session(self, result: dict[str, Any]) -> None:
        session = result.get("session")
        if isinstance(session, dict) and session.get("session_id"):
            self.variables["last_session"] = session["session_id"]
            self.variables["last_object"] = session.get("object_key")
        elif result.get("session_id"):
            self.variables["last_session"] = result["session_id"]
        if result.get("object_key"):
            self.variables["last_object"] = result["object_key"]
        if result.get("created_session"):
            self.variables["last_session"] = result["created_session"]["session_id"]
            self.variables["last_object"] = result["created_session"]["object_key"]
        if isinstance(result.get("version"), dict):
            self.variables["last_version"] = result["version"]["version"]

    def _execute(self, op: str, step: dict[str, Any]) -> dict[str, Any]:
        if op == "upload_file":
            source = self._source(step.get("source"))
            concurrency = int(step.get("concurrency", 4))
            result = self.client.upload_file(
                object_key=str(step.get("object_key", "")),
                source=source,
                chunk_size=int(step.get("chunk_size", 0)),
                tags=step.get("tags") or {},
                digest_algorithm=str(step.get("digest_algorithm", "sha256")),
                concurrency=concurrency,
                expected_base_version=step.get("expected_base_version"),
                commit=bool(step.get("commit", True)),
                inject=step.get("inject"),
            )
            self._remember_session(result)
            return result
        if op == "create_session":
            result = self.client.create_session(
                object_key=str(step.get("object_key", "")),
                source=self._source(step.get("source")),
                chunk_size=int(step.get("chunk_size", 0)),
                tags=step.get("tags") or {},
                digest_algorithm=str(step.get("digest_algorithm", "sha256")),
                expected_base_version=step.get("expected_base_version"),
            )
            self.variables["last_session"] = result["session_id"]
            self.variables["last_object"] = result["object_key"]
            return result
        if op in {"upload_session", "upload_chunks"}:
            session_id = str(step.get("session_id", self.variables.get("last_session", "")))
            source = self._source(step.get("source"))
            reupload = step.get("indices")
            result = self.client.upload_session(
                session_id=session_id,
                source=source,
                concurrency=int(step.get("concurrency", 4)),
                reupload_indices=[int(item) for item in reupload] if reupload is not None else None,
                inject=step.get("inject"),
            )
            self._remember_session(result)
            return result
        if op == "reupload_chunk":
            session_id = str(step.get("session_id", self.variables.get("last_session", "")))
            source = self._source(step.get("source"))
            result = self.client.upload_session(
                session_id=session_id,
                source=source,
                concurrency=1,
                reupload_indices=[int(step.get("chunk_index", 0))],
                inject=step.get("inject"),
            )
            self._remember_session(result)
            return result
        if op in {"abort", "abort_session"}:
            session_id = str(step.get("session_id", self.variables.get("last_session", "")))
            result = self.client.request(
                "abort_session",
                {"session_id": session_id, "reason": step.get("reason", "scenario abort")},
            )
            self._remember_session(result)
            return result
        if op in {"resume", "resume_session"}:
            session_id = str(step.get("session_id", self.variables.get("last_session", "")))
            result = self.client.request("resume_session", {"session_id": session_id})
            self._remember_session(result)
            return result
        if op in {"commit", "commit_session"}:
            session_id = str(step.get("session_id", self.variables.get("last_session", "")))
            params: dict[str, Any] = {"session_id": session_id}
            if step.get("expected_base_version") is not None:
                params["expected_base_version"] = step["expected_base_version"]
            if step.get("inject") is not None:
                params["inject"] = step["inject"]
            result = self.client.request("commit_session", params)
            self._remember_session(result)
            return result
        if op in {"delete", "delete_object"}:
            object_key = str(step.get("object_key", self.variables.get("last_object", "")))
            return self.client.request("delete_object", {"object_key": object_key})
        if op == "audit":
            return self.client.request("audit", {})
        if op in {"snapshot_save", "snapshot"}:
            params = {"path": step["path"]} if step.get("path") else {}
            return self.client.request("snapshot_save", params)
        if op == "snapshot_restore":
            params = {"path": step["path"]} if step.get("path") else {}
            return self.client.request("snapshot_restore", params)
        if op == "views":
            params = {"prefix": step["prefix"]} if step.get("prefix") else {}
            return self.client.request("list_views", params)
        if op == "graph":
            return self.client.request(
                "get_graph", {"include_events": bool(step.get("include_events", False))}
            )
        if op == "invalidations":
            return self.client.request("invalidations", {"limit": int(step.get("limit", 20))})
        if op == "sessions":
            return self.client.request("list_sessions", {})
        if op == "objects":
            return self.client.request("list_objects", {})
        if op == "status":
            return self.client.request("status", {})
        if op == "recovery":
            return self.client.request("recovery_status", {})
        if op == "rpc":
            rpc_op = step.get("rpc_op")
            if not isinstance(rpc_op, str) or not rpc_op:
                raise BusinessError(E_INPUT, "rpc scenario step needs rpc_op")
            params = step.get("params", {})
            if not isinstance(params, dict):
                raise BusinessError(E_INPUT, "rpc scenario params must be an object")
            return self.client.request(rpc_op, params)
        raise BusinessError(E_INPUT, "unknown scenario operation", {"op": op})


def load_scenario(path: str | Path) -> dict[str, Any]:
    scenario_path = Path(path)
    try:
        payload = json.loads(scenario_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BusinessError(
            E_INPUT,
            "scenario file is not readable JSON",
            {"path": str(scenario_path)},
        ) from exc
    if not isinstance(payload, dict):
        raise BusinessError(E_INPUT, "scenario root must be a JSON object")
    return payload

