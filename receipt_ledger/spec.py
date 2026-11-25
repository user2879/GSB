from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError
from .models import MergeRules, RequestSpec, SpecBundle, WorkerSpec


WORKERS_FILE = "workers.json"
REQUESTS_FILE = "requests.json"
RULES_FILE = "rules.json"


def is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _escape_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def path_to_pointer(path: Iterable[str | int]) -> str:
    parts = [str(part) for part in path]
    return "" if not parts else "/" + "/".join(_escape_pointer_token(part) for part in parts)


def display_field(path: Iterable[str | int]) -> str:
    text = ""
    for part in path:
        if isinstance(part, int) or (isinstance(part, str) and part.isdigit() and text):
            text += f"[{part}]"
        elif text:
            text += f".{part}"
        else:
            text = str(part)
    return text or "$"


def _tokenize_json(text: str) -> list[tuple[str, str, int]]:
    tokens: list[tuple[str, str, int]] = []
    index = 0
    line = 1
    length = len(text)
    while index < length:
        char = text[index]
        if char in " \t\r":
            index += 1
            continue
        if char == "\n":
            line += 1
            index += 1
            continue
        start_line = line
        if char == '"':
            start = index
            index += 1
            while index < length:
                current = text[index]
                if current == "\\" and index + 1 < length:
                    if text[index + 1] == "\n":
                        line += 1
                    index += 2
                    continue
                if current == '"':
                    index += 1
                    break
                if current == "\n":
                    line += 1
                index += 1
            tokens.append(("value", text[start:index], start_line))
            continue
        if char in "{}[]:,":
            tokens.append(("punct", char, line))
            index += 1
            continue
        if char == "-" or char.isdigit():
            start = index
            while index < length and text[index] in "+-0123456789.eE":
                index += 1
            tokens.append(("scalar", text[start:index], start_line))
            continue
        if char.isalpha():
            start = index
            while index < length and text[index].isalpha():
                index += 1
            tokens.append(("scalar", text[start:index], start_line))
            continue
        index += 1
    return tokens


def build_line_map(text: str) -> dict[str, int]:
    """Map JSON Pointer paths to source lines without third-party packages."""

    tokens = _tokenize_json(text)
    lines: dict[str, int] = {"": 1}
    stack: list[dict[str, Any]] = []
    token_index = 0

    def current_frame() -> dict[str, Any] | None:
        return stack[-1] if stack else None

    def child_pointer_for_value(token_line: int) -> str | None:
        frame = current_frame()
        if frame is None:
            lines[""] = token_line
            return ""
        if frame["kind"] == "object":
            key = frame.get("pending_key")
            if key is None:
                return None
            pointer = f"{frame['ptr']}/{_escape_pointer_token(key)}"
            frame["pending_key"] = None
            lines[pointer] = token_line
            return pointer
        pointer = f"{frame['ptr']}/{frame['index']}"
        frame["index"] += 1
        lines[pointer] = token_line
        return pointer

    while token_index < len(tokens):
        kind, value, token_line = tokens[token_index]
        if kind == "punct" and value in "{[":
            pointer = child_pointer_for_value(token_line)
            if pointer is None:
                token_index += 1
                continue
            lines[pointer] = token_line
            stack.append(
                {"ptr": pointer, "kind": "object" if value == "{" else "array", "index": 0, "pending_key": None}
            )
        elif kind == "punct" and value in "}]":
            if stack:
                stack.pop()
        elif kind == "value":
            frame = current_frame()
            next_is_colon = (
                token_index + 1 < len(tokens)
                and tokens[token_index + 1][0] == "punct"
                and tokens[token_index + 1][1] == ":"
            )
            if frame is not None and frame["kind"] == "object" and next_is_colon:
                try:
                    key = json.loads(value)
                except json.JSONDecodeError:
                    key = value.strip('"')
                frame["pending_key"] = str(key)
                pointer = f"{frame['ptr']}/{_escape_pointer_token(str(key))}"
                lines[pointer] = token_line
            else:
                child_pointer_for_value(token_line)
        elif kind == "scalar":
            child_pointer_for_value(token_line)
        token_index += 1
    return lines


@dataclass
class LoadedJson:
    data: Any
    text: str
    path: Path
    line_map: dict[str, int]

    def line_for(self, path: Iterable[str | int]) -> int:
        pointer = path_to_pointer(path)
        if pointer in self.line_map:
            return self.line_map[pointer]
        parts = pointer.split("/")[1:]
        while parts:
            candidate = "/" + "/".join(parts)
            if candidate in self.line_map:
                return self.line_map[candidate]
            parts.pop()
        return 1


def load_json_file(path: str | Path) -> LoadedJson:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"无法读取 JSON 文件: {exc}", file=str(file_path), field="$", line=1) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SpecError(
            "JSON 语法损坏，无法强校验",
            file=str(file_path),
            line=exc.lineno,
            column=exc.colno,
            field="$",
        ) from exc
    return LoadedJson(data=data, text=text, path=file_path, line_map=build_line_map(text))


class ErrorCollector:
    def __init__(self, loaded: LoadedJson) -> None:
        self.loaded = loaded
        self.errors: list[SpecError] = []

    def reject(self, message: str, path: list[str | int]) -> None:
        self.errors.append(
            SpecError(
                message,
                file=str(self.loaded.path),
                line=self.loaded.line_for(path),
                field=display_field(path),
                json_pointer=path_to_pointer(path),
            )
        )

    def raise_if_any(self) -> None:
        if not self.errors:
            return
        first = min(self.errors, key=lambda err: (err.details.get("line", 1), str(err.details.get("field", ""))))
        first.details["all_error_count"] = len(self.errors)
        raise first


def _require_object(loaded: LoadedJson) -> ErrorCollector | None:
    collector = ErrorCollector(loaded)
    if not isinstance(loaded.data, dict):
        collector.reject("顶层 JSON 必须是对象", [])
        collector.raise_if_any()
    return collector


def load_workers(path: str | Path) -> tuple[WorkerSpec, ...]:
    loaded = load_json_file(path)
    collector = _require_object(loaded)
    assert collector is not None
    data = loaded.data
    raw_workers = data.get("workers") if isinstance(data, dict) else None
    if not isinstance(raw_workers, list):
        collector.reject("workers 必须是数组", ["workers"])
        collector.raise_if_any()

    workers: list[WorkerSpec] = []
    seen_ids: dict[str, int] = {}
    seen_ports: dict[int, int] = {}
    assert isinstance(raw_workers, list)
    for index, raw in enumerate(raw_workers):
        base = ["workers", index]
        if not isinstance(raw, dict):
            collector.reject("工作进程声明必须是对象", base)
            continue

        worker_id = raw.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            collector.reject("worker_id 必须是非空字符串", [*base, "worker_id"])
        elif worker_id in seen_ids:
            collector.reject(
                f"worker_id 重复，首次出现于行 {seen_ids[worker_id]}",
                [*base, "worker_id"],
            )
        else:
            seen_ids[worker_id] = loaded.line_for([*base, "worker_id"])

        host = raw.get("host")
        if host != "127.0.0.1":
            collector.reject("host 必须是 127.0.0.1 字面量", [*base, "host"])

        port = raw.get("port")
        if not is_plain_int(port) or not (1024 <= int(port) <= 65535):
            collector.reject("port 必须是 1024-65535 的整数高位端口", [*base, "port"])
        elif port in seen_ports:
            collector.reject(
                f"回环端口全局重复，首次出现于行 {seen_ports[port]}",
                [*base, "port"],
            )
        else:
            seen_ports[int(port)] = loaded.line_for([*base, "port"])

        model_dim = raw.get("model_dim")
        if not is_plain_int(model_dim) or model_dim <= 0:
            collector.reject("model_dim 必须是正整数", [*base, "model_dim"])

        hidden_dim = raw.get("hidden_dim")
        if not is_plain_int(hidden_dim) or hidden_dim <= 0:
            collector.reject("hidden_dim 必须是正整数", [*base, "hidden_dim"])

        seed = raw.get("seed")
        if not is_plain_int(seed):
            collector.reject("seed 必须是整数", [*base, "seed"])

        param_version = raw.get("param_version")
        if not is_plain_int(param_version) or param_version <= 0:
            collector.reject("param_version 必须是正整数序号", [*base, "param_version"])

        quota = raw.get("quota")
        if not is_plain_int(quota) or quota <= 0:
            collector.reject("quota 必须是正整数", [*base, "quota"])

        batch_size = raw.get("batch_size")
        if batch_size is not None and (not is_plain_int(batch_size) or batch_size <= 0):
            collector.reject("batch_size 若提供必须是正整数", [*base, "batch_size"])

        worker_valid = (
            isinstance(worker_id, str)
            and bool(worker_id)
            and host == "127.0.0.1"
            and is_plain_int(port)
            and 1024 <= int(port) <= 65535
            and is_plain_int(model_dim)
            and model_dim > 0
            and is_plain_int(hidden_dim)
            and hidden_dim > 0
            and is_plain_int(seed)
            and is_plain_int(param_version)
            and param_version > 0
            and is_plain_int(quota)
            and quota > 0
            and (batch_size is None or (is_plain_int(batch_size) and batch_size > 0))
        )
        if worker_valid:
            workers.append(
                WorkerSpec(
                    worker_id=worker_id,
                    host=host,
                    port=int(port),
                    model_dim=int(model_dim),
                    hidden_dim=int(hidden_dim),
                    seed=int(seed),
                    param_version=int(param_version),
                    quota=int(quota),
                    batch_size=int(batch_size) if is_plain_int(batch_size) and batch_size > 0 else None,
                )
            )
    collector.raise_if_any()
    return tuple(sorted(workers, key=lambda worker: worker.worker_id))


def load_rules(path: str | Path) -> MergeRules:
    loaded = load_json_file(path)
    collector = _require_object(loaded)
    assert collector is not None
    data = loaded.data
    assert isinstance(data, dict)

    def positive_int(key: str, default: int | None = None) -> int | None:
        value = data.get(key, default)
        if value is None or not is_plain_int(value) or value <= 0:
            collector.reject(f"{key} 必须是正整数", [key])
            return None
        return int(value)

    batch_size = positive_int("default_batch_size", 2)
    global_concurrency = positive_int("global_concurrency", 2)
    ttl = positive_int("lease_ttl_ms", 5_000)
    renew = positive_int("lease_renew_ms", 1_000)
    algorithm = data.get("digest_algorithm", "sha256-v1")
    if algorithm != "sha256-v1":
        collector.reject("digest_algorithm 仅支持 sha256-v1", ["digest_algorithm"])
    collector.raise_if_any()
    assert batch_size and global_concurrency and ttl and renew
    return MergeRules(batch_size, global_concurrency, ttl, renew, str(algorithm))


def load_requests(path: str | Path, workers: tuple[WorkerSpec, ...]) -> tuple[RequestSpec, ...]:
    loaded = load_json_file(path)
    collector = _require_object(loaded)
    assert collector is not None
    data = loaded.data
    raw_requests = data.get("requests") if isinstance(data, dict) else None
    if not isinstance(raw_requests, list):
        collector.reject("requests 必须是数组", ["requests"])
        collector.raise_if_any()

    worker_map = {worker.worker_id: worker for worker in workers}
    seen_request_ids: dict[str, int] = {}
    seen_arrival_seqs: dict[int, int] = {}
    requests: list[RequestSpec] = []
    assert isinstance(raw_requests, list)

    for index, raw in enumerate(raw_requests):
        base = ["requests", index]
        if not isinstance(raw, dict):
            collector.reject("请求必须是对象", base)
            continue

        request_id = raw.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            collector.reject("request_id 必须是非空字符串", [*base, "request_id"])
        elif request_id in seen_request_ids:
            collector.reject(
                f"request_id 重复，首次出现于行 {seen_request_ids[request_id]}",
                [*base, "request_id"],
            )
        else:
            seen_request_ids[request_id] = loaded.line_for([*base, "request_id"])

        arrival_seq = raw.get("arrival_seq")
        if not is_plain_int(arrival_seq) or arrival_seq < 0:
            collector.reject("arrival_seq 必须是非负整数", [*base, "arrival_seq"])
        elif arrival_seq in seen_arrival_seqs:
            collector.reject(
                f"arrival_seq 全局重复，首次出现于行 {seen_arrival_seqs[arrival_seq]}",
                [*base, "arrival_seq"],
            )
        else:
            seen_arrival_seqs[int(arrival_seq)] = loaded.line_for([*base, "arrival_seq"])

        worker_id = raw.get("worker_id")
        worker = worker_map.get(str(worker_id)) if isinstance(worker_id, str) else None
        if worker is None:
            collector.reject("worker_id 必须指向已声明工作进程", [*base, "worker_id"])

        vector = raw.get("input_vector")
        if not isinstance(vector, list) or not vector:
            collector.reject("input_vector 必须是非空整数数组", [*base, "input_vector"])
        else:
            if any(not is_plain_int(item) for item in vector):
                collector.reject("input_vector 的每个元素必须是整数", [*base, "input_vector"])
            elif worker is not None and len(vector) != worker.model_dim:
                collector.reject(
                    f"input_vector 长度 {len(vector)} 与 model_dim {worker.model_dim} 不一致",
                    [*base, "input_vector"],
                )

        if (
            isinstance(request_id, str)
            and request_id
            and is_plain_int(arrival_seq)
            and arrival_seq >= 0
            and worker is not None
            and isinstance(vector, list)
            and vector
            and all(is_plain_int(item) for item in vector)
            and len(vector) == worker.model_dim
        ):
            requests.append(
                RequestSpec(
                    request_id=request_id,
                    worker_id=worker.worker_id,
                    arrival_seq=int(arrival_seq),
                    vector=tuple(int(item) for item in vector),
                )
            )

    collector.raise_if_any()
    return tuple(sorted(requests, key=lambda request: request.arrival_seq))


def load_bundle(directory: str | Path) -> SpecBundle:
    folder = Path(directory)
    workers = load_workers(folder / WORKERS_FILE)
    requests = load_requests(folder / REQUESTS_FILE, workers)
    rules = load_rules(folder / RULES_FILE)
    return SpecBundle(workers=workers, requests=requests, rules=rules, source_dir=str(folder))
