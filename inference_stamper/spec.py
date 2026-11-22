"""本地 JSON 声明与令牌请求集的加载及强校验。

任一不合法都抛出 E_SPEC，指明首个违规标识与字段，
校验全部通过后才返回对象，绝不部分载入。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SpecError
from .parameters import SUPPORTED_QUANTS
from .prng import BitStream, mix_seed


def _is_int(value: Any) -> bool:
    # 显式拒绝 bool（bool 是 int 的子类）
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    model_dim: int
    hidden_dim: int
    seed: int
    quant: str
    identity_version: int
    token_spec_version: int


@dataclass(frozen=True)
class TokenRequest:
    token_id: str
    backend_id: str
    input_vector: tuple[float, ...]
    request_seq: int


@dataclass(frozen=True)
class Spec:
    backends: tuple[BackendSpec, ...]
    requests: tuple[TokenRequest, ...]
    source: str

    def backend(self, backend_id: str) -> BackendSpec:
        for backend in self.backends:
            if backend.backend_id == backend_id:
                return backend
        raise SpecError("后端引用不存在", item_id=backend_id,
                        field="backend_id")


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"声明文件不存在: {path}", field="file") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            "声明 JSON 语法损坏",
            field="json",
            detail={"line": exc.lineno, "column": exc.colno},
        ) from exc

    backends = _parse_backends(raw)
    requests = _parse_requests(raw, backends)
    return Spec(backends=backends, requests=requests, source=str(path))


def _parse_backends(raw: dict[str, Any]) -> tuple[BackendSpec, ...]:
    if not isinstance(raw, dict) or "backends" not in raw:
        raise SpecError("声明缺少 backends 节", field="backends")
    backend_list = raw["backends"]
    if not isinstance(backend_list, list) or not backend_list:
        raise SpecError("backends 必须是非空数组", field="backends")

    seen_ids: set[str] = set()
    parsed: list[BackendSpec] = []
    for entry in backend_list:
        if not isinstance(entry, dict):
            raise SpecError("后端声明必须是对象", field="backends")
        backend_id = entry.get("backend_id")
        if not isinstance(backend_id, str) or not backend_id:
            raise SpecError("后端标识缺失或非字符串",
                            item_id=str(backend_id), field="backend_id")
        if backend_id in seen_ids:
            raise SpecError("后端标识不唯一", item_id=backend_id,
                            field="backend_id")

        model_dim = entry.get("model_dim")
        if not _is_int(model_dim) or model_dim <= 0:
            raise SpecError("模型维度必须是正整数", item_id=backend_id,
                            field="model_dim")
        hidden_dim = entry.get("hidden_dim")
        if not _is_int(hidden_dim) or hidden_dim <= 0:
            raise SpecError("隐藏层维度必须是正整数", item_id=backend_id,
                            field="hidden_dim")
        seed = entry.get("seed")
        if not _is_int(seed):
            raise SpecError("随机种子必须是整数", item_id=backend_id,
                            field="seed")
        quant = entry.get("quant")
        if not isinstance(quant, str) or quant not in SUPPORTED_QUANTS:
            raise SpecError(
                f"量化口径不受支持，允许值: {', '.join(SUPPORTED_QUANTS)}",
                item_id=backend_id, field="quant",
            )
        identity_version = entry.get("identity_version")
        if not _is_int(identity_version) or identity_version < 0:
            raise SpecError("身份版本号必须是非负整数", item_id=backend_id,
                            field="identity_version")
        token_spec_version = entry.get("token_spec_version", 1)
        if not _is_int(token_spec_version) or token_spec_version < 1:
            raise SpecError("令牌口径版本必须是正整数", item_id=backend_id,
                            field="token_spec_version")

        seen_ids.add(backend_id)
        parsed.append(BackendSpec(
            backend_id=backend_id,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            seed=seed,
            quant=quant,
            identity_version=identity_version,
            token_spec_version=token_spec_version,
        ))
    return tuple(parsed)


def _parse_requests(
    raw: dict[str, Any],
    backends: tuple[BackendSpec, ...],
) -> tuple[TokenRequest, ...]:
    if "requests" not in raw:
        raise SpecError("声明缺少 requests 节", field="requests")
    request_list = raw["requests"]
    if not isinstance(request_list, list) or not request_list:
        raise SpecError("requests 必须是非空数组", field="requests")

    dims = {b.backend_id: b.model_dim for b in backends}
    seen_token_ids: set[str] = set()
    seen_seqs: set[int] = set()
    parsed: list[TokenRequest] = []

    for entry in request_list:
        if not isinstance(entry, dict):
            raise SpecError("令牌请求必须是对象", field="requests")
        token_id = entry.get("token_id")
        if not isinstance(token_id, str) or not token_id:
            raise SpecError("令牌标识缺失或非字符串",
                            item_id=str(token_id), field="token_id")
        if token_id in seen_token_ids:
            raise SpecError("令牌标识不唯一", item_id=token_id,
                            field="token_id")

        backend_id = entry.get("backend_id")
        if not isinstance(backend_id, str) or backend_id not in dims:
            raise SpecError("后端引用不存在", item_id=token_id,
                            field="backend_id")

        vector = entry.get("input_vector")
        if not isinstance(vector, list) or not vector:
            raise SpecError("输入向量必须是非空数组", item_id=token_id,
                            field="input_vector")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool)
               for x in vector):
            raise SpecError("输入向量元素必须是数字", item_id=token_id,
                            field="input_vector")
        if len(vector) != dims[backend_id]:
            raise SpecError(
                f"输入向量长度 {len(vector)} 与模型维度 "
                f"{dims[backend_id]} 不匹配",
                item_id=token_id, field="input_vector",
            )

        request_seq = entry.get("request_seq")
        if not _is_int(request_seq):
            raise SpecError("请求序号必须是整数", item_id=token_id,
                            field="request_seq")
        if request_seq in seen_seqs:
            raise SpecError("请求序号互不重复，发现重复序号",
                            item_id=token_id, field="request_seq")

        seen_token_ids.add(token_id)
        seen_seqs.add(request_seq)
        parsed.append(TokenRequest(
            token_id=token_id,
            backend_id=backend_id,
            input_vector=tuple(float(x) for x in vector),
            request_seq=request_seq,
        ))
    return tuple(parsed)


def deterministic_input_vector(backend_id: str, request_seq: int,
                               model_dim: int, master_seed: int) \
        -> tuple[float, ...]:
    """按固定整数种子确定性生成请求输入向量（[-1, 1) float32）。"""
    seed = mix_seed(master_seed, request_seq, model_dim)
    stream = BitStream(mix_seed(seed, 0x746F6B656E))
    values = stream.float32_array((model_dim,))
    return tuple(float(x) for x in values)


def write_sample_spec(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def input_vector_digest(vector: tuple[float, ...]) -> str:
    """输入向量的确定性摘要，供对账与展示使用。"""
    array = np.asarray(vector, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()
