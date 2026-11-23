"""热更新：按后端整体替换参数并递增参数版本序号，隔离新旧请求。

- 派发时通过 bind() 冻结 (param_version, param_digest)，在途请求用冻结摘要完成；
- 热更新后新派发请求绑定新版本；
- 同一批次出现两个参数版本时由 pipeline 以 E_MIXED 拒绝。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .spec import BackendDecl


def compute_param_digest(
    seed: int, model_dim: int, hidden_dim: int, quant_profile: str
) -> str:
    """由种子确定性生成参数矩阵，按量化口径量化后取摘要。"""
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((model_dim, hidden_dim))
    if quant_profile == "int8":
        quantized = np.round(matrix * 127).astype(np.int8)
    elif quant_profile == "int4":
        quantized = np.round(matrix * 7).astype(np.int8)
    elif quant_profile == "fp16":
        quantized = matrix.astype(np.float16)
    else:
        raise ValueError(f"不支持的量化口径: {quant_profile}")
    return hashlib.sha256(quantized.tobytes()).hexdigest()


@dataclass(frozen=True)
class ParamBinding:
    backend_id: str
    param_version: int
    param_digest: str


class ParamRegistry:
    """跟踪每个后端的当前参数版本与历史版本冻结摘要。"""

    def __init__(self, backends: dict[str, BackendDecl]):
        self._decls = dict(backends)
        self._versions: dict[str, int] = {}
        self._digests: dict[tuple[str, int], str] = {}
        for backend_id, decl in backends.items():
            self._versions[backend_id] = 1
            self._digests[(backend_id, 1)] = compute_param_digest(
                decl.seed, decl.model_dim, decl.hidden_dim, decl.quant_profile
            )

    def bind(self, backend_id: str) -> ParamBinding:
        version = self._versions[backend_id]
        return ParamBinding(
            backend_id=backend_id,
            param_version=version,
            param_digest=self._digests[(backend_id, version)],
        )

    def hot_update(self, backend_id: str, new_seed: int) -> ParamBinding:
        """整体替换后端参数：版本序号递增，旧版本摘要保持冻结。"""
        decl = self._decls[backend_id]
        new_version = self._versions[backend_id] + 1
        self._versions[backend_id] = new_version
        self._digests[(backend_id, new_version)] = compute_param_digest(
            new_seed, decl.model_dim, decl.hidden_dim, decl.quant_profile
        )
        return self.bind(backend_id)

    def digest_for(self, backend_id: str, version: int) -> str:
        return self._digests[(backend_id, version)]

    def active_versions(self) -> dict[str, int]:
        return dict(sorted(self._versions.items()))
