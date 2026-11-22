"""生成本地确定性声明与令牌请求集样本。

所有输入向量由固定整数种子 MASTER_SEED 派生，
不下载任何数据集或模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_stamper.spec import (  # noqa: E402
    deterministic_input_vector,
    write_sample_spec,
)

MASTER_SEED = 20261123

BACKENDS = [
    {
        "backend_id": "backend-alpha",
        "model_dim": 8,
        "hidden_dim": 12,
        "seed": 1001,
        "quant": "fp32",
        "identity_version": 1,
        "token_spec_version": 1,
    },
    {
        "backend_id": "backend-beta",
        "model_dim": 6,
        "hidden_dim": 10,
        "seed": 2002,
        "quant": "int8",
        "identity_version": 1,
        "token_spec_version": 1,
    },
    {
        "backend_id": "backend-gamma",
        "model_dim": 10,
        "hidden_dim": 16,
        "seed": 3003,
        "quant": "int4",
        "identity_version": 1,
        "token_spec_version": 1,
    },
]

REQUEST_PLAN = [
    ("token-001", "backend-alpha", 9001),
    ("token-002", "backend-alpha", 9002),
    ("token-003", "backend-alpha", 9003),
    ("token-004", "backend-beta", 9004),
    ("token-005", "backend-beta", 9005),
    ("token-006", "backend-beta", 9006),
    ("token-007", "backend-gamma", 9007),
    ("token-008", "backend-gamma", 9008),
]


def build_payload(backends: list[dict]) -> dict:
    dims = {b["backend_id"]: b["model_dim"] for b in backends}
    requests = []
    for token_id, backend_id, request_seq in REQUEST_PLAN:
        vector = deterministic_input_vector(
            backend_id=backend_id,
            request_seq=request_seq,
            model_dim=dims[backend_id],
            master_seed=MASTER_SEED,
        )
        requests.append({
            "token_id": token_id,
            "backend_id": backend_id,
            "input_vector": list(vector),
            "request_seq": request_seq,
        })
    return {"master_seed": MASTER_SEED, "backends": backends,
            "requests": requests}


def main() -> None:
    sample_dir = ROOT / "samples"
    write_sample_spec(sample_dir / "spec.base.json",
                      build_payload(BACKENDS))

    # 演进变体 1：alpha 身份版本 +1（模拟参数版本演进）
    evolved_identity = [dict(backend) for backend in BACKENDS]
    evolved_identity[0] = dict(evolved_identity[0])
    evolved_identity[0]["identity_version"] = 2
    write_sample_spec(sample_dir / "spec.evolved-identity.json",
                      build_payload(evolved_identity))

    # 演进变体 2：alpha 令牌口径版本 +1
    evolved_spec = [dict(backend) for backend in BACKENDS]
    evolved_spec[0] = dict(evolved_spec[0])
    evolved_spec[0]["token_spec_version"] = 2
    write_sample_spec(sample_dir / "spec.evolved-token-spec.json",
                      build_payload(evolved_spec))

    # 演进变体 3：beta 量化口径 fp32 -> fp16
    evolved_quant = [dict(backend) for backend in BACKENDS]
    evolved_quant[1] = dict(evolved_quant[1])
    evolved_quant[1]["quant"] = "fp16"
    write_sample_spec(sample_dir / "spec.evolved-quant.json",
                      build_payload(evolved_quant))

    # 损坏声明：gamma 维度被改为非正整数
    corrupt = [dict(backend) for backend in BACKENDS]
    corrupt[2] = dict(corrupt[2])
    corrupt[2]["hidden_dim"] = 0
    write_sample_spec(sample_dir / "spec.corrupt.json",
                      build_payload(corrupt))

    print(f"样本已写入 {sample_dir}")


if __name__ == "__main__":
    main()
