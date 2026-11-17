"""固定种子生成三份同内容的 v1 / v2 / v3 管道样本文件。

缺失表达刻意不同以覆盖三版本语义差异：
- v1：缺失位 1（值仍写入千分比整数）
- v2：一部分缺失为缺键、一部分为显式 null
- v3：记录缺失标记位（值仍写入 float32）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import formats
from app.converters import round_half_up_permille
from app.ir import IR, FeatureSet, Step

SEED = 20261118
ENTITIES = [f"ent_{i:03d}" for i in range(12)]
FEATURES = ["f_age", "f_click", "f_price", "f_ctr", "f_depth"]

STEPS = [
    Step("normalize", {"features": ["f_age", "f_click", "f_price"]}),
    Step("cross", {"pairs": [["f_age", "f_click"]]}),
    Step("bucket", {"feature": "f_ctr", "boundaries": [0.25, 0.5, 0.75],
                    "output": "f_ctr_bucket"}),
]


def _is_missing(i: int, j: int) -> bool:
    return (i * 5 + j) % 11 == 0


def _base_values() -> dict[str, dict[str, float]]:
    import random
    rng = random.Random(SEED)
    return {e: {f: round(rng.random(), 6) for f in FEATURES} for e in ENTITIES}


def generate(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    values = _base_values()

    fs1 = FeatureSet()
    for i, entity in enumerate(ENTITIES):
        for j, feature in enumerate(FEATURES):
            v = values[entity][feature]
            fs1.values.setdefault(entity, {})[feature] = (
                round_half_up_permille(v) / 1000.0)
            fs1.precision.setdefault(entity, {})[feature] = "permille"
            if _is_missing(i, j):
                fs1.missing.setdefault(entity, set()).add(feature)
    ir1 = IR(features=fs1, steps=[Step(s.name) for s in STEPS], source_version="v1")
    (data_dir / "sample.v1.txt").write_text(formats.serialize_v1(ir1), encoding="utf-8")

    doc = {"version": 2, "entities": {}, "precision": {},
           "steps": [s.to_dict() for s in STEPS]}
    for i, entity in enumerate(ENTITIES):
        feats = {}
        prec = {}
        for j, feature in enumerate(FEATURES):
            if _is_missing(i, j):
                if (i + j) % 2 == 0:
                    continue
                feats[feature] = None
                continue
            feats[feature] = values[entity][feature]
            prec[feature] = "double"
        doc["entities"][entity] = feats
        doc["precision"][entity] = prec
    (data_dir / "sample.v2.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8")

    fs3 = FeatureSet()
    for i, entity in enumerate(ENTITIES):
        for j, feature in enumerate(FEATURES):
            fs3.values.setdefault(entity, {})[feature] = values[entity][feature]
            fs3.precision.setdefault(entity, {})[feature] = "float32"
            if _is_missing(i, j):
                fs3.missing.setdefault(entity, set()).add(feature)
    ir3 = IR(features=fs3, steps=[Step(s.name) for s in STEPS], source_version="v3")
    name_map = json.loads((Path(__file__).resolve().parent.parent
                           / "config" / "step_name_map.json").read_text("utf-8"))
    (data_dir / "sample.v3.bin").write_bytes(formats.serialize_v3(ir3, name_map))
    print(f"样本已生成到 {data_dir}（种子 {SEED}，{len(ENTITIES)} 实体 x "
          f"{len(FEATURES)} 特征，{len(STEPS)} 类步骤）")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent.parent / "data")
