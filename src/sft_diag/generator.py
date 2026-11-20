"""确定性候选样本生成器。

随机源：numpy PCG64（平台无关、序列稳定），种子由种子/槽位/配置/词表
指纹与任务类型混合派生。同一输入与配置下多次运行产出完全一致的候选
样本序列与采样序号。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from .models import WorkspaceState
from .utils import derive_seed, fingerprint, stable_id


class CandidateGenerator:
    def __init__(self, state: WorkspaceState) -> None:
        self.state = state
        self.fingerprints = state.fingerprints()
        # 采样随机性只依赖采样相关配置，显式排除 max_workers / 快照间隔 /
        # 去重参数等非采样字段，保证并发度调整不改变候选序列。
        sampling_config = {
            "task_types": list(state.config.task_types),
            "target_counts": dict(state.config.target_counts),
            "template_weights": state.config.to_dict()["template_weights"],
            "slot_weights": state.config.to_dict()["slot_weights"],
        }
        self.sampling_config_fp = fingerprint(sampling_config)

    def _rng(self, task_type: str, purpose: str) -> np.random.Generator:
        seed = derive_seed(
            self.fingerprints["seeds_fp"],
            self.fingerprints["slots_fp"],
            self.sampling_config_fp,
            self.fingerprints["vocab_fp"],
            task_type,
            purpose,
        )
        return np.random.Generator(np.random.PCG64(seed))

    @staticmethod
    def _weighted_choice(
        rng: np.random.Generator,
        ids: list[str],
        weights: dict[str, float],
    ) -> str:
        probabilities = np.array(
            [max(float(weights.get(identifier, 0.0)), 0.0) for identifier in ids],
            dtype=np.float64,
        )
        total = probabilities.sum()
        probabilities = probabilities / total
        return ids[int(rng.choice(len(ids), p=probabilities))]

    def generate_task(self, task_type: str) -> list[dict[str, Any]]:
        config = self.state.config
        target = config.target_counts.get(task_type, 0)
        seeds = self.state.seeds_by_task(task_type)
        template_table = config.template_weights.get(task_type, {})
        slot_table = config.slot_weights.get(task_type, {})

        template_rng = self._rng(task_type, "template")
        slot_rng = self._rng(task_type, "slot")
        template_ids = [seed.template_id for seed in seeds]
        seed_by_template = {seed.template_id: seed for seed in seeds}
        template_occurrence: dict[str, int] = defaultdict(int)
        candidates: list[dict[str, Any]] = []

        for order in range(target):
            template_id = self._weighted_choice(
                template_rng, template_ids, template_table
            )
            occurrence = template_occurrence[template_id]
            template_occurrence[template_id] += 1
            seed = seed_by_template[template_id]

            slot_picks: dict[str, dict[str, Any]] = {}
            for slot_name in seed.slots:
                sources = self.state.sources_by_task_slot(task_type, slot_name)
                source_ids = [source.source_id for source in sources]
                source_id = self._weighted_choice(
                    slot_rng, source_ids, slot_table
                )
                source = next(item for item in sources if item.source_id == source_id)
                value_index = int(
                    slot_rng.integers(0, len(source.values), endpoint=False)
                )
                slot_picks[slot_name] = {
                    "source_id": source_id,
                    "value_index": value_index,
                    "value": source.values[value_index],
                }
            values = {name: pick["value"] for name, pick in slot_picks.items()}
            instruction = seed.template.format(**values)
            answer = seed.answer_template.format(**values)
            candidates.append(
                {
                    "candidate_id": stable_id(f"{task_type}", order),
                    "task_type": task_type,
                    "sample_order": order,
                    "template_id": template_id,
                    "template_occurrence": occurrence,
                    "seed_id": seed.seed_id,
                    "slot_sources": {
                        name: {
                            "source_id": pick["source_id"],
                            "value_index": pick["value_index"],
                        }
                        for name, pick in slot_picks.items()
                    },
                    "slot_values": values,
                    "instruction": instruction,
                    "reference_answer": answer,
                }
            )
        return candidates

    def generate_all(self) -> list[dict[str, Any]]:
        all_candidates: list[dict[str, Any]] = []
        for task_type in self.state.config.task_types:
            all_candidates.extend(self.generate_task(task_type))
        return all_candidates


def build_reference_index(
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """记录每个模板 / 槽位来源 -> 引用它的候选 ID 集合，供精确失效判定。"""
    index: dict[str, dict[str, list[str]]] = {
        "template": defaultdict(list),
        "slot_source": defaultdict(list),
    }
    for candidate in candidates:
        index["template"][candidate["template_id"]].append(candidate["candidate_id"])
        for pick in candidate["slot_sources"].values():
            index["slot_source"][pick["source_id"]].append(candidate["candidate_id"])
    return {
        kind: {key: sorted(values) for key, values in table.items()}
        for kind, table in index.items()
    }
