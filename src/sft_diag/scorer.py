"""多维度确定性质量打分器。

四个维度：clarity（清晰度）、correctness（答案正确性）、format（格式合规性）、
length（长度合规性）。所有运算使用 IEEE-754 float64，无隐式随机或平台相关
分支；同一候选与参数表多次打分逐位一致。
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .models import WorkspaceState
from .tokenizer import Tokenizer
from .utils import derive_cache_key, float64, format_sig, round_sig


class QualityScorer:
    def __init__(self, state: WorkspaceState, tokenizer: Tokenizer) -> None:
        self.state = state
        self.params = state.score_params
        self.tokenizer = tokenizer
        fps = state.fingerprints()
        self.params_fp = fps["params_fp"]
        self.vocab_fp = fps["vocab_fp"]
        self.seeds_fp = fps["seeds_fp"]

    def cache_key(self, candidate: dict[str, Any]) -> str:
        """打分缓存键：候选内容指纹 + 参数/词表/种子指纹派生。"""
        content_payload = {
            "candidate_id": candidate["candidate_id"],
            "instruction": candidate["instruction"],
            "reference_answer": candidate["reference_answer"],
            "slot_values": candidate["slot_values"],
        }
        from .utils import fingerprint

        return derive_cache_key(
            "score",
            fingerprint(content_payload),
            self.params_fp,
            self.vocab_fp,
            self.seeds_fp,
        )

    def _clarity(self, candidate: dict[str, Any], baseline: float) -> float:
        instruction = candidate["instruction"]
        tokens = self.tokenizer.tokenize(instruction)
        coverage = self.tokenizer.vocab_coverage(instruction)
        token_count = max(len(tokens), 1)
        unique_ratio = len(set(tokens)) / token_count
        ends_clean = 1.0 if instruction.rstrip().endswith((".", "?", "!", "。", "？", "！")) else 0.0
        raw = 0.5 * coverage + 0.3 * unique_ratio + 0.2 * ends_clean
        return float64(0.8 * raw + 0.2 * baseline)

    def _correctness(self, candidate: dict[str, Any], baseline: float) -> float:
        answer = candidate["reference_answer"]
        values = candidate["slot_values"]
        expected_terms = sorted(values.values())
        present = all(term in answer for term in expected_terms)
        no_placeholder = "{" not in answer and "}" not in answer
        raw = 1.0 if present and no_placeholder else (0.5 if present else 0.0)
        return float64(0.85 * raw + 0.15 * baseline)

    def _format(self, candidate: dict[str, Any], baseline: float) -> float:
        instruction = candidate["instruction"]
        answer = candidate["reference_answer"]
        no_placeholder = "{" not in instruction and "{" not in answer
        non_empty = bool(instruction.strip()) and bool(answer.strip())
        balanced = instruction.count('"') % 2 == 0 and answer.count('"') % 2 == 0
        raw = (float(no_placeholder) + float(non_empty) + float(balanced)) / 3.0
        return float64(0.75 * raw + 0.25 * baseline)

    def _length(self, candidate: dict[str, Any], baseline: float) -> float:
        lower = self.params.length_lower
        upper = self.params.length_upper
        token_count = len(self.tokenizer.tokenize(candidate["instruction"]))
        if lower <= token_count <= upper:
            raw = 1.0
        elif token_count < lower:
            raw = max(0.0, 1.0 - (lower - token_count) / max(lower, 1))
        else:
            raw = max(0.0, 1.0 - (token_count - upper) / max(upper, 1))
        return float64(0.9 * raw + 0.1 * baseline)

    def score_one(self, candidate: dict[str, Any]) -> dict[str, Any]:
        specs = self.params.specs
        dimensions = {
            "clarity": self._clarity(candidate, specs["clarity"].baseline),
            "correctness": self._correctness(candidate, specs["correctness"].baseline),
            "format": self._format(candidate, specs["format"].baseline),
            "length": self._length(candidate, specs["length"].baseline),
        }
        total = float64(
            math.fsum(
                float64(dimensions[dim] * specs[dim].weight)
                for dim in self.params.dimensions
            )
        )
        return {
            "candidate_id": candidate["candidate_id"],
            "cache_key": self.cache_key(candidate),
            "dimensions": {
                dim: round_sig(float64(dimensions[dim])) for dim in dimensions
            },
            "dimensions_text": {
                dim: format_sig(float64(dimensions[dim])) for dim in dimensions
            },
            "total_score": round_sig(total),
            "total_score_text": format_sig(total),
        }

    def score_all(
        self,
        candidates: list[dict[str, Any]],
        max_workers: int,
        cached: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """并发打分。cached 提供可复用结果（按 cache_key 命中）。

        返回 (按候选 ID 排序的打分结果, 候选ID -> reused/recomputed)。
        """
        cached = cached or {}
        results: list[dict[str, Any]] = [None] * len(candidates)  # type: ignore[list-item]
        status: dict[str, str] = {}
        workload: list[tuple[int, dict[str, Any]]] = []
        for index, candidate in enumerate(candidates):
            key = self.cache_key(candidate)
            if key in cached:
                results[index] = cached[key]
                status[candidate["candidate_id"]] = "reused"
            else:
                workload.append((index, candidate))
                status[candidate["candidate_id"]] = "recomputed"
        if workload:
            workers = max(1, min(max_workers, len(workload)))
            if workers == 1:
                for index, candidate in workload:
                    results[index] = self.score_one(candidate)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(self.score_one, candidate): index
                        for index, candidate in workload
                    }
                    for future in futures:
                        results[futures[future]] = future.result()
        ordered = sorted(results, key=lambda item: item["candidate_id"])
        return ordered, status
