"""四个处理阶段的纯函数：特征哈希 / 成对打分 / 边界判定 / 批次汇总。

所有函数无副作用、不读取全局随机状态，输出完全由输入与配置决定，
因此同一 (偏好对, 配置, 档位) 多次运行逐整数一致。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal, localcontext
from typing import Any

from .config import AppConfig
from .fixedpoint import requant, to_fixed, fixed_to_str
from .hashing import BucketVector, bucket_vector, sha256_hex
from .pairs import PreferencePair

# 内部统一在 8 位定点精度计算，再按配置档位 requant，使 quant_mode 可观测
INTERNAL_PLACES = 8

STABLE = "stable"
CRITICAL = "critical"
CONFLICT = "conflict"
LABEL_RANK = {STABLE: 0, CRITICAL: 1, CONFLICT: 2}


# --------------------------------------------------------------------------
# 阶段 1：特征哈希
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureResult:
    pair_id: str
    prompt_vec: BucketVector
    chosen_vec: BucketVector
    rejected_vec: BucketVector
    cache_key: str
    referenced_buckets: tuple[int, ...]

    def bucket_digest(self) -> str:
        return sha256_hex([
            self.prompt_vec.digest, self.chosen_vec.digest, self.rejected_vec.digest,
        ])

    def summary(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "prompt": self.prompt_vec.to_dict(),
            "chosen": self.chosen_vec.to_dict(),
            "rejected": self.rejected_vec.to_dict(),
            "feature_key": self.cache_key,
            "referenced_buckets": list(self.referenced_buckets),
        }


def feature_cache_key(pair: PreferencePair, cfg: AppConfig) -> str:
    return sha256_hex([
        "feature", pair.content_fingerprint(), cfg.tokenizer_version,
        str(cfg.normalize), str(cfg.hash_buckets),
    ])


def run_feature(pair: PreferencePair, cfg: AppConfig) -> FeatureResult:
    vecs = []
    bucket_refs: set[int] = set()
    for text in (pair.prompt, pair.chosen, pair.rejected):
        vec = bucket_vector(text, cfg.tokenizer_version, cfg.normalize, cfg.hash_buckets)
        vecs.append(vec)
        bucket_refs.update(b for b, _ in vec.counts)
    key = feature_cache_key(pair, cfg)
    return FeatureResult(
        pair_id=pair.pair_id,
        prompt_vec=vecs[0], chosen_vec=vecs[1], rejected_vec=vecs[2],
        cache_key=key,
        referenced_buckets=tuple(sorted(bucket_refs)),
    )


# --------------------------------------------------------------------------
# 阶段 2：成对定点奖励打分
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreResult:
    pair_id: str
    chosen_score: int       # 配置档位下的定点整数
    rejected_score: int
    diff: int
    chosen_score_str: str
    rejected_score_str: str
    diff_str: str
    decomposition: dict[str, int]  # 逐槽位 l1 计数与定点贡献
    cache_key: str
    upstream_feature_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def weights_fingerprint(cfg: AppConfig) -> str:
    w8 = cfg.weights.to_fixed(INTERNAL_PLACES)
    return sha256_hex([
        "weights", str(w8["prompt"]), str(w8["chosen"]), str(w8["rejected"]),
        str(cfg.decimal_places), cfg.quant_mode,
    ])


def score_cache_key(feat: FeatureResult, cfg: AppConfig) -> str:
    return sha256_hex([
        "score", feat.bucket_digest(), weights_fingerprint(cfg),
        cfg.tokenizer_version, str(cfg.hash_buckets),
    ])


def _slot_score(vec: BucketVector, weight8: int) -> int:
    """固定运算顺序：按 bucket 升序逐项累加 weight*count。"""
    total = 0
    for _bucket, count in vec.counts:  # counts 已按 bucket 排序
        total += weight8 * count
    return total


def run_score(pair: PreferencePair, feat: FeatureResult, cfg: AppConfig) -> ScoreResult:
    w8 = cfg.weights.to_fixed(INTERNAL_PLACES)
    # 内部 8 位定点
    prompt8 = _slot_score(feat.prompt_vec, w8["prompt"])
    chosen_term8 = _slot_score(feat.chosen_vec, w8["chosen"])
    rejected_term8 = _slot_score(feat.rejected_vec, w8["rejected"])
    chosen8 = prompt8 + chosen_term8
    rejected8 = prompt8 + rejected_term8
    diff8 = chosen8 - rejected8

    # 量化到配置档位
    chosen_q = requant(chosen8, INTERNAL_PLACES, cfg.decimal_places, cfg.quant_mode)
    rejected_q = requant(rejected8, INTERNAL_PLACES, cfg.decimal_places, cfg.quant_mode)
    diff_q = chosen_q - rejected_q

    decomposition = {
        "prompt_l1": feat.prompt_vec.l1(),
        "chosen_l1": feat.chosen_vec.l1(),
        "rejected_l1": feat.rejected_vec.l1(),
        "prompt_contrib_fixed8": prompt8,
        "chosen_contrib_fixed8": chosen_term8,
        "rejected_contrib_fixed8": rejected_term8,
        "diff_fixed8": diff8,
    }
    return ScoreResult(
        pair_id=pair.pair_id,
        chosen_score=chosen_q,
        rejected_score=rejected_q,
        diff=diff_q,
        chosen_score_str=fixed_to_str(chosen_q, cfg.decimal_places),
        rejected_score_str=fixed_to_str(rejected_q, cfg.decimal_places),
        diff_str=fixed_to_str(diff_q, cfg.decimal_places),
        decomposition=decomposition,
        cache_key=score_cache_key(feat, cfg),
        upstream_feature_key=feat.cache_key,
    )


# --------------------------------------------------------------------------
# 阶段 3：边界样本判定
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BoundaryResult:
    pair_id: str
    label: str
    diff: int
    threshold: int
    at_threshold: bool
    tie_rank: int          # 恰等于阈值时按 pair_id 字典序的决断序号
    cache_key: str
    upstream_score_key: str
    referenced_pairs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def boundary_cache_key(score: ScoreResult, cfg: AppConfig) -> str:
    threshold_fixed = to_fixed(cfg.boundary_threshold, cfg.decimal_places)
    return sha256_hex([
        "boundary", score.cache_key, str(threshold_fixed), str(cfg.decimal_places),
    ])


def classify(diff: int, threshold: int) -> tuple[str, bool]:
    at_threshold = abs(diff) == threshold
    if diff > threshold:
        return STABLE, at_threshold
    if diff < -threshold:
        return CONFLICT, at_threshold
    return CRITICAL, at_threshold


def run_boundary(pair: PreferencePair, score: ScoreResult, cfg: AppConfig,
                 tie_rank: int) -> BoundaryResult:
    threshold = to_fixed(cfg.boundary_threshold, cfg.decimal_places)
    label, at_threshold = classify(score.diff, threshold)
    return BoundaryResult(
        pair_id=pair.pair_id,
        label=label,
        diff=score.diff,
        threshold=threshold,
        at_threshold=at_threshold,
        tie_rank=tie_rank,
        cache_key=boundary_cache_key(score, cfg),
        upstream_score_key=score.cache_key,
        referenced_pairs=(pair.pair_id,),
    )


def resolve_tie_ranks(pairs: list[PreferencePair], scores: dict[str, ScoreResult],
                      cfg: AppConfig) -> dict[str, int]:
    """对恰等于阈值（正/负）的临界样本按 pair_id 字典序给出决断序号。"""
    threshold = to_fixed(cfg.boundary_threshold, cfg.decimal_places)
    on_edge = sorted(
        p.pair_id for p in pairs
        if abs(scores[p.pair_id].diff) == threshold
    )
    return {pid: idx for idx, pid in enumerate(on_edge)}


# --------------------------------------------------------------------------
# 阶段 4：批次级汇总诊断
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupSummary:
    batch_id: str
    annotator_id: str
    time_partition: int
    total: int
    stable: int
    critical: int
    conflict: int
    stable_ratio: str
    critical_ratio: str
    conflict_ratio: str
    mean_diff_str: str
    mean_diff_fixed: int
    pair_ids: tuple[str, ...]
    cache_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio(n: int, total: int) -> str:
    with localcontext() as ctx:
        ctx.prec = 40
        return str(Decimal(n) / Decimal(total))


def _mean_fixed(values: list[int]) -> int:
    total = sum(values)
    # 精确平均，ROUND_HALF_EVEN
    q, r = divmod(abs(total), len(values))
    if r == 0:
        return q if total >= 0 else -q
    half = len(values) // 2
    if r * 2 > len(values) or (r * 2 == len(values) and q % 2 == 1):
        q += 1
    return q if total >= 0 else -q


def summarize_group(batch_id: str, annotator_id: str, partition: int,
                    members: list[tuple[PreferencePair, BoundaryResult]],
                    cfg: AppConfig) -> GroupSummary:
    members = sorted(members, key=lambda m: m[0].pair_id)
    pair_ids = tuple(p.pair_id for p, _ in members)
    labels = [b.label for _, b in members]
    n_stable = labels.count(STABLE)
    n_critical = labels.count(CRITICAL)
    n_conflict = labels.count(CONFLICT)
    total = len(members)
    diffs = [b.diff for _, b in members]
    mean_fixed = _mean_fixed(diffs)
    upstream = sorted(b.cache_key for _, b in members)
    key = sha256_hex(
        ["summary", batch_id, annotator_id, str(partition), str(cfg.time_partition_ms)]
        + upstream
    )
    return GroupSummary(
        batch_id=batch_id,
        annotator_id=annotator_id,
        time_partition=partition,
        total=total,
        stable=n_stable,
        critical=n_critical,
        conflict=n_conflict,
        stable_ratio=_ratio(n_stable, total),
        critical_ratio=_ratio(n_critical, total),
        conflict_ratio=_ratio(n_conflict, total),
        mean_diff_str=fixed_to_str(mean_fixed, cfg.decimal_places),
        mean_diff_fixed=mean_fixed,
        pair_ids=pair_ids,
        cache_key=key,
    )


def summarize_batch(batch_id: str, pairs: list[PreferencePair],
                    boundaries: dict[str, BoundaryResult],
                    cfg: AppConfig) -> list[GroupSummary]:
    """按 (标注者, 时间分区) 汇总；同分按标注者 ID 字典序、再按分区排序。"""
    groups: dict[tuple[str, int], list[tuple[PreferencePair, BoundaryResult]]] = {}
    for pair in sorted(pairs, key=lambda p: p.pair_id):
        partition = pair.event_ms // cfg.time_partition_ms
        groups.setdefault((pair.annotator_id, partition), []).append(
            (pair, boundaries[pair.pair_id])
        )
    result = [
        summarize_group(batch_id, ann, part, members, cfg)
        for (ann, part), members in groups.items()
    ]
    # 同分按标注者字典序，再按时间分区
    result.sort(key=lambda g: (g.annotator_id, g.time_partition, g.batch_id))
    return result
