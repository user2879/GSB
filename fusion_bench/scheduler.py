"""预算受限并行调度器。

确定性设计：线程池仅执行无副作用的纯计算；配额判定、全局预算扣减、
输出排序全部在提交阶段按样本 ID 稳定顺序串行完成，因此任意并发度与
任意预算耗尽时机下，预算扣减序列与最终产物与串行执行逐字段一致。
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .errors import PipelineError
from .features_image import extract_visual
from .features_text import encode_text
from .fusion import fuse


def sample_mac_cost(sample, cfg):
    """单样本乘加量（确定性闭式公式，与实际执行路径一致）。"""
    bs = sample.block_size
    n_blocks = sample.n_blocks
    c_stats = n_blocks * 5 * bs * bs
    c_proj = n_blocks * 5 * cfg.visual_dim
    n_assert = len(sample.assertions) if sample.assertions else 1
    c_text = (n_assert + 1) * cfg.text_dim
    c_fuse = cfg.visual_dim + cfg.text_dim + 16
    return c_stats + c_proj + c_text + c_fuse


def effective_quota(sample, cfg, plan):
    """单样本配额；zero_quota 注入将配额覆盖为 0。"""
    quota = cfg.per_sample_quota
    for inj in plan.for_sample(sample.sample_id, "zero_quota"):
        quota = 0
        plan.mark_hit(inj["id"])
    return quota


def compute_sample(sample, cfg, plan):
    """纯函数：特征抽取 + 融合打分。任务排序键 = (样本 ID, 阶段名)。"""
    feat_v, blocks, meta_v = extract_visual(sample, cfg, plan)
    vec_t, info_t = encode_text(sample.assertions, cfg)
    fused = fuse(feat_v, vec_t, cfg)
    return {
        "sample_id": sample.sample_id,
        "sort_keys": [(sample.sample_id, "feature_extract"),
                      (sample.sample_id, "fusion_score")],
        "visual_feature": feat_v,
        "visual_blocks": blocks,
        "visual_meta": meta_v,
        "text_feature": vec_t,
        "text_info": info_t,
        "fused": fused,
    }


def compute_parallel(samples, cfg, plan, max_workers):
    """线程池并发执行；返回 (结果字典, 失败清单, 完成顺序记录)。"""
    results = {}
    failures = []
    completion_order = []
    lock = threading.Lock()

    def run_one(sample):
        return compute_sample(sample, cfg, plan)

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(run_one, s): s for s in samples}
        for fut in as_completed(futures):
            sample = futures[fut]
            try:
                outcome = fut.result()
            except PipelineError as exc:
                with lock:
                    failures.append(exc.to_record(sample.sample_id))
            except Exception as exc:  # 未预期异常隔离：记录后继续其余样本
                with lock:
                    failures.append({
                        "sample_id": sample.sample_id,
                        "stage": "feature_extract",
                        "code": "E_SAMPLE_INVALID",
                        "message": f"unexpected: {exc!r}",
                        "injection_id": None,
                    })
            else:
                with lock:
                    results[sample.sample_id] = outcome
                    completion_order.append(sample.sample_id)
    return results, failures, completion_order
