"""基于贡献偏移的失效边界判定。

禁止"引用即失效"：目标层未被替换但上游层被替换时，以替换前后
该任务各层贡献份额（归一化）的最大偏移与阈值比较；
判定结果给出参与比较的层与偏移值。
"""

from .adapter import content_key
from .encoder import normalized_shares


def decide(task, artifact, prev_shares, current_shares, cfg, features_fingerprint):
    """返回 (decision, detail)。

    decision ∈ {"new", "retrain", "reuse"}
    detail: 失效判定细节（参与比较的层与偏移值），不适用时为 None。
    """
    if artifact is None:
        return "new", None
    if artifact["content_key"] != content_key(task, cfg, features_fingerprint):
        return "retrain", {"reason": "content_changed", "compared": None,
                           "max_shift": None, "threshold": None}
    before = normalized_shares(prev_shares)
    after = normalized_shares(current_shares)
    compared = []
    max_shift = 0.0
    for layer in sorted(before):
        shift = abs(after[layer] - before[layer])
        compared.append(
            {"layer": layer, "previous": before[layer], "current": after[layer],
             "shift": shift}
        )
        if shift > max_shift:
            max_shift = shift
    detail = {
        "reason": "base_shift",
        "compared": compared,
        "max_shift": max_shift,
        "threshold": cfg.contribution_shift_threshold,
    }
    if max_shift > cfg.contribution_shift_threshold:
        return "retrain", detail
    return "reuse", detail
