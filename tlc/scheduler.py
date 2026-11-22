"""失效任务调度：局部重训预算、待办保留与确定性并发训练。

- 每个失效任务分配其配置的局部重训预算（训练轮次*样本数）；
- 预算不足 -> 按任务稳定顺序保留待办，标记 E_RETRAIN_BUDGET_PENDING；
- 上一批待办任务优先处理；
- 线程池并发训练，最大并发数取自配置；结果按稳定排序键归并，
  与单线程按任务名+目标层名字典序执行的结果逐位一致。
"""

from concurrent.futures import ThreadPoolExecutor

from .adapter import train_adapter
from .errors import E_RETRAIN_BUDGET_PENDING


def stable_key(task):
    return task.sort_key()


def plan_training(candidates, cfg):
    """candidates: [(task, reason)]，按待办优先+稳定键排序传入。

    返回 (to_train, pending_entries)。
    """
    to_train, pending = [], []
    for task, reason in candidates:
        cost = cfg.epochs * len(task.train_ids)
        if cost > task.retrain_budget:
            pending.append(
                {
                    "task": task.name,
                    "reason": reason,
                    "required": cost,
                    "budget": task.retrain_budget,
                    "code": E_RETRAIN_BUDGET_PENDING,
                }
            )
        else:
            to_train.append(task)
    return to_train, pending


def train_all(to_train, encoder, bank, cfg, sensitivities, features_fingerprint):
    """并发训练；返回 {task_name: artifact}，顺序与并发度无关。"""
    if not to_train:
        return {}
    workers = min(cfg.max_concurrency, len(to_train))
    results = {}
    if workers <= 1:
        for task in sorted(to_train, key=stable_key):
            results[task.name] = train_adapter(
                task, encoder, bank, cfg, sensitivities, features_fingerprint
            )
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                train_adapter, task, encoder, bank, cfg, sensitivities,
                features_fingerprint,
            ): task
            for task in to_train
        }
        for fut, task in futures.items():
            results[task.name] = fut.result()
    return {name: results[name] for name in sorted(results, key=lambda n: (n,))}
