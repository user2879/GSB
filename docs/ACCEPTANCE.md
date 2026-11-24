# Acceptance Record

验证日期：2026-11-24  
运行环境：Linux 反馈容器中的 Python 3.11 + NumPy 1.26.4 + PyYAML 6.0.1；实现仅使用 Python 标准库、NumPy 与 PyYAML，无容器或网络数据依赖，可直接在 Windows 10/11 64 位 Python 3.11/3.12 运行。

## 回归测试

```text
$ python -m pytest
..................                                                       [100%]
18 passed in 1.25s
```

覆盖点：

- 固定种子合成数据两次生成逐列一致；
- 训练分区缺失中位数填充、标准化/分位数确定性；
- `average_rank` 并列规则写入产物元数据；
- KMeans/DBSCAN/GMM 重复运行逐样本标签一致；
- GMM 四类协方差及奇点正则不抛异常；
- 空簇、全噪声告警与不可比指标；
- 排名并列按方法名与规范超参决定；
- 单候选配额与全局预算待办；
- 线程池并发严格不超过配置；
- 修改单列只失效引用该列的候选；
- 超参数变化复用预处理、重算下游；
- 交互超时返回 `E_RESULT_STALE`，随后可返回新结果；
- 交互响应延迟严格落在配置预算内；
- 任意命令流增量结果等于空白缓存全量重建；
- 快照自动间隔、恢复、版本不匹配检测。

## 端到端命令

```bash
python -m cluster_workbench \
  --config samples/config.yaml \
  --data samples/data.yaml \
  --commands samples/commands.yaml \
  --injections samples/injections.yaml \
  --workspace artifacts/final_run \
  --report artifacts/diagnostic_report.json \
  --responses artifacts/responses.json
```

结果摘要：

```text
command_count: 12
candidate_count: 9
completed: 6
failed: 3
pending: 0
best_candidate: dbscan-dense
best_silhouette: 0.5718315461601768
failure_count: 3
warning_count: 2
stale_responses: 9
snapshot_versions: 1, 2, 3
equivalence_equal: true
all_responses_within_budget: true
```

规范指纹：

```text
incremental_fingerprint = c58a5cfb8791e86863d63269364b0102c3c4ef02d498d73f6c6d18a85924758d
full_rebuild_fingerprint = c58a5cfb8791e86863d63269364b0102c3c4ef02d498d73f6c6d18a85924758d
```

## 预算与复用

```text
global_iteration_budget = 900
consumed_total = 62
remaining = 838
reserved_total = 172
remaining_after_reservations = 728
unaffected_recompute_points = 0
recomputed_candidates = [gmm-4]
fully_reused_candidates = [dbscan-dense, dbscan-main, gmm-3, km-3, km-4]
```

最终流水线只重算命令 11 修改过 `n_components` 的 `gmm-4` 及其下游；其余可比较候选直接复用。三个隔离失败候选没有消耗聚类预算。

## 注入隔离

```text
E_COLUMN_DESCRIPTION_MISSING  dbscan-legacy
E_NONFINITE_FEATURE           dbscan-spike
E_CLUSTER_COUNT_ZERO          km-bad
```

同时产生全局非有限值数据告警和 `gmm-3` 协方差正则告警；所有候选仍被报告，批处理未中断。

## 交互延迟

12 条命令全部满足 `within_budget: true`，且有 9 条返回带 `E_RESULT_STALE` 的旧摘要。命令 1、3、6、7、8、9、10、11、12 的 `result_command_seq` 依次指向其最近可用的已完成状态；命令 2、4、5 返回与当前状态一致的新摘要。实测延迟范围为 31.321–201.084 ms，对应运行期间的 250 ms 或 100 ms 配置预算；等待路径会预留响应组装与线程唤醒的尾延迟，不以阻塞等待突破预算。

## 快照

`artifacts/final_run/snapshots/versions` 中存在命令序号 4、8、12 对应的三个完整版本；`CURRENT` 指向最后版本。每个版本包含 `state.json`、`report.json`、`manifest.json`，恢复时校验清单与文件哈希。最终报告的 `snapshot_check` 显示 `recovered: true`、`report_matches_final_state: true` 且无恢复告警。

## 交付物

- `artifacts/diagnostic_report.json`：最终诊断报告，含全量候选、标签、预算、复用账本、注入报告、交互响应和等价检查；
- `artifacts/responses.json`：12 条命令的即时响应；
- `artifacts/final_run/`：持久缓存及三个可恢复快照版本；
- `samples/`：可直接运行的配置、数据描述、命令流与注入清单；
- `tests/`：18 项本地回归测试。
