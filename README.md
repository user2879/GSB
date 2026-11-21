# DriftBench：推荐排序离线评测与 A/B 口径一致性诊断工作台

DriftBench 是一个完全离线的确定性工作台，用于定位推荐排序指标在以下三类定义来源之间发生漂移时的变化、来源与影响范围：

- 曝光日志中的排序位次；
- 实验口径清单中的版本声明；
- 分桶属性与哈希种子造成的用户/请求分组差异。

它保留原始点击与转化标记不变，通过版本化指标、相邻口径漂移检测、固定顺序贡献分解、内容指纹缓存和阶段快照，给出可复现的诊断报告。

## 快速开始

要求 Python 3.11+ 与 NumPy（样本生成使用固定种子的 `numpy.random.Generator`）。

```bash
python3 -m driftbench generate --out samples
python3 -m driftbench all --sample-dir samples --output-dir artifacts --snapshot-dir artifacts/snapshots
python3 -m unittest discover -s tests -v
```

`all` 一次完成编辑序列、逐批增量/全量比较、五类故障注入与并发一致性验证，并写出：

- `artifacts/final_report.json`：最终诊断报告；
- `artifacts/verification.json`：逐批增量与全量重建指纹；
- `artifacts/injection_report.json`：异常注入命中证据；
- `artifacts/concurrency_verification.json`：串行/并行结果对比；
- `artifacts/snapshots/`：四阶段快照、原子 `current` 指针与清单。

常用命令：

```bash
python3 -m driftbench run --sample-dir samples --full --output /tmp/full.json
python3 -m driftbench simulate --sample-dir samples --snapshot-dir /tmp/snapshots
python3 -m driftbench injections --sample-dir samples --output /tmp/injections.json
```

## 确定性样本

固定种子 `20261122` 生成 72 个用户、64 个候选物品、480 条初始曝光，并在编辑计划中追加 24 条曝光，最终 504 条。72 个用户中 12 个故意缺失分桶属性，用于覆盖回退明细。三个口径版本为：

| 版本 | 截断位次 | 归一化 | 去重 | 过滤 |
|---|---:|---|---|---|
| v1 | 10 | none | 否 | 无 |
| v2 | 5 | rank_inverse | 否 | 无 |
| v3 | 5 | rank_inverse | 是 | `event_ms >= 1760002142000` |

编辑计划共 7 批，覆盖新增曝光、单条修正、v3/v2 反复切换和分桶种子变更。初始活动版本为 v2，最终为 v3。

## 核心口径

- 分桶：`sha256(seed, user_id, property_value) mod bucket_count`；偶数桶为 control，奇数桶为 treatment。
- 属性回退：`user_id_only` 用 `seed + user_id` 重新哈希；`fixed_group` 则按配置进入 `missing_attribute_group`。每次回退均记录 `user_id`、原因与策略。
- 曝光的 `experiment_group` 只做配置声明校验；指标分组始终由重新分桶决定，日志分组与派生分组不一致会单独计数并列出。
- 指标流水线固定为：过滤 → 截断 → 按用户去重 → 归一化权重。`ctr` 分母为样本权重、分子为点击权重；`cvr` 分母为点击权重、分子为点击且转化权重。
- 比率内部使用定点整数：`round_half_up(numerator * 10^places / denominator)`，再按 `quantization_step` 量化。报告同时给出 `value_fixed` 和固定小数位字符串。
- 事件累加顺序固定为 `(request_id, item_id)` 字典序；每个请求任务、分桶任务和来源任务都携带稳定排序键。
- 漂移对为活动口径版本与其相邻前序版本。显著判定为：**非边界**、`abs(delta) > drift_tolerance`、且受影响样本占比不低于阈值。差值恰等于容差时强制不显著，并输出 `boundary_samples`。
- 归因按 `attribution_order` 顺序逐项应用四类来源。每步记录中间指标；贡献采用相邻中间指标之差，因此精确望远镜求和到总差值。交换来源顺序会改变贡献时，仍按配置顺序决断，同时把相关来源列入 `order_sensitive_sources` 并输出 `E_ATTRIB_AMBIGUOUS`（`WARNING`），不会静默置零。

## 增量与快照

缓存分为 `bucketing → metrics → drift → attribution` 四个阶段。缓存键分别由用户/属性内容指纹、分组事件内容指纹、口径指纹与相关配置指纹派生。新增曝光只影响相关分组的下游；单条修正只影响其所在分组；切换活动版本会复用全部指标但重建漂移/归因；分桶种子变化会使分桶及全部下游失效。

每次运行精确记录复用和重算键，`reuse.actual_compute_calls` 必须等于 `reuse.recomputed` 数量。逐批验证会对每个状态另起一个无缓存全量引擎，并比较分桶、指标、漂移、归因和失败清单的规范化 JSON 指纹。

每阶段产物先写入版本临时目录，随后原子替换 `current` 符号链接，最后原子替换 `MANIFEST.json`。清单包含全局版本、配置指纹、口径集合与各阶段产物版本。启动时不匹配会输出 `E_SNAPSHOT_STALE`，不加载旧缓存并重建。

## 验证结果

本次交付实跑结果：

- `14` 项单元测试全部通过；
- 初始 + 7 批编辑共 8 个状态，增量与全量规范化指纹全部一致，`all_equal=true`；
- 最终活动版本 v3，漂移对 `v2 → v3`。control 漂移 `+0.099290`，treatment 漂移 `-0.014749`，两组均显著；
- 两组贡献和分别精确等于总差值，`residual_fixed=0`，目标复算校验通过；treatment 的 `dedup_count` 与 `sample_filter` 被标记为顺序敏感；
- 单条修正批次中，分桶复用，control 指标复用、treatment 的三版本指标重算，漂移与归因各复用一组、重算一组；
- 回到已计算过的 v3 时 11 个阶段产物全部复用、重算 0；
- 串行并发度 1 与配置并发度 4 的规范化结果摘要完全相同，实际最大活动线程数不超过配置值；
- 注入 `I01`–`I05` 全部命中，分别产生 `E_LOG_INVALID` / `E_CONFIG_INVALID` 及精确字段路径；
- 快照续跑测试中 11 个产物全部从磁盘复用；配置种子变化后输出 `E_SNAPSHOT_STALE` 并重建。

更完整的约定、数据结构与失效矩阵见 [docs/DESIGN.md](docs/DESIGN.md)。

## 项目结构

```text
driftbench/
  loading.py      输入解析、校验、错误收集
  bucketing.py    确定性分桶与回退明细
  metrics.py      固定点指标与请求集合指纹
  drift.py        相邻版本漂移与来源影响
  attribution.py  四来源贡献分解与顺序敏感性
  engine.py       四阶段缓存编排与报告
  snapshot.py     阶段快照、原子替换与陈旧检测
  simulation.py   编辑序列与逐批全量对比
  injections.py   五类隔离故障注入
  samplegen.py    固定种子样本生成器
tests/            14 项标准库 unittest
samples/          确定性输入与编辑/注入计划
artifacts/        本次实跑报告与快照
```
