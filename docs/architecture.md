# 推荐排序离线评测与 A/B 归因工作台

面向本地确定性样本的离线评测工作台，覆盖分桶分配、分组指标统计、提升量与置信
区间、分层归因、内容指纹增量复用、阶段快照与异常注入复现。全程仅依赖 Python 3.11
标准库与 NumPy，无外部账号、无容器、无网络依赖（仅首次引导 pip 时访问 PyPI）。

## 目录结构

```
ab_workbench/
  errors.py        业务错误码与确定性失败记录
  fixedpoint.py    定点整数运算与量化档位
  fingerprint.py   BLAKE2b 内容指纹与缓存键派生
  hashing.py       确定性分桶哈希与分组映射
  config.py        实验配置模型与加载
  loader.py        JSONL 逐行加载与严格校验
  injection.py     异常注入（真实输入故障）
  stats.py         分组指标累加（并发/串行同序）
  estimation.py    点估计、置信区间、胜负判定
  attribution.py   分层归因、加权汇总与漂移修正
  cache.py         内容指纹缓存与精确失效
  snapshot.py      阶段原子快照与续跑校验
  pipeline.py      端到端编排（批处理、信号、报告）
  generator.py     固定种子样本生成器
  edits.py         确定性编辑操作（增量序列）
  cli.py           命令行入口
scripts/
  run_demo.py          端到端演示（并发+串行）
  equivalence_check.py 增量 vs 全量等价性矩阵
  serial_compare.py    串行对照摘要
  full_rebuild.py      全量重建入口
tests/               unittest 测试套件（25 个用例）
```

## 确定性保证

- 分桶：`BLAKE2b(seed|user_id)` 摘要转大整数后对桶数取模，组区间按整数权重连续切分。
- 累加顺序：请求 ID 字典序，再候选物品 ID 字典序；并发任务携带
  `stable_sort_key(stage, id)`，部分结果按相同规范序合并。
- 定点：所有比率/提升/区间端点内部为 `10**decimal_width` 缩放整数；区间下界向下取整、
  上界向上取整，保证覆盖；量化档位再向量化网格对齐。
- 指纹：规范化 JSON（排序键、固定分隔符）后 BLAKE2b，不依赖 `PYTHONHASHSEED`。
- 失败清单：按 `(stage, code, request_id, item_id, group, reason, injection_id, detail)`
  稳定排序，线程调度不影响顺序。

## 四个阶段与缓存键

| 阶段 | 上游指纹 | 作用域 | 记录的事件集合 |
| --- | --- | --- | --- |
| bucketing | 属性表指纹 + 配置指纹 | all-users | 用户集合 |
| stats | 分组事件指纹 + 配置指纹 | 每个分组 | 该组请求 ID |
| estimation | 对照/实验组统计指纹 + 配置指纹 | 每个实验组 | 两组请求 ID |
| attribution | 属性键 + 全量事件指纹 + 配置指纹 | all-keys | 全部请求 ID |

缓存键 = `stage:BLAKE2b(stage, upstream, config, caliber, scope)`。新增曝光或单条修正
仅令引用了受影响请求 ID 的分组产物失效；桶数/位宽/配置变更改变配置指纹，派生产物整体
失效；未命中失效的产物直接复用且不重算。失效、复用、新算清单在报告 `cache` 字段列出。

## 指标与统计口径

- 主指标 `ctr`（点击/曝光），护栏 `cvr`（转化/曝光），样本量为曝光数，同时输出请求数。
- 方差估计（两独立比例，不合池）：
  `var = p_t(1-p_t)/n_t + p_c(1-p_c)/n_c`，正态分位数取配置常量
  （默认 z=1.959963984540054，95%）。
- 最小样本量：`n >= min_sample_size` 即足够，恰好相等在报告 `boundary` 显式标注
  `exactly_at_threshold:sufficient`；不足时抑制提升展示并标记 `insufficient_sample`。
- 胜负：按指标方向，置信区间相对最小提升阈值判定 `win/tie/lose`。

## 分层归因与漂移

按每个分桶属性键分层，逐层计算差异与区间；层权重为该层对照+实验曝光占比，使用最大
余数法保证权重之和精确等于定点单位 1；份额 = 层提升 × 层权重（定点截断）。当份额之和
与整体提升偏差超过 `attribution_tolerance` 时输出 `E_ATTRIBUTION_DRIFT`，并按确定顺序
回填截断余数，使汇总精确等于整体提升，逐层修正明细写入 `correction`。

## 快照与异常

- 每个阶段先写临时目录再 `os.replace` 整体替换；清单 `manifest.json` 记录全局版本号、
  配置指纹、口径版本、已完成批次与各阶段产物版本。
- 配置指纹变化：合法配置变更走失效重建并开启新快照世系；仅异常注入的陈旧快照复用记录
  `E_CONFIG_MISMATCH`。
- 注入故障：非有限分数、分组越界、点击标记非 0/1 在加载层真实改坏记录后由校验管线产出
  `E_LOG_INVALID`；桶数变化后旧快照复用产出 `E_CONFIG_MISMATCH`。命中注入项编号写入
  `injection_hits`。
- 信号：SIGTERM/SIGINT 完成当前批次快照后以退出码 0 退出；强制终止后只能恢复已完整
  替换的快照。

## 业务错误码

| 码 | 触发场景 |
| --- | --- |
| E_LOG_INVALID | 日志字段非法、分数非有限、分组越界、标记非二元、时间戳非单调、重复条目等 |
| E_HASH_CONFIG_INVALID | 桶数非正、权重非正/超桶、种子非法 |
| E_CONFIG_MISMATCH | 陈旧快照复用、配置/口径指纹不一致 |
| E_ATTRIBUTION_DRIFT | 分层加权汇总偏离整体提升超过容差 |
