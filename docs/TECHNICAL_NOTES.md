# 偏好对齐与奖励建模诊断工作台 · 技术说明

面向本地确定性样本的偏好对齐 / 奖励建模可视化与诊断工作台。所有计算在整数域完成、
不依赖任何运行时随机源，因此同一（偏好对集合, 配置, 量化档位）在任意机器、任意次
运行、串行或并发下逐字段一致。

## 1. 目录与模块

```
reward_workbench/
  errors.py       业务错误码与确定性失败清单（去重 + 固定排序）
  vocab.py        受控词表（版本化，加载校验与样本生成共用）
  fixedpoint.py   定点小数解析、量化档位（trunc/half_even/half_up）、位宽切换
  hashing.py      确定性子词切分(v1/v2)、SHA-256 稳定哈希、桶向量、内容指纹
  config.py       配置 YAML 加载校验（E_CONFIG_INVALID）、config_hash
  pairs.py        偏好对 JSONL 逐行加载校验（E_PAIR_INVALID）
  stages.py       四阶段纯函数：特征哈希/成对打分/边界判定/批次汇总
  cache.py        内容指纹缓存 + 对级精确失效（记录引用对集合与桶集合）
  executor.py     线程池执行器（稳定排序键，串行/并发同序回收）
  scheduler.py    全局预算/单批配额/待办保留（E_TRAIN_BUDGET_EXHAUSTED）
  snapshot.py     临时目录写入 + os.replace 整体提交 + E_SNAPSHOT_STALE 判定
  commands.py     命令流加载与声明式状态变更
  injection.py    四类真实输入故障注入（INJ-01..04）与业务码映射
  pipeline.py     增量流水线引擎（同一引擎服务增量运行与全量重建）
  report.py       JSON 明细 + 自包含 HTML（内联 SVG，无外部依赖）
  samplegen.py    固定种子 LCG 样本生成器
  cli.py          命令行：init/run/replay/inject/verify
samples/ config/ commands/   固定种子生成的输入样本
tests/            17 个 unittest，覆盖 8 项业务要求
runtime/          快照(snapshots)与临时(tmp)可写目录
reports/          diagnostics.json / diagnostics.html
```

## 2. 确定性设计要点

- **零随机源**：哈希全部走 `hashlib.sha256`，样本生成用固定 LCG（非 `random`、
  非 numpy 全局状态），打分只读取单个偏好对自身文本。
- **整数域打分**：权重/阈值用 `Decimal` 精确解析为定点整数，内部统一在 8 位定点
  精度按“桶升序逐项 weight*count 累加”的固定运算顺序求和，再按配置位宽与
  量化档位 `requant`，彻底规避浮点二进制误差。
- **稳定哈希到桶**：`SHA256(token)[:8] % hash_buckets`，跨平台一致。
- **稳定排序**：所有输出（偏好对、汇总、失败清单、边界清单）都按
  `pair_id / annotator_id / 时间分区 / 阶段名` 的确定性键排序。

### 定点与量化
- 允许位宽 `decimal_places ∈ {4,6,8}`，档位 `quant_mode ∈ {trunc, half_even, half_up}`。
- 奖励差值 `diff = chosen_q - rejected_q`（均为配置档位定点整数）。
- 边界判定：`diff > 阈值→stable`，`diff < -阈值→conflict`，其余（含恰等于 ±阈值）
  一律 `critical`；恰等于阈值的临界样本按 `pair_id` 字典序给 `tie_rank`。

## 3. 内容指纹与精确失效（缓存键派生链）

```
feature_key  = H("feature", pair内容指纹, tokenizer版本, normalize, hash_buckets)
score_key    = H("score", 三桶向量digest, 权重指纹(含位宽/档位), tokenizer版本, buckets)
boundary_key = H("boundary", score_key, 定点阈值, 位宽)
summary_key  = H("summary", batch, annotator, partition, 组内各 boundary_key)
```

每个产物记录其引用的偏好对集合与特征桶集合。变更影响面：

| 变更 | 失效范围 |
| --- | --- |
| 替换/删除某偏好对 | 仅该对的 feature→score→boundary，及引用它的 summary |
| 权重/阈值/档位/位宽/子词版本 | 全部对的下游阶段（feature 仅在子词版本/buckets 变化时失效）|
| 切换批次 | 该对及新旧两个批次的汇总（重算前整体清批次，杜绝孤儿汇总）|

未失效对直接命中缓存，不重新计算；`reuse_events` 与 `invalidation_events`
分别记录复用与摘除的 cache_key。

## 4. 预算与待办

- `global_pair_budget` 只约束“从未计费的对”，已计费对重算不重复扣减（账本持久）。
- `per_batch_quota` 限制单批处理对数；待办队列跨运行保留，下批优先处理。
- 全局预算耗尽：未计费对进入待办并产出 `E_TRAIN_BUDGET_EXHAUSTED`，保留游标。

## 5. 快照与恢复

- 阶段：`load / feature / score / boundary / summary`。
- 每阶段先写 `runtime/tmp/<staging>`，再 `os.replace` 原子整体提交到
  `runtime/snapshots/v<version>/`，不逐文件覆盖在线快照。
- 清单 `manifest.json` 记录全局版本、config_hash、数据指纹、已处理游标、
  已消耗预算与各阶段版本。
- 启动以“版本号 + config_hash + 数据指纹”判定：匹配则续跑；不匹配输出
  `E_SNAPSHOT_STALE`，从失败清单/空白重建而非覆盖。

## 6. 并发与异常注入

- `ThreadPoolExecutor(max_workers=cfg.max_workers)`，任务携带
  `stable_sort_key = 阶段::pair_id`，结果按排序键回收；单任务异常被捕获隔离进
  失败清单，调度顺序/并发度不改变最终顺序。
- 注入（独立真实畸形输入，非 mock）：
  - `INJ-01` 非有限权重 → `E_CONFIG_INVALID`
  - `INJ-02` 桶数 > 特征维度 → `E_CONFIG_INVALID`
  - `INJ-03` 负边界阈值 → `E_CONFIG_INVALID`
  - `INJ-04` 操作不存在的对 → `E_PAIR_NOT_FOUND`（状态保持不变）

## 7. 增量 vs 全量重建等价口径

- **数据产物**（桶向量摘要、定点得分/差值、边界判定、批次汇总顺序、缓存清单）
  在任意命令序列后的增量结果，与“用最终偏好对集合 + 最终配置从空白一次性全量
  计算”的结果逐字段相等（`replay` 与 `test_equivalence` 断言）。
- **运维历史**（预算账本、失效/复用事件、过程中的瞬时 `E_TRAIN_BUDGET_EXHAUSTED`、
  `E_SNAPSHOT_STALE`）本质是有状态演进记录，全量重建不会经历，因此单独列示，
  不纳入数据等价判定；输入类失败码（PAIR/CONFIG/COMMAND/NOT_FOUND）两侧仍要求一致。
- 全量重建以 `full_compute` 模式表达“最终数据的完整计算结果”，不受增量过程的
  预算截断影响。

## 8. Windows 10+ 运行方式

```powershell
py -3.11 -m pip install numpy==1.26.4 pyyaml==6.0.1
py -3.11 -m reward_workbench init --root .
py -3.11 -m reward_workbench run            # 全量
py -3.11 -m reward_workbench run --snapshot # 启用阶段快照
py -3.11 -m reward_workbench replay         # 命令流 + 增量/全量等价核验
py -3.11 -m reward_workbench verify         # 并发/串行一致
py -3.11 -m reward_workbench inject         # 异常注入
py -3.11 -m unittest discover -s tests      # 单元测试
```

输入均为 UTF-8、`event_ms` 为非负毫秒整数；快照/临时目录已在 `runtime/` 预留。

## 9. 实际验证结果（本次运行）

- `unittest discover`：**17/17 通过**。
- `replay`：`incremental_equals_full_rebuild = True`（15 条命令）。
- `verify`：并发 vs 串行逐字段一致（max_workers=4）。
- `inject`：INJ-01..04 全部命中预期业务码。
- 预算：默认全局预算 9、单批配额 4，首批各处理 4 对，剩余 4 对进入待办，
  耗尽时正确产出 `E_TRAIN_BUDGET_EXHAUSTED`；放大预算后待办清零且不重复计费。
- 非法样本文件正确隔离出空提示词、负时间戳、词表越界、重复 ID、坏 JSON 五类
  `E_PAIR_INVALID`，且不中断整批。
