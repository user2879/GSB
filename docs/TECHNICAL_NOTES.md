# 本地确定性精排与重排组件 — 技术说明

## 1. 设计目标与冲突权衡

组件面向本地确定性样本，核心矛盾是**交互延迟**与**计算结果一致性**：

- 延迟侧：每条交互命令必须在 `latency_budget_ms` 内返回，预算不足时返回
  `E_INTERACT_BUDGET_PENDING` 中间状态，绝不阻塞，并在后续命令中补交。
- 一致性侧：任意命令序列后的增量结果，必须与从空白工作区按最终配置一次性
  全量重建的结果**逐字段一致**。

解法是把计算拆成四个可独立缓存、内容寻址的阶段，并让缓存键由
“请求内容指纹 + 相关特征指纹 + 参数指纹”三段派生。命令只使受影响阶段及其
下游失效，未命中变化的阶段直接复用已持久化产物，从机制上保证增量与全量等价。

## 2. 目录结构

```
rerank_core/
  errors.py        业务码与异常（E_CONFIG_INVALID / E_RECORD_INVALID /
                   E_CONFIG_MISMATCH / E_INTERACT_BUDGET_PENDING）
  fixedpoint.py    精确 Fraction 解析、对称四舍五入、定点量化、整数开方、指纹
  models.py        候选、配置产物、贡献分解、折减记录等数据模型
  config.py        配置与特征分组表加载校验、配置指纹
  loader.py        JSONL 逐行解析与记录级校验、失败清单
  normalize.py     minmax / zscore 精确归一化与定点量化
  fusion.py        定点加权融合、温度、裁剪、同分精确决断
  rerank.py        定点余弦相似度、窗口内贪心 MMR 重排与定点折减
  cachekeys.py     请求/特征/参数指纹与三阶段缓存键派生
  snapshot.py      临时目录写入 + os.replace 整体提交 + 恢复判定
  pipeline.py      单请求阶段计算（串行与并发共用，稳定排序键）
  engine.py        失效判定、预算调度、线程池、快照、报告
  clock.py         单调时钟（实测）与可控时钟（测试）
  commands.py      五类交互命令 + poll，配置变更为纯函数
rerank_cli/        full / interactive 命令行入口
scripts/           样本生成、物化最终输入、故障注入、报告比对
tests/             pytest 套件（17 个用例）
```

## 3. 定点整数运算语义

全程不使用 float 参与运算：

1. JSON 数值解析为 `fractions.Fraction`，非有限值在加载期直接拒绝。
2. 归一化统计量（min/max、均值、总体方差、标准差）均为精确分数；
   `isqrt_frac` 对分子分母分别整数开方，z 分数不引入浮点。
3. 量化：`fixed = round_half_away(normalized * 10^frac_bits)`，
   并按 `quant_levels` 对归一化值做对称档位封顶。切换 `frac_bits` 或
   `quant_levels` 会真实改变整数表示与最终得分，无浮点回退路径。
4. 融合：权重与温度同样量化为定点整数，按特征下标固定顺序
   `sum(round(feature_fixed * weight_fixed / scale))`，再应用温度与裁剪区间。
5. 重排相似度用定点余弦（整数点积、`math.isqrt` 整数范数），
   折减 `after = round(before * (scale - penalty) / scale)` 全在整数域。

输出包含每个候选的定点得分、逐特征贡献分解、裁剪标记与位宽元数据。

## 4. 确定性与同分决断

- 融合同分：先比最高召回通道的**精确分数**，再比通道名字典序，最后比物品 ID。
- 重排平局：折减后得分相等按物品 ID 字典序；相似度相等时参照候选按
  “相似度降序、参照物品升序、已选序位”确定唯一参照。
- 并发：候选任务携带由 `blake2b(request_id, item_id)` 派生的稳定整数键，
  线程池内按该键顺序执行候选打分，主流程按请求首次出现顺序、候选稳定键顺序
  组装，因此任意并发度下结果与单线程逐字段一致。线程池 `max_workers` 取
  `min(concurrency, 待处理请求数)`，绝不超过配置上限。

## 5. 增量失效与缓存键

| 命令 | 失效范围 |
| --- | --- |
| `set_weight` | 全部请求的 fusion + rerank，normalize 复用 |
| `set_normalization` | 全部请求三阶段 |
| `set_quant`（位宽/档位） | 全部请求三阶段 |
| `set_rerank_window` / `set_penalty` | 全部请求仅 rerank |
| `add_candidate` / `delete_candidate` | 仅该请求三阶段，其余请求全复用 |

缓存键：

- normalize：`hash(阶段名, 请求内容指纹, 归一化参数指纹)`
- fusion（逐候选）：`hash(阶段名, 请求指纹, 物品特征指纹, 物品ID, 融合参数指纹)`
- rerank：`hash(阶段名, 请求指纹, 融合排序指纹, 重排参数指纹)`
- 每次折减另有 `hash(rerank键, step, 被折减物品, 参照物品)` 子键

每次响应都带 `reused` 与 `recomputed` 清单，精确到“请求 × 阶段”。

## 6. 预算响应语义

- 时钟默认 `time.monotonic_ns()` 实测，测试可注入 `DeterministicClock`。
- 有界在途窗口调度：在途任务数不超过并发度，主循环每 2ms 检查截止时刻。
- 预算耗尽时不再提交新任务，已在途任务跑完并保留内存产物，返回最近一次
  已完成结果摘要（已完成请求数、首位候选、受影响请求、进度、复用/重算清单）。
- `poll` 命令在新预算内补交挂起工作；提交前会把内存中已完成但未落盘的产物
  对账补齐，保证快照完整可恢复。

## 7. 快照与恢复

- 阶段产物先写 `snapshots/<name>/tmp`，再 `os.replace` 原子并入
  `artifacts/<cache_key>.json`，manifest 最后整体替换，不逐文件覆盖。
- manifest 记录全局版本号（单调递增）、配置哈希、分组表哈希、候选集指纹、
  已完成请求游标与各阶段产物版本/缓存键。
- 启动恢复时三重哈希校验，任一不一致抛 `E_CONFIG_MISMATCH` 拒绝续跑。
- `snapshot_every_commands` 控制提交节奏；`flush_snapshot()` 与报告生成时
  保证最终状态一定落盘。

## 8. 报告内容

每次完整运行产出 JSON 报告，包含：逐请求融合序、定点得分、贡献分解、
重排最终序与每次折减明细、各阶段缓存键、阶段耗时、当前档位与高一档位
重算的得分差、注入命中清单、失败清单以及全部交互响应。

## 9. 实际验证结果

环境：Python 3.11.0rc1 + NumPy 1.26.4，仅本地样本，无外部服务/GPU/网络。

- `pytest tests/ -q`：17 passed。
- 定点融合、窗口重排、折减边界（相似度恰好等于门限触发）均有断言覆盖。
- 增量 9 条命令后的报告与物化最终输入的空白全量重建报告，在 11 类业务字段、
  6 个请求上逐字段一致（`scripts/compare_reports.py`）。
- 故障注入 7/7 通过：非有限特征、负稀疏计数、空请求 ID、请求内重复物品、
  坏 JSON、权重长度不匹配、位宽越界、配置哈希不匹配、相似度门限边界。
- 并发一致性：concurrency=1 与 4 的业务产物完全相同，且线程池并发度不越界。
- 预算：0ms 预算下首命令返回 `E_INTERACT_BUDGET_PENDING`，放大预算 poll 后
  补交完成，磁盘快照完整且可被新引擎恢复。

复现命令：

```bash
python -m scripts.generate_samples --out samples/sample_input
python -m pytest tests/ -q
python -m rerank_cli.main full --input samples/sample_input \
    --snapshot snapshots/full --report reports/full_report.json
python -m rerank_cli.main interactive --input samples/sample_input \
    --snapshot snapshots/inc --commands samples/sample_input/commands.json \
    --report reports/interactive_report.json
python -m scripts.materialize_final --input samples/sample_input \
    --commands samples/sample_input/commands.json --out samples/final_input
python -m rerank_cli.main full --input samples/final_input \
    --snapshot snapshots/final_full --report reports/final_full_report.json
python scripts/compare_reports.py \
    --incremental reports/interactive_report.json \
    --full reports/final_full_report.json
python scripts/run_fault_injection.py
```
