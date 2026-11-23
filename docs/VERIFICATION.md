# 验证结果（本地确定性样本）

环境：Python 3.11、NumPy 1.26.4、无 GPU、无外部网络、Linux/Windows 均可运行（仅用标准库 + NumPy）。

## 1. 自动化测试

```
PYTHONPATH=. python -m unittest discover -s tests
Ran 38 tests in 0.9s — OK
```

| 测试文件 | 覆盖要求 |
| --- | --- |
| `test_fixedpoint.py` | 量化、饱和、四类舍入、两次运行逐元素一致、温度缩放 |
| `test_validation.py` | 张量包往返/截断/CRC、位宽枚举、温度、候选上限、补齐、命令引用 |
| `test_hottable.py` | 容量逐出、同次决断、逐出后重新累计、保留压制、账本重建 |
| `test_engine.py` | 反馈收敛、未收敛业务码、同分决断、查询块失效、预筛变更复用、确定性 |
| `test_scheduler.py` | 全局/单查询块预算、待办优先、并发下扣减序列稳定 |
| `test_snapshot_injection.py` | 快照清单、篡改重建、非有限缩放隔离、强制溢出、截断中止 |
| `test_consistency.py` | 增量编辑链与空白全量逐字段一致、预算暂停恢复结果一致 |
| `test_cache_keys.py` | 三类缓存键确定性与输入敏感、依赖图后代失效、无关产物保留 |

## 2. 端到端 CLI 验收（实际运行结果）

全量与再次运行：

```
python -m hotblock_diag.cli --sample-dir samples/demo --workspace runs/verify_full --fresh
-> status=converged, rounds_completed=2, failure_codes=[], exit=0
```

无编辑再次运行的缓存审计（持久化算子全部命中）：

```
no-edit rerun audit: {'invalidated': 0, 'recomputed': 0, 'reused': 16}
```

块级故障注入（故障隔离，其余块对继续）：

```
--injections injections_block.json
-> failure_codes=['E_NONFINITE_SCALE','E_ACC_OVERFLOW','E_INJECTION']
-> block_pair_count=15（目标 0:1 被隔离），status=converged, exit=0
```

张量包截断（致命）：

```
--injections injections_trunc.json
-> E_PACK_TRUNCATED，status=fatal，exit=2
```

快照版本篡改：

```
-> failure_codes=['E_SNAPSHOT_STALE']，自动全量重建后 status=converged, exit=0
```

预算暂停与恢复：

```
首批 --max-batches 1 -> status=budget_paused, exit=3, 待办持久化
再次运行             -> status=converged, exit=0
-> total_consumed=16, pending_tasks=0
```

信号安全退出：预置退出标志后，当前批产物落盘，`status=signal_paused`，退出码 0，
`snapshots/committed/manifest.json` 存在。

非整除序列补齐：`samples/padded`（seq_q=18、seq_k=15、块长 4）：

```
layout: q_pad=2, padded_q=20, q_blocks=5; k_pad=1, padded_k=16, k_blocks=4
status=converged, block_pair_count=20
```

非法输入业务码：

| 输入 | 业务码 | 退出码 |
| --- | --- | --- |
| 温度=0、候选超上限、acc<=frac | `E_CONFIG_INVALID` | 2 |
| 形状描述与数据长度不符 | `E_TENSOR_LAYOUT` | 2 |
| 命令引用不存在块 | `E_COMMAND_REF` | 2 |
| 张量包少字节 | `E_PACK_TRUNCATED` | 2 |

## 3. 增量与全量一致性

`test_incremental_chain_matches_blank_rebuild` 施加查询块替换、键块替换、预筛参数调整、
量化档位 q8→q12、热块容量变更五类编辑，断言以下字段与从空白按最终张量/配置一次性全量
运行完全相同：

- 每块对定点累加值、缩放因子、舍入模式与指纹；
- 最终选择位图与完整位图轨迹；
- 热块表条目、逐出与压制记录；
- 失败清单、收敛状态与完成轮次。

预算路径下，受限运行的扣减集合与无约束全量运行一致，且每个批次内扣减顺序按稳定键排序，
单查询块配额约束被严格遵守。

## 4. 复现步骤

```
pip install -r requirements.txt
python scripts/generate_samples.py --sample-dir samples/demo
bash scripts/run_demo.sh
```
