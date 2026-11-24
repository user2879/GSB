# 验证记录（本地确定性样本）

环境：Python 3.11，无 Docker、无网络；样本由整数 LCG 种子本地生成。
运行：`python -m unittest discover -s tests`（共 52 个用例，全部通过）。

## 1. 单元/集成测试分布
| 模块 | 用例数 | 覆盖业务要求 |
|---|---|---|
| test_fixedpoint | 8 | 舍入方向、饱和、缩放、定点乘、整数对数精度/空值 |
| test_spec_validation | 11 | E_SPEC：重复 id、未知引用、环、尺度、窗口、位图、阈值 |
| test_determinism | 4 | 块切分/调度逐字节一致、空值传播、饱和标记 |
| test_stats_verdict | 7 | 分组顺序无关、整数均值/缺失率、TV、三态裁决、等号决胜 |
| test_indistinguishable | 4 | 不可区分区间、桶边界不确定随误差界扩大 |
| test_error_chain | 6 | 误差界非负整数、乘积放大、主导段阈值、顺序无关 |
| test_end_to_end | 12 | 持久化、崩溃恢复、E_REPLAY、失效传播、E_DIVERGE、三类异常、整体替换 |
| 合计 | **52** | 全部 OK |

## 2. 样本管线实测漂移结论（samples/pipeline.json，会话尺度 1e-4）
| 特征 | drift(bps) | error(bps) | 裁决 |
|---|---|---|---|
| std_price | 30548 | 34 | drift_confirmed |
| std_vol | 2069 | 2 | drift_confirmed |
| bucket_price | 254516 | 1129355 | indistinguishable（参考均值近零，相对界跨阈值） |
| log_size | 1232 | 1 | stable |
| interact_pv | 540000 | 3573571 | indistinguishable |
| clip_interact | 540000 | 3573571 | indistinguishable |

三类成因均在样本或单测中出现并带数值依据（区间上下界、阈值、误差界）。

## 3. CLI 与异常业务码实测
| 命令 | 结果 |
|---|---|
| `run` / `recover` | rc=0，6 特征全部入库 |
| `crash --stage log_size` | rc=0；该批次留 `unknown`，重开 recover 后补齐 |
| `verify`（增量 vs 全量影子） | rc=0，`{"match": true}` |
| `inject --fault data_corrupt` | rc=11，E_DATA（缓存摘要不符） |
| `inject --fault param_overflow` | rc=12，E_PARAM（sd 越界） |
| `inject --fault snapshot_digest` | rc=15，E_SNAPSHOT（快照摘要不符） |
| 篡改账本摘要后 recover | E_REPLAY（定位首个不一致阶段/字段） |
| 篡改漂移值后 verify | E_DIVERGE（定位首个不一致阶段/特征） |

## 4. 复现步骤（Windows PowerShell 或 bash）
```
pip install -r requirements.txt
python run_demo.py init-sample --outdir samples
python run_demo.py validate --spec samples/pipeline.json
python run_demo.py run      --spec samples/pipeline.json --workdir work/demo
python run_demo.py crash    --spec samples/pipeline.json --workdir work/demo_crash --stage log_size
python run_demo.py recover  --spec samples/pipeline.json --workdir work/demo_crash
python run_demo.py verify   --spec samples/pipeline.json --workdir work/demo
python run_demo.py inject   --spec samples/pipeline.json --workdir work/demo_f1 --fault data_corrupt
python -m unittest discover -s tests
```
所有输出为规范 JSON；结构化日志位于 `<workdir>/events.log.jsonl`。
