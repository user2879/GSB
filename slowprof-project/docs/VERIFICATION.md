# 验证记录（实际运行结果）

执行环境：Ubuntu 22.04 沙箱，Python 3.11.0rc1，sqlite 3.37.2，NumPy 1.26.4。
该沙箱无 Tcl/Tk 且无 sudo，`import tkinter` 不可用；诊断模块在该环境自动使用
只读文本回退，在标准 Windows Python（自带 tkinter，`python -m tkinter` 可弹窗）
下打开 Tkinter 窗口。两条渲染路径共用同一份只读数据模型。

复现命令：

```bash
python -m unittest discover -s tests -v     # 33 tests, OK
python scripts/demo.py                       # 含 150,000 行规模段
python scripts/demo.py --quick               # 跳过规模段
```

## 1. 单元/集成测试

`docs/test_output.txt` 为完整输出，结果 `Ran 33 tests ... OK`，覆盖：

- NDJSON 强校验：坏 JSON、零/负权重、负耗时、非法帧名/空段、缺字段、同服务
  时间戳回退、布尔伪装整数，均断言 `E_SAMPLE` 且携带行号与字段。
- 区间非法（start>end）断言 `E_RANGE`。
- 自/累计口径、重复帧只计一次、顺序无关、双路径逐字节一致、阈值边界。
- 排名总序、调用者/被调用者聚合与稳定排序。
- 定点边界、缩放因子校验、真实 Top-K 判定、误差受影响标注、NumPy 向量化复核。
- 强制终止只保留已提交批次、resume 后与一次性运行规范化字节一致。
- 参数/区间版本变化触发整体替换；快照篡改定位首个 `E_REPLAY` 不一致项。
- 生成器重复运行字节一致、摘要稳定；诊断渲染只读且幂等。

## 2. 双路径一致性与阈值边界（实测）

2,000 行、skewed 权重、seed=123 的实测：

- 输入摘要：total_weight=17257，unique_stacks=1764，
  digest=`47396c82...954061`。
- 选择路径：`full`（1764 < 5000 且 17257 < 200000）。
- `cross_check=True`，full 与 streaming 的 `canonical_bytes()` 逐字节相同，
  SHA-256=`4219290b...a97a2`。
- 边界函数实测：`select_path(4999,199999)=full`；
  `select_path(5000,199999)=streaming`；
  `select_path(4999,200000)=streaming`（达到阈值即流式）。

Top5（self 降序、cumulative 降序、名字典序）：

| rank | frame | self | cumulative | 定点 self (S=10000) |
| --- | --- | --- | --- | --- |
| 1 | billing.serialize | 1201 | 2941 | 695 |
| 2 | gateway.flush | 923 | 2184 | 534 |
| 3 | billing.db | 855 | 2949 | 495 |
| 4 | gateway.enqueue | 826 | 2043 | 478 |
| 5 | gateway.query | 812 | 2365 | 470 |

合计误差上界 `K/S = 5/10000`；单帧误差严格 `< 1/10000`；
`top_k_contains_true_top_k=True`（精确有理比较）。

## 3. 强制终止与恢复（5,000 行，batch=1000，演示实测）

- 在第 2 批提交后注入 `ForcedKill`：`recoverable_batches=2`，
  `last_committed_line=2000`，`needs_rerun_from_line=2001`。
- 重新打开账本只信任已提交批次；resume 后批次数回到 5，最终规范化字节与全新
  一次性运行完全一致（见 `SessionRecoveryTests.test_forced_kill_recover_resume`）。
- 账本表 `batches` 中未提交批次不可见（`test_uncommitted_batch_not_visible`）。

## 4. 重放与快照替换

- 正常重放：`{"matched": true, "checked_frames": 51, ...}`。
- 人为把快照中首帧 `self_weight` 加 999：重放抛出 `E_REPLAY`，定位到该帧名、
  字段 `self_weight`、期望值（篡改值）与实际重算值。
- 区间版本号变化：`snapshot_replaced=true`，原因为版本不同→整体替换。

## 5. 150,000 行规模段（实测，`scripts/demo.py`）

- 生成 150000 行（4 服务、深度 4..12、skewed、seed=99），total_weight=1312161，
  unique_stacks=143418；生成耗时约 2.35s。
- 剖析耗时约 22.4s；选择 `streaming`（143418 ≥ 5000）。
- 大数据双路径 `identical canonical bytes=True`；重放 `matched=true`。
- 同参数重新生成的样本流摘要完全一致（可复现）。

## 6. 异常注入实测（只改输入/参数）

- `inject-sample --kind bad_weight --target-line 17` 后 profile：
  `{"code":"E_SAMPLE","field":"weight","line_number":17,
  "message":"weight must be a positive integer, got 0"}`，退出码 2。
- `--start-ms 500 --end-ms 100`：
  `{"code":"E_RANGE","field":"start_ms",
  "message":"start_ms=500 is greater than end_ms=100"}`，退出码 2。

完整端到端文本见 `state/demo_output.txt`（运行 demo 后生成/覆盖）。
