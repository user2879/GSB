# orderflow 技术说明

面向指令洪峰场景的延迟指令流确定性裁决与预算调度工具。纯 Python 3.11 + NumPy，无网络、无容器依赖。

## 运行方式

```powershell
pip install numpy pytest
python run.py run --workspace ws                 # 生成样本 -> 解析 -> 裁决 -> 并行调度 -> 恢复 -> 复算
python run.py run --workspace ws --threads 8 --mem-budget-mb 40
python run.py run --workspace ws --crash-match M01_NORMAL --crash-after 7   # 模拟强制终止(退出码 3)
python run.py run --workspace ws                # 再次执行即按已提交快照恢复
python run.py recompute --workspace ws          # 独立复算器校验
python -m pytest tests/ -q
```

## 目录结构

```
orderflow/
  model.py       # Config(声明窗口/网格/补给上限)、业务码、canonical JSON + SHA-256 内容哈希
  parser.py      # JSONL 解析、每指挥官 seq 连续性(与到达顺序无关)、越界/非法类型/迟到/重复投递
  adjudicate.py  # 物理时间裁决队列、互斥仲裁、单位状态推进、逐步内容哈希快照
  ledger.py      # 补给账本汇总(NumPy): 峰值/越界次数/按步资源曲线
  storage.py     # 临时文件 + 原子改名提交、manifest 提交点、崩溃恢复
  scheduler.py   # 线程/内存预算调度、准入、降级记录、按对局 ID 归并
  generator.py   # 固定种子(20261120)样本集: 正常/迟到乱序/平局/补给越界/预算拒绝 五局
  recompute.py   # 独立复算器: 只读落盘快照与账本, 定位首个不一致步
  cli.py         # 命令入口
run.py           # python run.py <cmd>
tests/test_orderflow.py  # 11 个 pytest 用例
```

## 关键确定性规则

- **物理时刻**: `abs_time = tick * tick_ms(100) + latency_ms`；裁决序键为
  `(abs_time, -priority, commander, seq)`，绝不使用到达顺序或线程完成顺序。
  输入行整体打乱后裁决序列与内容哈希逐字节一致（`test_shuffle_determinism`）。
- **seq 连续性**: 按指挥官对 seq 集合排序后检查从 0 连续，因此与文件行序无关；
  跳号/冲突重复 → `E_SEQ_GAP`（含行号与指挥官），内容完全相同的重复投递 →
  `E_DUPLICATE_ORDER`（保留首份）。
- **迟到**: `latency_ms > max_latency_ms(250)` → `E_LATE_ORDER`，标记 expired
  进入快照异常记录，不静默丢弃、不进入裁决队列。
- **平局**: 时刻与优先级均相同 → `E_TIE_UNRESOLVED` 异常入账，按
  （指挥官字典序， seq) 确定胜出，任意线程数下分类与首个命中位置一致。
- **互斥仲裁**: MOVE 与 RETREAT 互斥；同单位在裁决窗口（150ms）内首条胜出，
  被覆盖方记录 `E_ORDER_OVERRIDDEN`（胜出方、胜出时刻、窗口）。
- **补给账本**: 每裁决步一条账本记录并与快照哈希一一对应（ledger.jsonl 的
  `snapshot_hash`）；余额 <0 或 >上限（200) → `E_SUPPLY_OVERFLOW`，
  `first_overflow` 标记首个越界步与单位；账本给出每单位峰值、越界次数与按步曲线。
- **预算调度**: 按 `(sched_priority, match_id)` 固定序准入，超出内存预算的对局
  返回 `E_BUDGET_DENIED`（不静默计为通过）；并行度 = min（线程预算， 内存预算//
  单局估值， 准入数），降级记录原因与前后并行度；结果按对局 ID 与裁决步归并。
- **原子提交与恢复**: 每步提交顺序为 快照文件 → 账本追加 → manifest 推进
  （manifest 即提交点），全部临时文件 + `os.replace` 原子改名；强制终止留下
  `.tmp` 半成品即未提交。重启只依据已提交快照，重放时逐哈希校验已提交前缀
  （不一致抛 `RecoveryMismatch`），截断账本未提交尾部后继续；崩溃运行不写
  report 与 CLOSED 标记，不伪造退出报告。

## 落盘布局

```
ws/
  samples/M0X_*.jsonl(+meta.json)   # 固定种子确定性样本
  matches/<id>/snapshots/step_NNNNNN.json  # {hash, content} 不可变快照
  matches/<id>/ledger.jsonl         # 每步账本, snapshot_hash 对应快照
  matches/<id>/ledger_summary.json  # 峰值/越界/曲线
  matches/<id>/manifest.json        # 提交点 {committed, last_hash}
  matches/<id>/result.json + closed.marker
  schedule.json  report.json  CLOSED
```

## 实际验证结果（本机 Python 3.11）

- `python -m pytest tests/ -q` → **11 passed**（解析错误码、迟到/重复、乱序
  确定性、平局与覆盖、补给越界、账本-快照对应、预算拒绝与降级、线程数
  1/2/4 不变性、崩溃恢复、CLI 端到端字节一致）。
- 两次干净运行 `diff -r ws1 ws2` → 逐字节一致。
- `--crash-after 7` 模拟强制终止：退出码 3，仅 7 步已提交快照 + 一个
  `.tmp` 半成品，无 report/CLOSED；重新 `run` 恢复后 `report.json` 与干净
  运行逐字节一致，`recompute` 全部 OK。
- 样本裁决结果：M01 正常 36 步；M02 6×E_LATE_ORDER + 1×E_DUPLICATE_ORDER；
  M03 1×E_TIE_UNRESOLVED + 1×E_ORDER_OVERRIDDEN；M04 4×E_SUPPLY_OVERFLOW
  （首越界 step=4, U8, balance=220）；M05 E_BUDGET_DENIED；调度降级
  `{reason: memory_budget, before: 8, after: 4}`。
