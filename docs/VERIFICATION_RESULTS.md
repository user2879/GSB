# 实际验证结果

验证环境：Python 3.11.0rc1，numpy 1.26.4，sqlite3(SQLite 3.37.2)，
Linux 容器（目标 Windows 环境仅差异在 Tkinter GUI 可用性，见下）。
所有样本由固定整数种子 `20261123` 本地确定性生成，无网络访问。

## 1. 测试套件
`python -m pytest tests/ -q` → **41 passed**（约 4s）。覆盖：
- 三版本编解码往返、四类 `E_FORMAT` 注入（未知版本/缺字段/位图长度/非法类型）
- 升级 + 检查点续做（模拟强杀后不重复处理、备份保留原始载荷、幂等）
- `E_DIVERGE` 定位到块/列/行；表达式规范化摘要；批切分合法性与越界拒绝
- 量化截断/饱和、单值与逐批误差界、稳定决胜与 `error_affected` 标注
- 调度器双约束（观测全程 running_units≤预算、每任务≤配额、计时字段完整）
- 执行器跨批大小 {1,7,33,100,1000} × 顺序/乱序执行结果摘要完全一致
- 除零置空 + `DIV_BY_ZERO` 标记、三值逻辑、聚合口径
- 账本事务、快照版本匹配/不匹配整体替换、`E_REPLAY`
- 端到端：样本 → 升级 → 并行执行 → 校验 → 崩溃恢复 → 快照替换

## 2. 本地样本验收（工作区内实跑）
`init-sample` 生成 6 块：blk-a1/blk-a2(b1)、blk-b1/blk-b2(b2)、
blk-c1/blk-c2(b3)，覆盖整数列、浮点列与含空值列。

`run` 报告（run_report.json）：
- 升级块：`['blk-a1','blk-a2','blk-b1','blk-b2']`，备份 4 份（只读）
- 任务结果（5 任务 / 26 批，全局预算 6 单元）：

| 任务 | 表达式 | 聚合 | 结果 | 空值行 | 除零行 | 批数 |
|---|---|---|---|---|---|---|
| t1 | `qty * price - 10` | none | digest ce3b324e… | 0 | 0 | 4 |
| t2 | `if(amount > 500 and rate < 0.5, …)` | sum | 104214101 | 78 | 0 | 7 |
| t3 | `(amount + 7) * rate / 3` | avg | 766491 | 87 | 0 | 8 |
| t4 | `k * v + k / (v - 1.5)` | none | digest a40221eb… | 11 | 5 | 3 |
| t5 | `qty * price >= 100 or not (qty < 5)` | count | 112 | 28 | 0 | 4 |

- 调度：观测峰值 running_units=2 ≤ 6，每任务峰值 ≤ 各自配额；每批均含
  mem_units/peak_mem_bytes/queued/start/end/queue_wait_ms 单调时钟毫秒。
- `verify`：6/6 块与 canonical.json（直接 b3 写入）逐字节一致。

## 3. 异常注入实测输出
- `E_FORMAT`（注入非法类型 int128）：
  `E_FORMAT: unsupported column type 'int128' [block_id='blk-b1', field='columns[0].type']`
- `E_DIVERGE`（篡改 canonical 中 blk-c1.amount 第 7 行）：
  `E_DIVERGE: column value mismatch [block_id='blk-c1', column='amount', row=7, expected=2800, actual=2799]`
- 崩溃恢复：第 7 批提交后模拟强杀（`--crash-after 7`），重启依据账本游标
  续跑，5 个任务最终摘要与不中断运行**逐字节一致**（digests_equal=true）。
- `E_REPLAY`（篡改账本 t3 结果后重放）：
  `E_REPLAY: resumed run diverges from ledger result [task_id='t3', recorded='deadbeef', replayed='637e7589…']`
- 快照版本不一致（2 → 3）：整体替换 26 条批轨迹与 5 条结果
  （`diff.results=['t1','t2','t3','t4','t5']`），事件写入 recovery_log；
  再次打开版本匹配，零替换。

## 4. 已知限制
- 本容器缺少 Tk 运行库，`gui` 子命令在此环境不可用；Windows python.org
  发行版自带 Tkinter，GUI 代码已按守护导入编写，不影响 CLI 与测试。
- 批执行极快时观测到的并发峰值较低（约束始终强制成立；并发压力场景由
  tests/test_scheduler.py 以 20ms 人工负载验证）。
