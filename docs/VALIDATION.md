# 实际验证记录

验证日期：2026-11-23（Asia/Shanghai）。验证环境是本地 WSL/Linux 容器；交付目标是 Windows 本地 Python。当前容器缺少 `libtk8.6.so`，因此未启动 Tk 原生窗口；Tk 代码惰性导入，Windows 官方 Python 包含 Tk 时可直接运行。与 Tk 无关的视图模型、排序、固定文本渲染已在无头模式验证。

## 环境

```text
Python 3.11.0rc1
NumPy 1.26.4
SQLite 3.37.2
pytest 8.3.3
Tk shared library: unavailable in validation container
network: not used
Docker/container runtime: not used by the workbench
```

单调计时确认：`time.monotonic_ns()` 连续调用递增，事件时间则完全来自样本整数毫秒，不混用墙钟。

## 自动化测试

命令：

```text
pytest -q
```

结果：

```text
28 passed
```

覆盖点：

- 声明首个违规字段、重复标识、引用、成环、非负整数、最小/最大配额、事件时序和序号连续性；
- 函数数组逆序时完整批次负载逐字节一致；
- 全局/函数上限、借出、借入到期、抢占边界、挂起重排、显式释放和唯一归还；
- 定点截断、单值误差、比较误差、借用延迟主导段和稳定决胜；
- 两版声明续跑、引用闭包失效、未受影响租约携带；
- pending 意向恢复、未知租约重算、摘要篡改 `E_SNAPSHOT`；
- 语义篡改后 `E_REPLAY` 首个事件/租约定位；
- Tk 无关视图重复渲染及输入数组逆序后的稳定输出；
- CLI 退出码、结构化日志、强制终止和恢复。

## 端到端演示

命令：

```text
python scripts/run_demo.py
```

结果摘要：

```text
status: PASS
checks: all true
events: 11
replay batches: 9
v1 prefix cursor: 6
v2 final cursor: 11
invalidated lease/queue/preemption objects: 5
current preemption statuses: {}
v1 prefix preemption statuses: {'SUSPENDED': 1}
release dispositions: {'CALL_COMPLETED': 6, 'PREEMPTED': 1}
loan returns: {'INVALIDATED': 1, 'RETURNED_EARLY': 2}
headless render sha256: ebf2856766117ad2ac45fa99c166871d50511facb981b6dad0c0d6c7236ee66a
forced termination exit: 17
recovery: CLEAN
```

演示中：

- `api` 在 `t=100` 需要 2 并发，但自身配额已满且 `report` 无可借空闲，因此生成抢占预约；
- `report` 的租约在 `t=200` 边界挂起，原事件以新队列项在 `t=600` 前恢复等待；
- `api` 在 `t=200` 借入 2 并发，归还截止 `t=600`，借入记录在释放时唯一归还；
- 第一版处理到事件 6 后切换第二版，`report` 配额、租约时长和 `api -> report` 归还截止变化；
- `L00000001`、`L00000003`、`L00000004`、旧队列 `Q00000002` 与旧版本挂起抢占记录被标记陈旧并给出原因；当前视图由版本 2 状态整体重建；
- 9 个批次、11 个事件逐项重放一致。

## 异常注入

| 注入 | 退出码 | 结果 |
| --- | ---: | --- |
| 抢占关系成环 | 2 | `E_DECL`，字段 `preemptions`，环 `api -> batch -> api` |
| 强制授予超限 | 3 | `E_QUOTA`，`event_seq=1`，`lease_id=L00000001` |
| 账本内容摘要不符 | 4 | `E_SNAPSHOT`，字段 `content_digest` |
| 语义重放差异 | 5 | `E_REPLAY`，`event_seq=1`，`lease_id=L00000001` |

所有错误都写入 `artifacts/*.log` 的 JSON Lines。`E_REPLAY` 演示先修改副本账本的授予并发并重算内容/链摘要，使完整性校验通过，再由语义重放发现差异。

## 强制终止

`--inject-crash-after-pending` 在 pending 意向写入后直接 `os._exit(17)`，不执行 SQLite 批次提交，也不生成退出报告。重启后：

- 已提交游标之前的数据可加载；
- pending 候选租约被列为 `UNKNOWN_RECOMPUTE_REQUIRED`；
- 从游标后重算全部批次；
- 成功后未知租约清零，恢复状态为 `CLEAN`；
- 报告中保留 `recomputed_unknown_lease_ids` 作为审计证据。

## 产物

- `artifacts/demo_report.json`：机器可读完整报告；
- `artifacts/demo_summary.txt`：摘要；
- `artifacts/demo.log`：批次与重放结构化日志；
- `artifacts/view_v1_prefix.txt`：第一版前缀视图；
- `artifacts/view_v2_final.txt`：第二版最终视图；
- `artifacts/demo_final_view.json`：第二版最终结构化视图；
- `artifacts/crash_demo_recovered.json`：强制终止后重算结果；
- `artifacts/*_demo.log` 与副本数据库：异常注入证据。
