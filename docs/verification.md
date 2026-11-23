# 实际验证记录

验证日期：2026-11-23（Asia/Shanghai）。全部验证使用本机回环地址，不访问外部网络。

## 1. 环境

```text
Python 3.11.0rc1
NumPy 1.26.4
SQLite 3.37.2
pytest 8.3.3
```

当前 Linux 验收容器的 Python 缺少 `libtk8.6`：

```text
ImportError: libtk8.6.so: cannot open shared object file
```

因此这里无法实际打开 Tk 窗口；`build_audit_view()` 的稳定只读投影已通过测试。目标 Windows 官方 Python 默认包含 Tkinter，可按 README 命令运行界面。

## 2. 测试

快速测试：

```text
$ python -m pytest -q
s.................................                                       [100%]
33 passed, 1 skipped in 2.46s
```

包含真实 5.6 秒回环网络演示的完整测试：

```text
$ RUN_DEMO_INTEGRATION=1 python -m pytest -q
..................................                                       [100%]
34 passed in 9.26s
```

覆盖点包括：

- 场景字段路径、唯一性、回环地址、端口范围、布尔/浮点拒绝、保留配额超额；
- 同源多租约份额上限；
- 续约提前于到期、到期后重复续约 `E_RACE`；
- 到期后其他来源重签发；
- 配额只回收空闲额度且保留底线不下降；
- 重复释放/重复回收不改变归属；
- SQLite 摘要链篡改 `E_SNAPSHOT`；
- 快照整体替换恢复、版本不匹配时拒绝替换且目标账本不变；
- 强制终止后的 `LEASE_UNKNOWN` 与 `exit_report_available=false`；
- 重放首差异的 `event_seq/source_id/field_path`；
- 外部漂移独立分类；
- 三种模拟服务模式与真实连接拒绝；
- Tk 只读投影稳定排序且重复读取一致。

## 3. 三阶段真实回环演示

命令：

```powershell
python -m leaseaudit demo --workdir artifacts\final-demo3 --base-port 56041
```

实际输出：

```text
SCENARIO cold_start code=OK issued=alpha:3,beta:3,gamma:1
SCENARIO expiry_reclaim code=OK expired=beta:3;reissued=alpha:1,delta:2
SCENARIO quota_reclaim code=OK reclaimed=beta:2
INVARIANT retained+global=capacity code=OK retained=9 global=3 capacity=12
WATERMARK code=OK samples=88 truncation=FLOOR_TOWARD_ZERO
AUDIT code=OK events=178 digest=5ad47a234f2c29421566e52a9eb85f6e16aef7ad1e27804d9c4a9c24c6aeed02
SNAPSHOT_RESTORE code=OK status=RESTORED digest=5ad47a234f2c29421566e52a9eb85f6e16aef7ad1e27804d9c4a9c24c6aeed02
NETWORK source=alpha requests=23 outcomes=success:23
NETWORK source=beta requests=10 outcomes=success:10
NETWORK source=delta requests=15 outcomes=success:15
NETWORK source=gamma requests=24 outcomes=connection_refused:24
OVERALL code=OK
```

结论：

- 冷启动签发 7 个单位；
- `beta` 的 3 个单位在到期后释放，`alpha` 增加 1、`delta` 新增 2；
- `beta` 连续低水位超过 2 轮后回收 2 个单位；
- 最终 `retained=9`、`global=3`、容量 12，恒等式成立；
- 所有有效来源总授予不超过容量，且 `beta` 回收后保留额度等于保留底线 1；
- 账本被独立重放为 `OK`；
- 对账本追加 1 个有效新事件改变摘要后，快照恢复重新得到原摘要。

## 4. 故障注入

命令：

```powershell
python -m leaseaudit fault-demo --workdir artifacts\final-faults --kind scene_corrupt
python -m leaseaudit fault-demo --workdir artifacts\final-faults --kind quota_overflow
python -m leaseaudit fault-demo --workdir artifacts\final-faults --kind ledger_digest
```

实际结果：

```text
scene_corrupt E_SCENE sources[0].address EXPECTED_FAILURE
quota_overflow E_QUOTA sources[1].reserve_quota EXPECTED_FAILURE
ledger_digest E_SNAPSHOT $.events[1].event_hash EXPECTED_FAILURE
```

结构日志写入各工作目录的 `structured.jsonl`，异常详情包含：

- 场景：违规地址值；
- 配额：`sum_reserve_quota=10`、`cluster_capacity=8`、`excess=2`；
- 摘要：事件摘要期望值与重算实际值。

`source_code_modified=false`、`partially_written_ledger=false`。前两类故障在创建账本前失败；第三类只读打开受污染账本副本。

## 5. 强制终止恢复

命令：

```powershell
python -m leaseaudit crash-demo --workdir artifacts\final-crash
```

摘要：

```text
crash OK OK exit 91 unknown 1 exit_report_available False audit OK
```

具体验证：

- 先完整提交一个有效租约；
- 子进程进入第二次签发，在 `PENDING` 意图提交后执行 `os._exit(91)`；
- 重启只恢复第一个已提交租约；
- 第二个未提交意图生成 1 个 `LEASE_UNKNOWN` 事件和 unknown 标记；
- `exit_report_available=false`，不会声称生成强制终止退出报告；
- 独立重放整个账本仍为 `OK`。

## 6. 复现注意事项

- `scripts/run_demo.py` 会动态寻找 `127.0.0.1` 高位端口，避免固定端口冲突；静态样本仍使用 55441..55444。
- 演示网络请求数量可能因本机调度略有变化，但阶段业务码、归属结论、摘要恢复结果与不变量保持稳定。
- GUI 应在 Windows 官方 Python 环境验证；Linux 容器缺少 Tk 动态库属于宿主环境限制。
