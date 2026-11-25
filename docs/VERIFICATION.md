# 实际验证记录

验证时间：2026-11-25（Asia/Shanghai）。

## 当前沙箱环境

```text
Python 3.11.0rc1
numpy 1.26.4
sqlite3 3.37.2
socket: standard library available
tkinter import: unavailable in this Linux sandbox
                 ImportError: libtk8.6.so: cannot open shared object file
```

这是当前最小 Linux 沙箱缺少 Tk 原生库，不是视图模型或账本问题。`lease_arbitrator/ui.py` 延迟导入 Tk；只读文本快照和全部非 UI 测试均可运行。目标 Windows 官方 Python 3.11 安装通常自带 Tk/Tcl。

补充检查中，将 Ubuntu 的 `libtk8.6`、`libtcl8.6`、`libxft2`、`libxss1` deb 包解压到 `/tmp/tklibs` 后：

```bash
LD_LIBRARY_PATH=/tmp/tklibs/usr/lib/x86_64-linux-gnu \
  python -m lease_arbitrator doctor --require-tk
# doctor_ok True, tkinter available, TkVersion 8.6
```

该方式只验证 `import tkinter/_tkinter`；沙箱没有 X display，因此未实际进入 `mainloop()`。

## 自动化测试

命令：

```bash
python -m unittest discover -s tests
```

实际结果：

```text
.................
----------------------------------------------------------------------
Ran 17 tests in 21.903s

OK
```

覆盖：

- `E_SPEC` 首个非法条目和字段；
- 摘要黄金向量及 node/key/shard 绑定；
- 稳定仲裁、同分标记、容量预占精确回滚、水印误差边界；
- `E_PORT`、真实 `E_HANDSHAKE` 超时、`E_FIELD`；
- 真实节点子进程写入、摘要回执、续约乐观版本冲突；
- 样本端到端仲裁、高水印回滚、`E_LEASE`、真实过期、配额撤销；
- `E_VERIFY` 容量篡改定位；
- `os._exit(97)` 强杀、unknown 标记、恢复重放和校验一致；
- 取消后回滚；
- 节点集合删除/新增和旧租约整体撤销。

## CLI 样本验收

命令序列：

```bash
python -m lease_arbitrator init-sample --dir samples/demo --force
STATE=$(mktemp -d /tmp/la-accept-XXXXXX)
python -m lease_arbitrator run --sample samples/demo --state-dir "$STATE"
python -m lease_arbitrator verify --sample samples/demo --state-dir "$STATE"
python -m lease_arbitrator ui --state-dir "$STATE" --text
```

实际关键结果：

```text
run completed 12 events
seq 1  -> node-c
seq 2  -> node-a
seq 3  -> node-b
seq 4  -> rolled_back on node-a
seq 7  -> renew_rejected E_LEASE
seq 9  -> expiry reclaimed 1 lease
seq 11 -> quota_replaced, revoked 2 leases
seq 12 -> node-a after quota replacement

verify: E_OK, ledger_format_version=1, records_checked=87, unknown_batches=0
```

第 4 事件水印/回滚：

```text
reserve watermark_fixed=1000/1000
rollback watermark_fixed=500/1000
```

恢复演练由 `test_force_kill_unknown_recover_and_replay_verifies` 自动执行：首次注入强杀退出码 `97`，`recover` 生成 open unknown batch，校验返回 `E_UNKNOWN`，重跑后第 6 事件续约成功、第 7 事件旧版本仍被拒绝，最终 `E_VERIFY/E_OK`。

## 待在 Windows 目标机确认

当前交付代码使用跨平台 `sys.executable -m ...`、`subprocess.CREATE_NO_WINDOW`（仅 Windows）和 stdin 父进程看门狗，但本次没有真实 Windows 主机可运行。请在 Windows 上执行：

```powershell
python --version
python -m pip install numpy==1.26.4
python -m unittest discover -s tests
powershell -ExecutionPolicy Bypass -File scripts/recovery-check.ps1
python -m lease_arbitrator doctor --require-tk
```
