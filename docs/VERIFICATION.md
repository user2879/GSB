# 实际验证结果

最近一次完整验证由以下命令生成：

```powershell
python -m pip install numpy
python scripts/verify.py
```

Linux/WSL 复核环境等价执行 `python3 scripts/verify.py`。另外已通过 WSL 互操作调用 Windows 侧 Python 3.14.7 + NumPy 2.5.2 + Tk 9.0，在 `%TEMP%\scale-review-284-*` 实跑同一脚本并通过。

机器可读结果：

- `artifacts/verification-report.json`：当前 Linux/WSL 运行报告。
- `artifacts/verification-summary.txt`：Linux/WSL 简明结果。
- `artifacts/acceptance-report.json`：Linux/WSL 逐项证据。
- `artifacts/windows-verification-report.json`：真实 Windows Python 运行报告。
- `artifacts/windows-verification-summary.txt`：Windows 简明结果。
- `artifacts/windows-acceptance-report.json`：Windows 逐项证据。
- `artifacts/unittest-output.txt`：完整单元测试输出。

本次交付的实际结果为：

- `unittest`：19/19 通过。
- 一键验收：13/13 检查项通过。
- 双算法：样本序列摘要逐字节相同，所选路径为 `incremental`。
- 并行调度：`max_observed_inflight=2`，未超过配置上限；等待区间以单调毫秒记录。
- 租约：错误版本返回 `E_LEASE` 且版本/到期不变；正确续约后版本从 1 增至 2。
- 唯一归属：冲突后全局余量恢复为预算值，`owner_set=[]`，每个租约只有一条生效释放事件。
- 顺序无关：alpha/beta/gamma 正序与逆序裁决得到相同配额余量和属主集合证明。
- 滚动追加：版本从 1 增至 5；两个追加批次分别只影响 `[18,19,20]` 和 `[21,22,23]`，旧需求/分配字节保持不变。
- 快照：篡改 `content_digest` 后返回 `snapshot_digest_mismatch` 并整体重建替换。
- 强杀恢复：Linux/WSL 子进程返回码为 `-9`（`SIGKILL`）；真实 Windows 子进程返回码为 `243`（`TerminateProcess` 退出码）。两端均恢复出 1 个 `unknown` 尝试和 1 个完整提交轮次，下一未完成分钟为 5。
- `E_REPLAY`：篡改分配字段后定位到 `services.alpha.minutes[4].allocated_replicas`。
- 回环端口：实际绑定 `127.0.0.1` 的操作系统分配端口，并请求 `/health` 成功。
- GUI：Windows Python 3.14.7/Tk 9.0 已实际构建 `WorkbenchApp` 并完成一次事件刷新，证据见 `artifacts/windows-gui-smoke.txt`。

当前 Linux/WSL 沙箱内缺少 libtk 原生库，因此该环境中的 `python -m scale_review.cli gui` 会返回结构化 `E_TKINTER` 提示；Windows 侧 Python 3.14.7 已实测可导入 Tk 9.0。可在 Windows 桌面会话运行 `scripts\run-workbench.bat` 或 `python -m scale_review.cli gui`。
