# 页级双写缓冲恢复诊断组件（DWB Diagnostics）

面向 Windows 本地环境的页级双写缓冲（double-write buffer）恢复诊断组件，聚焦
日志与崩溃恢复场景下的可复现验证与异常注入。纯 CPython 3.11+ 标准库实现：
`sqlite3` / `tkinter` / `hashlib` / `json` / `struct`，无第三方包、无容器、无网络服务。

## 快速开始

```bash
python -m venv .venv                 # Windows: .venv\Scripts\python.exe
.venv/bin/python run_tests.py        # 31 个用例
.venv/bin/python -m dwb_diag.cli demo --root data/demo
```

Windows 一键脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_demo_windows.ps1
```

## 能力对照

- 页级编解码与严格校验（`E_PAGE`，首个出错偏移+页号）：`dwb_diag/buffer.py`、`dwb_diag/paging.py`
- 三阶段双写写路径与崩溃点标记：`dwb_diag/storage.py`
- 页级恢复、`E_LOST`、确定性恢复报告：`dwb_diag/recovery.py`
- 视图依赖、拓扑失效传播、`E_DEPEND`/`E_VIEW`：`dwb_diag/views.py`
- SQLite 快照、版本识别、整体替换、`E_SNAPSHOT`：`dwb_diag/snapshot.py`
- 四类异常注入与 `E_REPLAY` 重放比对：`dwb_diag/fault_injection.py`
- Tkinter 诊断界面（缺失 Tk 时优雅降级）：`dwb_diag/gui.py`、`dwb_diag/gui_model.py`

详见 `docs/TECHNICAL.md`（含文件格式、业务码链路与实测验证结果）。

## CLI 速览

```
init | write | crash | demo | recover | view-rebuild | view-verify
validate-buffer | validate-main | snapshot-list | snapshot-open
inject (main_corrupt|buffer_truncate|depend_cycle|snapshot_stale)
replay [--tamper-page N] | keyspace
```
