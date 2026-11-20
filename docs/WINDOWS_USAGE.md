# Windows 本地运行指南

## 1. 环境准备（无需 GPU / 容器 / 外部服务）

```powershell
py -3 --version            # 需要 Python 3.10 或更高
py -3 -m pip install -r requirements.txt
py -3 -c "import tkinter; print('tkinter ok', tkinter.TkVersion)"
```

标准安装的 Windows Python 自带 Tkinter 与 sqlite3。若使用精简发行版导致 Tkinter 缺失，诊断命令会自动降级为控制台视图，调度/账本/测试功能不受影响。

## 2. 一键演示

```powershell
py -3 demo.py
```

所有产物仅写入工作区 `workdata/demo/`：

- `ledger.sqlite`：SQLite 调度账本；
- `snapshots/snapshot.json`：会话快照；
- `backup/`：版本漂移时的只读旧快照；
- `logs/events.jsonl`：结构化日志；
- `injected/`：异常注入产生的输入副本。

## 3. 分步 CLI

```powershell
py -3 -m gpu_scheduler run
py -3 -m gpu_scheduler stats
py -3 -m gpu_scheduler inject memory
py -3 -m gpu_scheduler inject quota
py -3 -m gpu_scheduler inject cycle
py -3 -m gpu_scheduler inject ledger
py -3 -m gpu_scheduler crash --after-round 3
py -3 -m gpu_scheduler recover
py -3 -m gpu_scheduler replace
py -3 -m gpu_scheduler gui
```

`crash` 模拟本地进程被强制终止：第 3 轮在提交前中断，重启后用 `recover` 仅依据已提交账本恢复。也可以在两个终端中手工复现：先运行 `crash`，再运行 `recover`，这等价于进程重启。

## 4. 运行测试

```powershell
py -3 -m unittest discover -s tests -v
```

## 5. 自定义声明

将 `samples/specs/` 下三份 JSON 复制到新目录，保持文件名 `session.json`、`pools.json`、`tasks.json`，然后：

```powershell
py -3 -m gpu_scheduler --specs C:\path\to\specs --workspace C:\path\to\work run
```

校验失败时整体拒绝加载，输出的每个问题都带 JSON 字段路径、期望约束与实际值，不会写入任何账本或快照。
