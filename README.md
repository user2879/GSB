# SQLite 连接池并行调度与资源预算工作台

面向 Windows 本地环境、以本地文件 SQLite 为后端的领域数据编辑工作台。用户通过 Tkinter 界面或本地 JSON 脚本提交一批数据库任务，连接池在「最大连接数 / 每工作负载配额 / 单连接最长持有时长」三重约束下并行调度，并持久化池状态、任务轨迹与配额账本，支持参数调整后续跑与历史快照整体替换恢复、重放校验。

仅依赖 Python 3.11+ 标准库（含 `sqlite3`、`tkinter`），无第三方包、无网络、无容器。

## 快速开始

```bash
python scripts/build_sample_db.py     # 构建确定性样本库
python -m pool_workbench.demo         # 三轮可重复演示
python -m unittest discover -s pool_workbench/tests -p "test_*.py" -v
python -m pool_workbench.gui          # Tkinter 界面（桌面环境）
```

Windows 可直接运行 `scripts/run_demo.bat`、`scripts/run_gui.bat`、`scripts/run_tests.bat`。

## 能力对照（业务要求）

1. 配置与任务脚本强校验，非法输入 `E_CONFIG` 整体拒绝并给出字段路径 — `pool_workbench/config.py`
2. 三重约束并行调度，排队记录单调时钟，轨迹含连接序号/起止/等待 — `pool_workbench/planner.py`、`pool_workbench/scheduler.py`
3. 获取超时 `E_GETTIMEOUT` 不占连接；连接失效先移除后补建且不超限 — `pool_workbench/pool.py`
4. 持有时长在语句边界中断 `E_HOLDTIMEOUT`；成功/获取超时/持有超时/执行错误四类计数 — 见 `TECH_NOTE.md §3.5`
5. 快照版本识别与整体替换恢复，稳定列出参数变化与受影响排队任务 — `pool_workbench/snapshot.py`
6. 克隆库重放，逐项比对结局/连接序号/等待毫秒，不可复现报 `E_REPLAY` 且不改样本表 — `pool_workbench/replay.py`
7. 启动参数异常注入 `E_SCHEMA/E_CONFIG/E_SNAPSHOT` 与结构化 JSONL 日志 — `pool_workbench/fault_injection.py`
8. Tkinter 控制台（连接状态/配额/排队/结局/快照）与三轮演示脚本 — `pool_workbench/gui.py`、`pool_workbench/demo.py`

业务错误码：`E_CONFIG`、`E_SCHEMA`、`E_GETTIMEOUT`、`E_HOLDTIMEOUT`、`E_EXEC`、`E_SNAPSHOT`、`E_REPLAY`。

详细架构、策略依据与验证结果见 [TECH_NOTE.md](TECH_NOTE.md)。
