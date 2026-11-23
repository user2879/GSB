# 跨环境配置仲裁一致性验证工作台

面向 Windows 本地环境的多环境配置键仲裁、定点误差判定、视图版本替换恢复与
异常注入验证工具。纯本地运行，仅依赖 Python 3.11+、NumPy 与标准库
（Tkinter、sqlite3、decimal、unittest），无需容器或外部网络服务。

## 快速开始

```powershell
python -m pip install -r requirements.txt
python run_tests.py     # 43 个单元测试
python run_demo.py      # 端到端：干净仲裁 + E_SPEC/E_GRID/E_VERSION + 崩溃恢复
python -m arbworkbench --gui
```

无显示环境（或 Python 缺少 Tk 库）下使用：

```bash
python -m arbworkbench --fresh          # 生成 output/arbitration_report.json
python -m arbworkbench --inject spec_corruption
python -m arbworkbench --inject grid_violation
python -m arbworkbench --inject version_mismatch
python -m arbworkbench --crash-view view-production
```

## 输入文件

* `sample_data/environments.json`：`scale_exponent`、环境标识、整数权重、每键
  观测列表（`seq` 整数序号、`timestamp_ms` 非负整数毫秒、`value` 数值或
  `[lo, hi]` 区间）。
* `sample_data/views.json`：视图标识、参与环境集合、每键口径
  （`latest_wins` / `weighted_majority` / `interval_intersection`）。

## 输出文件（`output/`）

* `arbitration_ledger.sqlite`：事务型仲裁账本（含格式版本与内容摘要）。
* `session_snapshot.json`：会话快照（版本指纹、逐视图结论摘要、内容摘要）。
* `arbitration_report.json`：权威值、采纳/否决、误差界、归因与 NumPy 分布统计。
* `events.jsonl`：结构化日志（异常码、注入与恢复事件）。
* `test_results.txt`、`demo_results.txt`：实际验证结果留档。

设计与误差界推导见 [DESIGN.md](DESIGN.md)。
