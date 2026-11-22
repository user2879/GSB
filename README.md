# 模型服务弹性伸缩决策工作台

面向本地 Windows 环境的单机工作台，围绕“数据规模变化下的算法取舍”，对推理服务集群
做按分钟副本需求估算、全局预算/并发配额裁剪、拒绝量核算、延迟定点统计、版本化账本
与崩溃恢复，并提供 Tkinter 诊断界面。仅依赖 Python 3.10+、NumPy 与标准库 Tkinter。

## 目录结构
- `workbench/`：核心包（加载校验、双路径估算、裁剪、预览、延迟、账本、引擎、GUI、CLI）
- `samples/`：确定性 JSON 样本（正常、小样本、强校验损坏、强预算压力）
- `tests/`：35 个 unittest 用例，覆盖全部业务要求与错误码
- `data/`：本地 SQLite 数据库目录
- `logs/`：JSON Lines 结构化日志目录
- `docs/design.md`：架构与规则说明；`docs/validation_report.md`：实测验收报告
- `run_workbench.bat`：Windows 一键运行脚本

## 快速开始（Windows）
```bat
py -m pip install numpy
py -m workbench.cli init-samples
py -m workbench.cli --db data\workbench.db --log logs\workbench.jsonl run samples\cluster.json
py -m workbench.cli --db data\workbench.db --log logs\workbench.jsonl replay samples\cluster.json
py -m workbench.cli --db data\workbench.db --log logs\workbench.jsonl gui
```
无显示环境可用 `gui --headless` 输出视图模型摘要。

## 命令与异常注入
- `run / replay / recover / gui / fault {schema|length|snapshot} / init-samples`
- 退出码：`E_SCHEMA=2, E_LENGTH=3, E_DIVERGE=4, E_PREVIEW=5, E_LATENCY=6, E_REPLAY=7, E_SNAPSHOT=8`

## 测试
```
python -m unittest discover -s tests -v
```
