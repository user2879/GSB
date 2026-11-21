# 特征漂移监控组件（drift_monitor）

表格模型线上监控的分窗漂移诊断领域组件。基于本地 YAML 配置与固定种子确定性
数据，完成分窗切片、逐列漂移度量、定点整数显著性判定、量化档位翻转诊断、
内容指纹增量复用、并发确定性计算、阶段快照整体替换与 JSON/SVG 诊断报告。

## 快速开始

```bash
# 1. 虚拟环境（Python 3.11 + NumPy 1.26.4）
/home/zsl/.feedback/python-base/bin/virtualenv --clear -p python3.11 .venv
.venv/bin/python -m pip install -r drift_monitor/requirements.txt

# 2. 一键演示（测试 + 端到端运行）
cd drift_monitor && ./run_demo.sh

# Windows PowerShell
./run_windows.ps1
```

## 目录

- `drift_monitor/src/drift_monitor/`：组件源码（配置、数据生成、分窗、度量、
  定点、策略、缓存、快照、引擎、可视化、交互 CLI）
- `drift_monitor/configs/default_monitor.yaml`：默认监控配置
- `drift_monitor/tests/`：33 个 pytest 回归用例
- `drift_monitor/scripts/`：样本生成与验收场景脚本
- `drift_monitor/docs/TECHNICAL_DESIGN.md`：技术设计说明
- `drift_monitor/docs/RUN_REPORT.md`：实际验收运行报告

## 交互命令

```bash
PYTHONPATH=src ../.venv/bin/python -m drift_monitor.interactive \
  --config configs/default_monitor.yaml --tier small --output output/cli
# 进入 REPL 后：run / resume / set-windows N / set-threshold X / set-tier NAME /
# set-metric COL METRIC / set-size-tier NAME / set-concurrency N / show / history / exit
```

## 业务错误码

`E_CONFIG_INVALID`、`E_DATA_GEN_FAILED`、`E_SNAPSHOT_STALE`、
`E_SNAPSHOT_CORRUPT`、`E_METRIC_FAILED`。
