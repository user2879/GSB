# 应急预案演练诊断工作台

纯 Python 标准库实现的本地应急预案演练诊断工作台：预案强校验、阶段状态机、增量失效传播、
仅失效子集重算、SQLite 持久化账本、强制终止恢复、快照重放校验、结构化日志与 Tkinter 诊断界面。

## 快速开始
```bash
python -m emergency_drill.cli generate-samples --sample-dir samples
python -m emergency_drill.cli run --plan samples/plan_valid.json \
  --events samples/events_valid.json --ledger work/drill.sqlite3 --output work/run.json
python -m emergency_drill.cli view --ledger work/drill.sqlite3 --text
python -m unittest discover -s tests
```

Windows 一键验收：`scripts/run_demo_windows.bat` 与 `scripts/crash_recovery_windows.bat`。
跨平台验收：`bash scripts/run_demo.sh`。

## 文档
- 架构、业务码、关键设计与验证报告：`docs_技术说明.md`
- 最近一次实际验证记录：`work/VERIFICATION_RESULTS.md`
- 原始任务需求：见 Git 初始提交中的任务说明。

## 命令一览
`generate-samples / validate / run / edit / verify-incr / new-round / snapshot /
open-snapshot / replay / tamper-snapshot / recover / rerun-after-crash / crash-demo / view`
