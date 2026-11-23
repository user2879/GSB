# 云原生服务告警关联与根因定位组件（本地确定性实验环境）

纯 Python 3.11/3.12 实现，仅依赖 `numpy`；SQLite、线程池、Tkinter 能力均来自标准库。
不使用容器、不需要网络服务，所有样本由 `sample/` 下声明 JSON 在本地确定性生成。

## 快速开始
```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v     # 45 个测试
python scripts/run_demo.py                  # 端到端：双路径/误差界/调度/账本重放
python scripts/crash_recovery_demo.py       # 强制终止恢复重放
python -m arcore.cli --declaration sample/declaration.json `
  --ledger ledger/cli.sqlite3 --rounds 3
```

## 目录
- `src/arcore/`：组件源码（声明校验、确定性生成、租约、调度、双路径关联、
  抑制、根因、定点统计、SQLite 账本）。
- `sample/`：`declaration_small.json`（精确路径）、`declaration.json`（近似路径）、
  `declaration_large.json`（大规模近似路径）。
- `tests/`：覆盖八项业务要求的单元测试。
- `scripts/`：可复现演示脚本。
- `docs/`：架构说明、Windows 运行手册、需求对照与实测结果。

## 关键设计
- 确定性：SplitMix64 随机源 + 规范化 JSON/SHA-256 摘要 + 稳定多键排序，
  相同声明逐字节复现；运行时钟类观测不参与重放比对。
- 租约与调度：单调时钟真实计时，乐观版本号续约，过期回收配额；
  全局与每服务信号量加租约三重门控，真实投射到线程池。
- 双路径：近似路径是精确谓词的子集限制，误关联结构性上界为 0（非零即 `E_DIVERGE`），
  漏关联给出实测/组合/精确对数三档上界。
- 账本：整数格式版本 + 内容摘要，不符即整体替换；未提交轮次标记 `unknown` 要求重跑，
  不生成退出报告，恢复后逐项重放比对。

详见 `docs/ARCHITECTURE.md`、`docs/WINDOWS_RUNBOOK.md`、
`docs/REQUIREMENTS_TRACEABILITY.md`。
