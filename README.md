# 冷热分层与迁移组件（Tier Mover）

面向 Windows 本地环境的纯 Python 组件：本地 SQLite 文件作为元数据账本，
本地目录中的对象文件作为数据载体，JSON 清单声明存储层/对象/迁移规则，
固定整数随机种子生成确定性对象内容。支持热度驱动的冷热迁移、原子文件搬运、
批次预算、版本演进与陈旧防护、强制终止恢复与一致性重放，并提供 CLI 与
Tkinter 专业桌面界面。无 Docker、无外部服务。

## 环境要求

- Python 3.11+，标准库 `sqlite3`、`tkinter`（GUI）、`threading`、`hashlib`
- numpy 可选（确定性内容的第二后端，默认纯标准库即可复现）
- 运行前确认工作区目录可写；账本、对象、日志均落在工作区本地

## 快速开始（Windows PowerShell）

```powershell
python -m tier_mover.cli init .\workspace --seed 20261122   # 建账本+确定性对象
python -m tier_mover.cli plan .\workspace                   # 推导迁移计划
python -m tier_mover.cli run  .\workspace                   # 按批次真实搬运
python -m tier_mover.cli status .\workspace                 # 账本当前视图
python -m tier_mover.cli replay .\workspace                 # 一致性重放
python -m tier_mover.cli inject .\workspace capacity        # E_CAPACITY
python -m tier_mover.cli inject .\workspace integrity       # E_INTEGRITY
python -m tier_mover.cli inject .\workspace version         # E_VERSION
python -m tier_mover.cli run .\workspace --crash-after-key obj-alpha  # 模拟强杀
python -m tier_mover.cli recover .\workspace --mode rollback
python -m tier_mover.gui                                    # 桌面界面
.\run_demo.ps1                                              # 一键全流程演示
.\run_tests.ps1                                             # 28 个测试
```

WSL/Linux 等价脚本：`./run_demo.sh`、`./run_tests.sh`。

## 业务错误码

| 码 | 含义 |
| --- | --- |
| `E_SPEC` | 清单强校验失败，首个违规字段+原因，整体拒绝不部分装载 |
| `E_CAPACITY` | 层容量不足，列出层 id、被拒对象、余量与需求字节 |
| `E_INTEGRITY` | 摘要/目录字节数/文件完整性被破坏 |
| `E_VERSION` | 账本格式版本或策略版本冲突 |
| `E_REPLAY` | 重放分歧，定位首个不一致对象键与字段 |

## 文档

- `docs/TECHNICAL_DESIGN.md`：架构、整数热度公式、原子搬运、账本事务、
  版本演进、崩溃恢复、重放与注入的完整设计
- `docs/VALIDATION_RESULTS.md`：28 个测试与全部场景的实际验证记录
- `docs/demo_report.json`：7 场景一键演示的结构化报告
- `tier_mover/sample_workspace/`：随仓样例清单（tiers/objects/seed）

## 目录

```
tier_mover/        核心包（errors/content/manifest/heat/planner/ledger/
                   engine/recovery/replay/faults/workspace/demo/cli/gui）
tier_mover/tests/  unittest 测试套件（28 用例）
docs/              设计与验证文档
run_demo.* / run_tests.* / run_gui.ps1  启动脚本
```
