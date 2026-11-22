# MeshBench：服务网格流量治理与错误预算诊断工作台

纯 Python 3.11 标准库 + NumPy 的本地工作台，在 127.0.0.1 回环上用真实 TCP
模拟上游，提供确定性加权分发、熔断状态机、逐窗口 SLI/错误预算回执账本，
以及强制终止后的权威重放逐字节复核。无容器、无外部服务。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: scripts\setup_windows.bat
pip install -r requirements.txt
python -m pytest -q
python -m meshbench.cli demo
python scripts/crash_recovery_demo.py
```

## 命令

- `python -m meshbench.cli validate --rules samples/rules_v1.json`：规则强校验。
- `python -m meshbench.cli demo`：正常分发 → 熔断 → 规则演进回执替换三场景。
- `python -m meshbench.cli recover --rules samples/rules_v2.json --rules-version 2`：
  强制终止恢复与重放校验。
- `python -m meshbench.cli inject {rule-corrupt|upstream-down|snapshot-corrupt}`：
  三类异常注入，分别报 `E_RULE` / `E_UPSTREAM` / `E_SNAPSHOT`，写 `logs/meshbench.log`。
- `python scripts/run_upstream.py`：单独运行确定性模拟上游。

## 错误码

| 代码 | 含义 |
| --- | --- |
| `E_RULE` | 规则声明非法，整体拒绝且不产生分发记录 |
| `E_CIRCUIT` | 熔断打开，快速失败且不发起连接 |
| `E_UPSTREAM` | 回环上游真实连接不可达 |
| `E_PREVAIL` | 缓存回执与权威重算逐字节不一致 |
| `E_REPLAY` | 恢复重放无法复现账本回执 |
| `E_SNAPSHOT` | 快照/账本内容摘要不符 |

## 目录

- `meshbench/`：领域组件（配置、哈希、上游、熔断、分发、预算、账本、重放、快照、视图）。
- `samples/`：确定性规则样本 `rules_v1.json` / `rules_v2.json`。
- `scripts/`：模拟上游、演示、强制终止演练与 Windows `.bat`。
- `tests/`：35 个单元/集成/NumPy 交叉校验测试。
- `docs/`：`ARCHITECTURE.md` 设计与 `VALIDATION.md` 验证记录。

设计要点（定点截断方向、量化误差上界、哈希链、整体替换语义）见 `docs/ARCHITECTURE.md`。
