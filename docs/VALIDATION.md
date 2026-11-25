# 实际验证结果

验证环境：WSL Ubuntu 22.04，Python 3.11.0rc1（项目虚拟环境 `.venv`），
numpy 2.4.6、pytest 9.2.0；标准库 sqlite3/socket/threading/tkinter；
Xvfb 通过本地用户命名空间在显示号 `:99` 提供无显示帧缓冲。

## 1. 环境准备结论
- 虚拟环境由 `virtualenv -p /usr/bin/python3.11 .venv` 创建（系统 Python 缺 ensurepip）；
- numpy、pytest 安装于 `.venv`；
- tkinter 依赖（libtk/libBLT/TCL/TK）与 Xvfb 均来自工作区外的本地预置目录，
  经 `scripts/env.sh` 注入 `LD_LIBRARY_PATH/TCL_LIBRARY/TK_LIBRARY`，无 sudo、无系统安装；
- 缺失的 `libxkbfile.so.1` 以 deb 解包到 `.runtime/xkbfiles`（仅本地库补齐，不做系统安装）；
- `scripts/start-xvfb.sh` 在 `unshare -Urmpf` + pivot_root 内启动 Xvfb，
  宿主侧通过抽象 socket 连接 `:99`，Tk 控件构建与 update 验证通过。

## 2. 自动化测试
`python -m pytest -v` 结果：**39 passed**（证据 `artifacts/evidence/pytest.txt`）。

| 测试文件 | 覆盖要求 | 用例数 |
| --- | --- | --- |
| `test_loader.py` | 1 强校验/E_SPEC 定位 | 7 |
| `test_budget.py` | 2/3 双路径、路径规则、偏差界、E_DIVERGE | 8 |
| `test_late.py` | 4 迟到批窗口归属、重算、稳定去重 | 5 |
| `test_lease.py` | 5/6 租约、双重上限、真实计时、唯一裁决 | 7 |
| `test_ledger_recovery.py` | 7/8 账本、陈旧、重放、崩溃恢复、整体替换 | 9 |
| `test_end_to_end.py` | 端到端链路 | 2 |
| `test_ui.py` | 无头 Tk 仪表盘 | 1 |

## 3. 验收场景实测
- 双路径判定：`checkout-api`(W=30) 选择精确路径；`auth-service`(W=90)、
  `search-edge`(W=120) 选择增量路径，状态占比分别 0.089/0.067，双路径 0 越界；
- 迟到批重算：每服务 5 条覆盖留痕，受影响窗口分别 109/138/138 个，按分钟稳定排序；
- 租约裁决：24 次 acquired、12 次 recycled-revoke、12 次 rejected-double-count，
  最大并发 2，未突破全局 3/每服务 2 上限；错误版本续约返回 E_LEASE 且不延期；
- 账本：单事务提交，格式版本 1，SHA-256 摘要 64 位，450 条活跃结论；
- 独立重放：`replay` 450 行逐字段一致（`artifacts/evidence/replay.json`）；
- 强制终止：SIGKILL 退出码 137，重启后 1 个 unknown 批次被识别并先回收租约配额，
  重跑后重放 450 行一致，`exit_report_generated=false`
  （证据 `artifacts/evidence/crash/`）。

## 4. 复现命令
```bash
source scripts/env.sh
scripts/start-xvfb.sh            # 无头 X 显示 :99
export DISPLAY=$SLO_DISPLAY
python -m slo_burn.cli sample samples/demo
python -m pytest -q
python -m slo_burn.cli run samples/demo --ledger artifacts/run/ledger.db
python -m slo_burn.cli replay samples/demo --ledger artifacts/run/ledger.db
python scripts/verify_crash_recovery.py crash samples/demo artifacts/crash
python scripts/verify_crash_recovery.py recover samples/demo artifacts/crash
```
