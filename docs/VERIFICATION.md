# 实际验证记录

最近一次完整命令：

```bash
bash scripts/verify.sh
.venv/bin/python scripts/property_check.py
```

完整控制台输出保存在：

- `artifacts/verification/full-verify-output.txt`
- `artifacts/unittest.log`
- `artifacts/property-check.log`
- `artifacts/verification/verification_summary.json`

## 本机环境

- WSL Ubuntu 22.04（由 `artifacts/workspace.json` 的 `wslDistro` 记录）
- 默认 `python3`: Python 3.11.0rc1
- `.venv`: `virtualenv` 创建的 CPython 3.11 环境
- NumPy: `1.26.4`（已按要求固定安装）
- Tk: 8.6.12，`DISPLAY=:0`，真实窗口创建成功
- GUI 冒烟：7 个页签均填充成功（误差矩阵 12 行、外推统计 21 行、边界样本 18 行、注入清单 5 行、初始复用账本 471 行）

环境限制：该镜像缺少 `ensurepip`，直接执行 `python3 -m venv .venv` 会提示安装 `python3.11-venv`；同时 `sudo` 需要密码，无法非交互安装 `python3-tk`。因此使用镜像中已有的 `virtualenv` 创建 `.venv` 并安装固定 NumPy；Tk 运行库通过 `scripts/setup_tk_userland.sh` 以 `apt-get download` + `dpkg-deb -x` 免 root 解包到 `.local-tk`，并由 `rope_lab/tk_bootstrap.py` 预加载。标准 Ubuntu/WSL 若有 sudo，仍可直接执行 `sudo apt install python3.11-venv python3-tk xvfb`，随后使用标准 `python3 -m venv .venv`。

## 验收结果

- 单元测试：`12` 项全部通过。
- 内置编辑序列：10 条命令全部执行，最终增量结果与空白全量重建逐字段相等。
- 串行与线程池：完整语义报告逐字段相等。
- 最终账本顺序：增量与空白重建均为 393 条，稳定键顺序和缓存键签名完全相等。
- 最终增量账本：393 条全部命中复用；空白重建：393 条全部为 `recomputed`，状态差异符合“空白无缓存”的预期。
- 随机命令序列：12 组、每组 12 步，覆盖方案增删、基频/缩放、训练/目标长度、样本替换、位宽/量化档和并发/预算调整；全部满足字段一致与账本顺序一致。
- 预算：预算 140、单样本成本 40 时首批准入 `S000,S001,S002`，待办 `S003,S004,S005`；第二批待办优先并全部完成，恢复后核心输出与充足预算全量结果相等。
- 配额：`sample_quota=20` 时 6 个样本均记 `E_POINT_QUOTA_EXCEEDED`。
- 快照：临时目录 + 原子符号链接提交成功；版本号和配置哈希匹配时 `recovery_decision.use_snapshot=true`。
- 注入：5/5 命中预期业务码；NaN 注入产生 18 个隔离失败任务，但三个方案的统计结论仍由健康样本生成。
- 翻转归因：`angle_quantization`、`rotation_rounding`、`accumulator_overflow` 三个固定案例均命中。

## 主要报告文件

- `artifacts/verification/verification_summary.json`：总检查入口。
- `artifacts/verification/incremental_semantic_report.json`：增量最终完整语义结果。
- `artifacts/verification/blank_serial_semantic_report.json`：空白串行全量结果。
- `artifacts/verification/blank_parallel_semantic_report.json`：空白线程池结果。
- `artifacts/verification/incremental_ledger.json`：最终条目化账本。
- `artifacts/verification/command_ledger_series.json`：初始状态和每条编辑命令后的完整条目化账本。
- `artifacts/verification/comparison.json`：一致性布尔结论。
- `artifacts/verification/budget_report.json`：首批待办、恢复和配额报告。
- `artifacts/verification/injection_report.json`：注入编号、错误码和隔离结果。
- `artifacts/verification/flip_attribution_report.json`：排序翻转来源证据。
- `artifacts/environment.txt`：实际 Python、NumPy、DISPLAY 和 Tk 开窗检查。
