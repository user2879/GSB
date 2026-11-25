# SLO 错误预算燃烧率诊断（本地单进程 + 评估器子进程）

面向按分钟滚动的 SLI 事件流，提供错误预算账本、双路径燃烧率判定、迟到批
重算、评估租约调度、唯一归属裁决、SQLite 持久化与强制终止后可重放能力。

## 环境准备（WSL Ubuntu，无 sudo / 无容器）
```bash
make venv                       # 创建 .venv 并安装 numpy、pytest
source scripts/env.sh           # 注入本地 tk/Xvfb 库路径
scripts/start-xvfb.sh           # 用户命名空间内启动 Xvfb :99
export DISPLAY=$SLO_DISPLAY
```
说明：系统 Python 缺 `ensurepip`，使用 `virtualenv` 建虚拟环境；tkinter 与
Xvfb 依赖来自本地预置目录，`libxkbfile` 通过 deb 解包到 `.runtime/xkbfiles`，
全程不做系统级安装，样本生成与运行不访问网络。

## 一分钟上手
```bash
python -m slo_burn.cli sample samples/demo
python -m slo_burn.cli validate samples/demo
python -m slo_burn.cli run samples/demo --ledger artifacts/run/ledger.db
python -m slo_burn.cli replay samples/demo --ledger artifacts/run/ledger.db
python -m slo_burn.cli lease-demo --socket /tmp/demo.sock
python -m slo_burn.cli dashboard samples/demo --ledger artifacts/run/ledger.db
python -m pytest -q
```

## 关键能力与错误码
- `E_SPEC`：声明/批次/参数非法，定位首个非法服务与批次物理行号；
- 精确路径（逐分钟环形累计）与分块增量近似路径（定长块环形缓冲复用），
  按窗口长度/批数/状态占比显式选路并输出理由；
- `E_DIVERGE`：双路径逐字段校验，偏差上界由尾部边界块请求量严格量化；
- 迟到批按窗口引用链重算，重复分钟稳定择一并记录覆盖批；
- `E_LEASE`：评估器子进程租约版本号乐观续约，双重并发上限真实投射线程池；
- 到期/撤销回收唯一归属裁决，任意服务顺序重放得到同一裁决序列；
- `E_LEDGER`/`E_REPLAY`：SQLite 事务 + 格式版本 + 内容摘要，
  快照不一致整体替换；强制终止后只认完整提交，未提交批次标 unknown 重跑。

## 目录
- `slo_burn/`：系统包（模块职责见 `docs/ARCHITECTURE.md`）；
- `tests/`：39 个 pytest 用例；
- `samples/`：固定种子生成的确定性样本；
- `scripts/`：环境脚本与崩溃恢复验证；
- `artifacts/evidence/`：实际验证证据；
- `docs/VALIDATION.md`：验证结果与复现步骤。
