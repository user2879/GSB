# 配置分发结果验证与差异回归分析平台

本地（Windows 优先，跨平台可跑）配置分发验证平台：管理一组以独立子进程运行、
通过 `127.0.0.1` 回环 TCP JSON Lines 协议通信的模拟节点，对两版配置（基线/候选）
做差异度量、定点误差界分析、真实采集、回归判定、派生链归因，并提供 SQLite 账本、
强制终止恢复与重放校验。不使用 Docker，不访问任何线上服务。

## 环境要求

- Python ≥ 3.11（SQLite3、Tkinter 随标准库；Tkinter 仅 GUI 需要，缺失不影响管线）
- NumPy ≥ 1.26：`pip install -r requirements.txt`

## 目录

- `config_regression/` 平台源码（模块职责见 `docs/DESIGN.md`）
- `samples/` 本地样本：注册表、会话参数、两版清单（覆盖新增/删除/取值/目标/系数变更）
- `tests/` pytest 测试（29 个，含真实子进程 TCP、故障注入、恢复与重放）
- `docs/DESIGN.md` 技术说明与定点/并发/恢复口径
- `artifacts/` 运行产物（报告 JSON、SQLite 账本）

## 快速开始

```bash
python -m config_regression.cli validate \
  --baseline samples/baseline_manifest.json \
  --candidate samples/candidate_manifest.json \
  --registry samples/registry.json --session samples/session.json

python -m config_regression.cli diff \
  --baseline samples/baseline_manifest.json --candidate samples/candidate_manifest.json \
  --registry samples/registry.json --session samples/session.json \
  --output artifacts/diff_report.json

python -m config_regression.cli run \
  --baseline samples/baseline_manifest.json --candidate samples/candidate_manifest.json \
  --registry samples/registry.json --session samples/session.json \
  --database artifacts/session.db --output artifacts/run_report.json

python -m config_regression.cli recover \
  --baseline samples/baseline_manifest.json --candidate samples/candidate_manifest.json \
  --registry samples/registry.json --session samples/session.json \
  --database artifacts/session.db --output artifacts/recover_report.json
```

退出码：`0` 无回归；`1` 检出 E_REGRESSION/no_receipt；`2` 平台错误（E_MANIFEST 等）；
`3` GUI 不可用。

## 错误码

- `E_MANIFEST` 清单/注册表强校验整体拒绝（稳定列出全部违规项与依据）
- `E_REGRESSION` 摘要或耗时超容差、目标分发变化
- `E_EXHAUST` 真实 socket 超时/连接失败在声明重试次数内未恢复
- `E_REPLAY` 恢复重放与账本结论不一致（定位到首个 node.key.field）
- `E_SNAPSHOT` 快照版本/摘要不匹配触发整体替换

## 测试

```bash
python -m pytest tests/
```

覆盖：清单强校验与环检测、顺序无关差异与影响链、整数定点误差界、真实节点 TCP 采集
与确定性、双重并发上限与排队时钟、重试语义与 E_EXHAUST、回归判定与 no_receipt、
逐字节归因确定性、快照整体替换、强制终止 unknown 恢复、E_REPLAY 定位。

## 确定性与可复核

所有结论经规范化 JSON（键排序、无空白）+ SHA-256 摘要持久化；差异、误差界、
findings、归因在同输入下逐字节一致（仅采集时钟字段随单调时钟变化，不进入结论摘要）。
节点聚合算法在 `config_regression/sampling.py`，平台与节点共用，可独立复核。
