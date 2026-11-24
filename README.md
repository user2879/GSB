# 跨故障域租约仲裁与配额预算诊断工作台

本地、离线、确定性的故障隔离平台：每个故障域运行独立的回环 TCP 仲裁子进程，
实例按事件流申请带到期时刻的租约，平台在全局并发上限与每域配额双重约束下完成
租约状态机、槽位唯一归属、FIFO 排队、定点预算诊断、SQLite 账本与强制终止恢复。

## 特性与需求对照

- 强校验声明/实例/NDJSON 事件流，违规以 `E_SPEC` 报告首个字段与行号。
- 真实单调时钟 + 真实回环 TCP 子进程，JSON 行协议；续约乐观版本校验，不符 `E_LEASE` 不延期。
- 租约状态机 `granted/renewed/released/expired` 按显式许可表迁移并记录迁移轨迹。
- 槽位唯一归属，冲突以 `E_ALIAS` 报告双方租约；双重并发上限 + FIFO 排队，禁止优先级插队。
- 三条仲裁路径（逐事件/分批/归并）共享同一状态机，结论 canonical JSON 逐字节一致。
- 独立校验器重放两路径并比对全部确定性字段，首个不一致以 `E_DIVERGE` 定位事件/实例/字段。
- 整数定点（10 的 0..3 次幂）预算诊断，显式截断方向与单值/比较误差界。
- 每批次单事务写 SQLite 账本，带格式版本与 SHA-256 摘要；强制终止后只按已提交记录恢复。

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
bash scripts/demo.sh                                 # Windows: scripts/demo.ps1
```

## 样本与目录

- `samples/small/`：2 域 5 实例 48 事件；`samples/medium/`：3 域 10 实例 160 事件。
- 样本由固定 numpy 种子生成，事件到达时刻单调不减，可逐字节复现。
- `artifacts/`：运行产生的 SQLite 账本与恢复演示产物。

## 命令一览

```bash
python -m faultisolation.cli gen-samples --out samples
python -m faultisolation.cli --spec <spec.json> --session <session.json> validate
python -m faultisolation.cli --spec <spec.json> --session <session.json> plan
python -m faultisolation.cli --spec <spec.json> --session <session.json> replay --path event --ledger out.sqlite --replace
python -m faultisolation.cli --spec <spec.json> --session <session.json> live   --ledger out.sqlite --replace
python -m faultisolation.cli --spec <spec.json> --session <session.json> verify --ledger out.sqlite
python -m faultisolation.cli --spec <spec.json> --session <session.json> diagnose
python -m faultisolation.cli --spec <spec.json> --session <session.json> recover --ledger out.sqlite
python -m faultisolation.gui      # Tk 可用时启动图形工作台，否则无头交互控制台
```

## 路径选择规则（显式）

工作量点（N=事件数，D=域数，B=批大小）：

- `event`: `N + N//4`
- `batch`: `N + ceil(N/B)*D + N//4`
- `merge`: `N + N*D//ceil(N/D) + N//3`

阈值 `T=ceil(N/10)`；任一候选与最低工作量差在 T 内时，按固定优先级
`event < batch < merge` 稳定决胜并在输出中标注。

## 文档

- `docs/ARCHITECTURE.md`：架构、双时钟模型、并发与状态机设计。
- `docs/VERIFICATION.md`：独立校验命令与验收对照。
- `docs/WINDOWS_SETUP.md`：Windows 安装与端口检查。

## 测试

```bash
python -m pytest tests -q
```

覆盖：强校验行号定位、状态机许可表、乐观续约拒绝、FIFO 不插队、槽位 E_ALIAS、
三路径逐字节一致、E_DIVERGE 定位、定点误差界、账本摘要/版本替换、强制终止恢复、真实回环遥测。
