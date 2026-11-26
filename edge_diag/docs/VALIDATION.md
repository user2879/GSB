# 实际验证结果

环境：WSL Ubuntu，Python 3.11（要求 >=3.10），NumPy 1.26.4，SQLite 标准库，
Tkinter 经项目内置 `.tk-python` + `.sysroot` 本地运行时启用。

## 自动化测试

`python -m pytest edge_cache/tests/ -v` ：**22 passed**

- `test_determinism.py`（4）：分片字节确定性、摘要绑定、NumPy 定点量化、
  种子顺序敏感。
- `test_spec.py`（9）：节点/端口/内容键唯一、回环字面量、正整数、整除、
  事件序号单调、引用完整性，全部以 E_SPEC 定位首个非法字段。
- `test_integration.py`（8）：真实子进程回环 TCP 握手与摘要、E_PORT 端口
  占用、E_TIMEOUT 真实读超时、端到端运行+重放、快照整体替换、强制终止
  恢复逐字段一致、乐观版本冲突 E_LEASE。
- `test_ui_smoke.py`（1）：Tk 真实建窗并渲染账本；无 Tk 时自动跳过。

## 演示样本核对（artifacts/ledger-evidence.txt）

- 真实排队：seq5 因全局并发上限 4 排队，`wait_delta_ms=24`（真实单调时钟）。
- 双路回收唯一归属：同一槽位序列中 `expired` 与 `quota_recycle` 互斥，
  每次归属变化均带 `basis/seq/quota_version`，无重复记账。
- 真实到期：短 TTL 租约由 `monotonic now_ms >= expire_ms` 裁决回收，
  无任何 sleep 或线程超时冒充。
- 配额换代：node-c 3->4、node-b 7->8，旧版本活跃租约全部配额回收。
- 预算约束：`budget_used=1792` 触及预算上限，越界写入落 `rejected/E_BUDGET`。
- 乐观续约：成功续约版本 1->2；失效租约续约被 E_LEASE 拒绝且不延期。
- 未知批次：全部批次先落 `unknown_batches`，完整提交后标记 committed。

## 强制终止恢复

`crash-demo --crash-seq 3 --crash-node node-b`：

- 崩溃点前 3 个租约完整提交；SIGKILL 后新引擎恢复
  `active_leases_restored=3`，节点重启并回放租约表。
- 续跑至 seq18 后独立重放 `events_replayed=21`，节点租约集合、容量余量、
  分片摘要、归属裁决逐字段一致（无 E_VERIFY）。

## 界面只读性

连续刷新 3 次前后对 SQLite 文件做 sha256，结果一致（read-only unchanged:
True），刷新不产生任何写事务。
