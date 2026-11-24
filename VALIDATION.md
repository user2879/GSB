# 实际验证记录

验证日期：2026-11-25  
验证环境：WSL2 Linux，Python 3.11.0rc1，NumPy 1.26.4，未使用 Docker。  
当前 Python 发行版缺少 `libtk8.6.so`，因此本容器无法实际创建 Tk 窗口；`leasecache/gui.py` 已通过字节码编译，Windows 启动命令与批处理已交付。后端、TCP 子进程、账本、调度与验收均已实际运行。

## 单元与集成测试

```text
python -m unittest discover -v

Ran 21 tests in 11.343s
OK
```

覆盖：

- `E_SPEC` 首个违规字段、引用环、缺失引用、回环与正整数约束。
- 同节点/制品键摘要逐位稳定、工作区文件与 manifest 一致。
- 真实 TCP 握手超时 `E_HANDSHAKE`、缺回执字段 `E_ACK`、端口占用 `E_PORT` 及结构化日志。
- 节点租约上限、真实单调时钟到期、续约、真实子进程重启后摘要一致。
- 摘要匹配/容量/节点序稳定仲裁与完整节点排队。
- 同一制品双节点授予被全局 claim 拒绝。
- 到期与配额回收并发竞争仅一个赢家、只产生一个终止事件。
- 摘要变化使活动租约和复用来源同时失效。
- 全局/阶段/节点配额下真实线程池、真实套接字请求和排队记录。
- 编辑后增量结果与全量重建业务快照一致。
- 删除一个节点和一个制品后的增量结果与全量重建一致，历史事件仍可独立重放。
- `os._exit(86)` 强制终止后无报告、未知租约恢复、恢复结果与全量一致。
- 篡改账本摘要后独立校验以 `E_VERIFY` 定位到 `a/node-a/summaries.digest`。

## 完整验收

```text
python -m leasecache demo --root reports/acceptance
ok: true
elapsed_mono_ms: 8686
```

| 检查 | 结果 |
| --- | --- |
| 基线业务快照 vs 同规格复用 | 0 差异 |
| 第二次运行网络请求 | 0 |
| 引用编辑增量 vs 全量重建 | 0 差异 |
| 增量受影响键 | `obj/lib`, `pkg/bundle` |
| 强制终止退出码 | `86` |
| 强制终止后退出报告 | 不存在 |
| 恢复未知租约 | `src/app`、`src/lib` 两条 ACTIVE |
| 恢复 vs 全量重建 | 0 差异 |
| 三个账本独立重放 | 全部 `ok` |
| 种子变化受影响键 | 5 个，稳定排序 |
| 实际重启后二次摘要变化 | 空 |

账本内容摘要（最终三次运行）：

- full edited：`8d789196c1f0b189a6b44a59237c3051108596439e4ae56330e311d72b076267`
- incremental edited：`1018a7fcce0d6af1c9b758ba1840463bc1ab0778fbae52e00dc28135d2751500`
- recovered edited：`beacba21e21feca0ef5c228261130bfafcbdc7b4875ad30900450f7ab3679acb`

## 原子整体替换

实际执行：

```text
python -m leasecache run --spec samples/demo_spec.json --db <tmp>/db.sqlite ...
python -m leasecache run --spec samples/demo_spec_edited.json --db <tmp>/db.sqlite ... --replace
python -m leasecache verify --spec samples/demo_spec_edited.json --db <tmp>/db.sqlite
```

结果：替换成功，旧账本保留为 `db.sqlite.previous`，最终独立验证 `ok`。

## 说明

业务一致性比较覆盖全部结论字段：节点摘要/容量/活动租约、制品规范摘要/引用/来源集合/复用结论/服务节点、当前活动租约、编辑哈希和完成键。审计事件总序号、实际耗时与时间戳天然会因中断恢复历史不同而不同，按 `docs/DESIGN.md` 第 12 节的口径不参与业务一致性比较。
