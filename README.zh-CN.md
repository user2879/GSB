# 分片计算平台调度语义诊断实现

本项目是分片下发中「就绪冻结 / 租约撤销 / 容量版本演进」的可独立复核诊断实现。
不依赖 Docker、外部服务或真实账号；样本由固定整数种子在本地用 NumPy 生成，
节点以独立子进程在 `127.0.0.1` 回环 TCP 上以 JSON 行协议提供计算。

## 1. 环境与目录

- 运行环境：Linux（WSL Ubuntu），Python 3.10+；本仓在 Python 3.11 验证。
- 依赖：仅 `numpy`（验证版本 1.26.4）；数据库使用 Python 标准库 `sqlite3`
  （底层 SQLite 3.37.2）。
- `sqlite3` 命令行：当前容器无 sudo 权限无法安装 CLI，因此以标准库模块为准，
  并提供 `tools/inspect_ledger.py` 做只读检查，能力等价。

目录结构：

```
config/declaration.json     节点、分片、冻结/撤销/预算声明（强校验入口）
samples/                    固定种子生成的 .npz 样本与 manifest.json
ledger/diag.db              主 SQLite 账本（WAL）
ledger/hard_kill.db         强杀恢复专用账本
ledger/backups/             快照整体替换时保留的只读备份（0444）
sharding_diag/              诊断组件源码
tests/                      30 个单元/端到端测试
tools/                      端口预检与账本检查工具
artifacts/                  每次运行的结构化诊断报告
```

## 2. 运行方式

```bash
pip install numpy
python3 -m tools.preflight config/declaration.json   # 端口可监听预检
python3 -m sharding_diag.demo_run                    # 端到端诊断
python3 -m unittest discover -s tests -v             # 全量测试
python3 -m tools.inspect_ledger ledger/diag.db       # 只读查看账本
```

端口被占用时，直接在 `config/declaration.json` 改选 1024–65535 的高位端口，
校验器保证 `127.0.0.1` 字面量与端口全局唯一。

## 3. 模块与八条业务要求的落点

| 要求 | 实现位置 | 关键语义 |
| --- | --- | --- |
| 1 强校验 | `model.py` | 固定顺序遍历，首个非法对象以 `E_SPEC` 报 `object_id`+`field`；拒绝 bool 冒充整数、非回环地址、重复端口、未知依赖与环 |
| 2 就绪冻结 | `freeze.py` | 全部上游结局+摘要可用才解冻；否则 `UPSTREAM_HELD` 并记录缺失上游；全部迭代按排序后的 id 进行，与字典/哈希顺序无关 |
| 3 租约 | `leases.py` | 租约含 id/分片/节点/授予毫秒/到期毫秒/版本；撤销乐观校验版本，不符抛 `E_LEASE`；冻结、过期、撤销、完成四条释放路径汇聚到同一幂等裁决点，禁止重复归还预算 |
| 4 三重约束调度 | `scheduler.py` | 全局预算、节点租约上限、节点并发上限同时约束；真实 `ThreadPoolExecutor` + 回环子进程；耗尽即排队并记录单调毫秒等待起止，无人工延迟冒充配额 |
| 5 撤销容差 | `revocation.py` | 区分容差内不可区分/租约版本冲突/上游未定；容差内按「分片 id 升序、租约版本降序」稳定选主并标注 `tolerance_affected`；冻结分片不进入撤销统计 |
| 6 失效传播 | `invalidation.py` | 容量版本、节点配额、分片需求三类变更沿依赖图与节点引用传播；输出稳定拓扑序失效集与每步 `basis`；仅失效子集重跑，未失效条目 `entry_hash` 逐字节保持 |
| 7 原子账本/快照 | `ledger.py` | 结局与冻结迁移在单事务内写多表；事件以 SHA-256 链式摘要串联，携带整数格式版本；重开身份/摘要不符即整库替换并保留只读备份，不做增量合并 |
| 8 强杀恢复/重放 | `recovery.py` | 只信已提交记录；未提交在途分片标 `UNKNOWN`、先回收预算再重跑；重算逐字段比对，首个不一致以 `E_REPLAY` 报分片 id 与字段 |

## 4. 关键设计决策（便于独立复核）

1. **定点口径**：样本列为 `(group_id, raw_value)`，定点值 = `raw_value * scale`，
   分组 `count/sum/min/max` 与总量全部为 int64，不存在浮点，撤销差值可精确比较。
2. **确定性**：RNG 种子 = `blake2b(node_seed : shard_id)` 的 63 位派生值；
   节点进程与重放校验共用 `sampling.py`，同输入必然同摘要。
3. **唯一释放裁决**：`LeaseManager._release_locked` 是唯一释放点，先到路径写入
   `release_reason`，其余路径观察到 `released=True` 直接返回 `None`。
4. **冻结与撤销隔离**：撤销入口先判冻结，冻结分片结论固定为
   `UPSTREAM_UNDETERMINED` 且只计入 `frozen_excluded`。
5. **容量演进的历史正确性**：重放按账本提交时记录的 `capacity_version` 与节点
   身份核对，重算摘要与 `summary_digest`；演进只使失效子集重算，不追溯改写历史。
6. **单调时钟**：调度用 `MonotonicClock`（`time.monotonic_ns`），测试用
   `ManualClock` 精确驱动过期边界，不依赖 wall-clock。

## 5. 错误码契约

- `E_SPEC`：声明非法，带首个非法对象 id 与字段。
- `E_UPSTREAM`：上游结局/摘要缺失导致保持冻结，带缺失上游标识。
- `E_LEASE`：租约版本乐观校验失败或无活跃租约。
- `E_REPLAY`：重算结果与已提交账本逐字段不一致，定位到首个分片 id 与字段。

## 6. 验证结果摘要

最近一次 `python3 -m sharding_diag.demo_run` 与
`python3 -m unittest discover -s tests` 均通过，结构化结果见 `artifacts/`，
完整数值见 `VERIFICATION.md`。
