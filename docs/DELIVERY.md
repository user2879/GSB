# 分库分表路由工作台 — 交付说明

## 组成

| 路径 | 职责 |
| --- | --- |
| `workbench_app.py` | tkinter 交互工作台（Windows Python 3.11+ 自带 tkinter） |
| `demo.py` | 无头验收演示，生成 `artifacts/run_record.md` |
| `shardbench/config_loader.py` | JSON 配置 + 依赖图加载与强校验（`E_CONFIG`） |
| `shardbench/predicates.py` | 谓词规范化（eq / in / range / and）与分片键提取 |
| `shardbench/algorithms.py` | 取模 / 范围（显式边界归属）/ 一致性哈希环（虚拟节点+环回绕） |
| `shardbench/planner.py` | 路由计划编译，摘要绑定 rule_version 与 ring_version |
| `shardbench/leases.py` | 租约签发/续约（版本乐观校验）/真实单调时钟过期回收（`E_LEASE`） |
| `shardbench/scheduler.py` | 全局并发 + 每库配额双重约束的线程池调度，单调时钟等待窗口 |
| `shardbench/ledger.py` | SQLite 账本：批次原子提交、陈旧整体标记、当前视图排除陈旧行 |
| `shardbench/versions.py` | 规则/环/依赖图/预算版本单调递增注册表与单元差异 |
| `shardbench/invalidation.py` | 依赖图失效传播（稳定拓扑序）与仅失效子集重算 |
| `shardbench/snapshot.py` | 会话快照（版本+内容摘要校验，`E_SNAPSHOT` 整体替换） |
| `shardbench/faults.py` | 异常注入：配置损坏（`E_CONFIG`）、租约版本冲突（`E_LEASE`） |
| `samples/` | 固定本地样本：分片配置 v1/v2、谓词集合 |
| `tests/` | 44 项 pytest 用例 |

## 业务码

- `E_CONFIG`：配置/依赖图任一违规，整体拒绝并给出首个违规字段（如 `routing_units[0].algorithm.modulus`）。
- `E_LEASE`：租约版本冲突或租约非持有态；不延长租约，detail 含期望/当前版本。
- `E_SNAPSHOT`：快照内容摘要不一致或版本不一致；调用方整体替换而非逐条修补。
- `E_PREDICATE`：谓词结构非法。

## 运行

```bash
pip install numpy pytest          # numpy 为环境要求；核心逻辑仅用标准库
python -m pytest tests/ -q        # 44 项测试
python demo.py                    # 生成 artifacts/run_record.md
python workbench_app.py           # Windows 本地 GUI
```

## 关键设计

- **确定性**：所有哈希经 sha256(canonical JSON)；谓词按语义规范化（大小写/空白/IN 顺序/AND 顺序无关）；目标与理由排序后输出，不依赖字典遍历序。
- **租约**：单调时钟毫秒判定到期；续约凭版本号乐观校验；回收记录 `ttl_expired` 依据；账本 leases 表持久化每次状态迁移。
- **调度**：`BoundedSemaphore` 实现全局与每库配额，任务真实排队（无线程睡眠替代），等待起止以单调时钟毫秒入账。
- **版本替换**：规则/环版本递进时 `UPDATE ... SET stale=1` 整体标记旧版本行，当前视图与汇总只读 `stale=0`；计划缓存整体清空；差异稳定列出 added/removed/upgraded。
- **失效传播**：沿依赖图 BFS 得直接与间接下游，按 Kahn 稳定拓扑序输出；仅失效子集重算，引擎逐项校验未受影响单元摘要逐字节不变。
- **恢复**：快照含四版本+租约集合+缓存摘要+账本游标+内容摘要；强制终止后仅已提交批次可恢复，未释放租约标记 `unknown` 并要求重跑。

## 验证结果（本容器实际执行）

- `python -m pytest tests/ -q` → **44 passed**
- `python demo.py` → 全流程通过，记录见 `artifacts/run_record.md`：
  - 三种算法定位与理由、语义等价谓词 digest 一致（True）
  - 租约签发/续约/版本冲突 `E_LEASE`/真实过期回收
  - 双重配额峰值未超限（global≤3, per_db≤1），等待窗口入账
  - 规则 v1→v2：旧账本整体陈旧、当前视图清零、差异与失效集合稳定输出
  - 未受影响单元 `orders_by_amount` 摘要逐字节一致（True）
  - 快照篡改 → `E_SNAPSHOT(content_digest)`；崩溃恢复标记 unknown 租约
  - 异常注入：配置损坏 → `E_CONFIG`，租约冲突 → `E_LEASE`，均写结构化日志
- 注：本 Linux 容器无 Tk 动态库，GUI 以 `py_compile` 验证；Windows 版 Python 自带 tkinter 可直接运行。
