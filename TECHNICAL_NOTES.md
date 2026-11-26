# shardcast 技术说明与验证记录

## 1. 环境与结构

- 运行环境：WSL Ubuntu，Python 3.11，NumPy 1.26.4，SQLite 标准库 3.37.2。
- 虚拟环境：`.venv`（通过 `virtualenv` 创建，因镜像内 `ensurepip` 不可用）。
- 包结构：
  - `shardcast/spec.py`：JSON 声明加载、强校验、无环检测、结构/内容摘要。
  - `shardcast/sample.py`：确定性整数样本与 NumPy 定点分组聚合。
  - `shardcast/protocol.py`：JSON-lines 分帧、真实 TCP 握手/超时、字段校验。
  - `shardcast/node.py`：节点子进程服务（多线程真实监听、租约乐观校验）。
  - `shardcast/worker_pool.py`：子进程编排、就绪探测、强制重启、端口占用判定。
  - `shardcast/leases.py`：租约生命周期、单调时钟过期、唯一裁决序号。
  - `shardcast/capacity.py`：容量视图整体替换、只读备份、差异列表、陈旧标记。
  - `shardcast/engine.py`：三重约束并行调度、依赖就绪、冻结/取消、等待计量。
  - `shardcast/propagation.py` / `rebuild.py`：失效传播与全量/增量逐字段对比。
  - `shardcast/ledger.py`：SQLite 原子提交、快照识别与整体替换。
  - `shardcast/recovery.py`：强制终止后仅基于已提交记录的恢复重放。

## 2. 确定性规则（可复现口径）

1. 样本熵：对 `"{seed}|{shard_id}|group"` 与 `"...|value"` 分别做 FNV-1a 64 位
   哈希，得到两个显式 uint64 熵字，灌入 `numpy.random.SeedSequence` 与 `SFC64`。
2. 样本：`groups = integers(0, 16, int64)`，`values = integers(-1000, 1001, int64)`，
   按行主序拼成 `(N,2)` int64；`sample_bytes` 输出小端原始字节，逐位可复现。
3. 参与规则：仅 `value >= 0` 的行参与聚合，参与行数显式写入摘要。
4. 聚合：稳定排序后用 `reduceat` 向量化求每组 `count/sum/min/max`；空组省略。
5. 批次无关：`merge_aggregate` 为纯函数（不改输入），满足结合律/交换律，因此任意
   批次切分、下发顺序、并发度下摘要逐字段一致，字节摘要（SHA-256）随账本持久化。
6. 归属裁决：每次 grant/renew/release/expire/revoke 都带全局单调 `adjudication`，
   同分片在任一时刻至多一个 active 租约；归属变化可凭该唯一序号回溯。

## 3. 错误码映射

| 码 | 触发点 | 路径定位 |
| --- | --- | --- |
| `E_SPEC` | 声明非法（唯一标识、回环字面量、端口范围/重复、非负/正整数、依赖存在且无环、事件引用） | 首个出错字段路径，如 `nodes[0].port`，校验阶段不启动任何子进程 |
| `E_PORT` | 启动前端口被占、bind 失败、连接被拒、就绪超时 | `nodes.<id>.port` |
| `E_TIMEOUT` | 握手或 JSON 行读取超过真实 socket 超时 | `socket` / `host:port` |
| `E_FIELD` | 回执缺字段、身份/版本不匹配、JSON 非法 | `receipt.<field>` |
| `E_LEASE` | 未知/版本不符续约、重复属主；节点侧拒绝时不延长到期时刻 | `leases.<id>.lease_version` |
| `E_DIVERGE` | 增量与全量重建、恢复重放首个差异 | `outcomes.<shard>.<field>` |

## 4. 关键语义落地

- 租约时间全部来自 `time.monotonic()` 毫秒；过期只由真实时钟比较判定，无人工 sleep 代替
  （测试中的 `_sleep` 事件仅用于制造真实在途窗口）。
- 容量升级：`CapacityManager.replace` 生成不可变备份与差异，整体替换预算与每节点
  `concurrency/lease_limit`；`revoke_before_version` 撤销旧版本未到期租约并回收资源/名额，
  旧摘要与水位打 `stale` 标记，不参与当前汇总。
- 调度：每轮 `_schedule_pass` 对同分片至多下发一次；就绪集合要求全部上游终态且摘要可用；
  冻结分片不参与也不预占；取消立即释放额度且不再重试；等待开始/结束以单调毫秒记录。
- 三重约束：全局在途资源 ≤ 当前版本预算、单节点 active 租约数 ≤ lease_limit、
  单节点在途分片数 ≤ concurrency，越界即排队。
- 失效传播：直接失效（样本行数/资源需求/依赖边/优先级/容量版本/配额参数）按稳定拓扑序
  与分片标识序扩散到全部间接下游，并记录每步 `reason/caused_by/depth`。
- 账本：每个分片结局在单事务内写 `shard_outcomes + water_marks + event_cursor + meta`，
  携带整数 `LEDGER_FORMAT_VERSION=1` 与摘要 SHA-256；WAL 模式、`BEGIN IMMEDIATE`。
- 快照识别使用不含事件流的 `structural_digest` 与 `node_set_version`；不一致即导出只读
  备份后整体替换。强制 `SIGKILL` 不生成退出报告；恢复只信任完整提交行，缺失分片标记
  `unknown`，先回收其额度，再重放并逐字段比对，差异定位到分片与字段。

## 5. 复现步骤

```bash
# 环境（仓库内已创建 .venv）
/home/zsl/.feedback/python-base/bin/virtualenv .venv
.venv/bin/pip install -r requirements.txt

# 生成演示声明
.venv/bin/python -m demo.generate_spec --out demo/out/spec.json

# CLI 单次会话
.venv/bin/shardcast run \
  --spec demo/out/spec.json \
  --ledger demo/out/cli-ledger.db \
  --output demo/out/cli-result.json

# 四类验收场景（容量替换、依赖调度、失效传播、强制终止恢复）
.venv/bin/python -m demo.acceptance

# 全量测试
.venv/bin/python -m unittest discover -s tests
```

## 6. 实际验证结果

- 单元/集成测试：51 个全部通过（真实子进程、真实回环 TCP、真实 SQLite、真实单调时钟）。
- 验收证据：
  - `demo/out/01-baseline.json`：6 个分片全部 `completed`，依赖顺序与等待计量可见。
  - `demo/out/02-capacity-switch.json`：在途租约被 `revoke`、在途集合清空、配额差异与
    只读备份落盘，最终全部完成且归属裁决序号唯一。
  - `demo/out/03-propagation.json`：`s-alpha` 资源变更直接失效，`s-gamma/s-delta/
    s-epsilon` 作为间接下游按拓扑序失效，未失效分片摘要字节保持。
  - `demo/out/04-crash-recovery.json`：SIGKILL 后 3 个已提交分片被信任，3 个未知分片
    回收 7 个资源单位后重放成功，`mismatches` 为空。
- 确定性：不同批次大小（7/19/23/37/64/99）与两次独立会话的摘要 SHA-256 完全一致。

## 7. 范围说明

- 仅使用本地回环与确定性数据，不访问外部网络/真实账号/外部数据集，不使用容器。
- 节点崩溃恢复不声称产生“强制终止瞬间的退出报告”，该边界在代码与验收输出中显式声明。
