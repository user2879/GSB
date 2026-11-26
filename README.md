# 推荐分片下发诊断组件

本项目实现一套可在单机 WSL/Linux 上真实运行的分片下发诊断链路。节点是真子进程，通过 `127.0.0.1` TCP 上的 JSON 行协议交换租约和聚合请求；样本、配置、事件和账本均由固定整数种子确定生成，不访问网络或外部数据集。

## 快速开始

```bash
cd /home/zsl/feedback/uidfc5909d3-default-374
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-build-isolation

reco-shard-diag init --workdir work/demo --force
reco-shard-diag run --workdir work/demo --fresh --outdir work/demo/artifacts
reco-shard-diag acceptance --workdir work/acceptance --outdir artifacts/acceptance
```

若不想安装包，也可使用 `PYTHONPATH=src python3 -m reco_shard_diag.cli ...`；本仓库的 `Makefile` 已设置该路径。

## 验收产物

最近一次完整验收结果位于：

- `artifacts/acceptance/REPORT.md`：人读验收报告。
- `artifacts/acceptance/acceptance-report.json`：结构化验收结论。
- `artifacts/acceptance/diagnostic-report.json`：定点归因、注入、租约代次、等待分布与业务码计数。
- `artifacts/acceptance/full-run.json` 与 `resume-run.json`：全量与强杀恢复后的权威结论。
- `artifacts/acceptance/node-ledger.json`：各节点租约、覆盖投递和错误账本。
- `artifacts/acceptance/samples.jsonl`：最终分片集合对应的确定性本地样本。

本次实测：20 个 pytest 测试通过；验收 16 项检查全部通过；全量、增量、二次全量和强杀恢复的规范视图逐字段一致。业务码计数为 `{"E_QUOTA_SOFT_LIMIT": 1}`，五个注入场景全部命中，各节点最终租约代次为 `node-a=2`、`node-b=1`、`node-c=1`。

## 工程结构

```text
src/reco_shard_diag/
  models.py          节点、分片、批次、配额版本、事件与项目配置
  validation.py      E_xxx 业务码校验，定位对象和字段
  state.py           事件归约到最终状态
  generator.py       固定种子的 NumPy 确定性样本
  fixedpoint.py      NumPy 分组聚合、位宽饱和、万分比水位
  node.py            子进程节点、JSON 行协议、租约状态机和回收
  client.py          回环客户端、节点子进程管理
  keys.py            内容指纹与阶段缓存键
  coordinator.py     拓扑调度、预算/批点数/水位约束、复用与检查点
  persistence.py     临时目录提交、CURRENT 原子指针、版本/配置哈希识别
  diagnostics.py     位宽归因与独立异常注入
  fixtures.py        确定性本地配置和样本
  acceptance.py      自动验收矩阵
tests/               20 个单元与集成测试
```

## 业务对象与规则

### 配置与事件

节点声明包含唯一 `node_id`、唯一回环端口、正整数 `capacity_slots` 与 `lease_limit`。配额版本包含单调递增 `version`、显式 `parent_version`、节点容量/租约映射、万分比软硬水印、样本聚合位宽 `width` 和 `points_per_resource_unit`。

事件流支持：

- `batch_append`、`batch_replace`、`batch_delete`
- `shard_replace`、`shard_delete`
- `dependency_update`
- `quota_version_add`、`quota_rollback`
- `node_adjust`
- `width_switch`

事件按严格递增 `seq` 归约；最终聚合、依赖、水位、复用和恢复结论只依赖最终集合，不依赖线程完成先后。

### 节点协议

每次 TCP 连接发送一行 JSON，并接收一行 JSON。常用操作：

```json
{"operation":"acquire","shard_id":"shard-a","ttl_ms":3000,"expected_generation":0}
{"operation":"aggregate","request_id":"r1","shard_id":"shard-a","quota_version":2,"width":12,"dimensions":3,"samples":[...],"lease_id":"lease-node-a-000001","expected_generation":1}
{"operation":"renew","lease_id":"lease-node-a-000001","expected_generation":1,"ttl_ms":3000}
{"operation":"release","lease_id":"lease-node-a-000001","expected_generation":1}
{"operation":"sweep"}
{"operation":"status"}
{"operation":"ledger"}
```

所有应答带 `node_id` 和节点当前公认 `lease_generation`。租约按自身代次严格校验；代次不匹配返回 `E_LEASE_GENERATION_MISMATCH` 且代次不变。超过到期毫秒后由 `sweep` 真正移除在途单元并记 `E_LEASE_EXPIRED`，后续取消失败请求仍返回过期码，新请求可复用空出的槽位。

### 定点聚合与水位

节点用 NumPy `int64` 按样本分组求和，再按声明的有符号位宽饱和量化。分组顺序固定为字典序。水位采用独立安全整数域：

```text
soft_limit = floor(capacity_slots * soft_watermark_bp / 10000)
hard_limit = floor(capacity_slots * hard_watermark_bp / 10000)
```

资源单位 `resource_units > soft_limit` 记预警 `E_QUOTA_SOFT_LIMIT`；`resource_units > hard_limit` 拒收新派发并记 `E_QUOTA_HARD_LIMIT`。固定取整与边界规则同时写入结果元数据。

### 缓存键与失效边界

阶段缓存键由分片样本指纹、上游摘要指纹、上游阶段产物键、活跃配额版本指纹、位宽和逻辑租约代次派生。上层替换、依赖改写、配额回滚或位宽切换会传播到所有下游分片。`reuse_ledger`/`invalidation_ledger` 是根据最终事件集稳定推导的内容判决；`cache_hit_ledger`/`cache_miss_ledger` 记录本次实际持久化命中，便于复核两者。

### 调度约束

派发同时受三重约束：

- 全局 `resource_budget` 资源单位。
- 单批累计 `max_batch_points` 处理点数。
- 每节点租约上限和容量槽位。

调度顺序由依赖图和分片标识确定，分片结论按标识排序输出，因此不依赖节点完成顺序。

### 快照与恢复

每次分片完成都会把结果、待办队列、已完成游标、预算、缓存和租约链写入新的临时快照目录；校验文件落盘后通过 `CURRENT` 指针原子替换提交。启动时只读已提交指针，并同时核对 `global_version` 与基础配置哈希。`phase=RUNNING` 被归类为强制终止恢复，`phase=COMPLETE` 被归类为可处理退出。

强制中止示例：

```bash
reco-shard-diag run --workdir work/demo --fresh --crash-after 2
# 进程以 137 退出，最新已提交快照仍为 RUNNING
reco-shard-diag resume --workdir work/demo --outdir work/demo/artifacts
```

## 异常注入

诊断器在独立配置或独立节点运行时中制造以下场景，任一失败不会污染主分片集合：

- `I_DEPENDENCY_CYCLE`：依赖环，命中 `E_DEP_CYCLE`。
- `I_QUOTA_PARENT_MISSING`：父版本不存在，命中 `E_QUOTA_PARENT_MISSING`。
- `I_SAMPLE_DIMENSION`：单请求维度不符，命中 `E_SAMPLE_DIMENSION`，同节点后续请求仍成功。
- `I_STALE_LEASE`：陈旧代次续期，命中 `E_LEASE_GENERATION_MISMATCH`，代次保持不变。
- `I_EXPIRED_RENEW`：过期后续期，命中 `E_LEASE_EXPIRED`，回收后槽位可复用。

## 验证

```bash
make test
make acceptance
```

测试覆盖校验定位、定点与水位、真实 TCP 协议、重复投递覆盖关系、租约过期回收、快照版本哈希、事件全量归约、选择性复用、增量/全量一致性和强杀恢复一致性。
