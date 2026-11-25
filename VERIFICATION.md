# 实际验证结果

以下数值来自最近一次本地运行（WSL Ubuntu，Python 3.11，NumPy 1.26.4）。

## 1. 全量测试

`python3 -m unittest discover -s tests`：
- Ran 30 tests in 2.674s
- OK
- 覆盖：声明强校验 12 项、冻结/租约 6 项、撤销/失效 7 项、账本 3 项、真实子进程 E2E 2 项。

## 2. 端到端诊断（demo_run）

### 调度与排队（要求 4）
- 调度轮次：6；全局预算：12；提交分片：8
- 排队记录：s4 因 budget_exhausted 等待 64 单调毫秒

### 冻结判定（要求 2）
- s3 保持冻结，缺失上游 ['s1']
- s4 保持冻结，缺失上游 ['s1', 's2']
- s5 保持冻结，缺失上游 ['s2']
- s6 保持冻结，缺失上游 ['s3', 's4']
- s7 保持冻结，缺失上游 ['s4', 's5']
- s8 保持冻结，缺失上游 ['s6', 's7']
- s6 保持冻结，缺失上游 ['s3', 's4']
- s7 保持冻结，缺失上游 ['s4', 's5']
- s8 保持冻结，缺失上游 ['s6', 's7']
- s6 保持冻结，缺失上游 ['s4']
- s7 保持冻结，缺失上游 ['s4']
- s8 保持冻结，缺失上游 ['s6', 's7']
- s8 保持冻结，缺失上游 ['s6', 's7']
- 运行结束冻结分片：[]；就绪分片数：8

### 租约释放裁决（要求 3）
- 释放原因分布：{'complete': 8}；重复释放返回：None（禁止重复归还）

### 撤销容差裁决（要求 5）
- s1: TOLERANCE_INDISTINGUISHABLE，tolerance_affected=True
- s2: DISTINGUISHABLE，tolerance_affected=False
- s3: LEASE_VERSION_CONFLICT，tolerance_affected=False
- s8: UPSTREAM_UNDETERMINED，tolerance_affected=False
- 统计：{'DISTINGUISHABLE': 1, 'LEASE_VERSION_CONFLICT': 1, 'TOLERANCE_INDISTINGUISHABLE': 1, 'frozen_excluded': 1, 'tolerance_affected': 1}
- 错误版本撤销返回错误码：E_LEASE

### 失效传播与子集重算（要求 6）
- 变更：capacity_version node-b.capacity_version 1->2
- 变更：quota node-c.concurrency 1->2
- 变更：demand s2.demand 5->6
- 直接失效：['s2']
- 拓扑序失效集：['s2', 's4', 's5', 's6', 's7', 's8']
- 未失效且 entry_hash 保持：{'s1': True, 's3': True}
- 重放校验：{'checked_shards': 8, 'fields_per_shard': ['node_id', 'lease_version', 'capacity_version', 'summary_digest', 'fixed_total', 'status']}

### 快照整体替换（要求 7）
- 替换原因：capacity_versions
- 只读备份权限：0o444；新库结局数：0

### 强杀恢复与重放（要求 8）
- 强杀前已提交：['s1', 's2']
- UNKNOWN 分片：[{'budget_reclaimed': 6, 'node_id': 'node-a', 'shard_id': 's3', 'state': 'DISPATCHED'}]
- 回收预算：6；恢复后占用：0
- 重跑：{'reran': ['s3'], 'rounds': 2}；重放：{'checked_shards': 3, 'fields_per_shard': ['node_id', 'lease_version', 'capacity_version', 'summary_digest', 'fixed_total', 'status']}；链式摘要校验：True
- 最终结局分片：['s1', 's2', 's3']

## 3. 账本身份
- 格式版本：1
- 节点集合版本：ns:node-a@127.0.0.1:19001|node-b@127.0.0.1:19002|node-c@127.0.0.1:19003
- 容量版本映射：{'node-a': 1, 'node-b': 1, 'node-c': 1}
- 链式内容摘要头：f7a936c85f88df4325a509020d3e657e703f4a8b683b4f90f1c43775a611236c
