# 实际验证记录

环境：WSL Ubuntu，Python 3.11.0rc1，虚拟环境 `.venv`，numpy 1.26.4，pytest 8.2.2。

## 环境准备

```bash
python3 -m venv .venv            # 系统 ensurepip 缺失时改用 virtualenv 创建
source .venv/bin/activate
pip install numpy==1.26.4 pytest==8.2.2
pip install -e .
```

回环端口 20001/20050/20099 绑定探测通过；样本节点使用 20001-20004。

## 测试结果

```text
$ python -m pytest -q
46 passed in ~30s
```

重复执行两轮均为 46 passed，无顺序抖动。

覆盖的验收点：

- 租约代数链：真实子进程 acquire/renew/release，标签错位拒绝且代数不变
  （`tests/test_node_protocol.py`）
- 到期真实回收：`E_LEASE_EXPIRED` 触发节点 tagged release，槽位与全局令牌释放
  （`tests/test_leases.py`、`tests/test_scheduler_e2e.py`）
- 三格式往返：c3->c1->c3 经 sidecar 无损还原，缺失标记与完整性标签生效，篡改被拒
  （`tests/test_receipts.py`）
- 权威裁决与依赖失效：配额/时钟/节点 id 优先级、并列保留、上游失效传播
  （`tests/test_arbitration.py`）
- 精确失效复用：单分片仅失效下游、配额调整仅影响对应节点、格式切换仅影响序列化
  （`tests/test_fingerprints.py`）
- 增量=全量：业务账本与失败清单逐字段一致，复用命中与失效均留审计
  （`tests/test_scheduler_e2e.py`）
- 强制终止恢复：SIGKILL 后仅采信已整体替换快照，tmp 残留忽略，报告标注缺失阶段
  （`tests/test_crash_recovery.py`、`tests/test_snapshot.py`）

## CLI 手工验证

```bash
python -m edge_cache validate --config samples/config
python -m edge_cache run --config samples/config --work-dir .local/run --format c3
python -m edge_cache rebuild --config samples/config --work-dir .local/rebuild --format c3
python -m edge_cache recover --work-dir .local/run
```

账本产物：

- `ledger/business.jsonl`：逐分片业务结论（按序号稳定排序）
- `ledger/failures.jsonl`：带业务码的失败清单
- `ledger/algebra.jsonl`：逐条租约代数事件
- `ledger/reuse.jsonl`：失效依据、复用命中、重算范围
- `ledger/exits.jsonl`：带业务码的退出记录
- `ledger/artifacts/`：内容寻址缓存产物
- `snapshots/committed/`：已整体替换的阶段快照与 manifest
