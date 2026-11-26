# ckptsim：确定性训练检查点、租约恢复与重放

`ckptsim` 是一个完全本地、无外部服务依赖的 Python 3.11 包。它把训练模拟拆成四个可独立验证的层次：

1. 强校验的 JSON 规格与事件流；
2. NumPy `int64` 定点前向与逐批损失摘要；
3. 每个分片一个真实 `127.0.0.1` TCP 子进程；
4. SQLite 原子账本、租约调度、检查点/回滚、快照替换与独立重放。

样例位于 `samples/`，测试位于 `tests/`，实际验证结果见 `docs/VALIDATION.md`。

## 快速开始

```bash
# 本环境自带 NumPy/pytest；用系统站点包建立本地 venv，避免联网安装
python3.11 -m venv --without-pip --system-site-packages .venv
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .

.venv/bin/python -m pytest -q
.venv/bin/python scripts/demo.py
```

在具备 `ensurepip` 的常规 Python 3.11 环境中可直接使用：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[test]'
```

也可直接使用 editable 安装后的命令：

```bash
.venv/bin/ckptsim validate --shards samples/shards.json \
  --layers samples/layers.json --checkpoints samples/checkpoints.json \
  --events samples/events.json

.venv/bin/ckptsim run --fresh --workdir artifacts/run \
  --shards samples/shards.json --layers samples/layers.json \
  --checkpoints samples/checkpoints.json --events samples/events.json

.venv/bin/ckptsim verify --ledger artifacts/run/ledger.sqlite \
  --shards samples/shards.json --layers samples/layers.json \
  --checkpoints samples/checkpoints.json --events samples/events.json
```

## JSON 规格

### 分片 `samples/shards.json`

```json
{
  "global_lease_limit": 2,
  "shards": [{
    "shard_id": "shard-a", "host": "127.0.0.1", "port": 19501,
    "lease_limit": 2, "concurrency": 2, "seed": 1001,
    "parameter_version": 1
  }]
}
```

- `shard_id` 唯一；`port` 全局唯一且 `host` 只能是字面量 `127.0.0.1`。
- 租约上限、并发配额必须为正整数；随机种子为整数。
- 分片启动版本必须存在于层参数清单。

### 层参数 `samples/layers.json`

```json
{
  "parameter_versions": [{
    "parameter_version": 1,
    "generation": 1,
    "layers": [{
      "layer_id": "token", "dimension": 4,
      "scale_factor": 1000, "derived_from": null
    }]
  }]
}
```

- 版本号唯一，`generation` 在清单中严格递增。
- 每层维度、缩放因子为正整数。
- `derived_from` 可为 `null`、层标识字符串或字符串数组；引用必须存在且整体无环。
- 拓扑序按层标识字典序稳定排序；父层参数摘要参与子层参数种子。

### 检查点与事件

```json
{"checkpoints": [
  {"checkpoint_id": "cp-base", "parameter_version": 1, "generation": 1}
]}
```

事件类型：

- `batch`：`seq, shard_id, batch_id, sample_ids, parameter_version?`
- `renew`：`seq, shard_id, batch_id, lease_version`
- `checkpoint`：`seq, checkpoint_id, generation?`
- `rollback`：`seq, checkpoint_id`
- `advance`：`seq, parameter_version, shard_ids?`

`seq` 必须非减；检查点/回滚只能引用清单中声明且已提交的检查点；提交世代必须递增。

## 核心语义

### 确定性定点计算

- 参数种子由分片种子、参数版本、层标识、维度、缩放因子、父层摘要共同决定。
- 样本由分片种子、批次标识、样本标识、根层标识决定，与参数版本无关。
- 前向使用 `int64` 矩阵乘法和整数整除缩放；损失使用逐元素平方与整数累加。
- 参数摘要覆盖规范化元数据和 `little-endian <i8` 张量字节；损失摘要绑定
  `shard_id, batch_id, sorted(sample_ids), parameter_version, parameter_digest, samples_digest, loss_units`。

因此，批次执行顺序、并发度、Python 字典遍历顺序均不影响参数摘要与损失摘要。

### 检查点与回滚

- 检查点提交冻结：参数版本、世代、每个分片的参数摘要、其拥有的批次及损失基线摘要。
- 回滚把全部分片当前版本恢复为目标检查点版本，并把会话世代推进 `+1`。
- 被撤销版本按 `generation` 降序输出；受影响的已提交批次按分片标识、批次序输出。
- 回滚后的新批次在预规划阶段即绑定恢复版本与推进后的世代。
- 检查点归属由事件序号预规划决定，因此事件流任意切分/并发执行不改变归属。

### 依赖失效与选择性重算

`ckptsim impact --edited-layers ...` 比较两份层参数清单：

1. 识别 `scale_factor`、`dimension`、`derived_from` 与层增删；
2. 沿派生图求直接/间接下游闭包；
3. 由“参数版本 → 批次 → 损失摘要 → 检查点”传播失效；
4. 仅重算失效批次与失效检查点基线，未失效记录直接复用；
5. 输出稳定步骤、依据、失效集合，以及未失效记录的复用前/后摘要证明。

### 三重租约调度

- 全局活跃租约上限、分片租约上限、分片在途批次并发配额分别计数。
- 不满足约束的批次进入按事件序号排序的等待队列；记录 `monotonic_ns` 毫秒等待起止。
- 租约含 `granted_ms / expires_ms / version`；后台 janitor 使用真实单调时钟到期并归还租约配额。
- 续约把版本号发给分片 worker 做乐观校验；版本不符返回 `E_LEASE`，worker 在扩展到期时间之前完成校验。
- 批次完成会释放调度配额；worker 保留租约见证至 TTL，以支持紧随批次的续约事件。

### SQLite 账本与快照

- 每次批次完成在一个 `BEGIN IMMEDIATE` 事务中写批次归属、损失、参数版本、检查点状态、租约归属与进度游标。
- 开始执行时先提交 `inflight` 标记；进程中断后该标记恢复为 `unknown`，相关租约改为 `released_recovered/released_failed`。
- 账本保存 `batch_attempts`，重跑成功后仍保留首次 `unknown` 证据。
- 快照身份包含整数格式版本、分片集合版本、参数版本口径和声明内容摘要。
- 任一身份字段变化时，不合并旧行：先对旧库执行 WAL checkpoint，生成 `0444` 只读备份，再删库整体重建并写出差异列表与时间戳等元数据到 `*.snapshot.json`。

### 独立重放校验

`ckptsim verify` 以 SQLite 只读 URI 打开账本，不启动 worker，不写账本。它独立重推：

- 参数摘要；
- 样本摘要与逐批损失摘要；
- 检查点参数摘要、损失基线、批次归属和回滚状态；
- 进度游标；
- 可选传入 `--edited-layers` 时重推失效报告并与账本中的 `impact_runs` 比较。

首个不一致以 `E_VERIFY` 返回，包含 `shard_id`、`batch_id`、`field`、期望值、实际值和作用域。

## 错误码

| 错误码 | 场景 |
|---|---|
| `E_SPEC` | JSON/规格/事件语义非法；包含首个非法条目与行号 |
| `E_PORT` | 回环端口被占用或 TCP 连接失败 |
| `E_TIMEOUT` | 真实套接字连接/读取超时 |
| `E_FIELD` | 请求/回执缺失或字段类型、形状、身份不一致 |
| `E_LEASE` | 租约不存在、已到期或乐观版本校验失败 |
| `E_VERIFY` | 独立重放发现账本差异 |

## 强制终止语义

若 worker 被强制终止：

- 已完成原子事务的批次、损失摘要与检查点状态仍可从账本读取；
- 已有 `inflight`/未提交批次标记为 `unknown`，必须先回收租约并重跑；
- 不生成或伪造强制终止瞬间的退出报告；运行结果中的 `forced_exit_report` 恒为 `null`；
- `scripts/demo.py` 会真实 `SIGKILL` 一个正在服务的 shard 进程，重启后重跑 unknown 批次，并由独立校验确认最终账本一致。

## 需求映射

| 要求 | 实现位置 | 验证 |
|---|---|---|
| JSON 强校验与行号 | `src/ckptsim/spec.py` | `tests/test_spec.py` |
| 真实 TCP 子进程与错误码 | `worker.py`, `protocol.py`, `process.py` | `tests/test_protocol.py` |
| 定点前向与确定性摘要 | `model.py` | `tests/test_model.py` |
| 检查点/回滚归属 | `plan.py`, `ledger.py`, `runner.py` | 端到端测试、演示 |
| 依赖失效子集 | `impact.py` | `tests/test_end_to_end.py` |
| 三重租约与并发 | `lease.py` | `tests/test_lease.py` |
| SQLite/快照替换 | `ledger.py` | `tests/test_plan_ledger.py` |
| 强杀恢复与独立重放 | `runner.py`, `verify.py` | 端到端测试、演示 |

## 演示产物

`scripts/demo.py` 在 `artifacts/demo/` 写出：

- `summary.json`：总摘要；
- `run.json`：首次运行；
- `deterministic-run.json` / `deterministic-repeat.json`：两次运行的确定性投影；
- `verify.json` / `verify-impact.json`：独立校验；
- `impact.json`：失效传播与选择性重算；
- `crash/`：强杀恢复账本与校验；
- `snapshot-replacement.json`：快照整体替换与只读备份信息；
- 只读备份文件 `ledger.sqlite.backup-*.sqlite`。
