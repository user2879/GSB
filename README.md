# GSB 本地训练分片检查点诊断

这是一个不依赖容器、账号或外部服务的确定性诊断实现。主进程为每个训练分片启动独立 Python 子进程；子进程在 `127.0.0.1` 上以 TCP JSONL 协议接收批次，主进程通过真实套接字调度、签发租约，并把完成批次原子写入 SQLite 账本。

## 已实现能力

- 带来源行号的强校验：唯一 ID、回环地址与端口、非负配额、DAG 引用/无环、层输入维度、检查点引用、事件顺序。首个错误统一报 `E_SPEC`，校验通过前不会启动子进程。
- 真实 TCP 子进程：绑定冲突 `E_PORT`、握手/读取超时 `E_TIMEOUT`、回执字段缺失 `E_FIELD`。
- NumPy 整数定点前向：参数由 `(shard, seed, parameter_version, layer)` 稳定派生；样本 ID 排序后再计算，批次下发顺序、样本集合顺序和并发度不影响参数摘要与损失摘要。
- 检查点提交/回滚：提交冻结参数摘要与按到达序排列的损失基线；回滚恢复目标参数版本并推进世代号。完成顺序不参与状态确定性。
- 租约：签发、乐观版本续约、释放、真实 `time.monotonic_ns()` 过期。版本不符报 `E_LEASE` 且不延长到期时刻；子进程复核版本。
- 三配额调度：全局资源单位、每分片租约上限、每分片并发配额；批次进入真实 `ThreadPoolExecutor`。预算/定点水位落入合成误差界时按 `(shard_id, batch_order, batch_id)` 决胜并标记 `error_affected`。
- 失效传播：按分片、批次稳定输出直接/间接失效批次、损失摘要、检查点及依据；未失效批次提供规范 JSON 字节一致性证明。
- SQLite 账本：每批在单事务内写批次归属、参数/损失摘要、检查点状态、租约归属、进度游标、整数格式版本与内容摘要。快照身份或内容摘要不符时整体替换；未提交批次在恢复时标为 `unknown` 并回收租约；重放差异定位首个批次、分片与字段，报 `E_REPLAY`。
- 可选 Tkinter 面板：`gsb gui`。核心诊断不依赖 GUI。

## 环境

已在 Linux/WSL Ubuntu 风格环境验证：

```text
Python 3.11.0rc1
numpy 1.26.4
```

安装本项目：

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
# 或直接使用源码目录中的模块
python3 -m gsb.cli --help
```

本机若没有 `ensurepip`/`venv` 或 pip 版本过旧，可直接在仓库根目录使用 `python3 -m gsb.cli`，无需安装包；NumPy 已由 `requirements.txt` 声明。

可选 GUI 依赖：

```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

当前执行环境没有 `libtk8.6.so` 且无 root 权限，因此 `gsb gui` 会明确报 `E_UI`；所有校验、网络、调度、账本和重放能力均不依赖 Tk。

## 快速验收

```bash
# 1. 只校验，不会启动子进程
python3 -m gsb.cli validate --config samples/session.json

# 2. 绑定样本端口并真实运行；端口被占用时返回 E_PORT
python3 -m gsb.cli run \
  --config samples/session.json \
  --ledger /tmp/gsb-session.sqlite

# 3. 重放与从空白全量重建逐字段对比
python3 -m gsb.cli rebuild-check \
  --config samples/session.json \
  --ledger /tmp/gsb-session.sqlite

# 4. 配置变更后的失效子集
python3 -m gsb.cli invalidate \
  --old samples/session.json \
  --new samples/session_modified.json \
  --output /tmp/gsb-invalidation.json
```

样本使用固定的 39101/39102 端口。若本机端口冲突，复制样本并改为高位空闲端口即可；冲突本身会按 `E_PORT` 报告。

## JSON 结构

完整样本见 `samples/session.json`。也支持 `--shards`、`--checkpoints`、`--batches`、`--events` 分文件加载，并保留各自原始行号。

```json
{
  "format_version": 1,
  "global_budget": 4,
  "shards": [
    {
      "shard_id": "shard-a",
      "host": "127.0.0.1",
      "port": 39101,
      "lease_limit": 2,
      "concurrency_limit": 2,
      "seed": 1001,
      "parameter_version": 2,
      "sample_dim": 4,
      "lease_ttl_ms": 30000,
      "layers": [
        {"layer_id": "embed", "input_dim": 4, "output_dim": 3, "scale": 1000, "depends_on": []},
        {"layer_id": "head", "input_dim": 3, "output_dim": 2, "scale": 500, "depends_on": ["embed"]}
      ]
    }
  ],
  "checkpoints": [
    {"checkpoint_id": "base", "shard_id": "shard-a", "parameter_version": 1, "loss_baseline_digest": "0000000000000000000000000000000000000000000000000000000000000000"}
  ],
  "batches": [
    {"batch_id": "b1", "shard_id": "shard-a", "sample_ids": [1, 2], "resource_units": 2, "order": 1, "waterline": 100}
  ],
  "events": [
    {"seq": 10, "type": "commit", "shard_id": "shard-a", "checkpoint_id": "base"},
    {"seq": 20, "type": "renew", "shard_id": "shard-a", "lease_id": "lease:shard-a:1", "lease_version": 1, "ttl_ms": 30000},
    {"seq": 30, "type": "cancel", "shard_id": "shard-a", "lease_id": "lease:shard-a:1"}
  ]
}
```

约定：

- `batch.order` 与 `event.seq` 共同构成到达序；同值时事件先于批次。事件必须按严格递增 `seq` 声明。
- 无依赖层的输入维度等于 `sample_dim`；有依赖层的输入维度等于其 `depends_on` 输出维度之和。
- 定点层计算为 `y = floor((x @ W + b) / scale)`，全程整数。叶子层目标由固定种子生成，损失为整数平方误差之和。
- 损失摘要绑定 `shard_id`、`batch_id`、排序后的 `sample_ids`、`parameter_version`、`parameter_digest` 与损失值。
- `loss_baseline_digest` 必须是 64 位小写十六进制字符串。提交时账本会用实际冻结基线覆盖声明值。
- 批次的 `depends_on_batches` 用于显式失效传播；`checkpoint_id` 必须属于同一分片。

## 错误码

| 错误码 | 含义 |
| --- | --- |
| `E_SPEC` | 首个非法声明字段或 JSON 错误，带 `field`、`source`、`line` |
| `E_PORT` | 绑定或连接回环端口失败 |
| `E_TIMEOUT` | 子进程就绪、握手或套接字读取超时 |
| `E_FIELD` | 请求/回执缺字段、类型错误或身份不匹配 |
| `E_LEASE` | 未知/过期租约或乐观版本冲突 |
| `E_REPLAY` | 重放、账本一致性或工作者结果与确定性本地结果不符 |
| `E_QUOTA` | 批次永远无法满足预算或配额 |
| `E_UI` | 可选 Tkinter 依赖/显示环境不可用 |

## 验证

```bash
python3 -m unittest discover -s tests -v
```

最近一次结果：`Ran 30 tests ... OK`，覆盖真实子进程、端口冲突、握手超时、缺字段、线程池配额、检查点回滚、租约真实过期、SIGKILL 后恢复、快照整体替换、`E_REPLAY` 定位、失效传播和全量重建对比。详见 `docs/verification.md`。

## 文档

- `docs/design.md`：状态机、确定性定义、调度与账本不变量。
- `docs/verification.md`：实际环境、命令与结果记录。
