# 分片隧道互操作组件

面向本地隧道与 VPN 转发场景的 Python 3.11+ 组件。它在一个可运行的最小系统中覆盖以下闭环：

- 本地 JSON 声明强校验与首错字段路径。
- f1、f2、f3 三种历史分片线格式的编解码、校验、归一和双向迁移。
- 固定种子内层样本、稳定报文标识、确定性分片和重组。
- Windows 兼容的 `127.0.0.1` 回环 UDP 子进程、真实套接字投递与真实超时。
- 全局在途上限和每隧道配额约束下的线程池并行转发。
- 声明演进后的依赖传播、精确失效集合和仅失效子集重算。
- SQLite 原子批次账本、强制终止恢复、快照整体替换、逐分片重放与三类异常注入。

## 快速开始

### Windows

```bat
python --version
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy==1.26.4 pytest==8.2.0
python -m pip install -e .
```

### Linux/macOS/WSL 等价命令

```bash
python --version
python -m venv .venv
source .venv/bin/activate
pip install numpy==1.26.4 pytest==8.2.0
python -m pip install -e .
```

运行测试与完整验收矩阵：

```bash
pytest -q
python -m tunnel_interop verify \
  --spec examples/spec.json \
  --evolved-spec examples/spec_evolved.json \
  --output-dir artifacts/verification \
  --log-file artifacts/verification/verify.log \
  --out artifacts/verification/verify-summary.json
```

`verify` 会真实创建 UDP 接收子进程，并依次运行正常顺序、反序到达、重复到达三种场景，再执行账本重放、格式迁移、失效传播和快照替换。

## 声明文件

示例见 `examples/spec.json`，演进后版本见 `examples/spec_evolved.json`。

```json
{
  "versions": {
    "declaration": 1,
    "endpoint_set": 1,
    "route_table": 1,
    "fragment_policy": 1
  },
  "endpoints": [
    {
      "id": "ep-edge-a",
      "host": "127.0.0.1",
      "port": 46371,
      "mtu": 1400,
      "encapsulation_overhead": 96,
      "replay_window": 64,
      "seed": 73013
    }
  ],
  "routes": [
    {
      "prefix": "10.40.0.0/16",
      "next_hop": "ep-edge-a",
      "priority": 100,
      "tunnel_id": "tun-alpha"
    }
  ],
  "fragment_policy": {
    "fragment_size": 512,
    "max_retries": 2,
    "reassembly_timeout_ms": 750
  },
  "runtime": {
    "global_inflight_limit": 3,
    "per_tunnel_inflight_quota": 2,
    "ack_timeout_ms": 400,
    "fragment_format": "f3",
    "max_message_bytes": 8388608,
    "receiver_idle_timeout_ms": 5000
  },
  "sample": {
    "bytes_per_message": 1600
  }
}
```

校验规则包括：端点标识与回环端口全局唯一；端口范围 `1..65535`；MTU 和封装开销为正整数且开销严格小于 MTU；重放窗口为正整数；路由前缀合法且下一跳存在；隧道标识唯一；分片档位不超过所有端点的最小有效载荷；重传次数和重组超时非负；调度配额不超过全局上限。非法输入首个违规项以 `E_SPEC` 返回，并给出如 `$.endpoints[0].port` 的字段路径。

`versions` 可省略，默认全为 `1`。任何端点、路由、分片策略、运行参数或样本参数变化都必须递增 `declaration`；对应的端点集合、路由表或分片策略版本也必须在内容变化时递增，否则拒绝为 `E_SPEC`。

## 命令行

```bash
# 强校验并输出声明摘要
python -m tunnel_interop validate --spec examples/spec.json

# f1/f2/f3 归一和迁移一致性检查
python -m tunnel_interop interop --spec examples/spec.json \
  --out artifacts/verification/format_interop.json

# 真实 UDP 发送、回执、重组并写入 SQLite 批次账本
python -m tunnel_interop forward \
  --spec examples/spec.json \
  --ledger artifacts/verification/session.db \
  --scenario normal \
  --log-file artifacts/verification/forward.log

# 声明演进后的精确失效集合与选择性重算
python -m tunnel_interop evolve \
  --old examples/spec.json \
  --new examples/spec_evolved.json \
  --out artifacts/verification/evolution.json

# 恢复：校验内容摘要、整体替换不兼容快照、重放已提交分片
python -m tunnel_interop resume \
  --spec examples/spec.json \
  --ledger artifacts/verification/session.db \
  --out artifacts/verification/resume.json

# 重跑被标记为 unknown 的未提交批次
python -m tunnel_interop resume \
  --spec examples/spec.json \
  --ledger artifacts/verification/session.db \
  --rerun

# 三类异常注入
python -m tunnel_interop inject --kind corrupt-spec \
  --spec examples/spec.json --output-dir artifacts/inject/corrupt
python -m tunnel_interop inject --kind dangling-route \
  --spec examples/spec.json --output-dir artifacts/inject/dangling
python -m tunnel_interop inject --kind snapshot-digest \
  --spec examples/spec.json --ledger artifacts/verification/session.db \
  --output-dir artifacts/inject/snapshot
```

`serve` 子命令也可单独托管一个声明端点：

```bash
python -m tunnel_interop serve --endpoint-id ep-edge-a --host 127.0.0.1 \
  --port 46371 --ready-file artifacts/receiver-ready.json
```

## Python 集成

```python
from tunnel_interop import build_bundle, evolve_bundle, load_spec

spec = load_spec("examples/spec.json")
bundle = build_bundle(spec, target_format="f3")

for message in bundle.messages:
    print(message.plan.message_id, message.reassembly.payload_sha256)

new_spec = load_spec("examples/spec_evolved.json")
result = evolve_bundle(bundle, old_spec=spec, new_spec=new_spec)
print(result.invalid_fragments)
print(result.unchanged_checks)
```

主要模块：

- `tunnel_interop/spec.py`：声明模型、强校验、版本单调性。
- `tunnel_interop/wire.py`：三种二进制格式、校验规则、统一视图和格式迁移。
- `tunnel_interop/samples.py`：固定种子样本、报文标识和推导依据。
- `tunnel_interop/fragmenter.py`：分片、三种格式变体、重组。
- `tunnel_interop/transport.py` 与 `tunnel_interop/udp_receiver.py`：真实回环 UDP 收发。
- `tunnel_interop/scheduler.py`：配额调度和可复现统计。
- `tunnel_interop/invalidation.py`：依赖传播与选择性重算。
- `tunnel_interop/ledger.py`：SQLite 原子账本、快照替换和重放。
- `tunnel_interop/workflow.py`：端到端集成流程与验收矩阵。

## 关键语义

三种线格式、校验码、样本和标识推导的逐字节规则见 `docs/PROTOCOL.md`。

调度统计分两层：`real_trace` 保存真实的 `time.monotonic_ns()` 毫秒审计值；`deterministic_statistics` 使用“每完成一个准入波次推进 1 ms”的逻辑单调时钟，保证相同任务、相同全局上限和相同隧道配额下逐字节一致。两层约束都由真实 `ThreadPoolExecutor` 执行，且调度器在准入阶段严格检查全局和单隧道配额。

SQLite 批次开始时只提交 `intent` 与计划分片；完成时在一个事务内写入分片归属、封包摘要、重组状态、调度轨迹和进度游标。强制退出后，`intent`/`failed` 批次被标记为 `unknown`，列出未知分片并要求重跑，`can_claim_exit_report=false`，`exit_report=null`。重新打开时先验证账本内容摘要；声明版本或内容不一致时生成新 generation 并整体替换活动快照，不把旧批次与新声明增量合并。

## 实际验证

当前工作区已执行 64 项 pytest 测试和完整 `verify` 矩阵，结果与复现命令见 `VERIFICATION.md`。测试环境、输出摘要和每类业务码均有记录。

## 边界说明

- f1 本身不携带总片数、报文标识和生成时刻。对非零偏移或不完整单片的归一必须提供消息清单上下文；组件以 `padding_basis` 明示补齐来源，且迁移后统一视图的关键字段保持一致。
- UDP 回执采用本实验的 JSON 控制面，隧道数据报本身是紧凑二进制格式。回执时间是接收侧真实墙钟毫秒，发送超时来自真实 socket 超时。
- `reassembly_timeout_ms=0` 表示不启用额外批次重组截止期；`ack_timeout_ms` 仍然真实生效。
- 反序验收场景主动把该批次有效并发压到 `1`，以稳定产生真实反序到达；这不绕过配额，而是使用更小配额进行确定性验收。
