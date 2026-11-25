# Lease Middleware v0.1

可集成的确定性分布式锁/租约中间件首版。多个缓存节点以独立子进程监听 `127.0.0.1`，中心通过真实 TCP JSON-line 协议驱动版本标签代数链；调度器受全局与节点配额约束，SQLite 账本保存完整代数依据，并支持强杀恢复、独立重放校验和本地异常注入。

## 快速开始

```bash
# 若系统缺少 ensurepip，可使用 virtualenv；仓库已验证 .venv 流程
bash scripts/setup_env.sh

# 检查样本端口 48101-48103、崩溃端口 48191
./scripts/check_ports.sh

# 校验 JSON 样本
.venv/bin/python -m lease_middleware validate --sample-dir samples

# 真实启动 3 个节点子进程，运行同一事件流的正序/逆序启动演示
bash scripts/run_demo.sh

# 全量测试、演示、重放、崩溃恢复和注入
bash scripts/full_check.sh
```

演示结论写入：

- `artifacts/demo/order-forward/conclusion.json`
- `artifacts/demo/order-reverse/conclusion.json`
- `artifacts/demo/comparison.json`
- `artifacts/demo/order-forward/ledger.sqlite`

## 常用命令

```bash
# 独立重放：只读原始声明和 SQLite 账本
.venv/bin/python -m lease_middleware verify \
  --sample-dir samples \
  --ledger artifacts/demo/order-forward/ledger.sqlite

# SIGKILL 中心进程并恢复未知租约
.venv/bin/python -m lease_middleware crash-demo --root .

# 稳定业务码注入
.venv/bin/python -m lease_middleware inject --kind spec     --root .
.venv/bin/python -m lease_middleware inject --kind lease    --root .
.venv/bin/python -m lease_middleware inject --kind snapshot --root .
.venv/bin/python -m lease_middleware inject --kind verify   --root .
.venv/bin/python -m lease_middleware inject --kind port     --root .
.venv/bin/python -m lease_middleware inject --kind timeout  --root .
.venv/bin/python -m lease_middleware inject --kind reply    --root .
```

## 稳定业务码

| 业务码 | 触发条件 |
| --- | --- |
| `E_SPEC` | JSON 语法、字段、唯一性、端口字面量或事件序号校验失败 |
| `E_PORT` | 回环端口被占用或 TCP 端口不可用 |
| `E_TIMEOUT` | 真实 socket 建连/握手/读回执超时 |
| `E_REPLY` | 回执不是 JSON、缺少必填字段或节点标识不符 |
| `E_LEASE` | 标签不匹配、代数跳号/回退、非法状态迁移或容量双重持有 |
| `E_VERIFY` | 独立重放与有效摘要账本中的业务记录不一致 |
| `E_SNAPSHOT` | SQLite 整数版本头或内容摘要不一致 |

快照 JSON 的版本/摘要不一致不会增量合并，而是从已完整提交的 SQLite 账本整体替换。

## 样本设计

`samples/nodes.json` 声明 3 个 `127.0.0.1` 节点、全局在途上限 3、节点租约/在途上限和初始节点代数；30ms 的确定性处理延迟用于稳定观察并行峰值。

`samples/lease_requests.json` 声明 6 份租约申请和容量权益 `equity-1..5`。

`samples/events.json` 覆盖：

- 5 个授予请求的真实并行调度；
- `lease-a1` 连续续约：标签 `1 -> 2 -> 3`；
- 释放后同一容量权益转移给换代节点上的 `lease-b3`；
- cache-b 换代先回收、TTL 扫描后抑制重复到期回收；
- cache-c TTL 先回收、换代事件后抑制重复旧代数作废；
- topology/quota 版本 `1 -> 2 -> 3` 的子集失效传播。

当前样本结论 SHA-256 以实际 `artifacts/demo/comparison.json` 为准；正序和逆序节点启动产生逐字节相同的 `conclusion.json`。

## 代码地图

- `lease_middleware/spec.py`: JSON 强校验和 `E_SPEC` 字段定位。
- `lease_middleware/node_server.py`: 节点标签权威和 JSON-line TCP 子进程入口。
- `lease_middleware/protocol.py`: 中心 TCP 客户端、超时和回执字段校验。
- `lease_middleware/cluster.py`: 节点子进程启动、握手、端口占用识别和清理。
- `lease_middleware/statemachine.py`: 显式状态迁移许可表与标签匹配。
- `lease_middleware/engine.py`: 纯确定性代数链、唯一归属和回收裁决。
- `lease_middleware/scheduler.py`: 全局/节点配额、单调时钟等待记录、线程池投射。
- `lease_middleware/propagation.py`: 版本演进后的子集传播和稳定推导步骤。
- `lease_middleware/ledger.py`: SQLite 事务、意图、链式摘要、快照整体替换。
- `lease_middleware/verify.py`: 只读重放和首个 `E_VERIFY` 差异定位。
- `lease_middleware/faults.py`: 本地异常注入。
- `lease_middleware/crash.py`: 真实 `SIGKILL` 与恢复演示。
- `tests/`: 22 个单元/集成测试。

## Tkinter

可选只读查看器：

```bash
.venv/bin/python -m lease_middleware.tk_monitor \
  --path artifacts/demo/order-forward/conclusion.json
```

Ubuntu/WSL 正式安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

无显示环境中不要启动窗口；导入验证或 JSON/CLI 流程不受影响。
