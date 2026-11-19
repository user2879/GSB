# dp-grad-sim

面向本地小型语言模型数据并行训练的确定性梯度通信仿真台。项目只使用本地合成样本、NumPy/PyYAML 与 `127.0.0.1` 上的 TCP 回环套接字，不下载模型权重或外部数据，不依赖 Docker 或容器环境。

核心能力：

- 固定种子初始化模型参数，按工作号、步号和分片摘要派生 `float32` 伪梯度。
- 对称/非对称定点量化，二进制压缩包头携带张量名、位宽、缩放因子和校验信息。
- 每个工作号绑定独立 TCP 监听端口，梯度压缩包通过真实回环套接字收发。
- 每窗口按链路预算、总窗口预算和误差容差选择位宽与聚合顺序，并记录候选评分和理由。
- 以 `float64` 精确聚合为真值，追踪打包、解压、跨工作号定点聚合三阶段误差。
- 保真梯度按窗口轮转保留，超代数时替换最旧代数并记录事件。
- 每个阶段写入临时目录，目录整体提交后原子替换根清单；支持中断恢复和强制终止后的完整快照判定。
- 线程池并发收发不改变聚合结果、归因顺序、决策顺序或失败清单顺序。

## 快速开始

Python 3.11+，依赖为 NumPy 与 PyYAML。可以使用本地 wheel 离线安装：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --no-index /path/to/numpy.whl /path/to/PyYAML.whl
.venv/bin/python -m pip install -e . --no-deps
```

生成确定性样本：

```bash
.venv/bin/dp-grad-sim sample \
  --out samples --workers 3 --steps 4 --layers 2 \
  --hidden-dim 8 --topology ring \
  --link-bandwidth-bytes 4096 --window-budget-bytes 65536
```

分两段运行，模拟中途停止与恢复：

```bash
.venv/bin/dp-grad-sim run \
  --config samples/config.yaml \
  --shards samples/shards.jsonl \
  --topology samples/topology.json \
  --snapshot-dir runtime/snapshots --max-windows 2

.venv/bin/dp-grad-sim run \
  --config samples/config.yaml \
  --shards samples/shards.jsonl \
  --topology samples/topology.json \
  --snapshot-dir runtime/snapshots
```

检查最新提交并验证结果：

```bash
.venv/bin/dp-grad-sim inspect --snapshot-dir runtime/snapshots
.venv/bin/dp-grad-sim verify \
  --config samples/config.yaml \
  --shards samples/shards.jsonl \
  --topology samples/topology.json \
  --snapshot-dir runtime/snapshots
```

`verify` 成功时退出码为 `0`，并检查清单/文件校验和、配置/拓扑/数据哈希、完成游标、预算、决策与收发记录对应关系、报告排序、参数有限性及保真梯度引用。

## 输入文件

### 仿真配置 YAML

```yaml
seed: 20261119
model:
  layers: 2
  hidden_dim: 8
  layer_tensors:
    - {name: weight, shape: [hidden_dim, hidden_dim]}
    - {name: bias, shape: [hidden_dim]}
steps: 4
data_parallel_workers: 3
compression:
  candidates:
    - {scheme: symmetric, bits: [8, 6, 4]}
    - {scheme: asymmetric, bits: [8, 6, 4]}
  window_steps: 1
  max_bytes_per_window: 65536
aggregation: {mode: mean, order: auto}
fidelity_generations: 2
concurrency: 4
snapshot_interval_steps: 1
communication_error_tolerance: 0.001
snapshot_dir: runtime/snapshots
```

支持的聚合模式为 `mean`/`sum`，聚合顺序为 `auto`/`name_asc`/`bytes_desc`。`auto` 在张量压缩包大小差异达到 1.5 倍时选择字节数降序，否则选择张量名字典序。字节数相同时始终以张量名字典序决断。

### 分片描述 JSONL

每行一个对象：

```json
{"worker_id":0,"shard_id":"w0-s0","samples":8,"feature_digest":"...","label_digest":"..."}
```

所有工作号必须至少有一个分片，分片 ID 必须唯一。伪梯度种子包含配置种子、步号、工作号、张量、分片 ID、样本数和两个摘要。

### 通信拓扑 JSON

环形示例：

```json
{
  "type": "ring",
  "workers": 3,
  "link_bandwidth_bytes": 4096,
  "links": [
    {"src": 0, "dst": 1, "bandwidth_bytes": 4096},
    {"src": 1, "dst": 2, "bandwidth_bytes": 4096}
  ]
}
```

`ring` 只允许前后相邻链路；`full` 需要所有有序工作号对之间的链路。也可省略 `links`，由 `link_bandwidth_bytes`/`bandwidth_bytes` 自动展开。

### 独立注入配置

`samples/injection.yaml` 仅在被显式传给 `--inject` 时生效。它支持非有限值、单链路收发失败、链路预算收紧和指定阶段后模拟崩溃，用于异常隔离与恢复测试。正常运行不读取该文件。

## 输出与恢复

快照根目录布局：

```text
runtime/snapshots/
  manifest.json                  # 原子替换的提交指针
  commits/v00000016-.../         # 完整提交目录
    manifest.json                # 阶段、版本、哈希和文件清单
    state.json                   # 游标、决策、归因、轮转和数组索引
    arrays/                      # 参数、梯度、接收整数、聚合与保真梯度
    packets/                     # 已压缩定点字节流
  reports/latest.json            # 最近一次运行的诊断报告
```

提交顺序为：写临时目录、写完整文件、目录整体重命名到 `commits/`、原子替换根 `manifest.json`。恢复时只信任根清单指向的完整提交；指针损坏时才回退扫描完整提交。未完成临时目录不会被当作有效状态。

启动时若配置、拓扑或分片哈希与最新快照不一致，输出 `E_CONFIG_MISMATCH` 并拒绝续跑。阶段中断时，已完成窗口不会重复收发；`communicated` 阶段已提交时只继续误差追踪与参数更新。

## 业务码

- `E_PARAM_NONFINITE`：参数或梯度含非有限值，记录工作号和张量名。
- `E_COMM_BUDGET_EXHAUSTED`：链路或窗口预算不足，记录窗口、链路、已发送字节和请求字节。
- `E_CONFIG_MISMATCH`：配置、拓扑或数据哈希与快照不匹配，拒绝续跑。
- `E_COMM_SEND_FAILED`、`E_COMM_RECEIVE_MISSING`、`E_COMM_PACKET_UNAVAILABLE`：单体收发异常被隔离并继续其余任务。
- `E_QUANT_HEADER_INVALID`：压缩包魔数、头校验、负载长度、CRC、位宽、缩放因子或期望元数据校验失败。

## 退出与信号

`SIGINT`/`SIGTERM` 会设置优雅停止标记。当前窗口继续完成派生、量化、收发、追踪和快照提交，随后以退出码 `0` 退出。`SIGKILL` 等不可捕获终止后，仅根清单已完整替换的提交可恢复。`--max-windows N` 为确定性验证提供了同等的优雅停止入口。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

测试覆盖量化往返与头校验、确定性梯度、预算与排序决策、真实回环套接字、异常隔离、崩溃恢复、配置不匹配、并发一致性、误差边界和保真轮转。实际验证结果见 `docs/VERIFICATION.md`，设计细节见 `docs/DESIGN.md`。

## Windows 说明

代码通过 `socket.bind(("127.0.0.1", 0))` 选择空闲本地端口，快照写入使用 `os.replace` 替换文件。Windows 下需允许 Python 进程连接 `127.0.0.1` 的本地端口，并将 `snapshot_dir` 指向可写临时目录。项目不使用 Docker、虚拟机或外部网络目标。
