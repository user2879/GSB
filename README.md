# Loopback Tunnel Lab

面向本地 Windows 回环环境的确定性隧道转发与诊断实验系统。系统以 JSON 声明多条隧道路由，在真实 `127.0.0.1` TCP 套接字上完成封装、分片、并行转发、四重预算调度、单次会话账本、快照恢复与严格重放。运行核心只依赖 Python 3.11+ 标准库；NumPy 仅作为离线环境准备项，不参与运行逻辑。

> 这是可审计的本地隧道路由与调度实验，不是生产 VPN。加密套件字段是封装元数据和受支持标识集合，本项目没有实现真实加密或密钥协商。请勿把它用于承载敏感或公网流量。

## 核心能力

- **强校验配置**：隧道 ID、回环端点、全局端口唯一性、MTU、加密套件、带宽配额与会话上限一次性校验；任一错误以 `E_CONFIG` 整体拒绝并返回 JSON 字段路径。
- **确定性封装**：84 字节固定头，包含隧道 ID、序号、负载长度、原始长度、分片偏移/大小/索引/总数和 16 字节摘要；同输入逐字节一致。
- **MTU 分片与重组**：支持乱序、精确重复、冲突重复、缺片超时；缺片以 `E_FRAG` 返回缺失偏移。
- **四重预算**：全局/隧道并发许可 + 全局/隧道整数定点令牌桶，任一维度不足都会排队，账本记录每次排队开始、结束与时长。
- **真实回环对端**：对端独立进程解帧、验摘要、重组并回送摘要；支持 `normal`、`delay-script`、`unreachable`、`digest-error` 四种行为注入。
- **固定点统计**：按隧道与全局输出吞吐和平均排队时长，缩放因子为 `1000`，并给出 `0.0005` 的量化误差上界。
- **完整快照**：路由版本、序号水位、预算用量、账本、会话状态和统计结论整体原子持久化；未结束会话恢复为 `UNKNOWN` 并要求重跑。
- **严格重放**：按快照路由版本、负载和时钟观测重编码/重组/重演对端行为，首个差异定位到隧道、会话和字段，以 `E_REPLAY` 报告。
- **Tkinter 诊断台**：展示路由、四重预算、会话状态、排队/吞吐统计、业务码与快照列表。

## 快速开始（Windows PowerShell）

```powershell
cd <项目目录>

# 1. Python 与 Tk 自检（Windows 官方安装包应自带 Tk）
py -3.11 scripts/check_env.py --require-tk

# 2. 将匹配 cp311/cp312 的 NumPy wheel 放入 wheels\，然后完全离线安装
.\scripts\install_numpy_offline.ps1 -Python "py -3.11"

# 3. 运行标准库测试
py -3.11 -m unittest discover -s tests -v

# 4. 清理本演示遗留的子进程与端口，再顺序执行全部场景
py -3.11 scripts/cleanup_ports.py --config data/demo_routes.json
py -3.11 scripts/run_demo.py --config data/demo_routes.json --state-dir artifacts\demo-state

# 5. 打开诊断台
py -3.11 -m tunnelvpn.ui --config data/demo_routes.json --state-dir artifacts\ui-state
```

演示结果写入 `artifacts\demo-state\demo_result.json`。正常、排队与分片场景也可单独执行：

```powershell
py -3.11 scripts\demo_normal.py
py -3.11 scripts\demo_queue.py
py -3.11 scripts\demo_fragment.py
```

## 配置示例

完整样例见 `data/demo_routes.json`。核心结构：

```json
{
  "schema_version": 1,
  "route_version": "2026-11-18-demo-v1",
  "global": {
    "bandwidth_kbps": 24,
    "max_sessions": 5,
    "fragment_timeout_ms": 300,
    "peer_timeout_ms": 500,
    "connect_timeout_ms": 300
  },
  "peers": [
    {
      "id": "peer-normal",
      "host": "127.0.0.1",
      "port": 19101,
      "behavior": "normal"
    }
  ],
  "routes": [
    {
      "id": "alpha",
      "listen": {"host": "127.0.0.1", "port": 18101},
      "target": {"host": "127.0.0.1", "port": 19101},
      "peer_id": "peer-normal",
      "mtu": 256,
      "cipher": "AES-256-GCM",
      "bandwidth_kbps": 8,
      "max_sessions": 2
    }
  ]
}
```

约束：所有端点必须是 `127.0.0.1`，端口为 `1024..65535` 且路由监听/目标端点全局唯一；MTU 为 `128..9000`；带宽和会话上限为正整数；受支持套件标识为 `NONE`、`AES-128-GCM`、`AES-256-GCM`、`CHACHA20-POLY1305`。

## 命令行

```powershell
# 仅校验配置
py -3.11 -m tunnelvpn validate --config data\demo_routes.json

# 运行全部入站监听；Ctrl+C 停止并可保存快照
py -3.11 -m tunnelvpn run --config data\demo_routes.json --state-dir state --snapshot-on-stop

# 启动单个模拟对端，可用命令行覆盖 JSON 行为而不改引擎
py -3.11 scripts\peer_server.py --config data\demo_routes.json --peer-id peer-normal --behavior digest-error
py -3.11 scripts\peer_server.py --config data\demo_routes.json --peer-id peer-delay --behavior delay-script --delay-ms 1200

# 快照、恢复报告与严格重放
py -3.11 -m tunnelvpn snapshots --state-dir state
py -3.11 -m tunnelvpn restore --config data\demo_routes.json --state-dir state
py -3.11 -m tunnelvpn replay --config data\demo_routes.json --state-dir state
```

## 业务码

| 代码 | 含义 |
|---|---|
| `E_CONFIG` | 配置、端口或启动参数非法；整体拒绝 |
| `E_ENCAP` | 魔数、长度、摘要、序号或分片元数据非法；含隧道 ID/首个偏移 |
| `E_FRAG` | 重组超时；含隧道 ID、序号和缺失分片偏移 |
| `E_PEER` | 对端不可达、连接或发送失败 |
| `E_TIMEOUT` | 对端响应或入站读取超时 |
| `E_DIGEST` | 对端回送摘要与本地封装摘要不一致 |
| `E_REPLAY` | 恢复重放的首个会话/统计字段不一致 |
| `E_STATE` | 快照、账本或内部状态不完整/冲突 |

## 目录

```text
tunnelvpn/       配置、编解码、预算、引擎、对端、账本、快照、重放、CLI、UI
data/            确定性路由、对端延迟脚本与负载样本
scripts/         对端、三类演示、总演示、端口清理、环境检查、离线安装
tests/           46 项标准库 unittest
docs/            协议、设计/不变量、实际验证记录
state/ logs/     默认运行状态与结构化 JSONL 日志目录
```

更细的字节布局、调度不变量与恢复语义见 `docs/PROTOCOL.md`、`docs/DESIGN.md`；本次实际验证输出见 `docs/VALIDATION.md`。
