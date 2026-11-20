# 灰度分流与镜像采样连接级仲裁组件

这是一个面向 Windows 10/64 位、Python 3.10+ 的本地回环组件。它不按请求逐条做一次性的路由判断，而是把灰度去向、镜像采样、连接复用、空闲回收和借用容量统一到“连接池水位”资源模型中。

核心性质：

- 稳定池、灰度池、镜像池互相独立；主请求和镜像都通过 `http.client` 真实访问 `127.0.0.1` 高位端口。
- 灰度池容量不足时可向声明了 `lendable_to` 的稳定池借用容量；借出方始终满足 `owned_active + lent_out <= max_connections`。
- 镜像池禁止借用，镜像失败/耗尽/超时只写入镜像字段，不改变主请求去向、主请求结局或主池统计。
- 每个完整批次在一个 SQLite 事务中写入请求、连接复用序号、借用归属、池水位、进度游标、格式版本和 SHA-256 内容摘要。
- 强制终止后只恢复已完整提交的批次；未提交批次中的连接/借用被标记为 `unknown`，必须重跑。
- 统计使用整数定点（尺度 1000、向下截断），权重/采样误差和哈希分桶离散误差按池水位与借用次数传播并给出贡献占比。
- Tkinter 窗口只读刷新账本和快照，不修改任何账本内容。

详细设计见 `docs/TECHNICAL.md`，实测记录见 `docs/VERIFICATION.md`。

## 目录

```text
gray_arbiter/       核心包：校验、连接池、引擎、统计、账本、CLI、Tkinter
tools/              两个回环目标脚本、确定性样例生成器
samples/            池/规则 JSON、36 条 NDJSON 请求
scripts/            Windows PowerShell 环境检查与演示脚本
tests/              16 项本地端到端/单元验收测试
snapshots/          JSON 快照目录（运行时生成，不入版本库）
artifacts/          演示输出
```

## Windows 准备

在 PowerShell 中：

```powershell
python --version          # 需要 3.10+
python -m pip install numpy
python tools/generate_samples.py
powershell -ExecutionPolicy Bypass -File scripts/check_env.ps1
```

`check_env.ps1` 检查 NumPy 和 127.0.0.1 的 19081/19082/19083 是否空闲。首次启动 Python 时如 Windows Defender 防火墙询问，只允许专用网络下的 127.0.0.1 回环通信；组件不会访问外部网络。

## 一键演示

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1
```

脚本会启动 `tools/target_stable_gray.py`（19081/19082）和 `tools/target_mirror.py`（19083），执行 36 条样例请求，写入 `state/demo.sqlite` 与 `snapshots/`，再运行账本重放并打开 Tkinter 诊断窗口。

也可以分三个 PowerShell 窗口手动运行：

```powershell
python tools/target_stable_gray.py --delay-ms 20
python tools/target_mirror.py --delay-ms 35

python -m gray_arbiter run `
  --config samples/config.json `
  --requests samples/requests.ndjson `
  --ledger state/demo.sqlite `
  --snapshots snapshots `
  --reset-ledger `
  --batch-size 12 `
  --workers 16

python -m gray_arbiter replay --ledger state/demo.sqlite
python -m gray_arbiter ui --ledger state/demo.sqlite --snapshots snapshots
```

池与规则也可拆为 `--config samples/pools.json --rules samples/rules.json`。

## 配置格式

最小完整配置：

```json
{
  "version": 1,
  "borrow_ttl_ms": 1500,
  "queue_timeout_ms": 5000,
  "main_timeout_ms": 1200,
  "mirror_timeout_ms": 450,
  "pools": [
    {
      "pool_id": "stable",
      "host": "127.0.0.1",
      "port": 19081,
      "max_connections": 3,
      "max_requests_per_connection": 4,
      "idle_timeout_ms": 1200,
      "lendable_to": ["gray"]
    },
    {
      "pool_id": "gray",
      "host": "127.0.0.1",
      "port": 19082,
      "max_connections": 2,
      "max_requests_per_connection": 4,
      "idle_timeout_ms": 1200,
      "lendable_to": []
    },
    {
      "pool_id": "mirror",
      "host": "127.0.0.1",
      "port": 19083,
      "max_connections": 2,
      "max_requests_per_connection": 3,
      "idle_timeout_ms": 900,
      "lendable_to": []
    }
  ],
  "rules": [
    {
      "rule_id": "catalog",
      "prefix": "/api/catalog",
      "stable_pool": "stable",
      "gray_pool": "gray",
      "mirror_pool": "mirror",
      "gray_weight_bp": 6500,
      "mirror_sample_bp": 3500
    }
  ]
}
```

`max_requests_per_connection` 是单条连接承载的总请求数；第一条请求的 `reuse_sequence` 为 0，后续为 1、2……。`watermark = active_connections + lent_out`。权重和采样率均为 `[0, 10000]` 的整数万分比。`prefix` 匹配采用最长前缀，再按 `rule_id` 升序决胜。

DSL 同时接受少量字段别名，例如 `id`、`capacity`、`reuse_limit`、`can_lend_to`、`target: "http://127.0.0.1:19081"`；规范字段以上述样例为准。

## NDJSON 请求

```json
{"request_id":"sample-001","path":"/api/catalog/items/1","client_key":"client-alpha","timestamp_ms":1760000000000}
```

请求在进入批次前按 `(timestamp_ms, line_no, request_id)` 稳定排序并重新编号。所有请求输入先整体校验，任一非法输入以 `E_POOL` 拒绝且 `field` 指向首个非法字段。

## 命令行

```powershell
# 正常执行；已有完整批次时必须显式 --resume
python -m gray_arbiter run --config samples/config.json --requests samples/requests.ndjson `
  --ledger state/demo.sqlite --snapshots snapshots

# 从最后一个已提交游标之后续跑；未提交批次中的资源先标记 unknown
python -m gray_arbiter run ... --resume

# 仅恢复并重放，不执行请求
python -m gray_arbiter recover --ledger state/demo.sqlite
python -m gray_arbiter replay --ledger state/demo.sqlite

# 三类异常注入
python -m gray_arbiter run ... --inject borrow-cycle
python -m gray_arbiter run ... --inject-target-unreachable stable
python -m gray_arbiter replay --ledger state/demo.sqlite --inject-snapshot-digest
```

异常对应业务码：借用环 `E_BORROW`、目标不可达 `E_TARGET`、账本摘要不符 `E_SNAPSHOT`；内部重放字段不一致为 `E_REPLAY`。日志为 NDJSON，固定包含 `business_code`、`request_id`、`pool_id`、`phase` 和单调 `sequence`。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试使用临时回环端口真实启动 `ThreadingHTTPServer`，覆盖：强校验、连接复用/空闲回收、借用上限与排队、镜像不可达/真实套接字超时、定点误差与稳定决胜、SQLite 摘要/首字段重放、崩溃未知资源、断点续跑、版本整体替换，以及三类 CLI 异常码。

## 主要假设

- 借用的是“连接容量”，实际 TCP 连接仍建立到借入池自己的目标端点；归还时该连接真实关闭，借出量减一。这样既保持灰色流量真实去往灰度目标，又让容量约束由稳定池承担。
- 镜像请求使用非阻塞获取；镜像池满时记录 `E_MIRROR`，不会排队拖住主请求。
- 借入连接的主 HTTP 超时会被压到借用归还截止时刻以内，避免无限期占用借出容量。
- 当前转发动词为 `GET`；请求流只提供路径，不包含请求体或方法。
- NumPy 仅用于可选验证辅助；运行路径完全使用标准库。
