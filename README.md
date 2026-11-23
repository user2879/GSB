# 网关限流与鉴权诊断工作台

面向 Windows 10+ 本地环境的诊断工作台：本地 JSON 声明上游服务与请求脚本，
以子进程方式在 127.0.0.1 回环端口启动确定性上游服务，网关侧执行整数定点
令牌桶限流与密钥版本兼容的签名鉴权，真实套接字转发放行请求，并在全局并发
上限与每服务并发配额双重约束下调度。界面快速视图基于增量缓存；权威重算从
SQLite 账本逐请求重放；两者不一致时定位首个偏差请求并给出业务码。

仅使用 Python 3.11 标准库（`sqlite3`、`socket`、`tkinter`、`threading`、
`subprocess`），无第三方依赖，不使用 Docker 或任何容器，不访问外网。

## 目录结构

- `workbench/` 核心包
  - `spec.py` JSON 加载与强校验（E_SPEC，含首个非法字段与行号）
  - `jsonparse.py` 带行号的极简 JSON 解析器
  - `tokens.py` 整数定点令牌桶（会话缩放因子 0..6）
  - `authn.py` 签名鉴权与密钥版本窗口
  - `upstream.py` 确定性上游服务子进程
  - `forward.py` 真实回环转发与回执校验
  - `scheduler.py` 双配额线程池调度（排队/判定/转发区间轨迹）
  - `engine.py` 网关编排、有序判定屏障、增量缓存
  - `ledger.py` SQLite 账本（批次单事务提交、格式版本、内容摘要、恢复）
  - `replay.py` 权威重算与双路径一致性比对
  - `snapshot.py` 会话快照（版本识别、整体替换、恢复后重放校验）
  - `stats.py` 整数定点统计（均值/P95/最大值）
  - `gui.py` tkinter 快速视图界面
  - `cli.py` 命令行入口
- `data/` 本地确定性样本（服务、密钥版本、请求脚本）
- `tools/make_sample.py` 样本生成器（按声明规则计算签名）
- `tests/` 52 项 unittest 测试

## 快速开始

```bat
py -3.11 tools/make_sample.py            :: 生成 data\ 样本
py -3.11 -m workbench.cli run            :: 启动上游子进程并分批执行脚本
py -3.11 -m workbench.cli view           :: 快速视图（增量缓存一次读取）
py -3.11 -m workbench.cli recompute      :: 权威重算 + 双路径一致性
py -3.11 -m workbench.cli snapshot-save  :: 保存会话快照
py -3.11 -m workbench.cli snapshot-open  :: 打开快照 + 恢复后重放校验
py -3.11 -m workbench.cli recover        :: 扫描未提交批次并标记 UNKNOWN
py -3.11 -m workbench.cli gui            :: 启动 tkinter 界面
py -3.11 -m unittest discover -s tests   :: 运行测试
```

通用参数：`--data-dir data --ledger out\ledger.db --snapshot
out\snapshot.json --scale 3 --batch-size 8 --global-limit 4
--per-service-quota 2 --upstream-timeout 2.0`。

## 声明的业务规则

### 令牌桶（整数定点）

- 桶身份为 `(service_id, bucket_key)`；每服务声明容量、整数毫秒补充间隔、
  每请求消耗令牌数；桶初始为满。
- 会话缩放因子 `s ∈ [0,6]`，1 令牌 = `10**s` 缩放单位，余量全程整数。
- 补充量 = `经过毫秒 // 补充间隔`（整数），补充后夹取到容量上限。
- 时钟来源为请求脚本声明的单调 `arrival_ms`，而非墙钟；每服务的判定在
  有序屏障下按请求序号串行求值，因此同一请求序列在任意处理顺序下令牌
  余量轨迹逐字节一致（`tests/test_dual_path.py` 实证）。

### 签名鉴权

- 期望签名：`sha256("{secret}|{client_key}|{service_id}|{seq}|{cred_version}")`
  的十六进制，与凭据签名逐字节比较；计算不依赖字典/哈希遍历顺序。
- 凭据版本低于当前密钥版本时：`arrival_ms <= retire_ms(版本) +
  signature_window_ms` 窗口内按旧密钥校验，超窗以 E_AUTH
  （`credential_expired`）拒绝；签名不符为 `signature_mismatch`；未知版本
  为 `unknown_version`。

### 回环转发与回执

- 请求行：`REQ <seq> <service_id> <client_key>\n`；回执：`RCPT <seq>
  <service_id> <cksum16>\n`，`cksum16` 为请求行（去换行）的 SHA-256 前
  16 位十六进制。网关校验序号、服务标识与校验和。
- E_UPSTREAM（连接失败）、E_TIMEOUT（真实套接字超时，非人为延迟）、
  E_RECEIPT（回执字段缺失/不匹配）分别上报；被拒请求不连接上游。
- 上游故障注入客户端键：`__timeout__`、`__badreceipt__`、`__wrongsvc__`。

### 调度

- 工作线程池真实执行；在途总数 ≤ 全局上限，单服务在途 ≤ 服务配额。
- 有序判定屏障在配额获取之前等待（锁顺序：轮次 → 全局信号量 → 服务
  信号量），杜绝锁序死锁且不绕过配额。
- 每请求轨迹：排队区间、判定区间、转发区间（单调时钟毫秒）。

### 账本与恢复

- 批次协议：`begin_batch` 先写标记行（独立事务），`commit_batch` 在同一
  SQLite 事务内写入该批全部逐请求结论、令牌余量、回执摘要、等待时长并
  推进进度游标、更新链式内容摘要。
- 账本携带整数格式版本（`LEDGER_FORMAT_VERSION=1`）、缩放因子、时钟基准
  声明（`time.monotonic_ns//1_000_000`）。
- 强制终止后重启只信任已完整提交的批次；未提交批次的请求标记 UNKNOWN
  并要求重跑；不为被终止的运行生成退出报告。

### 快照

- 快照持久化：服务集合版本（摘要）、密钥版本集合、缩放因子口径版本、
  逐请求结论摘要、令牌余量快照、账本游标与整体内容摘要。
- 打开时任一版本或摘要不一致 → 整体替换（不做增量合并），差异项按稳定
  排序列出；恢复后按账本重放并逐项比较，首个不一致以 E_REPLAY 报明
  请求序号与字段。

### 延迟观测与统计

- 每次界面查询（`gui_query`）与每次权威重算（`recompute`，含整体与每
  服务分段）的起止单调时钟毫秒差逐次写入账本 `observations` 表——真值
  来源即账本逐次记录。
- 聚合：按服务与整体输出均值、P95（最近秩）、最大值的 ×1000 整数定点
  表示；截断方向为向零截断；均值量化误差上界 1 个缩放单位（0.001 ms），
  P95/最大值为整数毫秒精确值。

## 业务码

`OK`、`E_SPEC`、`E_RATE`、`E_AUTH`、`E_UPSTREAM`、`E_TIMEOUT`、
`E_RECEIPT`、`E_REPLAY`、`UNKNOWN`。

## 实测验证结果（本仓库样本）

```
$ python -m workbench.cli run
executed 20 requests in batches of 8
decision_counts: {E_AUTH: 3, E_RATE: 4, E_RECEIPT: 1, E_TIMEOUT: 1, OK: 11}

$ python -m workbench.cli recompute
authoritative recompute OK: 20 requests replayed, cache view byte-identical

$ python -m workbench.cli snapshot-save / snapshot-open
snapshot saved: out/snapshot.json (cursor=20, conclusions=20)
snapshot accepted (cursor=20)
replay verification OK: 20 requests match ledger

$ python -m tests.crash_worker out\crash.db &  :: 模拟强制终止
$ python -m workbench.cli --ledger out\crash.db recover
marked UNKNOWN (re-run required): seqs=[101, 102, 103, 104]

$ python -m unittest discover -s tests
Ran 52 tests ... OK
```

样本覆盖：放行、桶耗尽 E_RATE、成本高于余量 E_RATE、旧版本窗口内放行、
凭据过期 E_AUTH、签名不符 E_AUTH、未知版本 E_AUTH、真实套接字超时
E_TIMEOUT、回执字段缺失 E_RECEIPT。
