# DNS 解析与缓存诊断工作台 — 技术说明

## 1. 环境与依赖

- Python 3.11+，仅使用标准库（`socket`、`socketserver`、`struct`、`sqlite3`、`tkinter`、`json`、`hashlib`、`threading`、`logging`、`dataclasses`、`unittest`）。
- Windows 官方安装包自带 `tkinter` 与 `sqlite3`，无需第三方包，无需联网。
- 本仓库 Linux 验证环境缺少 `libtk8.6.so`，GUI 在该环境自动降级并打印提示；全部非界面能力通过测试与演示验证。
- 虚拟环境（无 pip，纯标准库）：
  - Windows：`py -3 -m venv .venv && .venv\Scripts\python -m unittest discover -s tests`
  - 本环境：`python3 -m venv --without-pip .venv && .venv/bin/python -m unittest discover -s tests`

## 2. 模块结构与职责

| 模块 | 职责 | 对应要求 |
| --- | --- | --- |
| `dns_workbench/wire.py` | DNS 线格式编解码、名称压缩/指针防御、UDP 512 截断、TCP 分帧 | 1 |
| `dns_workbench/records.py` | v1/v2/v3 缓存记录与 u1/u2 上游应答强校验、归一、降级/补齐 | 2、3 |
| `dns_workbench/cache.py` | 单调时钟毫秒 TTL、过期回收、负应答 TTL 上限、稳定 LRU 逐出 | 4 |
| `dns_workbench/zone.py` | JSON 区文件加载、E_ZONE 校验、CNAME 链权威解析 | 1、8 |
| `dns_workbench/upstream.py` | 回环 TCP 转发、事务/问题/应答链校验、E_UPSTREAM/E_CHAIN/E_TXN、重试语义 | 5 |
| `dns_workbench/ledger.py` | 版本识别、整体替换、陈旧标记、新增/删除/变更差异清单 | 6 |
| `dns_workbench/snapshot.py` | SQLite 快照（版本集合、区版本、序列号、摘要、命中历史）与重放校验 | 7 |
| `dns_workbench/server.py` | 127.0.0.1 UDP/TCP 服务编排与查询处理 | 1、4、5 |
| `dns_workbench/gui.py` | Tkinter 四标签页稳定渲染 | 8 |
| `dns_workbench/structured_log.py` | JSON Lines 结构化日志 | 8 |

## 3. 关键设计与不变量

### 3.1 线格式与指针防御

- 名称解析逐字节校验：越界标签、非法标签类型、不完整指针均抛 `E_WIRE` 并带 `offset`。
- 压缩指针必须指向**先前**偏移：指针 `>= 名称起始偏移` 判为向前/自引用；重复访问同一指针判为环。
- UDP 截断为确定性过程：问题段始终保留，资源记录按 应答→权威→附加 的段内顺序逐条加入，加入后超过 512 即停止并置 TC；TCP 路径不截断（实测 40 条 A 记录：UDP 499 字节/29 条/TC=1，TCP 675 字节/40 条/TC=0）。

### 3.2 三版本归一与视图

- 内部模型 `CacheEntry` 统一承载全部字段，`content_digest` 对归一化内容计算 SHA-256，与来源版本解耦。
- v1 降级稳定剥离字段集合：`aa, ecs_prefix, hits, rcode, seq, source`（字母序）。
- v3 补齐显式默认值：`ecs_prefix=null, hits=0, seq=0`，v2→v3 仅补齐 `ecs_prefix, hits, seq`。
- 不变量：同一记录任意视图下 `name`、`qtype`、剩余 TTL 逐字节一致（测试 `test_views_keep_identity_bytes_equal`）。
- 强校验：未知版本 `E_VER`；缺字段/TTL 负值或超 `MAX_TTL=604800` 抛 `E_FORMAT` 并列字段；v3 同键 `seq` 倒退抛 `E_VER`。

### 3.3 TTL 缓存

- 时钟抽象 `Clock`，生产用 `time.monotonic_ns()`，演示/测试用可注入的 `DeterministicClock`，TTL 以毫秒存储、剩余秒向上取整输出。
- 负应答 TTL = `min(SOA minimum, negative_ttl_cap)`，默认上限 300 秒。
- 逐出规则显式稳定：先回收过期键；容量满时按 `(access_order, key)` 取最小值（LRU，键字典序兜底），并产出 `Eviction` 记录。

### 3.4 上游转发与链校验

- 仅 TCP，2 字节长度前缀；连接类异常按 `retries` 重发**同一字节报文**（单次转发语义不变），最终失败为 `E_UPSTREAM` 并记录尝试次数。
- 校验顺序：事务标识（`E_TXN`）→ 问题回显三元组 → 应答记录名/类型链（`E_CHAIN`），CNAME 的 RDATA 名称必须与下一条记录名逐跳衔接。
- u1/u2 区分：u2 附加 OPT 伪记录且 DO=1；`DnsMessage.opt_record()/dnssec_ok()` 识别。

### 3.5 整体替换与快照

- 触发条件：区版本单调递增，或缓存格式版本集合变化；旧代数 `stale=True` 进入历史，不参与当前展示与统计。
- 差异清单稳定按键排序，分 `added / removed / changed`（含新旧版本与摘要）。
- SQLite 三表 `meta / entries / hits`；恢复重放比较格式版本集合与逐条摘要，不一致即返回差异并交由账本整体替换（不做增量合并）。

## 4. 错误码对照

| 码 | 触发场景 | 附加信息 |
| --- | --- | --- |
| `E_WIRE` | 报文非法、指针环/向前/越界 | `offset` |
| `E_FORMAT` | 缺字段、TTL 越界、标志类型错误 | `fields[]` |
| `E_VER` | 未知版本、序列号倒退、区版本倒退、版本不匹配 | `fields[]`、上下文 |
| `E_ZONE` | 区 JSON 语法/结构损坏 | `offset`/`fields[]` |
| `E_UPSTREAM` | 回环上游不可达 | 尝试次数、原因 |
| `E_CHAIN` | 问题回显或应答名/类型链断裂 | 断裂字段与期望值 |
| `E_TXN` | 上游事务标识不一致 | expected/actual |
| `E_SNAPSHOT` | 指定快照不存在 | snapshot_id |

## 5. 验证结果

执行 `.venv/bin/python -m unittest discover -s tests -v`：38 个用例全部通过，覆盖线格式（8）、记录版本（11）、TTL 缓存（5）、区（4）、账本快照（6）、上游（4）。

执行 `.venv/bin/python scripts/demo.py` 与三种 `--inject` 异常注入，输出与预期一致；结构化日志位于 `data/run/workbench.jsonl`，快照位于 `data/run/ledger.sqlite3`。
