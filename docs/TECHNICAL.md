# 技术说明

## 1. 分层与数据流

```text
JSON 池/规则 ─┐
NDJSON 请求 ──┼─> 强校验/稳定排序 ─> 批次调度 ─> 连接池仲裁 ─> 127.0.0.1 HTTP
SQLite 账本 ──┘                          │             ├─> 借用账本/归还
                                          │             └─> 镜像独立池
                                          ├─> 定点水位/误差传播
                                          └─> 原子批次提交 + JSON 快照
```

模块职责：

- `config.py`：JSON/NDJSON 解析、唯一性/引用/端口/权重/镜像隔离/借用环校验。
- `hashing.py`：SHA-256 前 64 位的稳定分桶，桶空间 10001。
- `pool.py`：连接状态机、线程安全容量仲裁、借用/归还、队列、空闲回收。
- `engine.py`：主请求/镜像执行、批次调度、配置版本整体替换、快照。
- `stats.py`：定点水位、量化误差传播、稳定决胜、贡献占比。
- `ledger.py`：SQLite 事务账本、崩溃恢复、内容摘要、逐字段重放。
- `cli.py`/`ui.py`：可操作入口与只读诊断。

## 2. 强校验

在任何网络动作之前完成整文件校验。失败抛出 `E_POOL`，`field` 采用稳定路径，例如 `pools[2].idle_timeout_ms`、`rules[0].gray_weight_bp`、`requests[line=7].timestamp_ms`。

校验项：

- 池 ID、规则 ID 唯一；池引用存在；借用目标不得为自身；正常配置禁止借用环。
- 端口必须是 `127.0.0.1` 上的 `1024..65535`；也可提供 `target: http://127.0.0.1:PORT`。
- 权重/采样率必须为 `[0,10000]` 整数；布尔值不被当作整数。
- 镜像池必须与稳定池、灰度池不同；任何借用关系不得跨越镜像池。
- 请求 ID 唯一；路径必须以 `/` 开头，拒绝 `//` 与 CR/LF；时间戳为非负整数。
- 最长前缀匹配；同长度按 `rule_id` 字典序。

规则权重/采样映射 0..10000 共 10001 个桶。`0` 与 `10000` 作为端点分别强制为 0% 与 100%；内部值选择条件为 `bucket < rate_bp`。误差界仍按最保守的哈希离散公式计算。

## 3. 连接状态机与水位

状态：`established -> idle -> reused -> ... -> closed`。新连接建立后进入 `established` 并立即借给当前请求；请求结束且未到复用上限时进入 `idle`；再次借出进入 `reused`；达到复用上限、空闲超时、请求异常、借入归还或版本替换时真实 `HTTPConnection.close()` 并进入 `closed`。

每个池：

- `active_connections`：所有已建立、分配给该池的连接数（含 idle/in-use/借入）。
- `lent_out`：该池借给其他池的容量单位数。
- `borrowed_in`：该池当前占用的借入连接数。
- `watermark = active_connections + lent_out`。
- 借出不变式：`owned_active + lent_out <= max_connections`。
- 借入池可在自身 `active == max_connections` 后继续借入，故其 `active` 可短暂超过自身容量；这是借用容量要解决的问题。

同一 HTTP 连接被一个请求独占，所以连接上的请求天然串行。引擎用批次内到达序号闸门（`_TurnGate`）按序执行主池获取尝试；请求取得连接后立即释放闸门，因此不同连接上的 HTTP 仍可并发。

## 4. 借用仲裁

获取顺序固定为：

1. 先清理超时空闲连接。
2. 优先复用可用 idle 连接。
3. 若 `active < max_connections`，建立自有连接。
4. 若已满且主路径允许借用，按“剩余容量降序、池 ID 升序”选择借出池。
5. 否则线程进入条件队列；记录等待起止单调毫秒。超时返回 `E_QUEUE`。

借用记录包含借出方、借入方、数量、请求、创建时间、归还截止。借入连接建立到借入池自己的端点，但容量记在借出池。归还时连接真实关闭，借出池 `lent_out` 减一。借用的主请求超时上限会被限制在归还截止以内。

镜像获取使用 `allow_borrow=False, block=False`，池满直接记录 `E_MIRROR`。

## 5. 镜像隔离

镜像采样仅依赖 `stable_bucket(request_id, rule_id)`。主请求完成后，镜像使用独立池、独立连接和独立超时执行。异常只写入：

- `mirror_status`: `ok/http_error/error/pool_exhausted/timeout/not_sampled`
- `mirror_error_code`: `E_TARGET/E_MIRROR/E_TIMEOUT`
- `mirror_pool_id`、镜像连接、复用序号、响应摘要、耗时、水位。

镜像错误不会修改主请求的 `main_destination/main_pool_id/main_status/main_error_code`，也不会写入主池统计。镜像超时来自 `HTTPConnection` 套接字 `socket.timeout`，而非人造状态。

## 6. 定点统计与误差传播

固定点：

- 缩放因子 `1000`。
- 截断方向 `floor_toward_negative_infinity`。
- 平均值误差上界 1 个定点单位；计数本身为精确整数。
- 输出字段均带 `_scaled`，并显式声明 `fixed_point` 元数据。

两级来源误差：

1. 连续比率四舍五入到整数万分比：半宽 `0.5 bp`。对 N 条规则请求，任一侧请求数上界为 `N * 0.5 / 10000`。
2. 哈希桶离散：阈值 `w` 在 10001 个桶中选中 `w` 个，而声明的概率为 `w/10000`。灰度误差为 `N*w/(10000*10001)`，稳定侧为 `N*(10000-w)/(10000*10001)`；镜像同理。

传播链：

- 请求/采样数误差 -> 池平均水位定点误差：每个可能错位请求贡献至多 1 个水位单位，乘 1000 后加上定点截断 1；报告同时给出 observed average/borrow 与上下界区间。
- 请求数误差 -> 借用次数误差：每个额外目标请求最多触发一次借用，同样以定点表示。
- 镜像采样只进入镜像池误差，不进入主池。

报告按 `(pool_id, rule_id, level, component)` 排序并汇总。因为只依赖聚合计数，不依赖请求遍历顺序，反转请求列表得到的报告 JSON 完全一致。

稳定决胜：对每条规则的稳定池/灰度池平均水位，若 `|stable-grayscale| <= combined_error_bound`，固定输出 `[stable_pool, gray_pool]`、`winner_by_stable_tiebreak="stable"`、`within_error_bound=true`；否则按水位高者排序，同水位按池 ID 字典序。

## 7. 版本整体替换

`ArbiterEngine.replace_config(proposed, borrow_policy=...)`：

- 新版本号取 `max(旧版本+1, proposed.version)`，保证单调。
- 先校验完整新配置；任何非法返回 `E_POOL`。
- 对旧版本所有活跃借用按 `force_return`、`recompute`、`invalidate` 逐笔裁决并输出理由。
- 关闭旧管理器全部连接，建立全新管理器；当前视图整体替换，不做增量合并。
- 旧版本 epoch 在 SQLite 标记 `stale`，不参与当前报告；新 epoch 标记 `active`。
- 变动规则输出旧/新定义以及基于最近请求集合的稳定/灰度/镜像采样分布；变动池输出旧/新容量及变化字段。

## 8. SQLite 账本

每次批次开始前，先单独提交 `inflight_batches` 标记和资源快照；执行过程中每次池状态变化都会刷新该快照。批次完成后，以下内容在同一事务提交：

- `batches`：ID、配置版本、账本格式版本、起止游标、SHA-256 内容摘要、规范 JSON。
- `request_ledger`：逐请求去向、池、连接 ID、复用序号、借用 ID、借出方、状态、错误码、水位和完整文档。
- `mirror_ledger`、`pool_watermarks`、`borrow_ledger`、`connection_events`。
- 删除 inflight 标记。

摘要材料使用排序键、紧凑分隔符的 canonical JSON。写入失败整体回滚。

恢复：

- 只有 `batches` 中完整提交的行可以重放。
- 遗留 inflight 行移入 `interrupted_batches`，其快照中的活跃连接/借用逐项写入 `unknown_resources(state=unknown, needs_rerun=1)`。
- `exit_report_available` 在存在未知资源时为 false，不生成“正常退出报告”。
- `resume_cursor` 指向最小未提交批次起点；续跑只允许从该游标开始，之前已提交批次不重复执行。
- 未知资源在对应游标区间被重新提交后标记 resolved；在此之前重复恢复仍保持 `exit_report_available=false`。

重放：

- 先重算 canonical payload 的 SHA-256，不一致返回 `E_SNAPSHOT`。
- 再比较请求行与签名 payload 的去向、池、连接 ID、复用序号、借用归属、状态、错误码与水位。
- 同一请求多字段不一致时按字段固定顺序找出首个差异，返回 `E_REPLAY` 并给出请求 ID、字段、期望/实际值和判断依据。
- 另外比较水位行、借用行和连接生命周期事件；批次游标必须连续。

## 9. 结构化日志与诊断界面

日志每行一个 JSON 对象，`sequence` 是稳定顺序。错误在批次并发执行完后按请求 ordinal 顺序统一写出，避免线程调度决定日志顺序。

Tkinter 页面：

- Pool Watermarks：按事件样本绘制各池 `active` 实线与 `lent_out` 虚线。
- Requests：逐请求去向、池、连接、复用序号、借用、主状态、镜像状态。
- Borrow / Return：借出/借入、数量、截止、归还、是否逾期。
- Error-bound Heat：按组合误差上界着色，并展示各规则贡献占比与稳定决胜比较。
- Rules & Versions：当前配置、旧版本 stale、替换报告。
- Ledger：批次、格式版本、游标、摘要前 16 位、epoch 状态。

刷新仅执行 JSON 读取和 SQLite `mode=ro` 查询，不写快照/账本。
