# shardmeta 技术说明

本地多节点分片上传的元数据核算组件，从一次性脚本迁移为可诊断、可集成、可重复演示的 Python 包。
全程仅使用 127.0.0.1 回环、固定整数种子与本地文件/SQLite，不依赖外部网络、账号或容器。

## 1. 环境与目录

- Python >= 3.10（实测 3.11.0rc1），依赖 `numpy`、标准库 `sqlite3`、`tkinter`。
- 虚拟环境位于 `.venv`。本环境系统 Python 缺少 `ensurepip`，改用反馈环境自带
  `virtualenv` 创建（等价于 `python -m venv .venv` 的结果），再 `pip install numpy`。
- 无头 Tk：环境已运行 `Xvfb :99`，演示以 `DISPLAY=:99` 做冒烟；有显示时可直接进入 mainloop。
- 包代码 `shardmeta/`，测试 `tests/`，固定种子样本与账本 `samples/`，复现脚本 `run_demo.sh`。

## 2. 模块与职责

| 模块 | 职责 | 对应要求 |
| --- | --- | --- |
| `errors.py` | 统一 E_* 错误码，携带 node/key/field/line 定位 | 全部 |
| `spec.py` | 节点声明、对象清单、请求流 JSON/JSONL 强校验 | 1 |
| `metadata.py` | c1 定宽行、c2/c3 JSON 编解码与跨版本归一 | 2 |
| `lease.py` | 租约签发/续约/中止，乐观版本，单调时钟真实过期 | 3 |
| `node.py` | 节点子进程、回环 TCP JSONL、客户端与编排 | 4 |
| `content.py` | 固定种子分片字节、NumPy 整数定点摘要 | 4 |
| `adoption.py` | 并入裁决稳定排序、采纳/否决理由 | 5 |
| `engine.py` | 请求处理、缓存视图、权威重放、E_DIVERGE、恢复 | 4/6/8 |
| `evolution.py` | 版本/节点演进失效传播、增量与全量一致性 | 7 |
| `ledger.py` | SQLite 原子事务、快照版本/摘要、整体替换 | 8 |
| `ui.py` | Tkinter 一次缓存读取与毫秒延迟展示 | 6 |
| `generator.py` | 固定整数种子生成全部本地样本 | 准备 |
| `demo.py` | 端到端可重复演示与 JSON 报告 | 验收 |

## 3. 关键设计

### 3.1 确定性内容与摘要
- 分片字节：`numpy.random.default_rng` 由
  `mix64(node_seed, object_seed, sha256(key), shard_index)` 派生种子，生成 `[0,250)` uint8。
- 摘要全部使用 NumPy `uint64` 整数定点运算（回绕是预期行为，已关闭溢出告警）：
  - `content_check`：分片字节 FNV-1a 后 SplitMix64，16 位十六进制；
  - `fixed_mean_x1000`：字节均值 ×1000 的整数定点值（无浮点）；
  - `shard_digest`：绑定节点标识、对象键、分片序号与内容字节后 SplitMix64。
- 同一声明在任意调用顺序/批次切分下逐字节一致（有专门单测）。

### 3.2 三格式归一
- c1 为 46 列定宽行（key 24 + index 6 + digest 16），仅三字段；
- c2 增加 `merge_id/shard_bytes/meta_version`；c3 再增
  `content_check/parent_merge_id/commit_cursor`。
- 归一只补齐低版本缺失字段，并记录 `filled_fields` 与 `fill_basis`；
  c3 显式提供的值（含显式 `null` 父引用）不会被覆盖。
- 校验定宽宽度、键/序号一致、并入标识唯一、父引用存在且无环，错误带行号抛 E_FORMAT。

### 3.3 租约与真实过期
- 租约字段：`lease_id/object_key/shard_index/granted_ms/expires_ms/renew_count/lease_version`。
- 续约凭 `expected_lease_version` 乐观校验；版本不符抛 E_LEASE 且不延长到期。
- 过期是 `now_ms >= expires_ms` 的单调时钟比较，不使用人为睡眠模拟；
  到期回收同时释放容量槽位与租约计数并写账本。子进程模式使用真实单调时钟与短 TTL。

### 3.4 节点通信与错误码
- 每节点独立子进程，`-m shardmeta.node <base64 spec>` 启动，回环 TCP 上一行一个 JSON。
- 端口占用/拒绝连接 → E_PORT；握手或回执超时 → E_TIMEOUT；回执缺字段/ok=false → E_REPLY。

### 3.5 并入裁决
同一 `(object_key, shard_index)` 的候选按
`元数据版本降序 → 来源格式版本降序 → 并入序号升序 → merge_id 字典序`
选采纳分片；其余保留为对照并记录否决理由。排序全为显式全序，结论与派发/字典遍历顺序无关，
可序列化为逐字节一致的规范 blob（含 sha256 指纹）。

### 3.6 视图缓存与权威重算
- Tkinter/快速读取只查内存并入视图缓存，单次读取返回逐对象状态、采纳分片与租约水位，
  并记录单调时钟毫秒耗时，不触发全量重算。
- 权威路径从 SQLite 已提交 `shards` 行从零重建全部候选再裁决；
  两路结果逐字段比对，首个不一致的 key/shard_index/field 抛 E_DIVERGE。

### 3.7 演进失效传播
版本递增或节点集合变化沿 `节点 → 对象键/分片 → 父/并入引用 → 租约引用` 传播，
按对象键序、分片序输出失效集合与每步依据；未受影响分片的增量赢家与空白全量重建赢家
逐字段比对（`compare_field_by_field`），不一致即列出。

### 3.8 SQLite 账本与强制终止恢复
- 每次授予/续约/到期/中止/并入在单事务内写事件、租约、分片、节点余量与进度游标，
  并更新账本内容摘要；账本带整数 `format_version` 与 `content_digest`。
- 重开时版本或摘要不一致 → 整库替换（非增量合并）。
- 节点被 `kill` 后，未提交请求写 `unknown_requests` 要求重跑；重启节点只依据
  已提交租约通过 `restore` 重建自身状态，引擎再从账本恢复内存并复跑该请求。

## 4. 错误码对照

| 错误码 | 触发场景 |
| --- | --- |
| E_SPEC | 声明/清单/请求流非法（唯一性、回环字面量、正整数、整除、节点存在、序号单调） |
| E_FORMAT | c1/c2/c3 宽度/字段/并入标识/父引用环等结构错误（带行号） |
| E_LEASE | 容量不足、未知租约、乐观版本冲突、续约/中止非法状态 |
| E_PORT | 回环端口占用、节点不可达 |
| E_TIMEOUT | 握手/回执超时 |
| E_REPLY | 回执缺必填字段、JSON 损坏、ok=false |
| E_DIVERGE | 缓存视图与权威重算首个不一致 key/shard_index/field |

## 5. 运行方式

```bash
python3 -m venv .venv            # 本环境用 virtualenv 等价创建
.venv/bin/pip install numpy
DISPLAY=:99 ./run_demo.sh        # 测试 + 端到端演示 + 无头 Tk 冒烟
# 或单独：
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m shardmeta.demo --report samples/demo-report.json
```

## 6. 实测结果摘要

- 单元测试：38/38 通过，覆盖要求 1–8 与全部 E_* 错误码。
- 端到端演示：3 节点真实子进程 + 回环 JSONL，跨格式归一 7 条记录；
  强制终止 seq=7 后标记 unknown、按账本恢复并成功重跑；
  真实单调时钟到期回收租约；缓存视图与权威重算 blob sha256 一致（divergence=none）；
  演进失效仅命中受影响分片，增量/全量赢家零字段差异；无头 Tk 对 3 个对象各完成一次缓存读取。
- 详细数据见 `samples/demo-report.json`，账本见 `samples/shardmeta-ledger.sqlite3`。
