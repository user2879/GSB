# 缓存失效传播中间件技术说明

## 1. 目标与范围

组件完全基于 Python 3.11 标准库实现，不依赖第三方包、容器或在线服务。
核心命题是“缓存条目注释版本如何决定失效传播的可复现性”：

- 条目失效沿注释依赖图传播，每轮受影响条目各自递增一次注释版本。
- 命名空间刷新沿继承链展开全部下游命名空间的条目。
- 传播轨迹写入逐轮账本与版本化快照，恢复时整体替换并做一致性重放。

## 2. 目录结构

```
cache_invalidation/
  errors.py            # E_GRAPH/E_FORMAT/E_CLOCK/E_SNAPSHOT/E_INHERIT/E_REPLAY
  model.py             # EntryID/Entry/Namespace/Graph/RoundRecord
  jsonpos.py           # 带行号、字段路径与截断识别的 JSON 解析器
  loader.py            # 声明文件加载与图强校验
  propagation.py       # 可达闭包 + 最小堆 Kahn 稳定拓扑排序
  snapshot.py          # v1/v2/v3 快照编解码与归一
  store.py             # SQLite 持久化、整表替换与版本差异
  replay.py            # 恢复后逐轮重放校验
  engine.py            # 引擎：时钟、互斥、三类触发、恢复编排
  netloop.py           # 127.0.0.1 JSON Lines 回环服务
  fault_injection.py   # 只改输入副本/参数的三类异常注入
  logging_json.py      # JSON Lines 结构化日志
  gui.py               # Tkinter 诊断界面与可测试视图模型
  cli.py / demo.py / envcheck.py
samples/               # 确定性命名空间与条目声明
tests/                 # 45 个 unittest 用例
run_tests.py
```

## 3. 关键设计

### 3.1 可复现的传播顺序

1. 显式失效、过期回收、命名空间刷新合并为同一轮触发集合，并按
   `explicit > expired > refresh` 去重，同一条目每轮只归属一个来源。
2. 在注释边上求可达闭包，只保留当前仍存活的条目。
3. 对闭包子图用最小堆做 Kahn 拓扑排序，节点键为 `(namespace, key)`。
   排序只依赖图内容，不依赖提交顺序或线程调度；输入集合被刻意打乱时
   输出仍逐字节一致（见 `test_traversal_shuffle_does_not_change_result`）。

### 3.2 注释版本与条目级互斥

引擎公开操作均在同一把 `threading.RLock` 下串行化；每个条目带 `alive`
状态与注释版本计数：

- 已失效条目不会再次进入受影响集合，版本不会二次递增。
- 逻辑时钟由调用方以整数毫秒参数提供，引擎只接受单调不减输入；
   时钟倒退抛 `E_CLOCK` 且在任何状态变更前返回。

### 3.3 三版本快照口径

| 版本 | 条目字段 | 额外结构 |
| --- | --- | --- |
| v1 | `key`, `expire_ms` | 无命名空间，统一归一到 `default` |
| v2 | v1 + `annotations` 注释依赖集合 | 无命名空间，引用归一到 `default` |
| v3 | v2 + `namespace`、`digest_algorithm`、`annotation_version` | `namespaces[].parents` 继承链 |

未知格式版本、缺失必需字段、不支持的摘要算法均为 `E_FORMAT`，并给出
字段路径与行号；尾部截断由 `jsonpos` 识别为未闭合结构并报 `E_SNAPSHOT`。

### 3.4 整体替换恢复与重放

`Store.replace_all` 先删除 `namespaces/entries/annotation_edges/rounds`
全部业务表，再按快照重建，不做任何增量合并；替换前后的注释版本差异
稳定排序输出。重放从全量存活、版本为 0 的初始图开始，按账本逐轮重新
推导受影响集合和版本，首个不一致即定位到触发条目键、受影响条目键与
字段路径并返回 `E_REPLAY`。v1 升级缺少注释口径，按
`format_v1_lacks_annotation_sets` 如实报告不可复现。

## 4. 业务码对照

| 业务码 | 触发条件 | 定位信息 |
| --- | --- | --- |
| `E_GRAPH` | 标识/键重复、非法过期时刻、注释目标缺失、注释环或继承环、重复父级 | 出错条目键或环上序列 |
| `E_FORMAT` | 未知快照版本、必需字段缺失、摘要算法不支持 | 行号 + 字段路径 |
| `E_CLOCK` | 逻辑时钟倒退或非整数 | 当前值/给定值 |
| `E_SNAPSHOT` | 快照尾部截断、JSON 结构不完整 | 截断行号 |
| `E_INHERIT` | 跨命名空间继承引用缺失 | 出错命名空间与缺失父级 |
| `E_REPLAY` | 重放集合/版本不一致，或 v1 口径不可复现 | 首个触发条目、受影响条目与字段 |

## 5. Windows 本地运行

```powershell
python --version                 # 需 3.11+
python -c "import tkinter"       # 官方安装包默认成功
python -m cache_invalidation.cli env
python run_tests.py
python -m cache_invalidation.cli demo
python -m cache_invalidation.cli loopback
python -m cache_invalidation.gui
```

异常注入示例：

```powershell
python -m cache_invalidation.cli --log-file artifacts/faults.jsonl inject `
  --fault annotation_cycle --namespaces samples/namespaces.json `
  --entries samples/entries.json
python -m cache_invalidation.cli inject --fault truncated_snapshot `
  --snapshot artifacts/demo/snapshot_v3.json
python -m cache_invalidation.cli inject --fault missing_inherit `
  --namespaces samples/namespaces.json --entries samples/entries.json
```

## 6. 实际验证结果（本工作区）

- `python run_tests.py`：45 个用例全部通过，覆盖图校验、三版本快照、
  传播确定性、并发互斥、时钟、SQLite 账本、整体替换、重放、注入、
  回环网络与 GUI 视图模型。
- `cli demo`：三个场景顺序完成，账本总结落在
  `artifacts/demo/ledger_summary.json`。
- `cli loopback`：仅绑定 `127.0.0.1` 动态空闲端口，ping/invalidate/state
  请求响应一致。
- 三类注入分别得到 `E_GRAPH`、`E_SNAPSHOT`、`E_INHERIT`，结构化日志
  含 `time/code/key/line/details`，见 `artifacts/faults.jsonl`。

### 环境差异

当前验证容器为 WSL2 中的 Python 3.11.0rc1，`sqlite3/json/threading/
subprocess` 均可用，但该发行版未安装 `tkinter` 系统组件且无 sudo 权限。
GUI 因此采用延迟导入：界面内容由纯函数 `build_view_model` 生成并已在
无显示环境通过单元测试；在 Windows 官方 Python 或安装 `python3-tk`
的环境中执行 `python -m cache_invalidation.gui` 即可打开窗口。
