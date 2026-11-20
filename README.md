# ckptmeta — 训练检查点元数据互操作组件

`ckptmeta` 面向 Windows 本地环境，处理训练检查点清单在 `v1`、`v2`、`v3` 三种格式之间的导入、升级、归一、逐字节一致性校验、可续做升级和 SQLite 会话快照。

组件只处理训练元数据，不重跑训练。内容载体由固定整数种子生成：

- 64 条、8 维、3 类的本地合成分类数据；
- 一个 4 分片小型 MLP 的 `w1/b1/w2/b2` 参数与动量；
- 各分片参数摘要、动量摘要、PCG64 随机数状态、数据游标和父清单引用。

所有文件均在本地磁盘，不访问网络、外部账号或 GPU。

## 环境

- Windows 本机，Python 3.11
- `numpy`
- `pytest`（仅测试）

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
.venv\Scripts\python demo.py --workspace .interop-demo
```

Linux/macOS 下将 `.venv\Scripts\python` 改为 `.venv/bin/python`。项目不使用 Docker 或其他容器环境。

## 快速演示

```bash
python -m ckptmeta demo --workspace .interop-demo
```

演示脚本按固定顺序执行四个场景：

1. 当前版本 `v3` 直达写入、链校验与归一；
2. `v1`、`v2` 目录逐条升级后归一，并与 `v3` 直达视图逐字节比较；
3. 子进程在升级检查点提交后强制终止，重启后只处理未完成条目；
4. 保存 SQLite 快照、恢复快照并重放校验。

演示目录可重复执行；脚本只清理自己管理的子目录。

## 清单格式

当前版本为 `v3`。严格字段与 JSON 示例见 [docs/format-spec.md](docs/format-spec.md)。

| 版本 | 内容 |
| --- | --- |
| `v1` | `global_step`、`shard_count`、各分片 `parameter_digest` |
| `v2` | `v1` + 各分片 `momentum_digest` + `rng_state` |
| `v3` | `v2` + 各分片 `data_cursor` + `partition_mode` + `parent_manifest` + `manifest_id` |

解码要求：

- `format_version` 只能是整数 `1`、`2`、`3`；
- 分片数必须与 `shards` 数量一致，分片索引必须是连续的 `0..N-1`；
- 摘要必须是 64 位十六进制串；
- `global_step` 必须是非负整数；
- `v3` 必须存在 `parent_manifest` 字段；非创世步的父引用必须可解析且摘要匹配；
- 未知字段、缺失字段、未知版本均整体拒绝，不返回部分对象。

## 命令行

生成可重复样本：

```bash
python -m ckptmeta generate --workspace .interop-demo
```

打开目录并升级全部低版本清单：

```bash
python -m ckptmeta open \
  --catalog .interop-demo/historical_v1 \
  --config .interop-demo/config.json \
  --ledger .interop-demo/ledger/truth-ledger.json
```

只升级、不执行其他操作：

```bash
python -m ckptmeta upgrade \
  --catalog .interop-demo/historical_v1 \
  --config .interop-demo/config.json \
  --ledger .interop-demo/ledger/truth-ledger.json
```

归一单个清单：

```bash
python -m ckptmeta normalize \
  --manifest .interop-demo/historical_v1/step-00000020.v1.json \
  --ledger .interop-demo/ledger/truth-ledger.json
```

可用 `--expected-manifest <v3 直达清单>` 增加一次显式逐字节比较；不一致返回 `E_RESTORE` 并定位首个分片和字段。

比较两个清单：

```bash
python -m ckptmeta compare \
  --left .interop-demo/historical_v1/step-00000020.v1.json \
  --right .interop-demo/direct/step-00000020.v3.json \
  --ledger .interop-demo/ledger/truth-ledger.json
```

快照保存、恢复和重放：

```bash
python -m ckptmeta snapshot-save \
  --catalog .interop-demo/direct \
  --config .interop-demo/config.json \
  --ledger .interop-demo/ledger/truth-ledger.json \
  --snapshot-db .interop-demo/snapshots/interop.sqlite3

python -m ckptmeta snapshot-restore \
  --catalog .interop-demo/direct \
  --config .interop-demo/config.json \
  --ledger .interop-demo/ledger/truth-ledger.json \
  --snapshot-db .interop-demo/snapshots/interop.sqlite3

python -m ckptmeta snapshot-replay \
  --catalog .interop-demo/direct \
  --config .interop-demo/config.json \
  --ledger .interop-demo/ledger/truth-ledger.json \
  --snapshot-db .interop-demo/snapshots/interop.sqlite3
```

强制终止注入：

```bash
# 模拟在第一条清单已提交检查点后进程被终止
python -m ckptmeta upgrade ... --crash-after-completed 1

# 模拟清单已重写但检查点尚未提交，重启后会变为 unknown
python -m ckptmeta upgrade ... --crash-after-rewrite 1
```

异常注入会直接修改工作区内的输入文件，不修改实现代码，并以对应非零退出码结束：

```bash
python -m ckptmeta demo --workspace .interop-demo-integrity --inject-integrity  # 12 / E_INTEGRITY
python -m ckptmeta demo --workspace .interop-demo-chain --inject-chain          # 13 / E_CHAIN
python -m ckptmeta demo --workspace .interop-demo-upgrade --inject-upgrade      # 14 / E_UPGRADE
```

## JSONL 命令循环

`session` 从 stdin 逐行读取 JSON，向 stdout 逐行写回 JSON：

```bash
python -m ckptmeta session \
  --catalog .interop-demo/direct \
  --config .interop-demo/config.json \
  --ledger .interop-demo/ledger/truth-ledger.json \
  --snapshot-db .interop-demo/snapshots/interop.sqlite3
```

请求示例：

```json
{"request_id":"1","command":"import","catalog":".interop-demo/historical_v1","upgrade":true}
{"request_id":"2","command":"normalize","manifest":".interop-demo/historical_v1/step-00000020.v1.json"}
{"request_id":"3","command":"compare","left":".interop-demo/historical_v1/step-00000020.v1.json","right":".interop-demo/direct/step-00000020.v3.json"}
{"request_id":"4","command":"snapshot_save"}
{"request_id":"5","command":"snapshot_restore","verify_replay":true}
{"request_id":"6","command":"snapshot_replay"}
{"request_id":"7","command":"quit"}
```

支持的命令：`scan`、`open` / `import` / `upgrade`、`normalize`、`compare`、`snapshot_save`、`snapshot_restore`、`snapshot_replay`、`snapshot_list`、`ping`、`quit`。业务错误作为带 `business_code` 的 JSON 行返回；会话最终以非零码结束。

## Python API

```python
from ckptmeta import (
    Catalog,
    SnapshotStore,
    build_ledger,
    compare_upgrade_paths,
    decode_manifest,
    load_config,
    normalize_manifest,
    write_direct_manifest,
)

config = load_config("config.json")
ledger = build_ledger(config)
write_direct_manifest("direct/step-00000010.v3.json", ledger, 10)

catalog = Catalog("historical_v1", config, ledger)
opened = catalog.open()                    # 逐条升级 + 检查点续做 + 链校验 + 账本校验
manifest = decode_manifest("historical_v1/step-00000010.v1.json")
normalized = normalize_manifest(manifest, ledger)

equivalence = compare_upgrade_paths(
    decode_manifest("historical_v1/step-00000020.v1.json"),
    decode_manifest("historical_v2/step-00000020.v2.json"),
    decode_manifest("direct/step-00000020.v3.json"),
    ledger,
)

with SnapshotStore("snapshots/interop.sqlite3") as snapshots:
    snapshot = snapshots.save(catalog)
    restored = snapshots.restore(catalog)
    replay = snapshots.replay(catalog)
```

完整公开接口见 `ckptmeta/__init__.py`。

## 升级与续做

- 清单目录为平铺的 `*.json`；控制目录为 `.ckptmeta/`。
- 低版本清单逐条在内存中构造完整 `v3` 视图，再以同目录临时文件 + `os.replace` 原子重写。
- 原文件在 `.ckptmeta/backups/` 保留只读备份。
- 每完成一条，检查点 `.ckptmeta/upgrade-checkpoint.json` 原子更新。
- `pending` 条目继续处理；`done` 条目不再处理；`processing` 在重启时转为 `unknown` 并返回 `E_UPGRADE`，必须显式 `--retry-unknown`。
- 强制终止时不会生成成功报告；恢复报告明确标注 `previous_run_report: "unavailable"`。

## 快照与恢复

SQLite 快照保存：

- 清单格式版本集合；
- 清单文件内容摘要；
- 归一视图汇总摘要；
- 升级轨迹与已处理条目游标；
- 生成真值账本所需的配置；
- 每个清单的完整归一化 JSON 与视图摘要。

恢复时先比较版本集合和内容摘要。不一致时基于当前磁盘整体重建会话状态，旧快照标记为只读，并新建活动快照；随后按快照内嵌配置重新归一、逐项比较。不可复现时返回 `E_REPLAY`，并给出 `reason`、清单路径和首个不一致字段。

## 业务码

| 业务码 | 退出码 | 含义 |
| --- | ---: | --- |
| `E_CONFIG` | 10 | 配置越界、字段非法或配置与账本不一致 |
| `E_FORMAT` | 11 | 未知版本、字段缺失/多余、路径或格式校验失败 |
| `E_INTEGRITY` | 12 | 分片摘要或随机数状态与真值账本不一致 |
| `E_CHAIN` | 13 | 父引用缺失、摘要不匹配或链循环 |
| `E_UPGRADE` | 14 | 检查点不符、未知条目需要重跑或升级计划不完整 |
| `E_RESTORE` | 15 | 归一视图与预期视图不一致 |
| `E_DIVERGE` | 16 | 两条升级/直达路径逐字段比较不一致 |
| `E_REPLAY` | 17 | 快照内容或归一结果不可复现 |
| `E_IO` | 18 | 本地磁盘或 SQLite 读写失败 |
| `E_INTERNAL` | 19 | 未预期内部错误 |

结构化日志为 JSONL，每条至少包含 `timestamp`、`business_code`、`manifest_path`、`shard_index` 和 `details`。

## 项目结构

```text
ckptmeta/
  config.py      配置边界与步骤定义
  generator.py   确定性数据、MLP 参数和摘要
  ledger.py      真值账本与摘要校验
  view.py        v3 视图、规范字节和首差异定位
  formats.py     v1/v2/v3 严格编解码
  normalize.py   整体替换式归一与一致性比较
  catalog.py     目录扫描、链校验、可续做升级
  snapshot.py    SQLite 快照、恢复与重放
  logging.py     结构化 JSONL 日志
  cli.py         命令行与 JSONL 会话
  samples.py     固定样本与可重复演示夹具
tests/           45 项 pytest 验证
docs/            格式规范与实际验证记录
```

实际测试结果见 [docs/verification.md](docs/verification.md)。

## 说明

- 组件不会访问网络，也不会连接外部账号。
- 所有路径使用 `pathlib`，原子替换使用同目录临时文件和 `os.replace`，适配 Windows 本地文件系统。
- 清单目录、检查点目录和 SQLite 文件所在位置需要当前用户具备读写权限。
- 父引用使用逻辑 `manifest_id` 加归一视图摘要，不依赖绝对磁盘路径，因此可跨目录比较。
