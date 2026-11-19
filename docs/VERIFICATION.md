# 实际验证记录

## 环境

- 当前执行环境：Linux，Python `3.11.0rc1`，SQLite `3.37.2`。
- 当前容器未安装 Tkinter，也未提供图形显示；`docdb_index/gui.py` 已通过语法编译，Windows Python 安装含 Tkinter 时可直接启动。
- 当前环境没有 `python` 别名，使用 `python3` 执行；Windows 使用 `python`。
- 无第三方包、无网络访问。

## 自动化测试

命令：

```text
python3 -m unittest discover -s tests -v
```

结果：`Ran 11 tests in 3.4–3.8s`，`OK`。

覆盖：

1. `E_DOC` 非法路径/重复主键且不留下 `session.db`；
2. 相同输入两次初始构建的段标识与内容 BLOB 完全一致；
3. 单键更新只替换受影响段，其他段摘要与版本不变；
4. 变更顺序与 `--batch-size` 不同仍产生相同段集合与账本；
5. 停滞窗口查询与全量扫描一致，陈旧回收及账本幂等；
6. 重复索引键按主键稳定升序，跨段分页连续；
7. 版本不一致返回 `E_STALE` 且拒绝应用；
8. 中途变更失败时版本、段摘要、审计批次均回滚；
9. 快照创建、只读旧快照、差异明细与整体替换恢复；
10. `E_SEGMENT`、`E_BOUND`、`E_LEDGER` 三类注入及 JSONL 日志；
11. 内容表篡改后重放返回 `E_REPLAY`，定位首个段与字段。

## 48 条本地样本

生成：

```text
python3 samples/generate.py --scale 48 --seed 20261120 --capacity 5 --output-dir samples
```

初始构建：

- 文档 48 条，段 10 个，集合/索引版本均为 `7`。
- 第二次生成集合与变更文件 SHA-256 完全一致。

应用 `incremental` 批次（8 个变更，`--batch-size 2`）：

- 集合版本 `7 -> 8`；
- 受影响段 8 个，未受影响段 2 个；
- 识别并回收 6 条陈旧引用，账本 6 条；
- 未受影响段的摘要、版本与内容 BLOB 逐段一致。

保留停滞窗口 `isolation` 批次：

- 3 条物理陈旧条目仍留在段内；
- 索引查询与全量扫描均返回 47 条，`items` 逐字节一致；
- 查询过滤 3 条，诊断报告逐段显示对应主键/键值；
- `reclaim` 后物理陈旧条目为 0，历史账本仍为 13 条。

验证：

```text
structure: collections=1, segments=14, entries=47, ledger=13
replay: batches_checked=4, unaffected_segments_checked=31
```

## 500 条压力冒烟

```text
python3 samples/generate.py --scale 500 --seed 991 --capacity 7 --output-dir /tmp/sample500
```

- 初始 500 条文档生成 72 个段。
- 8 个变更的原子批次：10 个受影响段、64 个未受影响段、6 条陈旧引用。
- 优化推导链后的应用耗时约 `0.26s`。
- 查询 `0..10000` 返回 494 条，耗时约 `0.15s`。
- `verify` 检查 74 段、500 条目、6 条账本，并成功重放 1 个批次与 64 个未受影响段。

## 三类故障注入

| 注入 | 退出码 | 返回码 | 定位 |
|---|---:|---|---|
| 段摘要损坏 | 2 | `E_SEGMENT` | `segment_id=seg-00000001`, `field=digest` |
| 条目数不一致 | 2 | `E_BOUND` | `segment_id=seg-00000001`, `field=entry_count` |
| 账本行缺失 | 2 | `E_LEDGER` | 首个缺失主键/陈旧键 |

对应 `logs/events.jsonl` 均包含 `INJECT` 与预期业务码。

## `E_STALE`

将集合版本从 `7` 外部改为 `99` 后：

```json
{
  "code": "E_STALE",
  "stale_dimension": "index_version_vs_collection_version",
  "collection_version": 99,
  "index_version": 7
}
```

## `E_REPLAY`

直接篡改某段首个 `value_json` 后，独立重放：

```json
{
  "code": "E_REPLAY",
  "batch_id": "<final>",
  "context": "current_vs_replay",
  "segment_id": "seg-00000001",
  "field": "digest"
}
```
