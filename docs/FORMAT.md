# 清单格式与校验规则

## 1. 位图编码

- `bitmap_length = ceil(volume_block_count / 8)`。
- 块 `i` 的位是 `bitmap[i // 8] & (1 << (7 - i % 8))`。
- `bitmap` 字段是位图字节的小写十六进制字符串，不带 `0x`。
- 最后一个字节中超出 `volume_block_count` 的位必须为 0。

## 2. 文档骨架

```json
{
  "format_version": "m3",
  "volume_block_count": 16,
  "block_size": 512,
  "manifest_digest": "<64 hex>",
  "snapshots": []
}
```

### m1 条目

```json
{
  "snapshot_id": "s0",
  "volume_block_count": 16,
  "bitmap_length": 2,
  "bitmap": "0000"
}
```

为兼容“仅记录位图长度”的更早期 m1 变体，`bitmap` 也允许省略；此时归一必须提供 SQLite resolver，位图会由磁盘位图或摘要索引重建并记录到 `filled_fields/fill_basis`。

### m2 条目新增

```json
{
  "digest_table": [{"block": 3, "digest": "<64 hex>"}],
  "bitmap_checksum": "<64 hex>",
  "snapshot_version": 2
}
```

### m3 条目再新增

```json
{
  "parent_snapshot_id": "s0",
  "block_refs": [{"block": 3, "ref": "<64 hex>"}],
  "content_digest": "<64 hex>"
}
```

摘要值可接受裸 64 位十六进制或 `sha256:` 前缀；内部统一为裸值，输出显示时加前缀。

## 3. 校验顺序

1. 文档是对象，`format_version` 属于 `m1/m2/m3`。
2. 顶层与条目没有未知字段，所有本版本必填字段存在。
3. `volume_block_count > 0`，`block_size` 属于支持集合。
4. 每条 `volume_block_count` 与顶层一致。
5. `bitmap_length` 与块数相容，十六进制可解码，长度精确，尾部位为 0。
6. m2/m3 的 `digest_table` 项数和索引集合等于置位块集合。
7. m2/m3 的 `bitmap_checksum` 等于位图字节 SHA-256。
8. m3 的 `block_refs` 与摘要表索引集合一致。
9. m3 的 `content_digest` 与规范化条目内容一致。
10. 父引用存在且链无环。
11. m3 的 `manifest_digest` 与规范化完整文档一致。

任何失败都抛 `E_FORMAT`，`field` 是首个失败字段，`snapshot_id` 是条目标识；截断 JSON 尽量从已保留文本中提取首个 snapshot id。

## 4. 规范化摘要

摘要使用 UTF-8、键排序、无多余空白的 JSON 序列化（`ensure_ascii=False`, `sort_keys=True`, `separators=(',', ':')`）。

- `content_digest` 的 payload 是去掉自身后的 m3 规范化条目。
- `manifest_digest` 的 payload 是包含所有 m3 条目的文档，去掉顶层 `manifest_digest`。
- `bitmap_checksum` 是原始位图字节的 SHA-256，不是十六进制文本的 SHA-256。
- 内容对象引用 `ref` 与条目的 `digest` 在当前实现中都是旧块字节的 SHA-256。

## 5. 历史格式补齐

| 版本 | 丢弃/缺失字段 | 补齐来源 |
| --- | --- | --- |
| m1 | 摘要表、引用、父链、位图校验、内容摘要、快照版本 | SQLite `snapshot_blocks` / `snapshots`，校验和由位图计算 |
| m2 | 父链、数据引用、内容摘要、顶层清单摘要 | SQLite 父链/引用，内容摘要由规范化 m3 计算 |
| m3 | 无 | 无需补齐 |

归一结果示例：

```json
{
  "source_format": "m1",
  "filled_fields": ["snapshots.s0.digest_table", "snapshots.s0.block_refs"],
  "fill_basis": {
    "snapshots.s0.digest_table": "sqlite_snapshot_blocks",
    "snapshots.s0.block_refs": "sqlite_snapshot_blocks"
  }
}
```

`fingerprint` 不包含 `source_format/filled_fields/fill_basis`，只覆盖归一后的语义内容，因此用于证明三版本等价。
