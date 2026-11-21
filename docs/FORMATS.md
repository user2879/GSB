# 输入格式与稳定序列化

## 观测帧

固定头为小端 `<4sHIIQIIHHBBIH`，`struct.calcsize` 为 42。

字段顺序：

```text
magic, frame_len, flow_wire_id, seq, send_ms,
src_ip, dst_ip, src_port, dst_port, protocol,
flags, arrival_ordinal, payload_len
```

`flow_wire_id = crc32(flow_id) & 0xffffffff`，`flow_id` 为：

```text
{src_ip}:{src_port}->{dst_ip}:{dst_port}/{protocol}
```

合法标志位：

```text
0x01 DATA
0x03 DATA | RETRANSMIT
0x05 DATA | REORDER
0x07 DATA | RETRANSMIT | REORDER
```

## 清单文件

文件级必需字段：`version`、`source_id`、`normalization_defaults`、`records`。用于对齐的流可由文件级 `five_tuple` 或记录级 `five_tuple` 给出。`time_basis` 与 `alignment` 缺失时不会被推断，而是进入未对齐集合。

记录字段矩阵：

| 字段 | v1 | v2 | v3 |
|---|---|---|---|
| `event_seq` | 必填 | 必填 | 必填 |
| `loss_count` | 必填 | 必填 | 必填 |
| `five_tuple` | 文件级或记录级 | 文件级或记录级 | 文件级或记录级 |
| `link_id` | 显式默认 | 必填 | 必填 |
| `time_window_ms` | 显式默认 `null` | 必填 | 必填 |
| `retransmit` | 显式默认 | 显式默认 | 必填 |
| `reorder_depth` | 显式默认 | 显式默认 | 必填 |

## 规范序列化

```python
json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
```

摘要为上述 UTF-8 字节的 SHA-256。结论摘要不依赖文件遍历顺序、列表插入顺序或数据库行顺序。

