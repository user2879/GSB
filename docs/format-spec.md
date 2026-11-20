# 清单格式规范

## 1. 通用约定

- JSON 使用 UTF-8。
- 所有整数不能是布尔值。
- 摘要输出为小写十六进制；解码时接受大小写十六进制，长度固定 64。
- `shard_count` 范围为 `1..256`。
- `global_step` 范围为 `0..1_000_000_000`。
- 分片列表必须按 `shard_index = 0, 1, ..., shard_count-1` 排列。
- 解码器先完成整条记录的全部校验，再构造 `Manifest` 对象；失败不产生部分加载结果。
- 未声明字段会被拒绝，字段路径写入错误对象 `path` 与 `details.field`。

当前版本：`3`。

## 2. v1

```json
{
  "format_version": 1,
  "global_step": 20,
  "shard_count": 2,
  "shards": [
    {
      "shard_index": 0,
      "parameter_digest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    },
    {
      "shard_index": 1,
      "parameter_digest": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
    }
  ]
}
```

允许字段：

- 顶层：`format_version`、`global_step`、`shard_count`、`shards`
- 分片：`shard_index`、`parameter_digest`

v1 不包含动量、随机数状态、数据游标、划分方式或父引用。升级时这些字段从真值账本补齐。

## 3. v2

```json
{
  "format_version": 2,
  "global_step": 20,
  "shard_count": 1,
  "rng_state": {
    "algorithm": "PCG64",
    "seed": 20261120,
    "state": "00000000000000000000000000000000",
    "inc": "00000000000000000000000000000001",
    "has_uint32": 0,
    "uinteger": 0,
    "draw_step": 20
  },
  "shards": [
    {
      "shard_index": 0,
      "parameter_digest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "momentum_digest": "1111111111111111111111111111111111111111111111111111111111111111"
    }
  ]
}
```

`rng_state` 必须恰好包含：

- `algorithm`：当前只接受 `PCG64`
- `seed`：非负整数
- `state`、`inc`：32 位十六进制串
- `has_uint32`：`0` 或 `1`
- `uinteger`：`0..2^32-1`
- `draw_step`：非负整数

v2 仍不包含数据游标、划分方式或父引用。

## 4. v3

```json
{
  "format_version": 3,
  "manifest_id": "step-00000020",
  "global_step": 20,
  "shard_count": 1,
  "partition_mode": "modulo",
  "rng_state": {
    "algorithm": "PCG64",
    "seed": 20261120,
    "state": "00000000000000000000000000000000",
    "inc": "00000000000000000000000000000001",
    "has_uint32": 0,
    "uinteger": 0,
    "draw_step": 20
  },
  "parent_manifest": {
    "manifest_id": "step-00000010",
    "digest": "2222222222222222222222222222222222222222222222222222222222222222"
  },
  "shards": [
    {
      "shard_index": 0,
      "parameter_digest": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "momentum_digest": "1111111111111111111111111111111111111111111111111111111111111111",
      "data_cursor": {
        "epoch": 7,
        "offset": 3,
        "next_sample_index": 35
      }
    }
  ]
}
```

新增字段：

- `manifest_id`：必须等于 `step-{global_step:08d}`。
- `partition_mode`：`modulo` 或 `contiguous`。
- `parent_manifest`：必须存在。创世步 `0` 可为 `null`；其他步骤必须是对象。
- 分片 `data_cursor`：`epoch`、`offset`、`next_sample_index` 均为非负整数。

`parent_manifest.digest` 是父清单归一化 v3 视图的 SHA-256。父引用按 `manifest_id` 在目录内解析；目标不存在、摘要不符、自引用或链循环均为 `E_CHAIN`。

## 5. 规范化与视图摘要

归一视图只包含 v3 元数据字段，不包含 `format_version`：

```text
manifest_id
global_step
shard_count
partition_mode
shards[]
  shard_index
  parameter_digest
  momentum_digest
  data_cursor{epoch, offset, next_sample_index}
rng_state
parent_manifest{manifest_id, digest}
```

规范字节：

```text
UTF-8(JSON(sorted_keys=true, separators=(",", ":")) + "\n")
```

`view_digest = SHA-256(UTF-8(JSON(sorted_keys=true, separators=(",", ":"))))`，不带结尾换行。逐字节比较使用带结尾换行的规范字节。

## 6. 确定性生成

配置字段：

```json
{
  "seed": 20261120,
  "shard_count": 4,
  "partition_mode": "modulo",
  "steps": [0, 10, 20],
  "sample_count": 64,
  "input_dim": 8,
  "hidden_dim": 6,
  "classes": 3
}
```

- `seed`：`0..2^63-1`
- `shard_count`：`1..256`
- `sample_count`：`shard_count..1_000_000`
- `input_dim` / `hidden_dim`：`1..4096`
- `classes`：`2..1024`
- `steps`：唯一非负整数，必须包含 `0`
- `partition_mode`：`modulo` 或 `contiguous`

PCG64 的派生种子由 `SHA-256(seed|namespace|step)` 前 128 位构造。参数与动量按 C 顺序展平后在各分片间连续切分，摘要域中编码：

- 摘要类型（parameter/momentum）
- 配置种子、分片数、步数、分片序号
- 划分方式
- 张量名、切片长度、IEEE-754 小端 `float64` 字节

数据游标由步数、分片序号和该分片样本索引确定。相同配置重复运行产生相同摘要、随机数状态和游标。

## 7. 真值账本

账本条目包含：

- `global_step`
- `manifest_id`
- 完整 v3 `view`
- `view_digest`
- 逐分片 `parameter_digest` 与 `momentum_digest`

升级、归一同比较均以账本为真值来源。装载时先验证清单已有摘要与账本一致；不一致为 `E_INTEGRITY`，不会进入归一或重写。

## 8. 升级检查点

路径：`<catalog>/.ckptmeta/upgrade-checkpoint.json`。

关键字段：

- `schema_version`
- `target_version`
- `config_digest`
- `plan_digest`
- `run_state`
- `items[]`：`path`、`source_version`、`source_sha256`、`status`、`backup_path`、`target_sha256`

状态机：

```text
pending -> processing -> done
                    \-> unknown (重启后发现 processing)
```

`done` 条目必须仍与 `target_sha256` 一致，否则 `E_UPGRADE`。`unknown` 必须显式重试；重试优先使用只读备份恢复源记录。

## 9. SQLite 快照

表 `snapshots` 保存版本集合、内容摘要、归一摘要、状态 JSON、活动/只读标志。表 `snapshot_items` 保存每个清单的完整归一 JSON 和摘要。

快照状态包含：

- 内嵌配置
- 账本摘要
- 格式版本集合
- 磁盘内容摘要
- 归一视图汇总摘要
- 升级轨迹
- 已处理游标
- 逐清单归一化项

内容有变化时，恢复操作整体重建当前状态，将旧快照标记只读并创建新的活动快照；重放会先比较版本集合和内容摘要，再按内嵌配置重建账本并逐项比较。失败返回 `E_REPLAY`。
