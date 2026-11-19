# 分片组声明模式

## 顶层字段

| 字段 | 类型 | 规则 |
|---|---|---|
| `format` | string | 必须为 `shardrep-spec`，省略时按该值处理 |
| `format_version` | integer | 当前仅支持 `1` |
| `group_id` | string | 非空 |
| `group_version` | integer | 正整数 |
| `mirror_graph_version` | integer | 正整数 |
| `content.engine` | string | 样本生成规则名；当前为 `shake256-v1` |
| `content.seed` | integer | 固定整数随机种子 |
| `nodes` | array | 非空节点数组 |
| `blocks` | array | 非空块数组 |
| `relations` | array | 可为空的关系数组 |

## 节点与物理分片

| 字段 | 类型 | 规则 |
|---|---|---|
| `node_id` | string | 全组唯一、非空 |
| `directory` | string | 相对规格目录的路径；禁止绝对路径和 `..`；必须是已存在目录 |
| `pack_quota` | integer | `>= 0`，表示该节点同时运行的打包任务上限 |
| `repair_enabled` | boolean | 是否允许把修复内容写回该节点，默认 `true` |
| `shards[].shard_id` | string | 全组唯一物理副本标识 |
| `shards[].logical_id` | string | 逻辑分片标识；每个节点至多一个该逻辑副本 |
| `shards[].seq` | integer | `>= 0`；同一逻辑分片所有副本一致 |
| `shards[].bytes` | integer | 严格正整数；同一逻辑分片所有副本一致 |
| `shards[].digest` | string | `sha256:` 加 64 位小写十六进制；同一逻辑分片所有副本一致 |
| `shards[].file` | string | 相对节点目录的路径；禁止绝对路径和 `..`；同一节点内不得重复 |

分片文件缺失不是规格错误，而是在副本清单中登记为 `missing`。文件长度、摘要不匹配分别登记为 `truncated`、`length_mismatch`、`digest_mismatch`，并产生 `E_INTEGRITY`。

## 数据块

| 字段 | 类型 | 规则 |
|---|---|---|
| `block_id` | string | 全组唯一、非空 |
| `seq_start` | integer | `>= 0` |
| `seq_end` | integer | `>= seq_start`，不得越出逻辑序号范围 |
| `checksum` | string | 规范块声明摘要 |

所有块按 `seq_start` 排序后必须：

1. 第一块从序号 0 开始。
2. 每个后一块的 `seq_start == 前一块 seq_end + 1`。
3. 最后一块覆盖最大逻辑序号。
4. 区间内每个序号均存在逻辑分片。

块校验摘要的规范输入为：

```json
{
  "block_id": "block-alpha",
  "seq_start": 0,
  "seq_end": 2,
  "shards": [
    {"seq": 0, "logical_id": "s0", "bytes": 1024, "digest": "sha256:..."}
  ]
}
```

对规范 JSON 做 SHA-256，得到 `sha256:<hex>`。

## 镜像关系

派生镜像：

```json
{"kind": "mirror", "source": "s1", "target": "s2"}
```

拼接：

```json
{"kind": "join", "sources": ["s3", "s4"], "target": "s5"}
```

`source`、`sources[]` 与 `target` 必须指向已声明的逻辑分片。拼接来源不得重复。任何关系指向自身或关系图出现环，整体拒绝加载；顶层业务码为 `E_SPEC`，原因码为 `E_CYCLE`。

## 确定性内容

对每个逻辑分片，生成材料为：

```text
shardrep-shake256-v1\0<seed>\0<logical_id>\0<byte_count>
```

其 SHAKE-256 输出即分片字节。安装 NumPy 时用 `numpy.frombuffer(..., dtype=uint8)` 物化字节缓冲；未安装时直接使用标准库字节串，两者逐字节相同。文件摘要为 SHA-256。

