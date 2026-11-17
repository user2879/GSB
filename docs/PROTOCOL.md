# 封装与线协议

## 1. 固定封装头

所有整数为大端。结构为 `>4sBBBB32sQIIIIHH16s`，固定 84 字节：

| 偏移 | 长度 | 字段 | 说明 |
|---:|---:|---|---|
| 0 | 4 | `magic` | ASCII `TVPN` |
| 4 | 1 | `version` | 固定为 `1` |
| 5 | 1 | `flags` | bit0=`FRAGMENTED`，bit1=`LAST_FRAGMENT` |
| 6 | 1 | `cipher_id` | 0=`NONE`, 1=`AES-128-GCM`, 2=`AES-256-GCM`, 3=`CHACHA20-POLY1305` |
| 7 | 1 | `reserved` | 必须为 0 |
| 8 | 32 | `tunnel_id` | UTF-8，NUL 右填充；不得含 NUL，最长 32 字节 |
| 40 | 8 | `sequence` | 非负 64 位整数，隧道内单调 |
| 48 | 4 | `payload_length` | 当前帧负载字节数 |
| 52 | 4 | `original_length` | 未分片逻辑负载总长度 |
| 56 | 4 | `fragment_offset` | 当前分片在原始负载中的字节偏移 |
| 60 | 4 | `fragment_size` | 非末尾分片大小；用于精确推导缺失偏移 |
| 64 | 2 | `fragment_index` | 从 0 开始 |
| 66 | 2 | `fragment_count` | 总片数 |
| 68 | 16 | `digest` | 以下摘要的前 16 字节 |

帧摘要：

```text
SHA256("FRAME\0" || header[0:68] with digest field zero || payload)[0:16]
```

逻辑负载摘要（用于端到端对端回送比较）：

```text
SHA256("PAYLOAD\0" || original_payload)[0:16]
```

摘要不用于机密性认证，仅用于确定性实验中的完整性比较。

## 2. 分片规则

- 单帧总长必须 `<= route.mtu`，故分片负载容量为 `mtu - 84`。
- 一个逻辑消息的所有分片共享同一 `sequence`。
- 非末尾分片大小固定为容量，末尾分片可更短；`fragment_offset = index * fragment_size`。
- `LAST_FRAGMENT` 仅在 `index == fragment_count - 1` 时设置。
- 未分片帧不设置 `LAST_FRAGMENT`，且 `fragment_count=1`。
- 接收端接受乱序；字节完全相同的重复分片计入 `duplicates` 后忽略；同偏移不同内容为 `E_ENCAP`。
- 超时未齐片为 `E_FRAG`，`missing_offsets` 按 `i * fragment_size` 精确列出。
- 同一隧道的未完成分片组允许同序号帧；消息完成后重复同序号为 `E_ENCAP`。

不超限负载可强制走分片/不分片两条路径，最终重组负载和逻辑负载摘要一致。

## 3. 回环线协议

引擎与对端使用 TCP：

```text
4-byte big-endian frame_length || encapsulation_frame
... repeated ...
client half-close (SHUT_WR)
peer responds with one UTF-8 JSON line
```

正常回送示例：

```json
{"code":"OK","tunnel_id":"alpha","seq":0,"payload_bytes":1111,"payload_digest":"...","frame_digests":["..."],"fragmented":true,"duplicates":0,"fragment_count":7}
```

异常回送沿用 `E_ENCAP/E_FRAG` 代码，并带 `error_offset`、`missing_offsets` 等上下文。引擎将连接阶段失败映射为 `E_PEER`，回送等待超时映射为 `E_TIMEOUT`，成功回送但摘要不一致映射为 `E_DIGEST`。

## 4. 对端行为注入

行为只存在于对端进程/JSON 参数，不改变引擎或编解码：

- `normal`：验证帧、重组并回送正确摘要。
- `delay-script`：按 `delay_ms` 或脚本 `by_sequence` / `delays_ms` 延迟回送；超过引擎 `peer_timeout_ms` 即 `E_TIMEOUT`。
- `unreachable`：不监听目标端口；连接失败为 `E_PEER`。
- `digest-error`：正常接收但故意翻转回送摘要首字节；引擎为 `E_DIGEST`。

命令行 `--behavior`、`--delay-ms`、`--script` 可覆盖 JSON 对端定义。
