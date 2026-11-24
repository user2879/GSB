# 技术说明

## 1. 组件边界

本组件模拟 TLS 消息顺序和证书链验证，但不是真实 TLS 实现，也不提供可用于生产的密码学。证书中的模数、签名和签名算法共同构成一个确定性 toy-RSA 校验协议；全部路径、时间、得分和误差均使用 Python 任意精度整数完成，不使用浮点。

运行边界固定为 `127.0.0.1`。父进程按握手脚本发送客户端消息，子进程只返回脚本声明的服务端回执组。握手超时来自父进程真实套接字的 `settimeout`，不是线程计时器或人为延迟替代。

## 2. JSON 契约

### 2.1 证书链声明

```json
{
  "format_version": 1,
  "trust_anchors": ["root-ca"],
  "fixed": {
    "scale": 1000000,
    "max_bits": 256,
    "time_quantum_ms": 1,
    "time_epoch_ms": 1700000000000,
    "max_path_length": 5,
    "path_weight": 1,
    "overflow_policy": "saturate",
    "division_policy": "floor"
  },
  "certificates": [
    {
      "id": "leaf-server",
      "issuer": "intermediate-ca",
      "serial": 3001,
      "not_before_ms": 1699999999000,
      "not_after_ms": 1700001000000,
      "modulus": 123,
      "signature": 456,
      "algorithm": "TOY_RSA_E65537_SHA256"
    }
  ]
}
```

强校验规则：

- 证书对象只允许上述 8 个字段；ID 非空且唯一；签发者必须存在；根证书自签时必须是已声明信任锚；任何其他自环或互环都在 `issuer` 字段以 `E_SPEC` 拒绝。
- `serial`、`modulus` 为正整数，`signature` 为非负整数；有效性起止为整数且起点早于终点。
- 支持算法：`TOY_RSA_E3_SHA256`、`TOY_RSA_E5_SHA256`、`TOY_RSA_E17_SHA256`、`TOY_RSA_E65537_SHA256`。
- `scale < 2^max_bits`；截断策略只能是 `floor`，溢出策略只能是 `saturate`。
- 重复 JSON 键、额外字段、错误类型和损坏 JSON 都以首个非法证书 ID 与字段路径报告。

### 2.2 握手脚本

```json
{
  "format_version": 1,
  "socket_timeout_ms": 750,
  "messages": [
    {
      "seq": 1,
      "sender": "client",
      "client_seq": 1,
      "server_seq": 0,
      "arrival_ms": 1700000000000,
      "type": "CLIENT_HELLO",
      "cert_ids": ["leaf-server", "intermediate-ca", "root-ca"],
      "expected": "PASS"
    }
  ]
}
```

- `seq` 从 1 连续递增；`arrival_ms` 单调不减。
- `client_seq` 在客户端消息出现时递增 1；服务端消息保留最近客户端序号。`server_seq` 对称处理。
- `cert_ids` 是主候选链，必须非空、引用存在且链内不重复；可选 `alternative_cert_ids` 是第二候选链。
- `expected` 只能是 `PASS`、`FAIL`、`INDETERMINATE`。
- 脚本必须从客户端消息开始。一个客户端消息与其后连续服务端消息组成一个网络批次。

## 3. 整数定点模型

记会话缩放因子为 `S`（默认 `1,000,000`），整数位宽上限为 `B`，有符号饱和范围是 `[-M, M]`，其中 `M = 2^B - 1`。所有除法按下式向下取整：

```text
floor_div(a, b) = a // b,  b > 0
```

任何结果越界都饱和到 `M` 或 `-M`，并在证书/消息记录中记录饱和标志。发生饱和的证书状态至少为 `INDETERMINATE`，而不是把被截断结果当作确定值。

### 3.1 模数量化与签名

对于证书 `C`，实际验证密钥是其签发者的模数 `n`；自签根使用自身模数。设：

```text
D      = SHA-256(canonical_signed_payload(C)) 的整数
L      = pow(C.signature, exponent(algorithm), n)
R      = D mod n
Lq     = floor(L * S / n)
Rq     = floor(R * S / n)
resid  = abs(Lq - Rq)
```

每个归一化余数向下取整产生小于 1 个定点单位的损失，因此单证书签名比较误差界为：

```text
modulus_quantization = 2
```

`resid == 0` 为确定通过；`0 < resid <= 2` 为不可区分；更大为确定失败。签名验证使用任意精度整数模幂和整数归一化，不经过浮点。

### 3.2 时间窗口归属

设时间戳为 `t`，会话纪元为 `E`，时间量化为：

```text
q(t) = floor((t - E) * S / time_quantum_ms)
```

有效窗口采用左闭右开 `[q(start), q(end))`。起点和终点各有一个向下取整误差，因此单证书时间归属误差界为：

```text
time_window = 2
```

事件点距任一边界不超过 2 个定点单位时，固定点无法稳定区分边界归属，状态为 `INDETERMINATE` 并标注受误差影响。同一候选链内任意两个证书窗口边界之间的距离若不超过链级合成误差界，也记录 `window_boundary_ties`，按证书 ID 升序稳定展示，并将链状态提升为不可区分（确定失败仍保持失败）。

### 3.3 路径长度权重

链长度为 `N` 时：

```text
path_score = floor((max_path_length - N + 1) * path_weight * S / max_path_length)
```

长度超过上限为确定失败；路径权重归一化再引入一个单位的取整误差：

```text
path_length = 1
```

### 3.4 逐级与链级误差界

对第 `i` 个证书，累计比较误差界为：

```text
step_bound(i) = 4 * i
```

长度 `N` 的候选链：

```text
modulus_quantization = 2N
time_window          = 2N
path_length          = 1
chain_total          = 4N + 1
```

两条候选链做不可区分比较时，使用两条链误差界之和：

```text
pairwise_bound = chain_total(A) + chain_total(B) = 4N_A + 4N_B + 2
```

样本主链长度 3，因此单链总界 13，两条同长候选链的不可区分阈值为 26。最终记录的 `error_bound` 同时给出三段贡献、总界与每个证书的累计界。

## 4. 稳定裁决

候选链按以下得分向量降序排列：

```text
[通过证书数, -签名残差和, 最小时间边界余量, 路径得分, 锚匹配, -路径长度]
```

若得分向量仍相同，按链的证书 ID 序列字典序升序。实现使用显式排序键，不依赖集合迭代或哈希随机化。

判定规则：

- 单链且各证书/路径均确定通过：`PASS`。
- 任一组件确定失败：`FAIL`。
- 任一组件处于边界或饱和不可区分状态：`INDETERMINATE`。
- 多链最佳与次佳得分差不超过两条链的合成误差界：`INDETERMINATE`，仍按向量序、证书 ID 序选出展示链，并标记 `affected_by_error=true`。
- 多链全部确定失败：`FAIL`；最佳链显著领先时采用最佳链自身状态。

样本消息 6 的两条候选链每级证书值、窗口和路径相同，得分向量完全一致，因此按证书 ID 稳定选中 `leaf-b-1` 链，并标注受误差影响。

## 5. 失效传播与增量重算

证书依赖边为 `issuer -> child`，忽略受信任根的自环边。变更检测覆盖：

- `issuer`、`serial`、有效期、模数、签名、算法；
- 证书新增/删除；
- 会话固定点参数变化。
- 信任锚集合变化；
- 握手消息自身字段变化（到达时刻、类型、候选链、期望值等）。

传播算法：

1. 直接变更证书进入初始集合。
2. 沿旧、新两份 issuer 图向下传播所有后代。
3. 固定点参数变化直接使全部证书失效。
4. 信任锚集合变化不重算证书基础中间量，但使全部消息验证失效。
5. 消息自身字段变化时，该消息直接失效。
6. 消息若携带失效证书则直接失效。
7. 握手具有顺序依赖，消息失效后其后续消息也失效。
8. 证书节点使用以证书 ID 为堆键的稳定 Kahn 拓扑排序；消息按 `seq` 排序。

每个轨迹步骤包含 `order`、`kind`、`id`、`reason`、`field`，例如：

```text
direct_change:not_after_ms,signature
carries_affected_certificate:leaf-expired
follows_affected_message:4
fixed_parameters_changed
```

未失效证书的中间量字典和未失效消息的结果字典直接从上一会话保留，不重新构造。测试同时对增量结果与空目录全量重建逐字段比较，并对保留项做 canonical JSON 字节比较。

## 6. SQLite 账本

核心表：

| 表 | 作用 |
| --- | --- |
| `ledger_info` | 单行整数 `format_version` 与全账本 `content_digest` |
| `batches` | 批次号、进度游标、规范载荷、载荷摘要 |
| `messages` | 每消息记录及其摘要 |
| `session_snapshot` | 单行整体快照、格式版本与磁盘摘要 |

每个客户端/服务端回执组形成一个批次。提交事务一次写入：

- 本批逐消息验证中间量和误差界；
- 最终判定、不可区分标记、选择原因；
- 失效传播轨迹；
- 进度游标和规范载荷摘要；
- 更新后的全账本内容摘要。

账本启动时验证 `format_version` 为整数且等于 1，并重算批次内容摘要。单条记录还有独立摘要；任何批次或记录篡改都以 `E_REPLAY` 报告。

## 7. 快照整体替换

快照正文包含格式版本、全部证书、信任锚和固定点参数。打开账本时重新计算磁盘规范摘要：

- 快照缺失、版本不同或摘要不同：删除旧行，单事务插入完整新快照。
- 绝不逐证书合并。
- 返回稳定排序的 `added_cert_ids`、`removed_cert_ids`、`changed_cert_ids`。
- 账本批次本身不因快照替换而重写；若声明变化导致重放不一致，随后由 `E_REPLAY` 拒绝。

## 8. 恢复与重放

恢复只信任 SQLite 中已完整提交的批次：

1. 读取并校验账本版本、批次摘要、消息摘要。
2. 要求已提交消息构成脚本的连续前缀。
3. 用当前声明从零计算应然记录，与已提交记录的判定、误差界、证书中间量、失效轨迹逐字段比较。
4. 首个差异返回 `E_REPLAY`，带首个消息序号、证书 ID 和字段路径。
5. 游标之后的消息标记为未知，启动确定性服务重跑并逐批提交。
6. 全部批次完成后再次做全量账本复验。

强制终止无法执行退出报告；账本已提交事务由 SQLite 保证原子可见。优雅信号处理器只设置停止标志，主循环在批次边界停止并通过“临时文件 + `os.replace`”写完整退出报告。

## 9. 错误定位

| 代码 | 场景 | 定位字段 |
| --- | --- | --- |
| `E_SPEC` | JSON、证书、握手序、算法等非法 | 首个 `cert_id`、JSON 路径/字段、可选消息序号 |
| `E_PORT` | 回环端口占用或无法连接 | `socket.port` |
| `E_TIMEOUT` | 父进程真实套接字超时 | 当前客户端消息序号、超时毫秒 |
| `E_REPLY` | 连接关闭、JSON 错误、回执字段缺失/不匹配 | 首个相关消息序号与字段 |
| `E_REPLAY` | 账本摘要、记录或重放结果不一致 | 首个消息序号、证书 ID、字段 |

## 10. Tkinter 界面

界面分为 Overview、Certificate Chain、Verification、Invalidation、Ledger 五个页签，展示：

- 每条消息的确定/失败/不可区分结论与受误差影响标记；
- 每级证书的签名字节余数、定点残差、窗口状态和累计误差界；
- 候选链稳定选择与不可区分阈值；
- 失效传播步骤、未知/未提交消息；
- 账本批次、游标和内容摘要。

刷新只读打开账本并重建排序列表，不调用快照同步、不插入批次、不修改账本。`gui_model.py` 提供同样的无界面模型，便于在无显示环境测试渲染稳定性。

## 11. 集成 API

```python
from tlschain.spec import load_spec
from tlschain.session import build_full_session, recompute_session

spec = load_spec("samples/certificates.json", "samples/handshake.json")
full = build_full_session(spec)
counts = full.decision_counts()

new_spec = load_spec("samples/certificates_modified.json", "samples/handshake.json")
incremental = recompute_session(full, new_spec)
invalid_certificates = incremental.invalidated_cert_ids
invalid_messages = incremental.invalidated_message_seqs
```

账本、服务和 CLI 均可独立调用；详见 `tlschain/ledger.py`、`tlschain/service.py`、`tlschain/runner.py`。
