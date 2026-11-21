# TLS 信任锚轮换演练与链验证诊断工作台

这是一个面向 Windows 本地环境、完全基于 Python 标准库实现的 TLS 信任库轮换演练工作台。它在 `127.0.0.1` 高位端口上执行真实 TLS 握手，从同一条 TCP 连接的 TLS 记录中捕获服务端实际发送的证书链，再按本地 JSON 信任库修订逐级做离线链验证，并把结论、首失败层级、分类翻转、信任库版本切换和快照重放写入可判定账本。

核心结论不是“某张证书是否可信”的一次性判断，而是：

- 同一张真实握手链在相邻信任库修订下，结论和失败分类如何翻转；
- 翻转点如何按修订号稳定排序并逐点复核；
- 信任库回滚或重建后，旧判断如何整体失效、只读保留且不参与当前统计；
- 进程被强制终止后，如何只依据逐修订已提交结果恢复，并把缺失修订标成 `UNKNOWN`；
- 完整会话快照如何重新验证，并在校验口径变化时以 `E_REPLAY` 精确定位。

## 快速开始

要求 Python 3.11+，Windows 官方安装包建议勾选 `tcl/tk and IDLE` 以启用 Tkinter。

```powershell
python scripts/check_env.py
python scripts/run_demo.py --workdir data/demo
python -m tlstrust ui --workdir data/demo
```

也可以直接使用 PowerShell 包装脚本：

```powershell
.\scripts\demo.ps1 --workdir data/demo
```

演示会在调用结束时输出稳定排序的三场景 JSON，并写入：

- `data/demo/family/family.json`：确定性证书族；
- `data/demo/family/certs/`、`keys/`、`chains/`：证书、本地诊断私钥与服务端链；
- `data/demo/trust_revisions.json`：实际加载的六修订信任库；
- `data/demo/ledger.sqlite3`：逐修订提交账本；
- `data/demo/snapshots/`：真实握手输入快照与强制终止标记；
- `data/demo/sessions/`：完整会话快照；
- `data/demo/logs/events.jsonl`：结构化 JSONL 事件日志；
- `data/demo/demo_summary.json`：三场景总结。

`data/`、SQLite 和 PEM/KEY 文件默认不进入版本库。证书族和信任修订都可由给定参数与种子重新生成。

## 目录结构

```text
tlstrust/
  asn1.py          DER/ASN.1 编解码
  rsa.py           确定性 RSA 密钥生成、PKCS#1 v1.5 SHA-256 签名/验证
  x509.py          X.509 v3 证书构造与解析
  certfamily.py    证书族生成、PEM/链文件与清单
  trust.py         信任修订 JSON 校验、规范化与摘要
  handshake.py     标准库 ssl 真实握手 + 回环 TLS 1.2 Wire Capture
  verify.py        离线链验证、五类失败、翻转、稳定索引、差异比较
  ledger.py        SQLite 信任版本、逐修订提交、陈旧隔离、恢复与重放
  workspace.py     本地工作区准备与验证编排
  faults.py        E_CHAIN / E_REVOKE / E_SNAPSHOT 注入
  demo.py          三场景确定性演示
  crash_worker.py  子进程提交部分修订后 os._exit(90)
  ui.py            Tkinter 诊断界面与稳定文本渲染
  cli.py           命令行入口
tests/
  test_workbench.py
samples/
  demo_profile.json
  trust_revisions.template.json
scripts/
  check_env.py
  run_demo.py
  demo.ps1
```

## 关键实现

### 1. 确定性证书族

证书族参数包括根数、中间层数、常规叶子数、有效期天数、回环端口、随机种子和本地 RSA 位数。所有参数写入 `family.json`。

- 根证书自签，根路径长度约束为常规中间层数。
- 常规链在根下生成指定层数的中间 CA。
- 额外生成一张过期叶子、一张签名被故意翻转的叶子，以及一条比根约束多一层中间 CA 的过长链。
- 每张证书记录 SHA-256 DER 指纹、序列号、有效期、基本约束、路径长度、KeyUsage、父节点与服务端链文件。
- RSA 密钥由种子和证书 ID 经 SHA-256 计数器流确定性产生；同一参数和种子在同一实现中生成相同证书族。
- 这些密钥只用于本地诊断，不应作为生产密钥。

### 2. 真实握手取链，不使用声明的静态链

服务端由 `ssl.SSLContext(PROTOCOL_TLS_SERVER)` 加载叶子和完整链文件并在 `127.0.0.1` 监听。客户端由标准库 `ssl` 发起真实握手。

为取得“同一次握手中服务端实际发送的链”，工作台在客户端与 TLS 服务端之间加入回环 TCP 透传代理：

1. 客户端连接代理，代理连接 TLS 服务端；
2. 代理双向透传原始字节，不终止 TLS；
3. 对服务端到客户端方向增量解析 TLS 1.2 记录和 Handshake 消息；
4. 从 `Certificate` 消息提取 DER 证书序列；
5. 客户端握手返回后取得协商版本、密码套件、压缩、SNI 等参数；
6. 校验捕获链长度、末端锚、每级 DER 指纹与配置的服务端链一致。

握手被固定为 TLS 1.2，使 TLS 1.2 明文 `Certificate` 消息可被精确捕获。若握手失败、未捕获证书、证书链数量不足或末端不是配置锚，均返回 `E_HANDSHAKE`，不进入验证阶段。声明文件只用于检查捕获完整性，不能替代捕获链。

### 3. 信任库修订 JSON

实际格式示例：

```json
{
  "format_version": 1,
  "revisions": [
    {
      "revision": 1,
      "parent_revision": null,
      "trusted_root_fingerprints": ["sha256:<64 hex>"],
      "revoked_serial_numbers": ["0011223344556677"],
      "path_len_overrides": [
        {
          "certificate_fingerprint": "sha256:<64 hex>",
          "path_len": 0
        }
      ],
      "validity_evaluation_epoch_ms": 1735776000000,
      "label": "baseline"
    }
  ]
}
```

加载时整体校验，任一错误即拒绝整个文档并返回 `E_TRUST`，同时给出修订号和 JSON 字段路径：

- 修订号单调连续：首修订可为任意正整数，之后每个修订号必须等于前一修订号加一；
- 首修订父引用为 `null` 或 `0`，后续父引用必须指向更早且存在的修订，并显式做整图无环检查；
- 受信根指纹格式合法且必须是当前证书族中的根；
- 回收序列号为 2 到 64 位十六进制、正数；引用证书族中不存在的序列号返回 `E_REVOKE`；
- 路径长度覆盖项指纹必须属于证书族，`path_len` 为非负整数；
- `validity_evaluation_epoch_ms` 必须为非负整数毫秒且不能是布尔值。

`path_len_overrides` 也接受以指纹为键的对象形式；内部会规范化成稳定排序列表。修订内容摘要把集合型字段排序后计算，因此字典插入顺序和 Python 哈希种子不会改变摘要。

### 4. 离线链验证顺序

对固定快照和固定修订，检查顺序固定如下：

1. 锚：捕获链末端根指纹是否在修订受信根集合内；
2. 签名与基本约束：逐级 issuer 匹配、RSA-SHA256 签名、CA 的 `BasicConstraints`、`keyCertSign`、根自签；
3. 路径长度：从锚向下检查每级 CA 可跟随的中间 CA 数量，修订覆盖项优先于证书自身约束；
4. 有效期：所有证书相对修订评估毫秒的 `notBefore` 与 `notAfter`；
5. 回收：所有证书序列号是否命中修订回收集合。

失败分类只能显式取以下五类之一：

- `UNTRUSTED_ANCHOR`
- `BAD_SIGNATURE`
- `PATH_LEN`
- `EXPIRED`
- `REVOKED`

结果包含完整检查项、首失败层级、首失败检查名、原因、结论和稳定索引。同一快照与同一修订重复验证会得到逐字节一致的结果 JSON 与 `result_digest`。

### 5. 翻转账本

对同一快照按修订号排序后逐个验证，比较相邻修订：

- 结论由 `PASS` 与 `FAIL` 互变时记录 `CONCLUSION` 翻转；
- 结论不变但失败分类变化时单独记录 `CATEGORY` 翻转；
- 翻转列表按 `(from_revision, to_revision, kind)` 稳定排序；
- 每条翻转包含前后结论、前后分类、前后首失败层级和可读原因。

演示修订链产生：

```text
r1 PASS
r2 FAIL UNTRUSTED_ANCHOR   conclusion flip 1->2
r3 FAIL REVOKED            category flip 2->3
r4 FAIL EXPIRED            category flip 3->4
r5 FAIL PATH_LEN           category flip 4->5
r6 PASS                    conclusion flip 5->6
```

### 6. 稳定索引与差异比较

判定索引只由以下规范化内容推导：

```text
chain_fingerprints + revision_digest + failure_category_or_NONE
```

序列化使用 `sort_keys=True`、固定分隔符和 UTF-8，哈希为 SHA-256，带 `idx:` 前缀。索引不包含墙钟时间、单调时钟、进程 ID、字典遍历顺序或 Python 哈希值，因此不同进程和会话中保持一致。

比较两次判定时会先比较结论、分类、修订摘要、快照摘要，再逐检查项比较；发现差异时报告首个层级、判定项、字段路径、期望值和实际值。

```powershell
python -m tlstrust compare --ledger data/demo/ledger.sqlite3 `
  --left-index idx:<left> --right-index idx:<right>
```

### 7. 信任库版本、整体替换与陈旧隔离

- 每次信任库编辑、回收列表变化或回滚都会创建新版本，版本号为 `MAX(version)+1`，只增不减。
- 创建新版本的事务会先把旧版本、旧运行、旧输入快照、旧会话快照和旧 UI 缓存标记为陈旧。
- 当前视图只查询 `current AND stale=0` 的版本与运行；旧结论和翻转账本不参与当前显示和统计。
- 视图先删除当前版本缓存，再写入完整新 JSON，不做增量合并。
- 新版本记录受影响链、旧翻转点和旧失败分类的稳定排序审计。
- 回滚目标不在当前修订链时返回 `E_ROLLBACK`，不创建伪版本；被隔离的旧判定保留为只读，历史会话快照仍可显式重放。

### 8. 强制终止恢复与重放

每次运行先写入 `RUNNING`，每个修订的判定单独事务提交。只有全部修订已提交后才写入翻转和 `COMMITTED` 状态，并生成完整会话快照。

演示中的 `tlstrust.crash_worker` 会提交两个修订后直接 `os._exit(90)`，不执行正常关闭和最终报告。重启后的账本恢复规则：

- 遗留 `RUNNING` 运行标记为 `INCOMPLETE`，并设置 `recovery_required=1`；
- 已提交结果保留；缺失修订插入 `UNKNOWN`、`committed=0`，要求重跑；
- 不为未完成运行生成会话快照，也不声称存在强制终止时的退出报告；
- 已 `COMMITTED` 但缺少会话快照的运行可由已提交结果重建快照。

完整会话快照包含：

- 整数 `format_version`；
- 信任库版本与修订链摘要；
- 输入快照及其摘要、握手链摘要；
- 完整修订内容、逐修订判定、翻转、输入快照副本；
- 覆盖除 `content_digest` 外全部内容的 `content_digest`。

重放时先校验格式版本和内容摘要；再校验当前证书族仍包含快照链；随后对每个修订重新执行离线验证，并逐字段与快照结果及账本已提交结果比较。校验器版本、证书族或判定结果不一致时返回 `E_REPLAY`，其中带具体修订号、层级和字段。

## 常用命令

生成证书族与六修订信任样本：

```powershell
python -m tlstrust generate --out data/sample-family --roots 2 `
  --intermediate-levels 2 --leaves 1 --validity-days 90 `
  --port 0 --key-bits 2048 --trust-out data/sample-family/trust_revisions.json
```

执行真实握手并保存输入快照：

```powershell
python -m tlstrust handshake --family data/sample-family/family.json `
  --leaf leaf-normal-01 --port 0 `
  --snapshot-out data/sample-family/input-normal.json
```

全修订验证并写入账本与会话快照：

```powershell
python -m tlstrust validate --family data/sample-family/family.json `
  --trust data/sample-family/trust_revisions.json `
  --snapshot data/sample-family/input-normal.json `
  --ledger data/sample-family/ledger.sqlite3 `
  --session-dir data/sample-family/sessions
```

指定历史修订回滚并整体替换当前视图：

```powershell
python -m tlstrust rollback --ledger data/sample-family/ledger.sqlite3 --target 3
```

重放当前或指定完整会话快照：

```powershell
python -m tlstrust replay --ledger data/sample-family/ledger.sqlite3 `
  --family data/sample-family/family.json
```

打开诊断界面：

```powershell
python -m tlstrust ui --workdir data/demo
```

## 异常注入

```powershell
python -m tlstrust inject --kind chain    --workdir data/inject-chain
python -m tlstrust inject --kind revoke   --workdir data/inject-revoke
python -m tlstrust inject --kind snapshot --workdir data/inject-snapshot
```

预期结果：

- `chain`：真实握手完成后，捕获链中的一个指纹被注入为错误值；验证输入完整性和 DER 重算发现不一致，返回 `E_CHAIN`，`field_path=snapshot.chain[n].fingerprint`；
- `revoke`：修订回收列表加入语法合法但证书族不存在的序列号，返回 `E_REVOKE`，带修订号和数组字段路径；
- `snapshot`：完整会话快照在未同步内容摘要的情况下被修改，返回 `E_SNAPSHOT`，`field_path=session.content_digest`。

三类注入均写入 `logs/events.jsonl`，记录事件、错误码、修订号、字段路径、层级与细节。

## 测试

```powershell
python -m unittest discover -v
```

本机实际运行结果：

```text
Ran 11 tests in 5.1s
OK
```

覆盖点包括：

- 确定性证书族、链签名、根自签、坏签名、过期叶子和过长链；
- 真实 `ssl` 握手捕获完整 TLSv1.2 链；
- 五类失败及首失败层级；
- 结论翻转与纯分类翻转；
- 跨 `PYTHONHASHSEED` 的索引稳定性；
- 修订号、父引用、根指纹、序列号、评估时间和路径覆盖错误；
- 账本提交、回滚、陈旧快照、`E_ROLLBACK`、`E_REPLAY`、`E_SNAPSHOT`；
- 强制终止后的 `INCOMPLETE`/`UNKNOWN` 恢复；
- 三类异常注入；
- 三场景演示与稳定 UI 文本渲染。

## 默认 2048 位演示的实际结果

本机执行 `python scripts/run_demo.py --workdir /tmp/tlst-acceptance-2048`，得到：

```text
TLSv1.2, ECDHE-RSA-AES256-GCM-SHA384, captured chain length = 4

r1 PASS
r2 FAIL UNTRUSTED_ANCHOR (layer 3)
r3 FAIL REVOKED          (layer 0)
r4 FAIL EXPIRED          (layer 0)
r5 FAIL PATH_LEN         (layer 2)
r6 PASS

flips: 1->2 CONCLUSION, 2->3 CATEGORY,
       3->4 CATEGORY, 4->5 CATEGORY, 5->6 CONCLUSION

rollback target 3 -> new trust version 2
invalid rollback target 99 -> E_ROLLBACK
crash worker exit code 90
recovered run status INCOMPLETE, unknown revision [3]
full rerun replay -> ok=true
stale historical snapshot read-only replay -> ok=true
```

## 错误码与定位

- `E_TRUST`：信任库整体非法；带 `revision` 与 `field_path`。
- `E_HANDSHAKE`：握手失败或未取得完整链；不进入验证。
- `E_CHAIN`：捕获链或快照 DER/指纹/序列号不一致。
- `E_REVOKE`：回收列表引用不存在的序列号。
- `E_SNAPSHOT`：快照格式版本、完整性或内容摘要不合法。
- `E_REPLAY`：重放时证书族或校验口径变化，结果无法逐字节复现；带修订、层级、字段。
- `E_ROLLBACK`：回滚目标不在当前修订链；旧隔离数据仍只读保留。
- `E_PORT`：端口不是 `0` 或 `1024..65535`。
- `E_UI`：Tkinter 或工作区数据不可用。

所有结构化错误都包含 `ok=false`、`code`、`message`、`revision`、`field_path`、`layer` 和 `details`。

## 限制与安全说明

- 为了让标准库 `ssl` 的同一连接可捕获明文证书消息，握手固定为 TLS 1.2；不实现 TLS 1.3 密钥日志解密。
- 证书私钥由确定性种子生成并在本地工作区保存，仅适用于演练和测试。不要用于生产、公网服务或真实凭据。
- 所有监听地址固定为 `127.0.0.1`；端口使用 `0`（系统分配高位临时端口）或显式 `1024..65535`。
- 不使用 Docker、虚拟机、外部账号、在线 CA 或网络目标。
- 本机环境缺少 Tkinter，因此自动化测试验证的是稳定文本渲染器和所有业务闭环；Windows 上安装 Tcl/Tk 后可直接启动 `python -m tlstrust ui`。
