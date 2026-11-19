# 设计与一致性说明

## 1. 运行模型

引擎是单进程、本地文件系统实现，运行时只依赖 Python 标准库。工作目录包含：

```text
<workspace>/
  .backup-engine.lock          # 独占创建的单写者锁
  snapshots/                   # 内容寻址的规范化快照库
    v0000000001-<fingerprint>.json
  restore/                     # 不可变恢复代际
    state-000001-<state-prefix>.json
  checkpoint.json              # 原子提交指针，只指向一个完整恢复代际
```

`checkpoint.json` 是唯一提交点。查询只通过该文件定位恢复代际；在它被原子替换前，任何新恢复文件都不可见。因此进程在任意写入点死亡时，`status` 与 `verify-checkpoint` 仍只能读到旧检查点或完全没有检查点，不会读到半写状态。

## 2. 快照格式

生成器产生的规范字段如下：

```json
{
  "schema_version": 1,
  "version": 2,
  "algorithm": "sha256",
  "previous_version": 1,
  "previous_fingerprint": "<64 hex>",
  "entries": [
    {
      "path": "data/a.txt",
      "digest": "sha256:<64 hex>",
      "length": 8,
      "tampered": false,
      "content_b64": "YWxwaGEtMgo="
    }
  ]
}
```

兼容读取字段别名：`content`（UTF-8 文本）、`content_digest`、`declared_length`、`prev_version`、`prev_fingerprint`。写入和报告始终使用规范字段名。

### 版本与路径

- `version` 必须是大于 0 的 JSON 整数；布尔值不算整数。
- 路径统一转为 `/` 分隔，拒绝绝对路径、盘符、UNC、`.`、`..`、空段、控制字符、Windows 非法字符、保留设备名及结尾空格/点。
- 路径唯一性使用规范化后的 `casefold()` 键，避免 Windows 大小写不敏感文件系统覆盖。

### 条目校验

逐条目处理，单条失败不会中断其余条目：

| 业务码 | 条件 |
|---|---|
| `E_PATH` | 路径非法 |
| `E_DUPLICATE` | 规范化路径重复 |
| `E_ALGORITHM` | 摘要算法不在白名单，或 HMAC 缺少密钥 |
| `E_DIGEST_FORMAT` | 摘要不是 64 位十六进制 |
| `E_LENGTH` | 声明长度不是非负整数，或实际字节长度不一致 |
| `E_TAMPER` | 篡改标记为真、摘要与实际内容不一致、内容缺失/歧义/解码失败 |

每条条目拒绝记录均包含 `version`、零基 `entry_index`、一基 `entry_number`、路径（若可安全确定）和业务码。非法路径无法安全用于恢复，因此不会进入状态；摘要/长度/篡改标记失败的路径会从恢复状态中删除，防止较早版本的旧内容继续暴露。后续合法快照可以重新写回该路径。

## 3. 指纹与链图

快照内容指纹覆盖 `version`、规范化算法、`entries` 和未知扩展字段，**不覆盖** `previous_fingerprint`、`previous_version` 或可选自指纹。这样链指针独立于内容指纹，使自引用和互相引用在密码学上可表达并可被检测。

```text
fingerprint = SHA256(canonical_json(content_payload))
           或 HMAC-SHA256(key, canonical_json(content_payload))
```

规范 JSON 使用 UTF-8、键递归排序、紧凑分隔符。默认白名单为 `hmac-sha256` 与 `sha256`（排序后）；可通过 `--algorithms` 缩小。HMAC 密钥从 `--hmac-key-file` 原样读取，也支持 API 参数或 `BACKUP_ENGINE_HMAC_KEY` 文本环境变量；密钥不写入快照、检查点或报告，检查点只保存密钥身份的 SHA-256。

链图判定完全基于按 `(version, fingerprint)` 排序后的快照集合：

- `E_CHAIN_BREAK`：前驱版本缺失、版本跳号、根不是版本 1，或前驱指纹不在集合中。
- `E_FINGERPRINT`：`previous_version` 存在时，声明的前驱指纹与该版本实际指纹不一致；或可选自指纹不一致；或库文件内容与内容寻址文件名不一致。
- `E_CYCLE`：前驱指纹图存在环，包括自引用和版本自引用。
- `E_FORK`：同一前驱指纹有多个不同后继，或同一版本有多个不同快照。报告保留所有完整分支指纹，不选择任何分支继续生效。

所有指纹相关报告都包含 12 位实际计算摘要前缀（`actual_prefix` 或分支元数据中的 `fingerprint_prefix`）。

## 4. 恢复与回滚

恢复分两阶段：

1. **结构判定**：在排序后的完整导入集合上检测环、断链、指纹不匹配、分叉和回滚。结论只依赖集合内容，不依赖导入顺序。
2. **水位扩展**：从有效检查点的 `watermark` 和 `head_fingerprint` 继续，只接受版本恰好为下一版且前驱匹配的唯一合法快照。若遇到分叉、断链或缺口，只保留共同前缀。

回滚判定只读取持久化检查点水位：

- 显式提交或新导入的 `version < watermark` 快照返回 `E_ROLLBACK` 并列出版本与分支。
- 无参数增量恢复会把检查点中已经应用的旧快照视为历史幂等项，不重复报回滚；显式再次提交同一旧快照则按“收到旧版本”处理并返回 `E_ROLLBACK`。
- 回滚、分叉、断链均不修改已提交状态。

分叉发生时，共同前缀可以提交；所有分支都不生效。若检查点已经位于某个历史分支，后续出现的竞争分支不会导致状态回退，也不会静默改选。

## 5. 检查点与策略重建

恢复代际保存：

- 水位、头指纹、全部已应用快照 `(version, fingerprint)`；
- 最终条目集合（路径、内容、实际长度、声明摘要、来源版本）；
- 当前仍被排除的路径及各业务码；
- 自身 `state_sha256`。

检查点保存恢复文件名、文件摘要、状态摘要、策略和代际号。恢复开始时会完整校验：路径安全、文件摘要、状态摘要、应用集合顺序、水位一致性、条目顺序与内容摘要。

以下情况会丢弃旧基线并从完整导入集合重建：

- 检查点结构版本变化；
- 摘要算法白名单变化；
- HMAC 密钥身份变化；
- 检查点/恢复代际缺失、损坏或校验失败。

重建从空状态开始，不会把旧策略下的条目校验结论与新策略状态混合。重建成功后再原子切换检查点；重建失败也不会暴露半成品。若旧检查点必须重建但集合中没有可生效快照，会提交空水位检查点。

## 6. 原子提交与强制终止

提交顺序：

1. 校验新状态并计算 `state_sha256`。
2. 在 `restore/` 写临时文件，执行 `fsync`，回读校验字节与摘要。
3. 校验通过后 `os.replace` 为不可变代际文件。
4. 写检查点临时文件，校验后 `os.replace` 切换到新代际。

恢复代际从不覆盖；检查点是单文件原子切换。进程在 3/4 之间或临时文件写入中死亡时，旧检查点仍指向完整旧代际。下一次持锁操作会清理具有 `*.tmp-*` 名称的孤儿临时文件；不可变代际不参与清理。锁文件记录 PID 和随机所有者令牌，死进程遗留的锁会被回收，活进程或不可读锁返回 `E_LOCK`。

## 7. 一致性与排序

稳定键如下：

- 快照：`(version, fingerprint)`
- 条目/排除路径：`(casefold(path), path)`
- 业务记录：`(version, entry_index, code, path, fingerprint, message)`
- 分支：完整指纹排序

输出中的导入快照、生效快照、条目、排除路径、业务错误和分支均按上述稳定键排序。JSON 文件和报告使用递归键排序的规范 JSON，因此不依赖到达顺序或 Python 字典插入顺序。

对同一“导入集合 + 检查点基线 + 策略”，恢复计算是纯函数；文件写入只追加不可变代际并原子切换检查点。因此同一批快照无论以何种顺序导入，最终 `effective_snapshots`、`excluded_paths`、`E_ROLLBACK`/`E_FORK` 结论和恢复状态均一致。

## 8. 业务码与退出码

| 业务码 | 含义 |
|---|---|
| `E_VERSION` | 版本号或快照结构版本格式非法 |
| `E_PATH` | 相对路径非法 |
| `E_DUPLICATE` | 条目路径重复 |
| `E_ALGORITHM` | 算法不在白名单或 HMAC 无密钥 |
| `E_DIGEST_FORMAT` | 摘要格式非法 |
| `E_LENGTH` | 声明长度非法或实际长度不一致 |
| `E_TAMPER` | 篡改标记或摘要/内容完整性失败 |
| `E_CHAIN_BREAK` | 指纹链断裂或版本跳号 |
| `E_FINGERPRINT` | 指纹不匹配或内容寻址文件名不匹配 |
| `E_CYCLE` | 自引用或环 |
| `E_ROLLBACK` | 低于持久化水位 |
| `E_FORK` | 同前驱多后继或同版本多快照 |
| `E_LOCK` | 工作区写锁冲突 |
| `E_CHECKPOINT` | 检查点或恢复代际校验失败 |
| `E_CHECKPOINT_REBUILD` | 检查点损坏/结构版本变化，已从快照重建 |
| `E_POLICY_REBUILD` | 算法白名单或 HMAC 身份变化，已从快照重建 |
| `E_JSON` / `E_IO` | 本地 JSON 或文件读写错误 |

命令行退出码：`0` 全部成功（重建警告不算失败），`1` 至少一个业务错误，`2` argparse 参数错误。报告始终为单个 JSON 对象，业务错误位于 `errors`，可恢复的重建事件位于 `warnings`。

## 9. 命令行

```bash
python -m backup_engine import  --workspace WS [--report FILE] SNAP...
python -m backup_engine recover --workspace WS [--report FILE] [SNAP...]
python -m backup_engine status  --workspace WS [--report FILE]
python -m backup_engine verify-checkpoint --workspace WS [--report FILE]
python -m backup_engine samples --output DIR [--report FILE]
```

公共选项：`--algorithms sha256,hmac-sha256`、`--hmac-key-file FILE`。`recover` 不带路径时继续执行已导入集合并从最后检查点增量扩展；带路径时先校验并导入这些快照。显式路径中低于水位的版本会报 `E_ROLLBACK`。
