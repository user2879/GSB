# 实际验证记录

验证时间：2026-11-26（Asia/Shanghai）。

## 1. 环境与端口

命令与完整输出见 `artifacts/latest-validation/environment.txt`。摘要：

- Python 3.10.12：`/usr/bin/python3.10`。
- 虚拟环境：Python 3.11.0rc1，`.venv/bin/python`。
- numpy 2.4.6。
- pytest 9.2.0。
- sqlite3 3.37.2。
- 回环高位端口示例：`free_port=47005 bindable=True`。

## 2. 自动化测试

命令：

```bash
cd edge_ledger
.venv/bin/pytest -q
```

实际结果：`16` 个测试全部通过，退出码 `0`。覆盖：

- E_SPEC 重复节点、非法回环地址、事件乱序、非法 scale。
- v1/v2/v3 归一、补齐依据、E_FORMAT 行号/字段。
- NumPy 摘要在乱序、多批次和 8 线程下逐位一致。
- SQLite 摘要不符整体替换和只读备份。
- 真实 monotonic 到期、乐观续约冲突。
- E_REPLAY 定位篡改分片。
- 完整真实子进程验收流。

## 3. 端到端验收

命令：

```bash
./scripts/run-acceptance.sh
```

稳定机器可读结果：`artifacts/latest-validation/acceptance-report.json`。

关键实际结果：

```json
{
  "ok": true,
  "adjudication_race": {
    "double_winners_or_none": 0,
    "expiry_wins": 0,
    "quota_wins": 24
  },
  "receipts": {
    "adopted": [
      "receipts_v1.txt:1:adopted:exact_proof_match",
      "receipts_v2.jsonl:1:adopted:exact_proof_match",
      "receipts_v3.jsonl:1:adopted:exact_proof_match",
      "receipts_v3.jsonl:2:adopted:exact_proof_match",
      "receipt_replacement_v2.json:replaces_receipt_id=5:adopted:exact_proof_match"
    ],
    "rejected": [
      "receipt_bad_digest_v3.json:rejected:digest_mismatch"
    ],
    "format_errors": [
      {
        "code": "E_FORMAT",
        "field": "invalid_shard_ref.ref_id",
        "line": 99
      }
    ],
    "replacement": {
      "old_receipt_id": 5,
      "new_receipt_id": 6,
      "ok": true
    }
  },
  "renewal": {
    "new_version": 2,
    "stale_error": {
      "code": "E_LEASE",
      "field": "lease_version"
    }
  },
  "force_kill_recovery_v1": {
    "unknown_shards": ["shard-0000"],
    "rerun_digest_matches": true
  },
  "quota_evolution": {
    "long_lease_state": "quota_revoked",
    "basis": "lease_quota_version 1<2",
    "new_quota_version": 2,
    "stale_counts": {
      "capacity_margins": 42,
      "leases": 33,
      "receipts": 6,
      "shard_results": 4
    }
  }
}
```

`adjudication_race` 中每一对到期回收和配额回收调用都通过同一 SQLite CAS 竞争；
`double_winners_or_none=0` 表示恰好一个赢家。真实到期用例比较
`monotonic_now_ge_expires_at`，不是用线程等待超时冒充过期。

## 4. 独立重放

命令：

```bash
.venv/bin/edge-ledger-verify \
  --spec-dir artifacts/latest-validation/spec-v2 \
  --ledger artifacts/latest-validation/edge.db
```

实际结果保存在 `artifacts/latest-validation/replay-report.json`：

```text
ok=True
event_count=82
adjudications=39
verified_shards=8
unknown_shards=[]
```

`verified_shards=8` 是 quota v1 与 quota v2 两轮各 4 个分片；旧 quota 行已置
`stale=1`，当前界面统计只读取 quota v2 的非 stale 行。

## 5. 强制终止与未知标记

验收脚本发送 `crash_before_reply=true`，节点读完请求后对自身执行 `SIGKILL`。
客户端收到真实 `E_NODE_READ`。恢复事务把已提交授予但无已提交结果的分片写为
`unknown`，释放槽位和字节预留；监管器重新启动节点后重跑摘要成功。

未知时刻的数据库副本：

```text
artifacts/latest-validation/ledger-unknown-state.db
```

对应界面快照包含：

```text
shard-0003 | node-a | unknown | UNKNOWN/RERUN | - | - | recovery
unknown_shards=shard-0003
```

重跑后的最终快照中该行恢复为 `verified`。

## 6. Tkinter 界面运行结果

本机没有 root 密码，未能执行 `apt install python3-tk`；已用 `apt-get download`
把 Ubuntu jammy 系统包解压到项目本地 `.sysroot/`，并通过 WSLg `DISPLAY=:0`
启动真实 Tk 窗口。`scripts/edge-ui.sh` 会自动配置这些路径。

真实 Tk 冒烟证据：

```text
$ cat artifacts/latest-validation/ui-unknown-tk.txt
tk_smoke_exit=0

$ cat artifacts/latest-validation/ui-final-tk.txt
tk_smoke_exit=0

$ ./scripts/edge-ui.sh \
    --spec-dir artifacts/latest-validation/spec-v2 \
    --ledger artifacts/latest-validation/ledger-unknown-state.db \
    --smoke-ms 500
ui_wrapper_status=0
```

稳定文本渲染结果：

- `artifacts/latest-validation/ui-unknown-snapshot.txt`
- `artifacts/latest-validation/ui-final-snapshot.txt`

界面以固定顺序展示节点水位、分片结论、三版回执决策、unknown 标记和账本格式版本；
UI 使用只读 SQLite 连接，刷新不写账本。WSLg 下 Pillow/XGetImage 根窗抓屏返回
`X get_image failed`，因此未提交 PNG 截图；以真实 Tk 进程退出码和文本快照作为
可复核界面运行证据。
