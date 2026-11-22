# 顺序与事务消息账本（Windows 本地版）

这是一个只依赖 Python 标准库的本地事务消息领域组件。它从 JSON 加载分区与事务批次，把事务状态事件和逐条消息可见性记录写入 SQLite，并按分区位点推导连续可消费前缀。实现重点不是消息传输，而是 **状态推进、强杀恢复、快照整体替换和可验证的恢复结论**。

## 环境与快速开始

要求 Windows 本地 Python 3.10 或更高版本。运行时不访问网络，也不需要第三方包。

```powershell
cd <项目目录>
python --version
python -m unittest discover -s tests -v
python demo.py --workdir demo_workspace
python -m txnmsg view --workdir demo_workspace
python -m txnmsg ui --workdir demo_workspace
```

演示脚本每次开始都会清空 `ledger.sqlite`、快照、备份、日志和样本目录，因此重复运行得到同一事件序列与同一前缀结论。所有样本、账本、快照和日志都写入 `--workdir` 指定的目录。

若当前 Python 没有可加载的 Tk 运行库，`ui` 不会篡改账本，而是输出稳定只读诊断 JSON 作为降级结果。Windows 官方 Python 安装通常自带 Tk，可直接打开诊断窗口。

## 输入格式

分区声明：

```json
{
  "format_version": 1,
  "partitions": [
    {"partition_id": "orders", "partition_version": 3, "max_position": 20},
    {"partition_id": "payments", "partition_version": 1, "max_position": 10}
  ]
}
```

事务批次：

```json
{
  "format_version": 1,
  "transactions": [
    {
      "transaction_id": "tx-order-001",
      "txn_seq": 1,
      "partition_id": "orders",
      "local_txn_version": 1,
      "commit_time_ms": 1700000000001,
      "messages": [
        {"position": 0, "payload": {"order_id": "A-100", "amount": 10}},
        {"position": 1, "payload": {"order_id": "A-101", "amount": 11}}
      ]
    }
  ]
}
```

约束：

- 分区标识唯一；分区版本号与 `max_position` 必须是非负整数。
- `transaction_id` 唯一；`txn_seq` 为正整数，并在批次顺序及已加载账本中严格递增。
- 目标分区必须存在；`local_txn_version` 为正整数；`commit_time_ms` 为非负整数。
- 同一事务内位点严格递增；整个分区内不同事务也不得复用位点，否则为 `E_ORDER`。
- 位点从 `0` 开始，允许范围是 `0..max_position`。可消费前缀为包含式最大位点；没有可消费消息时为 `-1`。
- 消息载荷可以是任意可规范化 JSON 值。可选提供 `message_digest`，若提供则必须与 SHA-256 摘要一致。

任一批次不合法会在写入账本前整体拒绝：`E_SPEC` 指向首个非法事务标识和字段；同一分区重复位点使用 `E_ORDER`。

## 状态与可消费前缀

允许迁移：

```text
未开始 -> prepared -> committing -> committed
                    \-> aborted
          prepared -----------------> aborted
```

首次事件必须是 `prepared`。每次推进分配连续的全局 `event_seq`；事件记录包含事务标识、状态、本地事务版本号和分区。`committed` 事件与全部消息可见性记录在同一 SQLite 事务中写入。

前缀推导规则：

- 从位点 `0` 开始检查；中间缺失位点以 `missing` 截断。
- 任一 `prepared`、`committing` 或 `aborted` 事务都会阻断后续可见性。
- `committing` 在重启后作为未知事务单列，其位点区间不得计入前缀。
- 结论使用规范化 JSON 和稳定排序；同一事件序列不因批量推进边界或回放时的输入行顺序而改变字节结果。

## 包接口

```python
from txnmsg import (
    Ledger,
    create_snapshot,
    load_and_validate,
    replay_verify,
    restore_snapshot,
)

partitions, transactions = load_and_validate("partitions.json", "transactions.json")

with Ledger("ledger.sqlite") as ledger:
    ledger.initialise_partitions(partitions)
    ledger.ingest(transactions)
    ledger.advance_state("tx-order-001", "prepared")
    ledger.advance_state("tx-order-001", "committing")
    ledger.advance_state("tx-order-001", "committed")
    prefixes = ledger.query_prefixes()
    conclusion = ledger.conclusion()

    snapshot = create_snapshot(ledger, "snapshots/snapshot-0001.json")
    restore = restore_snapshot(ledger, snapshot.path)
    replay = replay_verify(ledger, snapshot.path)
```

`Ledger` 暴露事务、事件、可见性、前缀、恢复报告和整体结论；`create_snapshot`、`restore_snapshot`、`replay_verify` 以及只读 `build_diagnostics` 都可作为库接口集成。

## 命令行

```powershell
python -m txnmsg demo --workdir demo_workspace
python -m txnmsg view --workdir demo_workspace --json
python -m txnmsg ui --workdir demo_workspace
python -m txnmsg validate --partitions parts.json --transactions batch.json
python -m txnmsg snapshot --workdir demo_workspace
python -m txnmsg restore --workdir demo_workspace --snapshot snapshots/snapshot-0001.json
python -m txnmsg replay --workdir demo_workspace --snapshot snapshots/snapshot-0001.json
```

异常注入只创建输入文件或复制并篡改账本，不改生产源码。先运行过一次 `demo` 以生成样本：

```powershell
python -m txnmsg inject --workdir demo_workspace --kind corrupt-batch
python -m txnmsg inject --workdir demo_workspace --kind duplicate-position
python -m txnmsg inject --workdir demo_workspace --kind ledger-digest
python -m txnmsg inject --workdir demo_workspace --kind snapshot-digest
```

业务码与退出码：

| 业务码 | 退出码 | 含义 |
|---|---:|---|
| `E_SPEC` | 2 | 输入契约、状态迁移或格式非法 |
| `E_REC` | 3 | SQLite 记录长度、摘要、字段或原子状态不一致 |
| `E_ORDER` | 4 | 同一分区位点重复或顺序冲突 |
| `E_SNAPSHOT` | 5 | 账本/快照格式版本、内容摘要或完整性不一致 |
| `E_REPLAY` | 6 | 恢复后重放与快照/物化结论不一致 |

所有业务错误都会追加到 `<workdir>/logs/business.jsonl`，每条包含 UTC 时间、业务码、事件序号、分区、事务、字段和细节。

## 强杀恢复与快照

状态事件和可见性记录在 SQLite 单事务内原子提交；分区前缀是单独刷新的一致性缓存。因此存在真实的“事件已提交、前缀缓存尚未更新”窗口。强制终止后：

1. 打开账本时先校验记录长度、摘要、事件连续性、状态迁移和账本内容摘要。
2. 从已完整提交的事件与消息重建状态机、前缀和未知事务，不信任旧前缀缓存。
3. 最新状态为 `committing` 的事务写入未知标记，报告分区和悬置位点区间，并要求重跑。
4. 恢复报告明确 `exit_report_available: false`，不伪造强杀瞬间的退出报告。

快照包含整数格式版本、分区集合版本、事务集合版本、事件游标、全部事件/可见性/事务目录、各分区前缀和双层摘要。恢复时先验证快照自身内容摘要和其中账本摘要，再把旧账本备份为只读 SQLite，最后在一个 SQLite 事务中整体删除并替换状态，绝不增量合并。差异按稳定顺序列出新增、删除、版本变化和事务最终状态变化。

`replay_verify` 从事件序号 1 重放至游标，比较每个事务最终状态、每分区前缀、截断位点和未知事务。首个差异定位到事件序号、事务标识和字段并报 `E_REPLAY`；未知事务单列且不计入复现失败。

更细的持久化不变量与崩溃窗口说明见 [docs/DESIGN.md](docs/DESIGN.md)。

## 实际验证结果

验证环境：Python 3.11，SQLite 内置模块，无第三方依赖；当前容器的 Python 缺少 `libtk8.6.so`，因此 Tk 窗口本身未在该容器显示，诊断模型和只读刷新路径已验证，Windows 自带 Tk 的 Python 可直接运行。

```text
python -m unittest discover -s tests -v
Ran 28 tests
OK
```

`python demo.py --workdir demo_workspace` 的结果为 `PASS`，关键证据：

- 正常提交后 `orders` 前缀推进；中止事务使后续位点保持不可消费。
- 两次真实子进程以退出码 `74` 强杀，均在事件提交后、前缀缓存刷新前终止。
- 第一次恢复修复陈旧前缀；第二次将 `tx-order-003` 标为未知，悬置区间为 `[4,5]`，报告不提供退出报告。
- 未知事务重跑后前缀恢复推进；最终快照事件游标为 `17`，重放 17 个事件为 `PASS`。
- 快照后继续提交 `tx-pay-003` 使摘要和版本变化；恢复快照后该事务被整体删除、游标回到 17，旧账本备份权限为只读。
- 三类注入分别得到 `E_SPEC`、`E_ORDER`、`E_SNAPSHOT`，结构化日志包含 `timestamp`、`code`、`event_seq`、`partition_id`。
- 重复构建诊断视图不会改变账本摘要或渲染顺序。

完整机器可读结果位于 `demo_workspace/demo-summary.json`；验收摘要另见 [artifacts/verification-results.json](artifacts/verification-results.json)。

## 目录

```text
txnmsg/          领域包、SQLite 账本、快照、重放、诊断、Tk UI、CLI
tests/           28 个标准库 unittest 用例
demo.py          确定性演示入口
docs/DESIGN.md   持久化与一致性设计说明
artifacts/       实际运行与验收记录
```
