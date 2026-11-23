# 实际验证记录

验证日期：2026-11-24（Asia/Shanghai）。验证环境是 WSL2/Linux 开发机，目标运行环境为 Windows 10/11；核心、SQLite、线程调度和 CLI 均在当前环境实测，Tk 前端因当前 Python 缺少 `libtk8.6.so` 无法创建窗口，Windows 桌面端需另行做一次人工界面冒烟测试。

## 环境

```text
Python 3.11.0rc1
numpy 1.26.4
sqlite3 3.37.2
pytest 8.3.3
Linux/WSL2 x86_64
```

## 自动化测试

命令：

```bash
pytest -q
```

结果：

```text
..........................................                               [100%]
46 passed (about 6.3s)
```

覆盖范围：

- 合法声明、字段级 `E_SPEC`、对象键唯一、种子数量和类型、租约时长关系、受支持大小集合。
- 三聚合路径在 64 KiB / 128 KiB 不同批边界下的分片根与对象根一致。
- 路径选择：1 MiB 选 `direct`，32 MiB 边界同分选 `sliding`，256 MiB 选 `hierarchical`。
- 租约重复申请、全局/对象越界、错误版本续约不延长、真实单调时钟过期与配额回收。
- 最终落盘复核跨过租约到期点时，旧批撤销、租约重签、分片从已提交游标重跑并最终成功。
- 全局/对象租约峰值、累计吞吐上限、缺少预算时 `E_BUDGET`、批次字段完整落库。
- 刷新前后账本摘要和分片行完全一致；表格顺序稳定。
- `E_DIGEST` 不提交错误批次，目标分片保留 `unknown`。
- 已完成会话再次启用摘要注入时执行只读完成态探针，仍稳定产生 `E_DIGEST` 且账本摘要不变。
- 独立子进程 `os._exit(137)` 强杀后，inflight 恢复、旧租约回收、游标截断；续跑后的分片文件 SHA-256、分片摘要和对象摘要均与干净执行一致。
- 注入版本冲突与真实磁盘字节损坏都会整体替换，旧账本备份、只读权限和替换分片集合正确。
- 已初始化工作台再次启用快照注入仍会执行替换；`ledger_digest` 被清空也按损坏整体替换。
- 三类 CLI 注入退出码和 JSONL 业务日志。
- 定点截断、64 位饱和、误差界和误差敏感字典序排序。

## 规模样本

命令：

```bash
/usr/bin/time -f 'ELAPSED_SECONDS=%e' \
  python -m ossdiag run --spec samples/demo_spec.json --workdir .demo-scale-work
```

实际输出摘要：

```text
ok=True  ledger_format=1
throughput=303038464/335544320 bytes  peak_leases=4/8
- scale/large-256m.bin: complete path=hierarchical boundary=False completion=1.000000 digest=PASS unknown_shards=0
- scale/medium-32m.bin: complete path=sliding boundary=True completion=1.000000 digest=PASS unknown_shards=0
- scale/tiny-1m.bin: complete path=direct boundary=False completion=1.000000 digest=PASS unknown_shards=0
path_consistency=scale/large-256m.bin:PASS,scale/medium-32m.bin:PASS,scale/tiny-1m.bin:PASS
ELAPSED_SECONDS=23.45
```

关键数字：

| 对象 | 叶/批/分片 | 工作量 direct / sliding / hierarchical | 选择 |
|---|---|---|---|
| tiny 1 MiB | 16 / 1 / 1 | 59 / 118 / 289 | direct |
| medium 32 MiB | 512 / 32 / 16 | 1264 / 1248 / 1248 | sliding（差距 0 < 32，边界） |
| large 256 MiB | 4096 / 256 / 32 | 10576 / 9040 / 7808 | hierarchical |

运行目录约 290 MiB，SQLite 账本约 780 KiB。累计构造字节 `303038464` 未超过吞吐预算 `335544320`；全局租约峰值 `4 <= 8`；对象峰值分别为 `1/1`、`3/3`、`4/4`。三路径对三个对象均给出相同对象摘要。

## 三类异常注入

```bash
python -m ossdiag run --spec samples/small_spec.json --workdir .verify-lease --inject-lease-overflow
python -m ossdiag run --spec samples/small_spec.json --workdir .verify-digest --inject-digest-mismatch
python -m ossdiag run --spec samples/small_spec.json --workdir .verify-snapshot
python -m ossdiag run --spec samples/small_spec.json --workdir .verify-snapshot --inject-snapshot-conflict
```

实际结果：

```text
lease_exit=3
E_LEASE at demo/alpha.bin#2

digest_exit=3
E_DIGEST at demo/alpha.bin#0

snapshot_clean_exit=0
snapshot_fault_exit=3
E_SNAPSHOT at demo/alpha.bin#0; reconcile.replaced=True
```

每个错误 JSONL 记录均含 `monotonic_ms`、`code`、`object_key`、`shard_index`。快照备份文件权限实测为 `0444`，替换集合为 8 个旧分片，按 `(object_key, shard_index)` 稳定排序。

## 强杀与恢复

演示命令先用未修改源码的 CLI 参数强杀：

```bash
python -m ossdiag run --spec samples/small_spec.json --workdir .verify-crash --crash-after-batches 2
```

实际退出码 `137`。重新打开前账本状态：

```text
inflight_before_resume=1
unknown_before_resume=0       # unknown 只能由重启恢复流程依据 inflight 推导
active_leases_before_resume=3
```

随后执行：

```bash
python -m ossdiag run --spec samples/small_spec.json --workdir .verify-crash
```

恢复结果：

```text
resume_exit=0
resume_ok=True
reconcile_replaced=False
recovered_unknown=[('demo/alpha.bin', 1, requires_rerun=True)]
reclaimed_leases=[('demo/alpha.bin', 0, 1), ('demo/alpha.bin', 1, 2), ('demo/beta.bin', 0, 3)]
all_complete=True
path_consistency=[('demo/alpha.bin', True), ('demo/beta.bin', True)]
unknown_after=0
inflight_after=0
```

自动化测试还比较了强杀恢复与干净一次执行的全部 4 个分片摘要和对象摘要，结果完全相同。系统不生成强杀时退出报告。

## Tkinter 说明

当前 WSL Python 导入 Tk 时报：

```text
Tkinter 不可用；Windows 官方 Python 3.11 发行版应自带 Tk。
```

GUI 数据模型与只读刷新已由 `tests/test_gui_model.py` 实测；实际窗口渲染依赖 Windows 桌面 Tk 运行库，未在当前无 Tk 的 Linux 环境中声称通过。Windows 验收命令：

```powershell
python -m ossdiag gui --spec samples/small_spec.json --workdir .ossdiag-work
```

界面对象/分片/快照行均按对象键、分片序号、快照 ID 稳定排序；刷新前后账本摘要和分片行经测试保持不变。
