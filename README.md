# 对象存储分片上传诊断工作台

这是一个面向 Windows 10/11 本地环境的确定性诊断项目。它从本地 JSON 读取对象上传声明，在真实单调时钟、本地线程池、SQLite 和 Tkinter 之上模拟分片租约、批级内容构造、三路径摘要聚合、三重配额调度、账本整体替换和强杀恢复。

所有字节都由本地确定性算法生成，不访问网络，不需要账号、容器或 GPU。规模演示样本为 **289 MiB**，快速验收样本为 **512 KiB**。

## 功能概览

- 严格加载对象键、总字节数、受支持分片大小、分片种子、批大小和租约策略；全量校验失败统一返回 `E_SPEC`，并指出首个非法对象键与字段。
- 租约签发、重复申请登记、版本号乐观续约、真实单调时钟过期与配额回收。
- `direct`（逐分片前沿聚合）、`sliding`（有界滑动窗口）、`hierarchical`（分层 Merkle 树）三条路径共享固定 64 KiB 内容叶和规范 Merkle 根，保证分片摘要与对象摘要逐字节一致。
- 按字节数、分片数和批次数计算整数工作量，自动选择路径；接近阈值时使用稳定优先级并标记边界情形。
- 全局租约配额、对象租约配额、对象并发上限、累计吞吐预算四重约束。
- 每批先写盘、回读校验，再用同一 SQLite 事务写批次、叶摘要和进度游标。
- SQLite 账本携带格式版本、会话声明摘要和整体内容摘要；版本/摘要/磁盘数据不一致时整体替换，旧账本保留为只读备份。
- `inflight` 意图表支持强杀恢复：未提交批次所在分片标记为 `unknown`，旧租约先回收，再从已提交游标重跑。
- 整数定点统计、截断与 64 位饱和规则、单值误差、比较误差界和对象级聚合误差界。
- Tkinter 工作台展示对象/分片/租约/摘要/路径/定点统计/未知分片/快照；刷新只读。
- 命令行参数注入 `E_LEASE`、`E_DIGEST`、`E_SNAPSHOT`，结构化 JSONL 日志包含单调时间、对象键和分片序号。

## Windows 环境准备

在 PowerShell 中执行：

```powershell
winget install Python.Python.3.11
python --version
python -m pip install -r requirements.txt
```

Tkinter 由 Windows 官方 Python 发行版提供，SQLite 使用标准库 `sqlite3`。Tk 界面需要本地桌面会话。运行 289 MiB 演示及数据库/备份时建议至少保留 2 GB 可用空间。

## 快速开始

在项目根目录运行：

```powershell
python scripts/make_samples.py
python -m ossdiag run --spec samples/small_spec.json --workdir .ossdiag-work
```

打开 Tkinter 诊断工作台：

```powershell
python -m ossdiag gui --spec samples/small_spec.json --workdir .ossdiag-work
```

一键演示规模路径分化（约 25 秒，生成约 290 MiB 本地数据）：

```powershell
python scripts/demo.py --large --workdir .demo-scale-work
```

演示样本的默认路径选择：

| 对象 | 大小 | 分片 | 路径 | 说明 |
|---|---:|---:|---|---|
| `scale/tiny-1m.bin` | 1 MiB | 1 | `direct` | 小规模逐分片成本最低 |
| `scale/medium-32m.bin` | 32 MiB | 16 | `sliding` | 与 `hierarchical` 同分，落入 32 单位阈值，稳定优先选滑动窗口 |
| `scale/large-256m.bin` | 256 MiB | 32 | `hierarchical` | 叶节点规模增大后分层聚合明显占优 |

## 上传声明示例

```json
{
  "session": {
    "global_lease_quota": 4,
    "throughput_budget_bytes": 104857600,
    "scale_exponent": 6,
    "tie_threshold_work": 32,
    "worker_threads": 4,
    "batch_size": 1048576
  },
  "objects": [
    {
      "object_key": "demo/object.bin",
      "total_bytes": 8388608,
      "shard_size": 1048576,
      "shard_count": 8,
      "shard_seeds": [1001, 1018, 1035, 1052, 1069, 1086, 1103, 1120],
      "concurrency_limit": 2,
      "batch_size": 1048576,
      "lease_policy": {
        "lease_ms": 2000,
        "renew_interval_ms": 400,
        "max_concurrent_leases": 2
      }
    }
  ]
}
```

`shard_count` 可选；出现时必须等于 `ceil(total_bytes / shard_size)`。`shard_seeds` 必须恰好包含该数量的整数。分片大小支持 `64 KiB` 到 `64 MiB` 的二次幂集合，批大小支持 `64 KiB` 到 `16 MiB` 的二次幂集合。`renew_interval_ms` 必须小于 `lease_ms`。

## 命令行与业务码

```powershell
python -m ossdiag run --spec samples/small_spec.json --workdir .ossdiag-work --inject-lease-overflow
python -m ossdiag run --spec samples/small_spec.json --workdir .ossdiag-work --inject-digest-mismatch
python -m ossdiag run --spec samples/small_spec.json --workdir .ossdiag-work --inject-snapshot-conflict
```

三项注入都只改变命令行参数，不修改源码。业务码：

- `E_LEASE`：租约越界或版本校验失败。
- `E_DIGEST`：批次/分片/对象摘要不符；未提交批次不会写入账本。
- `E_SNAPSHOT`：账本格式版本、内容摘要或磁盘分片不一致；先备份旧账本，再整体替换。
- `E_SPEC`：上传声明不合法，整体拒绝。
- `E_BUDGET`：累计吞吐预算不足，不再构造后续字节。

退出码：`0` 成功；`2` 规格错误；`3` 注入或恢复类业务错误；`5` 未完成；`1` 环境/内部错误；`130` 键盘中断。

指定摘要异常目标：

```powershell
python -m ossdiag run --spec samples/demo_spec.json --workdir .demo-work `
  --inject-digest-mismatch --target-object scale/medium-32m.bin --target-shard 3
```

## 强杀恢复演示

`--crash-after-batches` 仅用于恢复演示：指定的批次写盘后立即 `os._exit(137)`，不会生成退出报告。随后用同一工作目录重新运行即可依据账本恢复：

```powershell
python -m ossdiag run --spec samples/small_spec.json --workdir .recovery-work --crash-after-batches 2
echo $LASTEXITCODE   # 137
python -m ossdiag run --spec samples/small_spec.json --workdir .recovery-work
```

恢复时，`inflight` 中未提交批次所属分片会变为 `unknown`；该分片文件截断到账本游标；旧活跃租约在账本中标记为 `reclaimed_after_crash`；新租约使用更高版本号。恢复后的分片摘要和对象摘要与一次不间断执行一致。

## 输出与目录

默认布局：

```text
.ossdiag-work/
  ledger.sqlite                  # WAL SQLite 账本
  data/objects/<key-hash>/...    # 确定性分片文件
  events.jsonl                   # 结构化事件与业务错误
  run_summary.json               # CLI 运行摘要
  ledger_backups/                # 只读整体备份
```

`python -m ossdiag inspect ...` 会输出仪表盘 JSON，但初始化阶段可能执行恢复或整体替换。Tk 界面的“刷新（只读）”只读取账本与租约内存快照，不改变账本内容。

## 测试

```powershell
python -m pip install pytest
pytest -q
```

测试覆盖三路径/不同批边界摘要一致、租约乐观版本与真实过期、配额与预算、账本整体替换、强杀恢复、三类 CLI 业务码、定点误差排序和 GUI 数据模型只读刷新。实际验证记录见 [`docs/VERIFICATION.md`](docs/VERIFICATION.md)。

## 设计说明

算法、事务协议、恢复语义、定点口径、误差界和成本模型详见 [`docs/DESIGN.md`](docs/DESIGN.md)。项目不包含 Docker、容器、网络访问或外部服务。
