# RTS 资源征收与单位调度验证管线

这是一个纯本地、固定种子、可复算的即时战略事件处理工具。它生成多阵营同一 tick 内提交的资源征收、单位生产与调度事件，按 tick 区间切块，在线程池中并行合成权威状态，再按时间轴归并账本，并对并行/串行结果、独立真值和资源预算做完整核对。

## 能力概览

- 确定性事件流与阵营规则生成；两次生成逐字节一致。
- 逐 tick 因果合成，以 `actor` 字典序与 `local_seq` 构成稳定提交序。
- 每个事件的随机值仅由 `(对局种子, tick, rng_seed_delta)` 的 SHA-256 子流派生。
- tick 区间分块、入口/出口资源与队列快照、块内容哈希与确定性清单。
- 预算感知 `ThreadPoolExecutor`：线程数和内存预算不足时固定优先级降级，完全不足时逐块返回 `E_BUDGET_DENIED`。
- 资源为负、超过上限、队列容量不足或调度欠额时拒绝该事件并记录 `E_RESOURCE_OVERFLOW`。
- 按 `(阵营, tick)` 比较资源与队列哈希，定位首个分歧并附参与事件、衔接快照和降级记录。
- 账本原子提交；退出报告最后写入；重启恢复只读取已提交并通过自哈希/RNG 复算的账本。
- 一条命令生成并验证正常、越界、高并发争用、预算紧张四类样本；完整重跑后逐文件比对字节哈希。

## Windows 环境

要求 Python 3.10 或更高版本，并安装 NumPy：

```bat
py -3.10 -m pip install -r requirements.txt
```

NumPy 用于峰值资源和队列深度的向量化统计；若未安装，代码会自动退化到标准库统计，业务结果和账本哈希不受影响。内容哈希统一使用标准库 SHA-256 与规范 JSON。

## 一键运行

```bat
run_samples.bat
```

或直接执行：

```bat
py -3.10 -m rts_pipeline sample --out samples
py -3.10 -m unittest discover -v
```

`sample` 会在临时目录再完整执行一遍四场景和线程/预算矩阵，并逐文件比较两次运行。正常情况下命令返回 `0`；越界局自身返回业务码 `E_RESOURCE_OVERFLOW`，但样本总报告仍将“已正确检出越界”视为通过。

## 常用命令

```bat
py -3.10 -m rts_pipeline generate --out case01 --seed 20261119 --ticks 36 --factions 4 --chunk-ticks 7
py -3.10 -m rts_pipeline run --workspace case01 --threads 4 --memory-budget 1048576
py -3.10 -m rts_pipeline verify --ledger case01\output\parallel_ledger.json --compare case01\output\serial_ledger.json
py -3.10 -m rts_pipeline recover --workspace case01
```

`run` 的退出码与首个业务结果对应：正常为 `0`，越界为 `23`，预算拒绝为 `24`。账本和验证报告仍会原子落盘。运行器在写入任何账本前会撤下旧的退出/错误标记；退出报告只在新账本、串行账本、预算报告、核对报告和产物哈希全部写入后生成。

## 目录结构

```text
case01/
  input/
    rules.json
    events.jsonl
    game.json
  truth/
    ledger.json
    factions/A.json ...
  chunks/
    manifest.json
    chunk-000000.json ...
  report/
    generation.json
  output/
    parallel_ledger.json
    serial_ledger.json
    scheduler_parallel.json
    scheduler_serial.json
    budget_report.json
    verification.json
    truth_comparison.json
    run_report.json
    artifact_hashes.json
    exit_report.json
```

## 事件与规则

事件每行一个 JSON 对象，字段固定为：

```json
{"tick":0,"actor":"A","action":"collect","target":"resource_field","amount":3,"rng_seed_delta":123,"local_seq":0}
```

`local_seq` 是阵营内全局连续序号，起始为 `0`。合成按 `(tick, actor, local_seq)` 排序；同一 tick 内按 `(actor, local_seq)` 串行提交。块入口快照携带每个阵营的下一个序号，因此块可以独立验证连续性。

规则为每个阵营声明 `resource_cap`、`collect_rate`、`queue_capacity`、`unit_costs` 和 `build_ticks`。征收收益为 `amount * collect_rate + rng_roll % collect_rate`；生产扣除单位成本并写入带剩余建造 tick 的队列；调度只移除已就绪单位。越界事件被拒绝、标记并继续审计，因此可以报告越界总数和首个越界事件。

## 错误码

| 代码 | 含义 |
| --- | --- |
| `E_SEQ_GAP` | 未初始化账本或某阵营 `local_seq` 不连续 |
| `E_RNG_DRIFT` | 随机派生记录无法由种子、tick、delta 复算 |
| `E_RESOURCE_OVERFLOW` | 资源越界、队列容量越界或调度欠额 |
| `E_BUDGET_DENIED` | 内存/线程预算不足以容纳单块 |
| `E_MALFORMED_EVENT` | 事件缺字段、类型错误、重复排序键或 JSON 损坏 |
| `E_LEDGER_HASH` | 账本、tick 哈希、业务哈希或退出报告完整性失败 |
| `E_CHUNK_BOUNDARY` | 块清单哈希、块内容哈希或衔接快照不一致 |
| `E_SCHEMA` / `E_IO` / `E_INTERNAL` | 配置、I/O 或内部不变量失败 |

## 实际验证

当前环境为 Ubuntu 22.04/WSL、Python 3.11.0rc1、NumPy 2.4.6；实现只使用 Windows/POSIX 均支持的 Python API。最近一次完整验证：

- `python3 -m unittest discover -v`：17 项测试全部通过。
- 四类样本全部提交退出报告，串并行及生成期真值全部一致。
- 正常局在 1/2/4/6/8 线程与多内存预算矩阵下 `business_hash` 均为 `101b4fe9ba83401433aa15411296d90446dbc91e40d4d7599153c5c67e5fd664`；不同块长也保持该业务哈希。
- `sample` 内部完整重跑后，117 个场景文件逐字节一致；场景树哈希为 `537d99aedf8c2176d2b094e539f05b7835cedc2bf74641385116ace4afe6965f`。
- 对同一输出目录连续执行两次相同 `sample` 命令，`sample_report.json` 的 SHA-256 均为 `c9b418fbb3d3cd7dd5b024fc8d6bccf5201fa6d642a54f9c9fdeed1eb7d89e79`。
- 最终样本总报告：`artifacts/local_samples/sample_report.json`，报告自哈希 `edb60c956dadabd6906f2e789a81f3d017e168c10f7a5ace71660759e8611f5a`。

更详细的算法、正确性不变量和验证步骤见 [TECHNICAL.md](TECHNICAL.md)。
