# 增量 Embedding 与 Gather 算子组件

面向推荐与检索场景的本地确定性实现：分块 Embedding 表、定点域 Gather
聚合（求和/均值）、行/块内容指纹驱动的精确失效、带版本号的阶段化快照、
编辑命令会话下的增量运行、全局点数预算并行调度、float64 可观测对比与
异常注入。纯本地运行，无容器、无网络、无外部数据集。

## 环境

- Python 3.10+（实际验证：Python 3.11）
- NumPy 1.24+（实际验证：2.4.6），仅用于 float64 参考路径与误差度量
- 安装：`python -m pip install -r requirements.txt`

## 目录结构

```
src/embedgather/     组件库
  config.py          JSON 配置加载与校验（结构字段参与 config_hash）
  fixedpoint.py      定点量化/反量化、定点均值舍入（半进位远离零）
  fingerprints.py    行/块/布局/配置/Gather 结果内容指纹（SHA-256）
  table.py           分块表加载、校验、编辑（换行/换块/追加块）
  gather.py          Gather 算子：定点聚合 + 逐 ID 贡献分解
  scheduler.py       线程池调度、全局点数预算与单请求配额
  snapshot.py        阶段化快照：临时目录写入 + 整体替换提交 + manifest
  edits.py           编辑命令应用与阶段影响判定
  injection.py       异常注入（4 类故障 -> 业务码映射）
  observability.py   float64 参考路径、最大绝对误差、舍入偏差位置
  pipeline.py        阶段编排、恢复校验、增量失效、诊断报告
  signals.py         可处理退出信号（SIGTERM/SIGINT）
scripts/
  gen_data.py        固定种子(204)生成表/请求/编辑序列/配置/注入配置
  run_full.py        全量运行（校验 -> 指纹 -> Gather）
  run_incremental.py 增量编辑会话运行，可导出最终状态
tests/run_tests.py   14 项确定性测试（无需 pytest）
configs/  data/  reports/  snapshots/  final_state/
```

## 快速开始

```bash
python scripts/gen_data.py .                                   # 生成确定性样本
python scripts/run_full.py --config configs/config.json        # 全量运行
python scripts/run_incremental.py --config configs/config.json \
    --edits data/edits.json --final-state final_state          # 增量会话
python scripts/run_full.py --config final_state/config.json \
    --report final_state/report.json                           # 最终态全量重建
python scripts/run_full.py --config configs/config_inj.json \
    --inject configs/injection.json                            # 异常注入运行
python tests/run_tests.py                                      # 测试套件
```

## 设计要点

**定点语义**：`q = round_half_away(x * 2^F)`，裁剪至量化档位区间
`[-levels/2, levels/2 - 1]`；求和按 ID 序列顺序在整数域累加；均值在求和
后按半进位远离零做定点除法。同一输入与档位多次运行逐字段一致。

**指纹与精确失效**：行指纹 = H(维度, 量化档位, 小数位宽, 定点向量)；
块指纹 = H(块序号, 每块行数, 有序行指纹列表)；每个 Gather 结果记录其
引用的行指纹集合与聚合模式。行替换仅使引用该词 ID 的结果失效；整块
替换按块内新旧词 ID 并集失效；追加块只重试此前因 E_ID_MISSING 失败
且缺失 ID 现已存在的请求；切换量化档位/聚合模式只使 Gather 阶段失效，
表校验与指纹阶段不失效（指纹按档位参数化重算，不重新校验数据）。

**阶段化快照**：validate / fingerprint / gather 三阶段产物先写
`snapshots/tmp/` 再 `os.replace` 整体替换提交；manifest 记录全局版本号、
配置哈希、量化档位、块布局指纹与各阶段产物版本。恢复时任一不符即
`E_SNAPSHOT_MISMATCH` 并拒绝按旧产物续跑。

**预算调度**：配额分配与预算扣减在请求序号稳定顺序下于 worker 启动前
完成，执行结束后按请求序退还未用配额；任务携带 `(请求序号, ID 位置)`
稳定排序键。不同并发度下结果顺序、贡献顺序、扣减序列完全一致。超配额
请求以 `E_POINT_BUDGET_EXHAUSTED` 中止并按请求序保留在待算队列。

**退出语义**：SIGTERM/SIGINT 置协作标志，当前阶段产物原子提交并写
`progress.json` 后以退出码 0 退出；被强杀后重启仅依据已完整替换的
快照与进度记录恢复，报告输出恢复点版本号。

## 业务码

| 业务码 | 含义 |
|---|---|
| `E_EMBED_INVALID` | 表记录非法（ID 重复/维度不符/非有限数/块不连续） |
| `E_ID_MISSING` | 请求引用了表内不存在的词 ID（记录请求与位置） |
| `E_SNAPSHOT_MISMATCH` | manifest 版本/配置哈希/档位/布局不符，拒绝续跑 |
| `E_POINT_BUDGET_EXHAUSTED` | 请求累计访问行数超出配额 |

## 实际验证结果（本仓库样本，种子 204）

- `python tests/run_tests.py`：**14/14 通过**（定点舍入、表校验、
  Gather 确定性、缺失 ID、失效精确性、快照恢复与拒绝、增量↔全量等价、
  并发度 1/4 预算序列一致、配额耗尽与待算队列、4 类注入、误差指标、
  报告模式、SIGTERM 优雅退出与恢复、SIGKILL 后重启恢复）。
- 增量会话 6 条编辑命令后，与最终态空白全量重建 **12 个请求 × 12 个
  字段逐字段一致**（定点值、逐 ID 贡献分解、指纹、引用行指纹集合等）。
- 失效清单：`replace_row(1003)->[0,6]`；`replace_block(2)->[0,1,5,7,8,9,11]`；
  `append_block->[10]`（重试原 E_ID_MISSING 请求）；`set_quant_level`/
  `set_agg_mode`->全部且仅 Gather 阶段失效；`replace_row(1026)->[10]`。
- 快照：增量会话全局版本递增至 17；量化档位或块布局变化后重启均输出
  `E_SNAPSHOT_MISMATCH`（退出码 2）并拒绝续跑。
- 注入运行：INJ-001..004 全部命中并分别映射到 4 个业务码，单项故障
  不中断其余请求；float64 参考路径最大绝对误差 ≈ 1.88e-04（< 0.5 ulp
  量级，frac_bits=12）。
