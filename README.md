# 多波段遥感瓦片语义分割增量推理组件

面向本地确定性样本的滑窗语义分割与增量推理实现。输入为固定尺寸的四波段 PNG
瓦片（红、绿、近红外、高程）、瓦片索引 JSON 与分割配置 JSON；输出每瓦片类别图/
置信度/得分图、跨瓦片融合后的全局类别栅格、连通域后处理记录、版本化快照与诊断
报告。全部计算确定性可复现，且任意编辑序列下增量结果与空白工作区全量重建逐像素
一致。

## 环境准备（全程离线）

```bash
python3.11 -m venv .venv
scripts/bootstrap_offline.sh
```

`wheelhouse/` 内置本地缓存装配的 wheel：

- numpy 1.26.4（cp311 manylinux wheel）
- Pillow 13.0.0（由本地已安装包用 `tools/repack_wheel.py` 重打包）
- pytest 9.2.0 及 iniconfig / pluggy / packaging / pygments 依赖

## 生成样本与运行

```bash
.venv/bin/python scripts/generate_samples.py --out data/sample_set
.venv/bin/python -m rsseg.cli --sample-dir data/sample_set --work-dir data/run_inc --mode incremental
.venv/bin/python -m rsseg.cli --sample-dir data/sample_set --work-dir data/run_full --mode full
.venv/bin/python scripts/run_acceptance.py --sample-dir data/sample_set --out data/acceptance
```

样本目录包含 9 张 32x32 瓦片（3x3 网格、8 像素邻接重叠）、`tile_index.json`、
`segmentation_config.json`、`injections.json` 与 `edit_sequence.json`。

## 目录结构

```
rsseg/            算法与引擎实现
scripts/          样本生成、离线引导、验收驱动、信号演示
tests/            42 个 pytest 用例（9 个测试文件）
wheelhouse/       离线依赖 wheel
tools/            wheel 重打包工具（仅用标准库）
docs/design.md    设计与确定性说明
data/             本地样本、运行工作区与验收报告
```

## 业务码

| 代码 | 触发条件 |
| --- | --- |
| `E_INDEX_INVALID` | 索引字段缺失、行列号重复、邻接引用缺失或不对称 |
| `E_TILE_SHAPE_MISMATCH` | 波段数/尺寸不符或整数精度被破坏 |
| `E_TILE_READ_FAILED` | PNG 截断或无法解码（含注入 INJ-001） |
| `E_NON_FINITE_BAND` | 波段补丁出现 NaN/Inf 浮点值（INJ-002） |
| `E_RECOMPUTE_BUDGET_PENDING` | 预算不足，瓦片/融合进入下批次清单（INJ-004） |
| `E_SNAPSHOT_INVALID` / `E_CONFIG_INVALID` | 快照损坏 / 配置非法 |

## 实际验证结果

`pytest`：42 passed（索引校验 6、特征分类 5、融合后处理 7、增量一致性 4、预算
调度 4、指纹与策略 4、注入隔离 5、快照并发 5、信号处理 2）。

`scripts/run_acceptance.py` 输出的 `data/acceptance/diagnostic_report.json`：

- full_rebuild：通过，9 瓦片 + 12 融合区域，全局栅格 80x80 且四类齐全
- edit_sequence_pixel_equality：6 个编辑步骤全部逐像素一致
- budgeted_convergence：预算 2 分批收敛，挂起项均带业务码
- injection_isolation：INJ-001/002/003/004 全部命中并写入报告
- snapshot_recovery：临时崩溃目录被忽略，恢复点只取完整快照
- concurrency_equivalence：4 线程与单线程输出顺序、失败清单、栅格完全一致

## 关键设计

详见 `docs/design.md`。核心保证：全量重建就是清空缓存后的增量运行，二者共用同一
套确定性函数；依赖指纹精确覆盖波段内容、索引波段版本、配置、邻接关系与融合策略，
未失效项只读缓存绝不重算。
