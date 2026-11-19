# 多波段遥感瓦片语义分割增量推理组件 — 技术说明

## 1. 范围与约束

组件只依赖 `numpy`、`Pillow` 与 Python 标准库（测试使用 `pytest`），全部样本由
固定种子的本地生成器产出，不访问网络或真实遥感数据。瓦片为 4 波段 8 位 PNG，
波段顺序固定为红、绿、近红外、高程；像素在进入特征计算前始终保持 `uint8` 整数。

## 2. 模块划分

| 模块 | 职责 |
| --- | --- |
| `rsseg/errors.py` | 业务码（`E_INDEX_INVALID`、`E_TILE_SHAPE_MISMATCH`、`E_TILE_READ_FAILED`、`E_NON_FINITE_BAND`、`E_RECOMPUTE_BUDGET_PENDING`、`E_SNAPSHOT_INVALID`、`E_CONFIG_INVALID`）与失败记录 |
| `rsseg/config.py` | 配置 dataclass、规范化 JSON 指纹、阈值/权重校验 |
| `rsseg/tiles.py` | 索引解析、行列唯一性与邻接对称性校验、整数精度 PNG 加载、波段补丁 |
| `rsseg/features.py` | 确定性滑窗、NDVI/NDWI/边缘密度/高程坡度、加权打分、同分低类别编码优先、置信度 |
| `rsseg/fusion.py` | 融合区域构建、距离权重投票、平票字典序决断、连通域后处理 |
| `rsseg/fingerprint.py` | 波段内容指纹、瓦片缓存键、融合区域指纹、邻接版本、磁盘结果缓存 |
| `rsseg/snapshot.py` | 临时目录写入 + 原子 `os.replace`、manifest、崩溃恢复点判定 |
| `rsseg/engine.py` | 失效分析、预算分批、线程池调度、信号协作、快照提交 |
| `rsseg/injection.py` | 独立故障注入配置（截断 PNG、非有限值、邻接不对称、零预算） |
| `rsseg/edits.py` | 阈值/权重/邻接/波段补丁编辑序列回放 |
| `rsseg/runner.py` / `rsseg/cli.py` | 组装入口与命令行 |

## 3. 确定性设计

1. 特征使用整数分子/分母并按 1000 缩放，避免浮点带来的跨平台差异。
2. 滑窗核心区对瓦片做无重叠、无遗漏的精确划分；上下文仅用于特征计算。
3. `np.argmax` 在同分时天然返回最小类别下标，等价于“类别编码升序决断”。
4. 融合权重为“像素到瓦片最近边界距离 + 1”；同权重时保留行列号字典序较小瓦片。
5. 连通域使用栅格扫描顺序的 BFS 标记，碎区回填取邻接票数最高者，再平票取最小编码。
6. 所有 JSON 指纹使用排序键 + 固定分隔符的规范编码后做 SHA-256。

## 4. 增量与全量的等价性

全量重建等价于“清空工作目录后的一次增量运行”，二者调用完全相同的加载、分类、
融合与后处理函数，因此不存在两套算法。

* 瓦片缓存键 = SHA-256(波段内容指纹 + 索引波段版本 + 配置指纹 + 行列/范围)。
* 融合指纹 = SHA-256(两侧瓦片缓存键 + 融合策略 + 邻接关系版本)。
* 波段补丁只改变目标瓦片键；配置变更改变全部瓦片键；邻接编辑只改变融合指纹。
* 未失效瓦片直接读取 `cache/tiles`，未失效融合直接读取 `cache/fusions`。

## 5. 预算与调度

每个运行先按行列号稳定顺序求失效集合，再按 `recompute_budget` 切分本批次：

* 预算为 0 时所有失效瓦片进入 `pending_tiles` 并写 `E_RECOMPUTE_BUDGET_PENDING`；
* 预算不足时剩余项保持稳定顺序进入下一批次清单，绝不静默丢弃；
* 瓦片失败（坏 PNG / 非有限值）同样进入挂起清单，直到输入修复后才能发布全局栅格；
* 线程池只改变执行并发，结果按提交顺序收集，输出与失败清单和单线程一致。

## 6. 快照与恢复

快照写入 `snapshots/snapshot.tmp.<pid>_<version>`，完整写入 `global_grid.npy`、
`postprocess.json` 与 `manifest.json` 后通过 `os.replace` 原子改名成
`snapshots/v<version>`。恢复时忽略并清理所有临时目录，只承认含完整 manifest 与
栅格文件的最高版本目录，因此强制终止只会回滚到上一个完整快照。

## 7. 复现实验

```bash
python3.11 -m venv .venv
scripts/bootstrap_offline.sh
.venv/bin/python scripts/generate_samples.py --out data/sample_set
.venv/bin/python -m pytest
.venv/bin/python scripts/run_acceptance.py --sample-dir data/sample_set --out data/acceptance
```

验收产物：`data/acceptance/diagnostic_report.json`，包含六项检查与四个注入编号命中。
