# 运行与验收报告

- 执行日期：2026-11-21（Asia/Shanghai）
- 运行环境：Python 3.11.0rc1、numpy 1.26.4、Pillow 11.0.0、Linux 本地线程池
- 依赖来源：仅本地 `numpy` / `Pillow`，无外部服务、无网络访问、无容器

## 1. 测试执行结果

命令：`python3 -m unittest discover -s tests`

结果：**32 个用例全部通过（OK）**，无跳过、无失败。

| 测试模块 | 用例数 | 覆盖业务要求 |
| --- | --- | --- |
| `tests/test_fixedpoint.py` | 6 | 定点量化、四种舍入、tie 边界、加权混合、uint8 预览 |
| `tests/test_noise.py` | 4 | 坐标+指纹确定性、有界性、频率/指纹敏感性 |
| `tests/test_validation.py` | 7 | E_TEX_DECODE / E_LAYER_INVALID / E_WEIGHT_INVALID / E_CONFIG_INVALID |
| `tests/test_pipeline_e2e.py` | 8 | 逐像素确定性、元数据、舍入/档位真实改像素、双档归因、预算耗尽 |
| `tests/test_scheduler_snapshot.py` | 5 | 稳定排序键、预算 pending、并发=串行、快照恢复与 E_SNAPSHOT_STALE |
| `tests/test_incremental.py` | 2 | 五类编辑序列增量==全量、精确下游失效范围 |

## 2. 定点渲染与确定性

- 量化后全部为 int64 定点整数，层序升序、像素行优先；重复运行层图像、通道
  图像、BLAKE2b 指纹与缓存键逐字段一致（`test_repeated_runs_pixel_identical`）。
- 元数据记录位宽与档位：默认 `frac_bits=8, unit_levels=256, signed_levels=512`，
  并写出 `unit_step/signed_step`。
- 切换舍入模式（half_even -> half_up）与量化档位后，真实像素发生改变
  （`test_round_mode_changes_actual_pixels`、
  `test_quantization_tier_changes_actual_pixels`），而非仅显示变化。
- 边界样本逐像素标注并汇总，例如 32x32 样本报告中
  `mat_slate:diffuse=263`、`mat_bronze:normal=98` 个边界像素。

## 3. 误差传播与归因

默认 8 位宽、容差 2e-3：

- `channel_diff_max=0.001953125`、`noise_diff_max=0.0`，8192 个材质像素全部
  `within_tolerance`，无最终不一致。

粗位宽压力场景（frac_bits=3，对照 4 位，16x16、2 材质 x4 通道）：

- 最终不一致像素：914；
- 归因统计：`quantization_truncation=686`、`rounding_accumulation=228`、
  `within_tolerance=1134`；
- 首次越界环节在 `noise_accumulation` 与 `material_blend` 之间逐像素标注；
- 最大逐像素差值：噪声与通道均为 0.125，与 3 位小数量化分辨率吻合。

## 4. 增量失效与全量一致性

演示编辑序列（同一份持久缓存，32x32）每步都与“从空白工作区按最终输入一次性
全量重建”逐字段相等（`matches_full_rebuild=true`）：

| 编辑 | base | 噪声层 | 通道 | 合成 | 复用 | 重算 |
| --- | --- | --- | --- | --- | --- | --- |
| 纹理替换 | 1 | 3 | 8 | 2 | 4 | 13 |
| 层频率/振幅 | 0 | 1 | 8 | 2 | 6 | 11 |
| 混合模式 | 0 | 0 | 8 | 2 | 7 | 10 |
| 位宽档位 | 0 | 4 | 8 | 2 | 3 | 14 |
| 舍入模式 | 0 | 4 | 8 | 2 | 3 | 14 |

纹理替换仅使被替换的 1 个 base 与引用它的 3 个噪声层失效，未引用层直接复用，
失效范围与全量重建推导的键划分完全一致（`test_incremental.py`）。

## 5. 并发、预算与串行等价

- 任务稳定排序键 `L{layer:06d}-B{batch:06d}`，预算扣减序列按层序稳定。
- 小预算（128 点 / 层，每层 256 点）触发 `E_POINT_BUDGET_PENDING`，待算批次
  进入 pending 并在下一预算轮恢复；并发与单线程参考的预算日志、pending、
  失败码、像素完全一致（`serial_equivalent=true`）。

## 6. 快照原子替换与恢复

- 产物先写 `versions/.stage-*` 临时目录，再原子改名并切换 `current` 符号链接；
  manifest 记录全局版本、配置哈希、位宽、量化档位、批次游标与产物指纹。
- 同配置续跑识别版本（`global_version=1, stale=false`）；切换混合模式改变配置
  哈希后判定 `E_SNAPSHOT_STALE` 并以新版本号重建（v1 -> v2），不逐文件覆盖。

## 7. 复现实验命令

```bash
python3 -m unittest discover -s tests -v
python3 tools/generate_samples.py --out workdata/sample --tiers 16 32 64 --layers 4
python3 -m texmat_pipeline.cli run --input workdata/sample --work workdata/run --resolution 32
python3 tools/run_demo.py --workspace outputs/demo --resolution 32
```

机器可读摘要：`outputs/demo/demo_summary.json`；单轮报告：
`outputs/demo/full_work/report.json`。
