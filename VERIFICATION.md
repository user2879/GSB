# 实际验证记录

验证日期：2026-11-19。环境：CPython 3.11、NumPy 1.26.4，本地文件系统，无容器、无运行时网络访问。

## 自动测试

```bash
python -m unittest -v tests.test_volrender
```

结果：`Ran 10 tests in 7.719s`，`OK`。

覆盖项：

- 非法 JSON 任务字段返回对应 `E_xxx`，且不创建 state/output 文件。
- 重叠编辑按清单顺序后者覆盖、重复应用逐体素一致。
- 传递函数变化只重渲非空分块，全零分块复用。
- 并发度 1 与 4 的灰度图、深度图逐像素完全相同；采样缓冲峰值不超过预算。
- 局部编辑增量结果与从零全量渲染逐像素相同；旧块写入归档目录。
- 篡改块摘要后检测 `E_DIGEST`，自动补渲并保持最终图像正确。
- 模拟分块边界中断，写中断标记且不生成输出；重启恢复 3 个已完成块，只补渲剩余 6 块。
- 任一分块异常后不再启动待处理块。

## 样本链验证

```bash
python -m volrender generate-samples --samples samples --seed 20261119
python -m volrender verify-samples --samples samples
```

样本：`48 × 48` 输出、`8 × 8` 分块，共 36 块；体数据 `40 × 36 × 32`，固定种子 `20261119`。每一轮增量运行都与同配置的全量渲染比较。

| 任务 | 变化 | 复用 | 重渲 | 逐像素/深度一致 |
| --- | --- | ---: | ---: | --- |
| `00_initial.json` | 首次全量 | 0 | 36 | 是 |
| `01_region_edit.json` | 245 个真实变化体素 | 32 | 4 | 是 |
| `02_transfer_edit.json` | 传递函数变化 | 12 | 24 | 是 |
| `03_camera_edit.json` | 相机变化 | 0 | 36 | 是 |
| `04_combined.json` | 体数据 + 传递函数 + 相机 | 0 | 36 | 是 |

关键逐块结果：

- `01_region_edit`：4 个块被光线命中变化体素并标记 `volume_edit`；32 个块复用。另有 6 个块的体素包围范围与编辑区域相交，但没有任何像素中心光线命中变化体素，全部正确复用，证明没有采用粗包围盒保守失效。
- `02_transfer_edit`：24 个非空块全部重渲并标记 `transfer_function`；12 个全零块全部复用。
- `03_camera_edit`：36 个块全部重渲并标记 `camera`。
- `04_combined`：36 个块均标记 `camera`，其中 33 个非空块同时标记 `transfer_function`，4 个块同时标记 `volume_edit`，报告保留并集原因。
- 五轮 `incremental_vs_full_gray_equal`、`incremental_vs_full_depth_equal` 和 `strict_shapes_equal` 全部为 `true`。

完整逐轮数据与摘要位于 `samples/verification.json`，运行时报告位于 `samples/verification/incremental_out/report.json`。
