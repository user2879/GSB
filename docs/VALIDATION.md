# 实测验证记录

验证环境：

- Python 3.11.0rc1
- NumPy 2.4.6
- Linux/WSL2；代码仅使用 Windows 10/11 与 Python 3.10/3.11 均支持的 API
- 工作区可用磁盘远高于 2 GiB

执行命令：

```bash
python3 -m unittest discover -s tests -v
python3 -m terrain_diag demo --output artifacts/demo --force
```

## 单元与端到端测试

12 项测试全部通过，覆盖：

- 257/1025 高程生成的确定性与 `uint16` 范围；
- 四叉树清单确定性、样本范围、极端高程和极差；
- Hi-Z 逐级最大降采样、误差单调性、持久化哈希与非法层级；
- 深度偏置错误码；
- 卸载后重载的绘制摘要一致性；
- 可见工作集超预算拒绝；
- 非可见块 LRU 淘汰和账本审计；
- 逐块首分歧定位（两个报告汇总计数相同但状态互换）；
- 全部七个业务错误码注入；
- 两档小规模端到端双跑目录逐字节一致；
- `UNCERTAIN` 必在可见集合中，且误差分量可加和核对。

## 双跑复现结果

本次 `demo` 最终结果：

```text
reproducible: true
run_a/run_b: 全部文件逐字节一致
运行文件数: 116
run_a 总字节: 36,982,270
整树哈希: 21c23c93cec322012f670965e00ef5624626879c47fe740df63701595829aae3
repro_summary SHA-256: ee09129a9b22ec34578f5f697b5993d15827a258bbc463fef77f644ebb544c18
257 报告 SHA-256: 44f50d1cf5b537f8068592fce989869d8543da3be8057032538a040fbc6adf1a
1025 报告 SHA-256: 752367368557ebe5ca49c9797f625e36e0331ca38f768821e1f06fc0e39f21f8
异常注入: 7/7 通过
```

降级覆盖：

```text
LOWER_HIZ:        run_a 69 次, run_b 69 次
EXPAND_BBOX:      run_a 69 次, run_b 69 次
REDUCE_RESIDENT:  run_a 4 次,  run_b 4 次
```

## 各档统计

257x257，24 帧：

- 可见块帧数：632
- 遮挡剔除块帧数：0
- 视锥+遮挡总剔除块帧数：904
- 不确定保留块帧数：0
- 加载 51 次、卸载 92 次、投机预取跳过 0 次
- 最大内存预算使用率：28.48%
- 最大显存预算使用率：36.95%
- 默认预算：内存 524,288 字节，显存 393,216 字节

1025x1025，24 帧：

- 可见块帧数：2086
- 遮挡剔除块帧数：2396
- 视锥+遮挡总剔除块帧数：22,490
- 不确定保留块帧数：122
- 加载 618 次、卸载 1173 次、投机预取跳过 123 次
- 最大内存预算使用率：66.53%
- 最大显存预算使用率：99.79%
- 默认预算：内存 1,164,902 字节，显存 755,630 字节

## 内容资产哈希

```text
257 高程 SHA-256:
55d1d7b92e88f5e4890f4f29914b813a9065c8c89405aeb44a147b646d945183

1025 高程 SHA-256:
bf400c55b622a8cbeffaa14028758641223e27a4340af65b5d4eb58a68e13650

257 四叉树清单 SHA-256:
eeb3d94b492cc503526000d8b3006612e31320b2599735cac5ee8e088e76c22a

1025 四叉树清单 SHA-256:
5ad9ab68e0166f153a92f6039341bf1c476d4601c738c3fe5ac27ba3fcf488f5

257 Hi-Z 清单 SHA-256:
a8d1d7d404c1dab9b52d514f6ce6afdc9a4c122dd55cfe15dc240695b4bc80b8

1025 Hi-Z 清单 SHA-256:
66a8f0c6b5301e41c4a538fe49cb248a88dad89f632fa74a140a07eab93ea82a
```

## 错误码注入摘要

`artifacts/demo/anomalies/anomaly_report.json` 中七项检查均为 `passed: true`：

```text
E_HEIGHT_RANGE    高程序列包含 70000
E_BLOCK_MISSING   查询 B999_99999_99999
E_HIZ_LEVEL       请求第 99 级
E_DEPTH_BIAS      传入 -0.01
E_BUDGET_DENIED   1 字节预算加载一个可见块
E_RESIDENT_LEAK   增加 1 字节未记账驻留
E_TRACK_DIVERGE   同计数但单块状态由 VISIBLE 改为 OCCLUDED
```

## 真实轨迹分歧定位

另用 257 地形、6 帧分别运行 `canonical` 与 `yaw_offset` 轨迹，再执行：

```bash
python -m terrain_diag compare \
  TRACK_A/reports/terrain_257_000000005eed1234.frame_reports.jsonl \
  TRACK_B/reports/terrain_257_000000005eed1234.frame_reports.jsonl \
  --output divergence.json
```

命令返回码为 `2`，分歧记录含 `code: E_TRACK_DIVERGE`，首个分歧定位为：

```text
frame: 0
block: B003_00003_00007
left:  VISIBLE
right: FRUSTUM_CULLED
left error components:  projection_float=2.171393e-06
                        bbox_expand=0.001435374879
                        hiz_sampling=1.549721e-06
                        depth_bias=1e-05
                        total=0.001449095992
right error components: 全部为 0，因为该块未通过右轨迹视锥
```

记录同时保存两侧相机矩阵/视点、块输入包围盒、参数版本戳和 Hi-Z 查找参数。

实际文件位置：

- 逐帧报告：`artifacts/demo/run_a/reports/terrain_{257,1025}_000000005eed1234.frame_reports.jsonl`
- 驻留账本：`artifacts/demo/run_a/ledgers/terrain_{257,1025}_000000005eed1234.residency_ledger.jsonl`
- 驻留历史：`artifacts/demo/run_a/ledgers/terrain_{257,1025}_000000005eed1234.residency_history.json`
- 双跑摘要：`artifacts/demo/repro_summary.json`
- 异常报告：`artifacts/demo/anomalies/anomaly_report.json`
