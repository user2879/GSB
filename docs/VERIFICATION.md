# 实际验证记录

## 平台

- 执行环境：WSL2 / Ubuntu 22.04，Windows 主机上的 Linux 子系统。
- Python：`3.11.0rc1`（满足 >=3.11）。
- NumPy：`1.26.4`。
- Pillow：`11.0.0`。
- 无 Docker、无 GPU、无网络、无外部账号。
- Windows 兼容性通过纯标准库文件替换、硬链接失败回退复制、相对路径与 PowerShell 脚本保证；本次没有 Windows 原生 Python 可供执行，因此不声称完成 Windows 主机实测。

## 样本

- 固定种子：`734921`。
- 分辨率：`256x256`。
- 3 张 PNG：`scene_alpha.png`, `scene_beta.png`, `scene_gamma.png`。
- 6 个编辑区域，覆盖 warm/cool/vivid/mono、mean/edge_extend/noise_guard、浮点坐标、旋转和羽化。
- 同一逻辑结果生成 v2 JSON、v1 定宽文本及旁路 JSON。

## 自动化测试

```text
$ python3 -m pytest -q
14 passed
```

离线打包与安装形态也已验证：使用本机旧版 pip/setuptools，不加网络依赖构建出
`image_edit_workbench-0.1.0-py3-none-any.whl`，安装到独立 `--target` 目录后，
在空目录完成 `generate-samples`、一次 `run` 和 HTML 报告生成。

CLI 双层转换也已实测：

```text
v2 -> v1 : sidecar_sha256=dce80eb6426bc0e9562e579713ef2106e54b1bdd12aa9fc1edb38bf7435de106
v1 -> v2 : sidecar_hits=6, sidecar_misses=0, logical_equal=true
```

覆盖点：

- v1 固定宽度、码表、统一 IR、三类业务错误。
- v1→v2 字段精确升级。
- v2→v1 残差/精度/旋转/羽化旁路，再升级与 v2 逐字段一致。
- 两次全新冷渲染 PNG 字节、像素、阶段摘要、内容指纹一致。
- 量化档位和羽化半径改变真实阶段数组/像素。
- 单区域编辑只重算 1 个区域；量化配置只重算 style/compose，复用 mask/repair/attribute。
- 像素预算 pending、稳定顺序、下次运行 pending 优先、恢复后续跑。
- 快照、清单字段和旁路哈希。
- 串行与 4 线程结果一致，最大并发不超过配置。
- 命令序列每条后与冷启动全量语义对照。
- v1/v2 坐标直方图、最大强度偏差、修补映射、旁路命中和边界归因。
- 4 个独立注入用例全部命中。
- 坏 v1 行与合法区域隔离，合法区域继续渲染。

## 最终命令序列

```bash
PYTHONPATH=src python3 -m image_edit_workbench run --root . \
  --commands scenarios/incremental_sequence.jsonl --reset-state
```

11 条命令包含：v2 初始渲染、量化档位 16→8、单区域强度修改、单区域矩形修改、修补方式切换、羽化修改、旋转新增、预算收紧至 25000、v2→v1、v1→v2、量化档位恢复 16。

结果：

| 指标 | 结果 |
|---|---|
| 每条命令语义等价 | 11/11 true |
| 像素相等 | 11/11 true |
| 阶段摘要相等 | 11/11 true |
| 内容指纹相等 | 11/11 true |
| pending 顺序相等 | 11/11 true |
| 最终 pending | 空 |
| 全过程最大并发 / 上限 | 4 / 4 |
| 快照提交 | 6 次，间隔 2 条命令 |
| 最终内容指纹 | `52b2cd48b9b9b4c3904f0b6cce209469a19f6d9749da1d904556b84c861f6310` |

最后一次命令只重算 `scene_gamma.png#001` 的 style/compose；其余五个区域从历史内容寻址缓存直接复用。这验证了阶段级失效边界，而不是整图重算。

## 跨版本诊断

| 指标 | 结果 |
|---|---|
| 旁路恢复命中 | 6 |
| 旁路未命中 | 0 |
| 恢复后与 v2 字段一致 | 6/6 |
| 修补方式/码表恢复 | 6/6 |
| 风格强度最大绝对偏差 | 0.0005 |
| 旋转/羽化信息丢失区域 | 5 |
| 边界敏感样本 | 5 |
| 业务码 | `E_BOUNDARY_SENSITIVE` |

边界样本归因覆盖：

- `coordinate_integerization`
- `strength_permille_truncation`
- `dropped_rotation_or_feather`

## 注入

| ID | 期望码 | 结果 |
|---|---|---|
| `INJ-V1-TRUNCATED` | `E_FIELD_TRUNCATED` | 命中 |
| `INJ-V2-STRUCTURE` | `E_STRUCTURE_INVALID` | 命中，5 个合法区域继续通过 |
| `INJ-V2-OUT-OF-RANGE` | `E_REGION_OUT_OF_RANGE` | 命中，5 个合法区域继续通过 |
| `INJ-V1-UNKNOWN-REGION` | `E_REGION_OUT_OF_RANGE` | 命中，5 个合法区域继续通过 |

## 预算事件

最终报告中记录了 5 次 `E_RECOMPUTE_BUDGET_PENDING` 事件。例如：

- `CMD-009-V1`：`scene_gamma.png#001` 需要 8448 像素，剩余 8182，进入 pending。
- `CMD-011-QUANT16`：下一批将旧 pending 标记为 `priority=pending`，预算重新开始为 25000 后完成。

待办最终为空，但历史扣减序列、剩余预算和业务码完整保留。

## 报告

- `report/index.html`：离线可视化入口。
- `report/diagnostics.json`：完整轨迹、预算、失败、注入、跨版本和快照数据。
- `report/diagnostics.json` 的 `final_stage_metadata` 逐区域保留量化档位、定点小数位宽、舍入方式、修补半径/轮次、羽化和合成 alpha 参数。
- `report/preview.png`：源图/增量结果叠加区域框。
- `report/cross_version_preview.png`：v2、v1 与 4 倍差异放大图。
- `report/abnormal/index.html`：独立异常命令序列报告；混合 v1 中 1 条截断记录进入失败清单，6 个合法区域继续完成并保持冷/热一致。

## 关于“复用清单与全量运行完全相同”

原始执行日志不可能同时满足两点：增量运行必须显示真实复用，冷启动全量必须显示真实重算。本报告不伪造该字段，而是分别保留 `incremental_reused`、`incremental_recomputed`、`reference_recomputed` 和 `raw_reuse_lists_equal`，并以像素、阶段摘要、内容指纹、pending 顺序和归一化语义作为等价性判定。所有 11 条命令的这些语义字段均一致。
