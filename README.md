# 大规模地形遮挡剔除与流式加载诊断工作台

本项目是一个仅依赖 Python 3.10/3.11、标准库与 NumPy 的确定性诊断工作台。它从固定种子生成本地 `uint16` 高度场，构建四叉树分块清单、低分辨率深度图与 Hi-Z 金字塔，执行视锥/遮挡剔除、流式驻留调度、预算降级、逐帧误差记账、首个分歧定位和业务错误码注入。

工作台不访问网络、不使用 Docker、不需要外部素材。默认支持 `257x257` 与 `1025x1025` 两档地形，所有诊断数值、失败原因和参数版本戳都会写入本地报告。

## 快速开始

Windows 10/11 的 PowerShell 或 CMD：

```bat
python --version
python -m pip install -r requirements.txt
python -m terrain_diag demo --output artifacts\demo --force
```

Linux/WSL 中若没有 `python` 别名，可将 `python` 替换为 `python3`：

```bash
python3 -m pip install -r requirements.txt
python3 -m terrain_diag demo --output artifacts/demo --force
```

`demo` 会一次完成：

- 生成 257 与 1025 两档地形、四叉树清单、相机轨迹与 Hi-Z 金字塔；
- 执行 24 帧剔除、流式加载/卸载、LRU 淘汰和预算降级；
- 把同一实验完整运行两次到 `run_a`、`run_b`；
- 对两个运行目录做逐文件字节比较；
- 注入并核对全部七个业务错误码；
- 输出可核对的帧报告、驻留账本、内容哈希与 `repro_summary.json`。

默认运行约 40 秒，生成不足 100 MB 的诊断数据。程序启动时会创建并校验工作区子目录，同时检查至少 2 GiB 可用磁盘；测试或受限环境可显式传入 `--allow-low-disk`。

## 输出目录

```text
artifacts/demo/
  run_a/
    terrain/                 uint16 高程原始文件
    manifests/               地形元数据与四叉树清单
    hiz/                     每帧深度图金字塔、逐级误差上界与内容哈希
    reports/                 两档地形各自的逐帧 JSONL 报告
    ledgers/                 搬入搬出账本、每块驻留历史、审计结果
    meta/                    工作区布局、分档汇总、整树内容清单
  run_b/                     与 run_a 目录结构和全部文件字节一致
  anomalies/anomaly_report.json
  repro_summary.json         双跑比较、降级覆盖、异常注入汇总
```

报告文件名带规模与种子，例如：

```text
terrain_1025_000000005eed1234.frame_reports.jsonl
terrain_1025_000000005eed1234.residency_ledger.jsonl
terrain_1025_000000005eed1234.residency_history.json
terrain_1025_000000005eed1234.run_summary.json
```

## 常用命令

生成资产但不执行完整轨迹：

```bat
python -m terrain_diag generate --output artifacts\generated
```

单次运行任意一档地形或轨迹变体：

```bat
python -m terrain_diag run --output artifacts\run257 --size 257 --frames 24
python -m terrain_diag run --output artifacts\yaw257 --size 257 --frames 24 --track yaw_offset
```

定位两段轨迹的首个逐块剔除分歧。命令会按帧、再按块 ID 比较状态，不会只比较汇总计数：

```bat
python -m terrain_diag compare ^
  artifacts\run257\reports\terrain_257_000000005eed1234.frame_reports.jsonl ^
  artifacts\yaw257\reports\terrain_257_000000005eed1234.frame_reports.jsonl ^
  --output artifacts\divergence.json
```

单独重放错误码注入：

```bat
python -m terrain_diag anomalies --workspace artifacts\demo\run_a
```

运行测试：

```bat
python -m unittest discover -s tests -v
```

## 数值与可追溯口径

- 世界坐标、观察/投影矩阵使用 `float64`；深度缓冲和 Hi-Z 使用 `float32`；高程输入为 `uint16`。
- 变换顺序固定为 `clip = projection @ view @ [x,y,z,1]`，采用列向量约定。
- Hi-Z 每级对上一级 `2x2` 区域取最大值，并把基础量化误差、`float32` 舍入误差和层级采样误差写入该级元数据。
- 每个块的外扩包围盒同时计入投影浮点误差、屏幕/世界外扩量和 Hi-Z 层级采样误差。差值落入误差区间时状态为 `UNCERTAIN`，该块必定保留下发。
- 每帧深度预渲染还会保存块归属图。若候选块仍拥有任何基分辨率前表面像素，算法必须保留该块；这是防漏可见块的关键保守证据。
- 预算降级按固定顺序执行：降低 Hi-Z 精度（增大层级索引）→ 增大保守外扩 → 减少投机驻留。每项决策记录触发块、参数版本戳和当次误差/预算值。
- 流式调度先完成可见块加载，再执行非可见块 LRU 淘汰；可见工作集本身超出预算时返回 `E_BUDGET_DENIED`，不会静默漏块。
- 确定性 JSON 使用排序键、固定分隔符、浮点 12 位十进制量化，不写入运行时绝对路径、时间戳或墙钟耗时。

详细算法与误差公式见 `docs/DESIGN.md`，本次实际验收结果见 `docs/VALIDATION.md`。

## 业务错误码

| 错误码 | 触发条件 | 注入示例 |
|---|---|---|
| `E_HEIGHT_RANGE` | 高程/形状/字节长度超出 `uint16` 或配置范围 | 输入 70000 的高程 |
| `E_BLOCK_MISSING` | 请求的分块、地形或清单不存在/哈希不符 | 查询不存在的块 ID |
| `E_HIZ_LEVEL` | Hi-Z 输入规格错误或请求非法层级 | 请求第 99 级 |
| `E_DEPTH_BIAS` | 深度偏置非有限或不在 `[0, 0.25]` | 传入负偏置 |
| `E_BUDGET_DENIED` | 可见工作集无法装入内存/显存预算 | 1 字节预算加载一个块 |
| `E_RESIDENT_LEAK` | 驻留集合与账本/字节记账不一致 | 人为增加未记账驻留字节 |
| `E_TRACK_DIVERGE` | 两次轨迹报告存在首个逐块剔除状态差异 | 改写单块状态后比较 |

## 性能说明

确定性产物中刻意不包含墙钟耗时，以保证两次运行可逐字节比较。需要比较帧耗时时，可在关闭后台高负载程序后用外部命令分别计时 `terrain_diag run`。本机参考值：新鲜 1025x1025 资产生成约 0.5 秒，24 帧诊断约 15 秒，完整双跑演示约 38 秒。
