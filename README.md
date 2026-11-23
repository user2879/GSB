# 多视图三维几何重建工作台（MVRB）

面向**本地确定性样本**的多视图三维几何重建工作台，用于在反复编辑视图、相机参数与重建
配置时，诊断重建结果的**跨版本互操作**与**增量失效边界**。仅依赖 `numpy` 与 `pillow`，
不使用容器、不访问网络或外部数据集。

## 1. 环境准备

### Windows（任务指定环境）
1. 安装 Python 3.11：从 https://www.python.org/downloads/windows/ 下载官方安装包，
   安装时勾选 **Add python.exe to PATH**。
2. 打开 PowerShell 或 CMD 确认：
   ```
   python --version    # Python 3.11.x
   ```
3. 安装依赖（版本以 pip 解析到的 Windows 兼容 wheel 为准）：
   ```
   python -m pip install -r requirements.txt
   ```

### Linux / macOS（本次实际验证环境）
```
python3 --version            # Python 3.11.0rc1（实测）
python3 -m pip install numpy pillow
```
代码仅使用标准库 + numpy + Pillow，路径处理全部基于 `os.path`，Windows/Linux 行为一致。

## 2. 目录结构

```
mvrb/                      工作台核心包
  errors.py                业务错误码、异常与稳定排序失败清单
  utils.py                 稳定哈希、定点换算(ROUND_HALF_UP)、原子 JSON I/O
  config.py / camera.py    重建配置 / 相机内外参与投影
  ir.py                    统一中间表示(IR)与范围/整除/非有限校验
  exchange_v1.py           v1 定宽文本行解析与序列化
  exchange_v2.py           v2 JSON 解析与序列化
  metrics.py               整数度量码 <-> 命名度量 码表
  convert.py               v1<->v2 双向转换与降级旁路(sidecar)
  synthetic.py             固定种子场景/相机/PNG 渲染
  features.py              阈值化 + 连通域特征提取与描述子
  match.py                 跨视图 NCC 匹配，千分比得分
  triangulate.py           DLT 三角化与最小视差校验
  depth.py / voxel.py      深度图量化融合 / 体素级点云合并
  cache.py                 内容指纹缓存（命中即复用，绝不重算）
  scheduler.py             全局预算 + 单视图配额并行调度器
  pipeline.py              五阶段编排与精确失效判定
  snapshot.py              阶段化快照整体替换与 E_SNAPSHOT_STALE 判定
  inject.py                五类真实输入故障注入
  diagnostics.py           跨版本一致性校验与原因码诊断
  workbench.py / cli.py    门面编排 / 命令行入口
scripts/
  generate_sample.py       固定种子样本生成（PNG/相机/配置/v1/v2）
  run_demo.py              端到端验收演示与诊断报告
tests/                     23 个 unittest 用例
samples/demo/              生成的确定性样本
reports/                   诊断报告 JSON 与文本摘要
```

## 3. 快速开始

```
# 生成确定性样本（6 视图、16 个几何图元）
python scripts/generate_sample.py --out samples/demo

# 一键验收：双版本往返 / 增量失效 / 预算 / 快照 / 并发一致 / 异常注入
python scripts/run_demo.py

# 运行全部测试
python -m unittest discover -s tests -v

# 命令行
python -m mvrb.cli run --sample samples/demo --work work/w --report reports/r.json
python -m mvrb.cli convert v2-to-v1 --input samples/demo/result_v2.json \
    --output work/v1.txt --sidecar work/v1.sidecar.json
python -m mvrb.cli convert v1-to-v2 --input work/v1.txt \
    --output work/v2.json --sidecar work/v1.sidecar.json
python -m mvrb.cli diagnose --input samples/demo/result_v2.json --output work/diag.json
```

## 4. 关键设计与业务要求映射

### 4.1 双版本交换与统一 IR（要求 1、2）
- **v1 定宽文本**：`F|/M|/V|` 三类记录，字段零填充、固定宽度；解析时逐行核对总宽度与
  标签，宽度不足或字段错位输出 `E_FIELD_TRUNCATED` 并进入失败清单，不中断其它行。
- **统一 IR 校验**：视图/特征唯一性、坐标落在图像范围内、得分 `[0,1000]`、体素代表点
  可被 `voxel_size` 整除（含定点量化半 LSB 容差）、非有限数 `E_NON_FINITE`。
- **v1→v2**：整数坐标/千分比精确换算为浮点并打精度标签；度量码按 `metrics.py` 码表
  映射命名度量，非法码报 `E_UNKNOWN_METRIC`。
- **v2→v1**：浮点 `ROUND_HALF_UP` 还原整数；截断残差、精度标签、命名度量、舍入模式写入
  `*.sidecar.json`。带旁路再次升版后与原始 v2 **逐字段一致**（测试
  `test_v1_v2_roundtrip_with_sidecar_exact` 断言）。

### 4.2 内容指纹与精确失效（要求 3、4）
- 缓存键 = `阶段名 | 上游内容指纹 | 该阶段实际参数指纹`。参数指纹按阶段拆分
  （特征只依赖 `feature_threshold`、体素只依赖 `voxel_size/fixed_point_bits` 等），
  因此修改体素尺寸不会重算特征。
- 仅噪声变化（图像字节变化但质心不变）只令特征/匹配失效；真正的坐标变化才会向下传播到
  三角化、深度与体素。命中缓存直接读盘复用，不调用任何重算函数。
- 任意编辑序列后的增量结果与“相同最终视图集合+配置”的空白全量重建**逐字段一致**
  （`test_full_rebuild_equals_incremental_edit_sequence`）。

### 4.3 预算并行调度（要求 5、8）
- 预算切分在提交线程池**之前**按视图序号稳定完成，故并发度、线程完成次序、预算耗尽时机
  都不影响最终顺序；输出顺序统一按稳定排序键回归。
- 单视图超配额 → `E_VIEW_BUDGET_EXHAUSTED`；全局预算耗尽 → 未完成视图记
  `E_POINT_BUDGET_EXHAUSTED` 并保留 `todo_queue`，下一批 `resume()` 优先续跑待办。
- `max_workers` 取自配置且实测峰值并发不超过该值；`force_serial=True` 走单线程路径，
  两者结果逐字段一致（`test_serial_equals_parallel_field_by_field`）。

### 4.4 跨版本诊断（要求 6）
输出逐视图坐标偏差直方图、匹配得分最大偏差、度量码还原情况、旁路命中数；当相邻两版本
特征数量或三维点不一致时，区分原因码：
- `R_COORD_ROUNDING` 坐标换算舍入
- `R_METRIC_MAPPING` 度量码映射问题
- `R_SIDECAR_MISS` 旁路元数据未命中

### 4.5 阶段化快照（要求 7）
阶段划分为 `load / features_match / triangulate / depth_voxel`，产物先写
`snapshot/tmp/<stage>` 再 `os.replace` 整体替换。清单记录全局版本号、配置哈希、相机
哈希、交换版本指纹、已完成视图与各阶段产物版本；提交前比对阶段版本，相同则
`unchanged` 而非逐文件覆盖。启动时哈希不一致输出 `E_SNAPSHOT_STALE` 并从受影响阶段
重算（配置变化从 `features_match` 起，相机/交换变化从 `load` 起）。

### 4.6 异常注入（要求 8）
独立注入配置对指定视图/阶段施加**真实输入故障**，注入项编号（`INJ-00x`）写入报告：

| 注入故障 | 真实手法 | 业务码 |
| --- | --- | --- |
| truncated_png | 截断 PNG 字节 | `E_TRUNCATED_PNG` |
| missing_camera | 从 cameras.json 删除视图 | `E_MISSING_CAMERA` |
| disparity_overflow | 抬高最小视差至不可满足 | `E_DISPARITY_OUT_OF_RANGE` |
| non_finite_feature | 交换文件写入 NaN 特征 | `E_NON_FINITE` |
| zero_voxel_size | 配置体素尺寸置 0 | `E_VOXEL_ZERO_SIZE` |

## 5. 实测验证结果

环境：Python 3.11.0rc1、numpy 1.26.4、Pillow 11.0.0（Linux 实测，代码跨平台）。

- `python -m unittest discover -s tests`：**23 passed**。
- `python scripts/run_demo.py`：
  - v2 无损往返 `True`，带旁路精确还原 `True`，匹配得分最大偏差 `0.0`，
    旁路命中 `96/80/16`（feature/match/voxel）。
  - 图像编辑失效 `depth/features/match/triangulate/voxel`；仅改阈值只失效 `features`；
    仅改体素只失效 `voxel`；增量与空白全量逐字段一致 `True`。
  - 全局预算 30 时待办队列 `[3,4,5]`、峰值并发 `3`，正确产生
    `E_POINT_BUDGET_EXHAUSTED`。
  - 快照首跑全 `replaced`、二跑全 `unchanged`、配置变更后 `stale=True` 并从
    `features_match` 重算。
  - 串行/并行逐字段一致 `True`。
  - 五类异常注入全部 `observed=True`，注入编号与业务码一一对应。
- 完整报告：`reports/diagnostic_report.json` 与 `reports/diagnostic_report.txt`。

## 6. 确定性与可复现说明

- 样本生成使用固定种子 `numpy.random.default_rng(seed)`；所有集合输出前按
  `(视图序号, 特征/配对/体素键)` 规范化排序。
- 浮点落盘统一精度（三角化 6 位小数、定点 `2^fixed_point_bits` 量化、
  `ROUND_HALF_UP`），JSON 采用排序键规范化序列化后再哈希，保证跨进程缓存键稳定。
- 失败清单条目携带由 `阶段|视图|对标识|错误码` 派生的稳定排序键，输出顺序与线程
  调度无关。
