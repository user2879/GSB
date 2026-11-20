# 多视图三维场景重建工作台（mvrwb）

面向**本地确定性样本**的多视图三维重建工作台。在反复编辑相机参数、视图内容与
重建配置的过程中维护重建状态，支持阶段级内容指纹的精确失效复用，并在中断与
强制终止后依据持久化快照恢复出与一次性全量重建**逐字段一致**的结果。

- 纯 Python 3.10+（开发与验证环境为 Python 3.11），运行期仅依赖 NumPy 与 Pillow
- 无 GPU、无容器、无网络下载
- 全部随机数来自固定种子（`numpy.random.default_rng(20261121)`），重复生成逐字节一致

## 目录结构

```
mvr_workbench/
├── mvrwb/                  # 工作台包
│   ├── errors.py           # 业务错误码与阶段异常
│   ├── util.py             # 定点量化、规范化 JSON、内容指纹、原子提交、稳定排序键
│   ├── config.py           # 重建配置解析、合法性校验、参数分区指纹
│   ├── camera.py           # 相机模型、视图扫描、PNG/内参/旋转校验、灰度解码
│   ├── sampler.py          # 固定种子样本 + 备选资产 + 配置/编辑流/注入配置生成
│   ├── features.py         # 确定性角点检测、非极大值抑制、单视图配额截断、描述子
│   ├── matching.py         # 互最近邻匹配、对极内点筛选、DLT 三角化、最小视差
│   ├── depth.py            # 投影深度图、量化离散、固定次数确定性平滑
│   ├── cloud.py            # 体素划分、同体素稳定排序取代表点、全局点云
│   ├── budget.py           # 全局点数预算与单视图配额的确定性账本
│   ├── injection.py        # 独立故障注入配置与注入项命中记录
│   ├── snapshot.py         # 阶段临时目录整体替换、清单版本判定与恢复
│   ├── editor.py           # 增量编辑命令流（替换/增删视图、改阈值/体素）
│   ├── pipeline.py         # 线程池编排、精确失效复用、信号处理、诊断报告
│   └── cli.py              # 命令行入口
├── tests/                  # 37 个 pytest 用例（确定性/增量/预算/快照/注入/校验）
├── scripts/
│   ├── make_alt_assets.py  # 备选视图资产（资产已内联进 sampler，保留作独立演示）
│   └── acceptance.py       # 一键验收，产出 acceptance_runs/acceptance_summary.json
├── sample_input/           # 生成的本地样本（视图、配置、编辑流、注入配置）
├── requirements.txt
└── pytest.ini
```

## 快速开始

```bash
# 1) 建立虚拟环境（本机 ensurepip 不可用时用 virtualenv）
python3 -m pip install virtualenv
python3 -m virtualenv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 2) 生成固定种子样本（6 视图 PNG + 相机参数 + 配置 + 编辑流 + 注入配置）
python -m mvrwb.cli generate --output sample_input

# 3) 从空白全量重建，输出 JSON 诊断报告
python -m mvrwb.cli run \
  --input sample_input --work runs/work --snapshot runs/snapshot \
  --report runs/report.json

# 4) 携带增量编辑流（同一快照反复运行即增量）
python -m mvrwb.cli run \
  --input sample_input --work runs/work --snapshot runs/snapshot \
  --report runs/report_inc.json --edits sample_input/edit_stream.json

# 5) 携带故障注入
python -m mvrwb.cli run \
  --input sample_input --work runs/work_inj --snapshot runs/snap_inj \
  --report runs/report_inj.json --injections sample_input/injection_config.json

# 6) 一键验收 / 测试套件
python scripts/acceptance.py
python -m pytest -q
```

## 确定性设计要点

- **定点量化**：所有几何量（坐标、置信度、视差、深度、体素尺寸）经
  round-half-away-from-zero 量化为定点整数再入指纹，跨平台、跨线程稳定。
- **稳定排序**：特征按 `(响应降序, y, x)`；匹配对按视图对标识；三维点按
  `point_key`；点云按体素键空间排序。线程池只做无状态计算，结果在主线程
  按稳定排序键归集，故并发度/调度顺序不影响最终输出顺序。
- **规范化 JSON**：键排序、紧凑分隔符、`allow_nan=False`，内容指纹为
  规范化字节的 SHA-256。

## 缓存键与精确失效

每个阶段产物的 `cache_key` 由**上游产物 cache_key 链 + 参数分区指纹**派生，
并记录 `upstream` 引用集合：

| 阶段 | cache_key 依赖 | 失效触发 |
| --- | --- | --- |
| feature | 图像字节指纹 + 相机指纹 + 特征参数指纹 | 替换图像/相机、改检测阈值/配额/位宽 |
| matching | 两视图 feature cache_key + 相机指纹 + 匹配参数指纹 | 上游特征失效、改内点/视差阈值 |
| depth | 引用到的 matching cache_key + 相机指纹 + 深度参数指纹 | 上游匹配失效、改深度分辨率/量化/平滑 |
| cloud | 全部 depth cache_key + 体素参数指纹 | 上游深度失效、改体素尺寸 |

- 未失效产物**直接复用且不重算**，逐项进入报告 `invalidation.reused`
- 重算产物进入 `invalidation.recomputed`
- 改体素尺寸只让 cloud 失效，depth 必须复用；体素键 `floor(xyz_q / voxel_q)`
  保证体素划分与代表点选取真实随尺寸变化
- 预算截断属于运行时调度结果，放在指纹外的元数据，不污染下游缓存键

## 预算调度

`BudgetLedger` 在主线程按视图序号稳定顺序 `reserve`/`settle`，线程池不参与
预算决策，因此扣减序列与并发度无关、全程可审计。超预算视图输出
`E_POINT_BUDGET_EXHAUSTED`，保留在待处理队列稳定位置，跳过三角化/深度/点云。

## 快照、中断与强制终止

- 各阶段先写 `snapshot/tmp/<stage>.tmp/`，再 `os.replace` 整体替换到
  `snapshot/<stage>/`（POSIX 原子 rename）
- `manifest.json` 记录快照版本号、配置哈希、相机集哈希、已完成视图、
  已消耗预算与各阶段产物版本
- 启动时仅以**快照 schema 版本**判定续跑资格：版本不支持输出
  `E_SNAPSHOT_STALE` 并给出恢复点视图序号，随后全量重建；配置/相机变更
  走参数指纹精确失效而非整体判陈旧
- 可处理退出信号（SIGTERM/SIGINT）：完成当前视图阶段产物落盘后以退出码 0 退出
- 强制终止（测试用 `MVR_CRASH_AFTER_STAGE/VIEW` 触发 `os._exit(137)`）：
  不写清单、不生成退出报告；重启仅依据已完整替换的快照恢复

## 故障注入与业务码映射

独立注入配置 `injection_config.json`，注入项编号随失败记录写入报告：

| 注入类型 | 目标 | 业务码 |
| --- | --- | --- |
| corrupt_png | 视图（load） | E_VIEW_INVALID |
| missing_camera | 视图（load） | E_VIEW_INVALID |
| non_finite_feature | 视图（feature） | E_NON_FINITE_FEATURE |
| parallax_out_of_range | 视图对（matching） | E_PARALLAX_OUT_OF_RANGE |
| 预算耗尽 | 视图（budget） | E_POINT_BUDGET_EXHAUSTED |
| 快照版本陈旧 | 快照（snapshot） | E_SNAPSHOT_STALE |

故障在阶段边界被隔离：失败视图进入 `failures`，其余视图继续处理。

## 诊断报告

每次运行产出 JSON 报告，含：各视图阶段摘要、匹配对摘要、点云点序与指纹、
失效复用/重算清单、预算账本（含扣减序列与队列顺序）、恢复点、失败清单、
注入命中映射与各阶段耗时（毫秒）。
