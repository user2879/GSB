# 多源特征交换流水线技术说明

## 1. 环境与目录

- 运行时：本机 Python 3.11（实测 3.11.0rc1）+ NumPy 2.2.6，安装于工作区本地
  虚拟环境 `.venv`，未使用 Docker 或其他容器。
- 任务描述提到在 Windows 验证；当前实际执行环境为 Linux/WSL2。代码仅使用
  跨平台标准库（`pathlib`、`os.replace`、`signal`、`concurrent.futures`），
  不依赖 shell 专有语义，同一命令在 Windows 的 Python 3.11 下可直接运行，
  仅需把脚本前缀 `.venv/bin/python` 替换为 `.venv\\Scripts\\python.exe`。
- 目录职责：
  - `feature_pipeline/`：解析器、互转、预处理、合并、诊断、缓存、快照、并发。
  - `tools/`：样本生成、流水线入口、信号演示、业务码验证。
  - `tests/`：27 个 `unittest` 用例。
  - `samples/`：固定种子 `20261118` 的确定性三版本样本与配置/元数据。
  - `artifacts/`：运行产物、快照与审计清单。

## 2. 三种交换格式

### v1 定宽文本 `samples/features_v1.txt`

- 行宽固定 47：实体 ID 12 列、特征名 24 列、千分比整数 10 列、缺失位 1 列。
- 首行 `FEATV1 width=47` 声明布局；布局变化会触发 `E_LINE_OVERFLOW`。
- 数值域换算：`canonical = permille / 1000`，缺失位 `1` 统一映射为
  `Cell(missing=True, value=NaN)`，原始整行保留在 `Cell.raw_text`。

### v2 JSON `samples/features_v2.json`

- `format_version="v2"`，`entities` 为实体到特征字典的映射。
- 每个值是 `{"value", "precision"}`；显式空值带 `explicit_null=true`，
  缺键由解析器补成缺失单元并以 `raw_text='{"absent_key":true}'` 区分。
- 未知精度标签抛 `E_PRECISION_INVALID`，非有限值抛 `E_NONFINITE_INPUT`。

### v3 二进制分区 `samples/features_v3.bin`

- 小端布局：魔数 `FEATV3` + uint16 格式版本 + 分区数/特征数 + 全局特征名表；
  每分区依次为实体计数/特征计数、实体名表、uint32 偏移表、float32 数据块。
- NaN 槽位表示缺失，原始 4 字节保留在 `Cell.raw_bytes`，分区边界与特征顺序
  保留在 `EntityTable.partition_boundaries/feature_order`。
- 魔数/版本变化、偏移越界、块越界分别以格式错误或
  `E_OFFSET_OUT_OF_RANGE` 拒绝，避免混用旧特征块。

## 3. 统一中间表示与互转

- 统一模型 `feature_pipeline/models.py` 的 `Cell` 与 `EntityTable`，始终保留
  来源版本、原始文本/字节、缺失位、精度标签、分区 ID、填充标记。
- v1→v2：千分比整数精确除以 1000，精度标签取自元数据允许集合。
- v2→v1：四舍五入到千分比，截断残差、原精度标签、来源标记写入旁路
  sidecar JSON；仅在确有残差时登记 `sub_permille_fraction` 降级损失。
- v3 互转：保留分区边界与特征名顺序；float64→float32 的精度损失逐条记录，
  v3→v2/v1 无法表达偏移表/分区头时把 `partition_offsets`、
  `partition_structure` 列入 `downgrade_losses`，不静默丢弃。

## 4. 缺失归一、填充与量纲统一

- 三类缺失（0/1 位、缺键/显式 null、NaN）统一为单一 `missing` 标记。
- 填充在合并前执行，统计量仅使用非缺失值；支持 `zero/mean/median/constant`，
  审计记录策略、依据、填充值、样本量、填充单元数。
- 量纲方式：
  - `raw`：恒等转换。
  - `zscore`：总体标准差；零标准差固定映射为 0，决策写入 `decision`。
  - `percentile`：经验 CDF，支持 `average_rank/min_rank/max_rank` 固定决断；
    边界相等（所有值相同）固定返回第 50 百分位。

## 5. 冲突裁决与合并产物

- 候选按 `source_order` 排序；差值（max-min）不超过 `numeric_tolerance`
  视为一致，取优先级最高来源，理由记为 `within_tolerance/exact_match`。
- 超容差时按 `source_version`、`precision_label`（源顺序兜底）或
  `abs_value`（绝对值最大，源顺序打破平局）裁决。
- 快照内输出：
  - `merged_table.json`：合并行、批次计划、填充审计、转换统计。
  - `conflicts.json`：冲突候选清单与胜出理由、被抑制项、逐实体逐特征来源追踪。
  - `interop_report.json`、`failures.json`、`incremental_audit.json`。

## 6. 跨版本互操作诊断

对同一逻辑特征集分别序列化为 v1/v2/v3 再解析，比较实体数、特征名集合、
缺失位置、数值偏差与精度标签：

- 逐特征偏差直方图（固定桶边界）、均值/最大偏差、最大偏差特征与实体。
- 缺失位置差异清单；精度标签区分“精确还原”和“可预期存储降级”
  （v1 仅 per-mille、v3 仅 float32），后者必须同时出现在降级损失清单。
- 判定字段 `cross_version_consistent` 与 `consistency_judgement`
  （`CONSISTENT/INCONSISTENT`）。样本实测：24 实体、6 特征、最大偏差约
  4.95e-4（v1 千分位量化），在容差 0.02 内，判定 `CONSISTENT`。

## 7. 指纹、增量失效与快照恢复

- 单元缓存键：来源字节/原文指纹 + 解析版本 + 配置指纹
  （`cell_cache_key`）；合并依赖指纹由参与单元键与裁决规则指纹派生
  （`dependency_fingerprint`）。
- `max_workers/batch_size` 属执行参数，不进入数据指纹，调整并发不会失效。
- 实测：
  - 第二次无变化运行：576 个坐标全部 `reused`，0 重算。
  - 替换 v2 中两个单元：仅 4 条记录失效（2 个 cell + 2 个 merge 依赖），
    其余 572 条复用。
  - 白名单移除特征：96 个坐标以 `feature_left_whitelist` 裁剪。
  - 容差/优先级变更：经配置指纹使数据与合并结果全量重算。
- 快照采用“临时目录构建 → `os.replace` 整体替换 → current 指针原子切换”。
  manifest 记录全局版本、配置/元数据哈希、来源指纹、阶段产物版本与产物哈希；
  换入前比较全局版本号，拒绝旧版本回滚。
- SIGTERM/SIGINT 在批次边界被消费，当前批次与 manifest 替换完成后以退出码 0
  结束（见 `tools/signal_demo.py`，测试实测退出码 0）。强制 kill 后重启只信任
  current 指针指向的完整快照；残留 staging 不会被采纳，恢复点信息由
  `--recovery-info` 输出，不承诺生成“退出报告”。

## 8. 并发、异常隔离与业务码

- `ThreadPoolExecutor` 并发解析校验、量纲与合并任务，线程数取
  `merge_config.json:max_workers`。
- 每个任务携带稳定排序键 `(entity_id, stage)`；结果与失败清单按键排序，
  实测 `max_workers=1/2/4/8` 的合并表、冲突、失败 JSON 完全一致。
- 单实体异常被隔离为 `StageFailure(entity_id, stage, code, message, sort_key)`，
  其余实体继续。
- 业务码契约：
  - `E_CONFIG_INVALID`：缺字段、枚举越界、数值非法。
  - `E_PRECISION_INVALID`：精度标签不在元数据允许集合。
  - `E_OFFSET_OUT_OF_RANGE`：v3 偏移/块越界、魔数或版本不符。
  - `E_NONFINITE_INPUT`：inf/-inf/非法浮点输入。
  - `E_LINE_OVERFLOW`：定宽行超长/缩短或声明宽度不符。

## 9. 验证命令

```bash
.venv/bin/python tools/generate_samples.py --out samples
.venv/bin/python tools/run_pipeline.py
.venv/bin/python tools/validate_error_codes.py
.venv/bin/python -m unittest discover -s tests
bash run_demo.sh
```

Windows PowerShell 等价入口：

```powershell
.venv\Scripts\python.exe tools\generate_samples.py --out samples
.venv\Scripts\python.exe tools\run_pipeline.py
.venv\Scripts\python.exe -m unittest discover -s tests
```
