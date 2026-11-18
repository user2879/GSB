# ML 训练数据管线诊断工作台

面向本地 Windows 环境的确定性 ML 训练数据管线诊断工作台。管线在本地 JSON 中声明
（数据源、顺序处理阶段、整数参数、版本号、上游引用、分片规模），工作台把规格编译为
确定性执行计划，在本地确定性样本上分块流式执行，全部浮点中间量以整数定点表示，
并把规格指纹、逐阶段观测与分片结果写入工作区目录内的 SQLite 账本。

仅依赖 Python 3.10+ 标准库；Tkinter 用于图形界面，NumPy/Tkinter 的可用性通过
`check-env` 子命令检查。不访问网络、不使用容器或 GPU。

## 目录结构

- `workbench/spec.py`：规格与样本加载、强校验（E_SPEC/E_PARAM）、确定性指纹推导
- `workbench/fixedpoint.py`：基于 `decimal.Decimal` 的精确整数定点运算，显式截断方向
- `workbench/canonical.py`：稳定序列化与带域分隔的 SHA-256 组合哈希
- `workbench/engine.py`：执行计划编译、分块流式阶段执行、列统计、分片打包
- `workbench/ledger.py`：SQLite 账本（指纹版本、运行、阶段、分片、事件、进度）
- `workbench/runner.py`：演进分析、失效传播、强制终止恢复、重放校验、报告数据
- `workbench/report.py`：稳定顺序的文本诊断报告
- `workbench/inject.py`：三类异常注入，只改输入文件或参数，不改源码/执行逻辑
- `workbench/demo_data.py`：确定性样本与管线规格生成器
- `workbench/cli.py` / `workbench/ui.py`：命令行入口与 Tkinter 诊断界面
- `tests/test_workbench.py`：12 个验收测试
- `scripts/demo.py`：九场景可重复演示脚本
- `run_demo.bat` / `run_tests.bat` / `launch_ui.bat`：Windows 命令提示符入口

## 定点与确定性语义

- 浮点输入先用 `Decimal(str(value))` 解析，避免宿主机二进制浮点误差。
- 类型规范化阶段把数值统一为整数定点：`int_cell * fixed_scale`，
  `float_cell` 经 `Decimal * fixed_scale` 后按声明方向截断（`toward_zero`）。
- 归一化阶段：`out = trunc(value * scale / divisor)`，截断方向由参数显式声明，
  支持 `toward_zero` 与 `floor`。
- 异常截断按固定整数上下界逐单元格处理；异常比例以基点统计：
  `outlier_bp = floor(10000 * outlier_cells / total_cells)`。
- 快照为规范 JSON-lines（键排序、固定分隔符、LF 换行），输出摘要为带行号与
  域前缀的链式 SHA-256；相同输入多次执行的快照文件逐字节一致。
- 分片指纹绑定行区间、目标规模、余数规则、来源阶段输出摘要与内容哈希；
  行数不能整除时，`keep_partial` 保留余数分片，`drop_partial` 丢弃并记录
  `dropped_rows`。

## 指纹与版本

- 样本内容摘要覆盖声明列与逐行精确单元格（顺序敏感）。
- 阶段指纹 = 哈希(阶段类型、版本、上游引用、规范化参数)。
- 管线指纹 = 哈希(样本摘要, 全部阶段指纹)，与加载顺序、并发无关。
- 指纹首次出现时在 `fingerprints` 表分配单调递增的版本序号，重复规格复用版本。
- 演进时沿上游边做下游闭包推导受影响阶段；未受影响阶段摘要保持可校验不变，
  当前结论整体替换，报告稳定列出变化阶段、旧/新指纹与失效分片数。

## 业务码

| 代码 | 触发条件 | 定位信息 |
| --- | --- | --- |
| `E_SPEC` | 结构/引用/环/类型等规格非法 | 阶段 id + JSON 指针字段路径 |
| `E_PARAM` | 阶段整数参数越界（如 `scale=42`，边界 [0,9]） | 阶段 id + 参数名 |
| `E_DATA` | 样本文件缺失、JSON 损坏、单元格类型不兼容 | 数据源 id + 字段路径 |
| `E_OUTLIER` | 异常基点比例超过 `max_outlier_bp` | 阶段 id + 参数名 |
| `E_SHARD` | 分片内容/指纹与来源阶段摘要绑定不一致 | 分片阶段 + 分片索引 |
| `E_REPLAY` | 恢复/重放与账本记录不一致 | 首个不一致阶段与字段 |

所有事件同时写入 SQLite `events` 表与工作区 `events.jsonl` 结构化日志。

## 强制终止与恢复

- 每个阶段在单个事务内提交阶段记录并推进 `progress`；裁剪阶段先写 `work/`
  临时文件，提交时才原子替换为正式快照，因此被强制终止时不会留下半提交阶段。
- 重启恢复时：已完整提交的阶段先在隔离临时目录重跑并逐项比对摘要/列统计；
  未提交阶段在账本中标记为 `UNKNOWN` 并要求重跑，不会生成虚假退出报告。
- 重放差异定位到首个不一致阶段与字段并报 `E_REPLAY`；分片篡改在重放时报
  `E_SHARD`。

## 使用方式（Windows 命令提示符）

```bat
:: 环境自检（python/numpy/tkinter/sqlite）
python -m workbench.cli check-env

:: 生成确定性夹具（fixtures/ 下的样本与 pipeline.json）
python -c "from workbench.demo_data import generate_fixtures; generate_fixtures('fixtures')"

:: 执行管线并输出诊断报告
python -m workbench.cli run fixtures\pipeline.json --workspace workspace --report report.txt

:: 强制终止模拟（在第 3 个阶段后停止）与恢复
python -m workbench.cli run fixtures\pipeline.json --workspace workspace --kill-after 2
python -m workbench.cli recover fixtures\pipeline.json --workspace workspace

:: 三类异常注入（仅改输入/参数）
python -m workbench.cli inject fixtures\pipeline.json --workspace ws_data --kind data
python -m workbench.cli inject fixtures\pipeline.json --workspace ws_param --kind param
python -m workbench.cli inject fixtures\pipeline.json --workspace ws_outlier --kind outlier

:: 账本重放、无界面诊断导出、Tkinter 界面
python -m workbench.cli replay 1 --workspace workspace
python -m workbench.cli ui-dump --workspace workspace --out ui_model.json
python -m workbench.cli ui --workspace workspace

:: 一键演示与测试
run_demo.bat
run_tests.bat
```

无显示环境时 `ui` 会给出明确提示，可改用 `ui-dump` 获取同一视图模型的 JSON。

## 验证结果

在本机（Python 3.11，标准库 sqlite3）执行：

- `python -m unittest tests.test_workbench -v`：12/12 通过，覆盖规格校验、
  指纹稳定性、阶段与分片逐字节可复现、演进失效与保留、样本演进全链失效、
  三类注入业务码、强制终止 UNKNOWN/恢复、重放首差异定位、分片篡改 E_SHARD。
- `python scripts/demo.py`：九场景全部成功，输出保存于
  `artifacts/demo_output/demo_run.txt`，诊断报告样例见
  `artifacts/demo_output/diagnostic_report.txt`。
- 关键确定性观测（13 行、目标分片 4）：分片分布 `[4,4,4,1]`；归一化 `scale=3`
  下 `visits` 定点和 336000、裁剪后 303000（`clipped_high=3`，`outlier_bp=769`）；
  二次运行的阶段摘要与分片指纹全部相同，指纹版本保持 v1。
