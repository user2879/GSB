# 技术说明：本地确定性视觉表征与检索增量编辑应用

## 1. 范围与约束

- 仅依赖 Python 3.10+ 标准库、NumPy 与 Pillow；不使用 GPU、网络、容器。
- 输入为本地合成 PNG（固定分辨率、浅色背景、固定种子几何图元/纹理块）、
  区域真值 JSON、表征配置 JSON、编辑命令流 JSON。
- 所有中间产物、缓存与快照写入工作区子目录，源样本目录只读。

## 2. 模块架构

```
vrep/
  errors.py       业务错误码（E_IMAGE_FORMAT / E_TRUTH_INVALID / E_CONFIG_INVALID /
                  E_POINT_BUDGET_PENDING / E_NONFINITE_FEATURE / E_QUANT_LEVEL / E_IO）
  config.py       表征配置 dataclass 与白名单校验
  jsonio.py       确定性 JSON（sort_keys、固定分隔符、原子写）
  hashing.py      内容指纹、配置子集指纹、独立确定性 RNG（不触碰全局随机状态）
  loader.py       PNG 容器头/模式/尺寸校验、真值整数矩形校验、失败隔离
  samplegen.py    固定种子合成样本、配置/命令流/注入配置生成、阈值探针
  features.py     行优先切块、6 维统计、固定投影、半向上定点量化、嵌入聚合、余弦相似度
  index.py        阈值决断、边界样本、相似度降序+名字字典序检索
  scheduler.py    单批特征点预算、稳定预订、待办队列与审计
  storage.py      不可变内容寻址 blob、复用账本、版本化快照与 latest 指针
  injection.py    四类故障注入、命中记录与 JSON 诊断报告
  engine.py       三阶段流水线编排、精确失效判定、线程池、信号停止、快照恢复
  cli.py          generate-samples / run / query 命令行
tools/
  run_acceptance.py   一键验收（全量 vs 增量逐字段比对 + 预算 + 注入）
  compare_ledgers.py  两个账本的递归逐字段比对
tests/                  49 个 unittest，按业务要求编号 test_01 ~ test_10
```

## 3. 关键设计与业务要求映射

### 3.1 加载与校验（要求 1）

- PNG 先校验 8 字节签名再解码，强制 RGB，统一尺寸；文件名去扩展名唯一。
- 真值 `box=[x,y,w,h]` 必须为整数（拒绝 bool）、正宽高且完全落在图内。
- 配置使用白名单：量化档位 `{8,16,32}`、定点位宽 `{8,12,16}`、
  分块整除边长、维度/预算/并发为正整数、阈值落在 `[-1,1]`。
- 单图/单真值失败生成结构化 `Failure` 并排序，不中断其余图像处理。

### 3.2 确定性特征与定点量化（要求 2）

- 切块严格行优先 `(row,col)`，序号 `seq` 从 0 起；超出 `max_tiles_per_image`
  的分块按行列号截断并记录 `truncated_seqs`。
- 每块 6 维统计：RGB 均值、梯度强度均值/标准差、纹理密度；
  经 `(feature_dim x 6)` 固定投影矩阵映射，矩阵种子来自
  `feature_config_fingerprint + 维度`，使用独立 `np.random.Generator`。
- 量化采用半向上取整 `floor(x+0.5)`，按 `quant_level`（有符号总位宽）截断，
  逐元素记录舍入残差与位宽；同图同配置多次运行逐元素一致（测试
  `test_repeated_extraction_elementwise_identical`）。

### 3.3 嵌入与检索（要求 3）

- 嵌入矩阵同样由配置指纹派生并定点化；按 `seq` 升序加权整数累加，
  权重相等时由序号决断，最后定点缩回并按档位截断。
- 余弦相似度统一四舍五入到 9 位小数再与阈值比较；
  `similarity >= threshold` 入选，恰等于阈值标注 `boundary=true`。
- 结果按相似度降序、同分按图像名字典序；边界样本取并集并排序，
  全量与增量共用 `index.py` 纯函数，保证逐字段一致。

### 3.4 缓存键、失效判定与复用账本（要求 4）

- 分块键：`tile|内容指纹|特征配置指纹|档位|row|col|seq`
- 特征束键：`feat|内容指纹|特征配置指纹|档位`
- 嵌入键：`emb|特征束键|嵌入配置指纹|档位`
- 索引条目键：`idxentry|嵌入键|索引配置指纹`；索引键 additionally 绑定全部嵌入键集合。
- 配置指纹拆为特征/嵌入/索引三个子集：
  - 仅阈值变更 → 只重建索引，特征与嵌入复用；
  - 嵌入维度变更 → 嵌入与索引失效，特征复用；
  - 分块尺寸/量化档位变更 → 特征、嵌入、索引级联失效；
  - 单图替换/增删 → 仅该图像的特征束及其下游失效。
- 账本 `runtime.reuse/recompute` 逐条记录阶段、图像与缓存键，可供审计。

### 3.5 增量与全量逐字段一致（要求 5）

- 两条路径共用同一引擎与同一套纯函数，内容寻址保证同键同内容。
- 比对字段：配置哈希、量化档位、逐图嵌入向量与缓存键、索引顺序、
  边界样本、失败清单、最终待办；过程事件（复用轨迹、版本号、快照历史）
  不参与最终一致性判定。`tools/compare_ledgers.py` 做递归逐字段 diff。

### 3.6 预算调度与待办保留（要求 6）

- 一个分块特征点消耗 1 预算；按图像名字典序 + `seq` 升序预订，
  历史待办优先于新任务。
- 超预算分块写入持久化“部分特征束”（独立 `partials/` 目录，键含已完成 seq 集合），
  标记 `E_POINT_BUDGET_PENDING`；后续批次从断点续算，完成后删除部分束。
- 预算审计逐条记录 reset/consume/defer，任何批次 consume 累计不超过上限；
  有待办的图像不生成嵌入、不进入索引。

### 3.7 并发与串行一致（要求 7）

- 预算预订在主线程按稳定顺序完成，线程池只承担无共享状态的分块计算与嵌入聚合。
- 每个任务携带 `阶段|图像|r####|c####|s####` 稳定排序键，结果回主线程后排序提交。
- 最大并发取 `worker_count`（上限 32）并按任务数收缩；
  `worker_count=1` 与 `=8` 的最终账本逐字段一致（`test_06_concurrency.py`）。

### 3.8 快照、恢复与异常注入（要求 8）

- 产物先写 `tmp/` 暂存目录，再 `os.rename` 整体替换为 `snapshots/snap-v<N>/`，
  最后原子更新 `latest` 指针；内容寻址 blob 不可变，同键写不同内容直接报错。
- 快照清单含全局版本、配置哈希、量化档位、已完成图像列表、各阶段版本。
- 强制终止后重启只读取 `latest` 指向的完整快照，未完成暂存目录被忽略，
  账本记录 `last_recovery` 恢复点。
- SIGTERM/SIGINT 触发停止事件：完成当前批次提交后以退出码 0 退出
  （子进程测试 `test_10_subprocess_signal.py`）。
- 注入四类真实故障并输出诊断：
  | 注入类型 | 业务码 |
  |---|---|
  | truncate_png | E_IMAGE_FORMAT |
  | truth_out_of_bounds | E_TRUTH_INVALID |
  | nonfinite_feature | E_NONFINITE_FEATURE |
  | quant_level_out_of_range | E_CONFIG_INVALID（诊断期望码 E_QUANT_LEVEL） |

## 4. 运行方式（Windows / Linux / macOS）

```bat
:: 1. 环境准备（仅两个第三方依赖）
python -m pip install -r requirements.txt

:: 2. 生成样本（samples/）
python -m vrep.cli generate-samples --samples samples

:: 3. 全量重建与增量执行
python -m vrep.cli run --samples samples --workspace workspace\full ^
    --config samples\config.json --commands samples\commands.json --mode full
python -m vrep.cli run --samples samples --workspace workspace\incr ^
    --config samples\config.json --commands samples\commands.json --mode incremental

:: 4. 逐字段比对
python tools\compare_ledgers.py workspace\full\store\state\ledger.json ^
    workspace\incr\store\state\ledger.json

:: 5. 一键验收
python tools\run_acceptance.py

:: 6. 全量测试
python run_tests.py
```

## 5. 实际验证结果

- 单元/集成测试：49 个全部通过（约 147 秒），覆盖要求 1-8。
- 一键验收（12 张 96px 图像、11 条编辑命令、预算 24 点/批）：
  - 全量 v20 与增量 v26 最终账本逐字段一致；
  - 边界样本稳定出现并在两条路径完全相同；
  - 预算审计无超支，最终待办为 0；
  - 四类注入 4/4 命中，注入编号与业务码映射写入 `diagnostics.json`。
- 确定性：重复生成样本逐字节一致；并发度 1 与 8 的嵌入、索引、失败清单、
  账本顺序逐字段一致。

## 6. 已知边界

- 注入量化档位越界通过命令流中的配置文件触发（启动配置非法仍按致命错误拒绝），
  以保证其余三类注入可以在同一运行中全部命中。
- 持续失败的图像（如截断 PNG）无法完成特征，预算排空循环在“无进展”批次后停止，
  避免空转；其失败保留在失败清单、不进入索引。
