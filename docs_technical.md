# AED 流式边界切分与事件轨迹重建 — 技术说明

## 1. 目标与范围

应用把按固定分片长度独立检测得到的 AED 帧级概率包重建成全局事件轨迹，重点解决三类问题：

1. 跨分片边界处同一真实事件被切开、概率不连续；
2. 分片头部的帧网格版本、类别码表版本与当前配置不一致；
3. 长流程中断后只能依据完整提交的快照恢复，且结果必须与不中断运行逐字段一致。

只依赖本地确定性合成 WAV 与 JSON/YAML 数据，不访问网络或外部服务，不使用容器。

## 2. 目录结构

```
aedflow/
  config.py      当前配置、码表/网格版本定义、配置指纹
  errors.py      固定业务码目录与失败记录构造
  generator.py   固定种子 WAV 与分片结果包生成
  loader.py      分片校验、语义码表映射、外网格重采样、全局网格对齐
  boundary.py    边界连续性判决、逐帧类别判决、线程池并发
  trajectory.py  边界概率平滑、RLE 聚合、跨边界合并、轨迹过滤
  matching.py    轨迹-真值匹配与确定性冲突决断
  snapshot.py    阶段快照临时目录 + 整体原子替换、身份校验
  pipeline.py    四阶段编排、可处理退出、恢复
  report.py      固定字段顺序 JSON 诊断报告、原子写盘
  cli.py         generate / run / drill / stale-drill 命令
config/          当前配置与注入样例
samples/         样本清单、全局真值、WAV、分片结果包
tests/           14 个 unittest 用例
output/          快照、诊断报告、演示输出
```

## 3. 确定性样本

- WAV：16 kHz、单声道、16-bit PCM，时长 4000 ms，64000 个采样点。
- 帧网格（当前 `frame_grid_v1`）：帧长 20 ms、帧移 10 ms，共 401 帧。
- 类别固定为 `speech`、`music`、`noise`、`alarm`，内部规范顺序按字母序
  `alarm/music/noise/speech`。
- 生成器种子写在 `samples/sample_manifest.yaml`（20261118）。每个分片在全局基线
  概率上叠加由 `(种子, 分片序号, 分片标识长度)` 派生的确定性扰动，制造边界跳变；
  重新生成与既有文件字节级一致（测试已验证）。
- 样本清单给定 5 个有效分片（800 ms 一段）与 3 个异常分片；切分边界位置只由清单决定。

### 头部版本构造

| 分片 | 码表版本 | 网格版本 | 用途 |
| --- | --- | --- | --- |
| shard_01 | codebook_v2 | frame_grid_v1 | 当前基线 |
| shard_02 | codebook_v1 | frame_grid_v1 | 码位不同、语义名相同，验证语义映射 |
| shard_03 | codebook_v2（含 50=dog_bark） | frame_grid_v1 | 无法映射编码，验证 E_CLASS_MAP_UNMAPPED |
| shard_04 | codebook_v2 | frame_grid_v2（帧移 8 ms） | 外网格重采样 |
| shard_05 | codebook_v2 | frame_grid_v1 | 基线 |
| shard_bad_align | v2 | v1 | 起止 125/525 ms 不整除帧移 |
| shard_bad_prob | v2 | v1 | 概率 1.7 越界 |
| shard_bad_len | v2 | v1 | 帧数与跨度不一致 |

## 4. 业务规则落点

### 4.1 加载与帧网格对齐（loader.py）

- 校验分片标识唯一、概率长度等于 `(end-start)/native_shift+1`、概率落在 `[0,1]`。
- 有效分片按全局时间轴排序；相邻有效分片要求 `prev.end == curr.start`，否则记
  `E_TIMELINE_GAP_OR_OVERLAP`。异常分片进入失败清单，不中断其余分片。
- 起止毫秒不整除当前帧移时记 `E_FRAME_GRID_MISALIGN`，并保留
  `start_remainder_ms/end_remainder_ms` 对齐残差。
- 边界帧按半开区间归属，理论上无重叠；若出现同帧竞争，采用确定性的左分片优先
  （left-wins）并写 `duplicate_notes`，不做静默覆盖。

### 4.2 头部版本兼容

- 码表兼容：读取分片**自身声明**的码表，把编码解析为语义类名，再按类名放到规范列。
  绝不按码位直连。无法映射到四类的编码记 `E_CLASS_MAP_UNMAPPED`，对应帧概率不进入
  规范类别（不可用），兼容记录保留不可用编码与命中帧。
- 网格兼容：`frame_grid_v2`（8 ms）在相同物理时间边界上线性插值重采样到 10 ms 网格，
  逐类别记录重采样前后在原生采样时刻的平均绝对残差，并记
  `E_GRID_VERSION_RESAMPLE`；禁止静默按当前位置拼接。

### 4.3 边界判决与逐帧判决（boundary.py）

- 每个边界取前后各 `neighborhood_frames` 帧，逐类别计算左右窗口均值与均值差。
- 主导类别取左右均值和最高者，同分按类别名降序固定决断。
- 连续判据：主导类别左右均值均不低于 `boundary_mean_min` 且均值差不超过
  `boundary_mean_diff_max`。输出边界标识、排序键（边界全局毫秒）、左右均值、判据、
  主导类别与是否连续。判决阶段不读取真值清单。
- 逐帧判决：取后验最高类别；所有类别同分按类别名降序；最高后验低于
  `frame_posterior` 标为背景帧。

### 4.4 轨迹重建（trajectory.py）

- 对判决为连续的边界先做 `linear_crossfade` 概率平滑，窗口帧数可配，平滑参数与
  实际左右帧号写入轨迹元数据。
- 在平滑后的网格上重新逐帧判决并 RLE 聚合；跨连续边界的同类片段合并为单一轨迹，
  记录跨越的全部边界标识与各边界判据。
- 同类轨迹间隔小于 `min_gap_ms` 做短间隙融合；时长小于 `min_trajectory_ms` 的轨迹
  进入 `dropped_short_trajectories`。
- 同起止同类别轨迹按拥有分片标识字典序保留，冲突写入 `duplicate_resolutions`。
- 轨迹起止毫秒由首末帧推导：`start = first_frame*shift`，
  `end = (last_frame+1)*shift`。

### 4.5 真值匹配（matching.py）

- 只在同类间匹配，重叠率采用交并比 IoU，阈值 `overlap_ratio` 可配。
- 冲突决断顺序：重叠毫秒降序 → 真值起始毫秒升序 → 轨迹起始毫秒升序 → 标识升序，
  形成全序，保证一对一分配可复现。
- 输出逐轨迹匹配真值、重叠毫秒/比率、整数毫秒起止边界误差、未匹配轨迹与未匹配真值。
- 评估阶段是唯一读取 `samples/global_truth.yaml` 的阶段。

### 4.6 诊断报告（report.py）

- 顶层字段、边界条目、轨迹条目均使用固定字段顺序；列表按稳定排序键序列化。
- 通过临时文件 `*.tmp` + `os.replace` 原子落盘，并 fsync。
- 包含业务码目录与计数、边界判据、类别分布、轨迹计数、跨边界轨迹与平滑参数、
  兼容/重采样/对齐残差、评估结果、失败清单、退出状态。

### 4.7 阶段快照（snapshot.py / pipeline.py）

- 四阶段：`load_align`、`boundary_decision`、`trajectory_build`、`match_evaluation`。
- 每阶段写入独立 staging 目录：先复制上一快照全部产物（累积快照），再覆盖当前阶段
  文件，最后用目录级 `os.replace` 整体替换；清单含全局版本号、配置指纹、分片头部
  版本集合、已完成分片标识、阶段产物版本。
- 启动时先派生当前输入身份，再校验快照的配置指纹与头部版本集合；任一头部版本变化
  即抛 `E_SNAPSHOT_STALE` 并拒绝续跑，不做逐文件覆盖。
- `drill` 在 `boundary_decision` 后制造可处理退出，再恢复并与全新运行比对，报告
  字节级一致。

### 4.8 并发、异常隔离与退出

- 边界判决与轨迹平滑使用 `ThreadPoolExecutor`，并发数取 `runtime.max_workers`。
- 任务携带稳定排序键：边界用边界全局毫秒，轨迹用首帧全局帧序号；结果在返回后统一
  排序，worker=1/2/4/8 的报告经测试逐字段相等。
- 单任务异常被捕获为 `E_TASK_FAILURE`，记录主体标识、阶段、异常类型并继续其余任务；
  平滑失败的边界不会中断轨迹构建，轨迹元数据单列 `failed_boundary_ids`。
- 收到 SIGTERM/SIGINT（或注入 `exit_after_stage`）时完成当前阶段快照原子替换与进度
  记录，以退出码 0 结束；强制 kill 后只能依据已完整替换的快照恢复。

## 5. 业务码

| 业务码 | 含义 |
| --- | --- |
| E_SHARD_ID_DUPLICATE | 分片标识重复 |
| E_SHARD_LENGTH_MISMATCH | 概率帧数与分片跨度不一致 |
| E_PROBABILITY_RANGE | 概率超出 [0,1] |
| E_TIMELINE_GAP_OR_OVERLAP | 时间轴不连续 |
| E_FRAME_GRID_MISALIGN | 边界不整除帧移 |
| E_CLASS_MAP_UNMAPPED | 类别编码无法做语义映射 |
| E_GRID_VERSION_RESAMPLE | 外网格版本已重采样 |
| E_TASK_FAILURE | 单边界/单轨迹任务异常被隔离 |
| E_SNAPSHOT_STALE | 快照身份失效，拒绝续跑 |
| I_BACKGROUND_FRAME / I_SMOOTH_MERGED | 信息类标记 |

## 6. 使用方式

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=.            # Windows PowerShell: $env:PYTHONPATH="."
python -m aedflow generate     # 生成 WAV 与分片包
python -m aedflow run          # 全流程，写 output/diagnostic_report.json
python -m aedflow drill        # 中断-恢复一致性演练
python -m aedflow stale-drill  # 头部版本变化 -> E_SNAPSHOT_STALE 演练
python -m unittest tests.test_pipeline
```

注入配置见 `config/injection.yaml`，可注入边界/平滑任务异常或阶段后可处理退出。

## 7. 实际验证结果（本机）

- 环境：Python 3.11.0rc1，numpy 1.26.4，PyYAML 6.0。
- `generate`：写出 1 个 WAV + 8 个分片包；重复生成字节级一致。
- 干净 `run`：5 条轨迹，5/5 主事件匹配，短事件 evt_06（40 ms）被最短轨迹规则过滤，
  失败清单 5 条，业务码与样本构造一一对应。
- 跨边界：b00/b01/b02 判决连续并合并为跨分片轨迹，平滑窗口与边界证据完整；b03
  因均值差 0.242 > 0.18 判决不连续，未错误合并。
- 兼容：shard_02 语义映射成功；shard_03 的 dog_bark 记 E_CLASS_MAP_UNMAPPED；
  shard_04 重采样残差为 speech 0.001616 ~ alarm 0.002995。
- 快照：恢复演练 `field_identical=true`，run_id 一致；失效演练存储层与流水线层均
  返回 E_SNAPSHOT_STALE。
- 并发：worker 1/2/8 诊断报告逐字段相等；14 个 unittest 全部通过。
