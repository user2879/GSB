# 多阶段确定性渲染诊断工作台 — 技术说明

## 1. 总体架构

```
config.json + vocab.txt (+ init.png) + injections.json
        |
        v
 config.parse_config ............ 严格校验，失败不落盘（需求5）
        |
        v
 dag.build_graph ................ 阶段节点 + 稳定排序键 + 依赖边
        |
        v
 cache.plan_cache .............. 参数指纹/依赖指纹 -> 内容指纹 -> 命中判定（需求2）
        |
        v
 orchestrator._execute ......... 线程池并发 + 信号量限流 + 失败隔离（需求6/8）
        |
   +----+-----------------------------+
   v                                  v
stages.* 纯确定性计算          store 原子版本化快照 + progress 账本
   |
   v
 PNG + report.json + dependency_graph.dot + compare.json
```

模块清单：

- `dbench/errors.py`：稳定业务码与 `WorkbenchError`。
- `dbench/hashing.py`：规范化 JSON、NDArray 内容指纹、数组摘要。
- `dbench/rng.py`：基于 SHA-256 的 `DeterministicRNG`，不读全局随机状态。
- `dbench/config.py`：配置 dataclass、严格校验、配置指纹。
- `dbench/stages.py`：编码/初始化/图编码/去噪/解码的纯函数。
- `dbench/store.py`：临时文件 + `os.replace` 原子提交、版本连续性校验。
- `dbench/dag.py`：t2i / img2img 依赖图与拓扑/下游传播。
- `dbench/cache.py`：每阶段参数指纹与基于上游指纹的失效判定。
- `dbench/executor.py`：单阶段计算 + 真实故障注入。
- `dbench/injection.py`：独立注入清单、唯一编号 -> 业务码映射。
- `dbench/orchestrator.py`：并发调度、缓存复用、失败隔离、恢复点。
- `dbench/reporting.py`：报告、DOT 依赖图、双运行逐阶段对比。
- `dbench/cli.py`：`run` / `compare` / `verify-snapshots`。

## 2. 阶段与产物（需求1）

阶段：`encode_prompt` -> `init_latent` -> `denoise_s001..N` -> `decode`；
img2img 额外有 `image_encode`，前 K 步去噪以图像潜变量替代。

每个阶段产物为不可变版本：

```
<store>/<stage>/v0001.manifest.json
<store>/<stage>/v0001.data.npy
<store>/progress.json
```

manifest 含：`stage`、`version`、`param_fingerprint`、
`dependency_fingerprints`（全部上游）、`content_fingerprint`、
`array_summary`（shape/dtype/有限元计数/min/max/mean/内容指纹）。
写入顺序为：同目录临时文件 -> flush/fsync -> `os.replace` 整体替换。
版本必须从 1 连续，manifest/data 缺一即 `E_SNAPSHOT_VERSION` 并拒绝续跑。

## 3. 指纹与失效传播（需求2）

- 阶段参数指纹 = 规范化后的自身输入（提示词、种子、噪声表条目、补丁等）。
- 内容指纹 = SHA256(参数指纹 + 排序后的全部上游内容指纹)。
- 命中条件：存储中存在内容指纹完全相等的历史版本 -> 直接 `load_array`
  复用，不调用任何计算函数；否则标记重算。
- 上游指纹变化经内容指纹自动级联，使该阶段及全部下游进入重算集合。
- 报告逐阶段写入 `decision = reuse | recompute`，并给出 `reuse`/`recompute`
  清单与完整 `plan`。

并发度 `max_workers` 与局部补丁、img2img 替代步数被刻意排除在基础配置指纹
之外（它们分别是调度参数、步级局部输入、DAG 路由），保证无关上游阶段不会
被误判失效；补丁仅进入其目标去噪步的参数指纹（需求4）。

## 4. 确定性约束（需求3）

- 词袋嵌入：受限词表顺序计数 + OOV 词 SHA-256 哈希回退，L1 归一化。
- 初始潜变量：`DeterministicRNG` 以（提示词指纹, 种子, 配置指纹, 形状）为键，
  Box-Muller 生成；不使用 `np.random` 全局状态。
- 每步噪声/引导矩阵/解码投影均由显式键派生 RNG，跨平台逐元素一致。
- 同输入多次运行指纹相同，测试以 `assert_array_equal` 与报告指纹比对证明。

## 5. 局部补丁（需求4）

矩形区域 `(top,left,height,width)` + 数值数组，配置加载期即校验：
越界 -> `E_PATCH_OUT_OF_RANGE`；数组形状与区域不符 ->
`E_PATCH_SHAPE_MISMATCH`。补丁只出现在目标去噪步的参数指纹中，因此仅该步
及其下游失效，编码/初始化/更早去噪步保持命中（见 `test_caching.py`）。

## 6. 校验码与失败隔离（需求5/6）

| 触发条件 | 业务码 |
| --- | --- |
| 缺字段/类型错误/结构错误 | E_CONFIG_INVALID |
| 种子越界 uint32 | E_SEED_OUT_OF_RANGE |
| 引导强度非有限 | E_GUIDE_NOT_FINITE |
| 步数为 0 | E_STEP_ZERO |
| 噪声表长度/条目不符 | E_NOISE_TABLE_MISMATCH |
| 补丁越界 / 形状不符 | E_PATCH_OUT_OF_RANGE / E_PATCH_SHAPE_MISMATCH |
| 快照缺失/不连续 | E_SNAPSHOT_VERSION |
| 注入：非有限潜变量 | E_NON_FINITE_LATENT |
| 注入：初始图尺寸不符 | E_IMAGE_SIZE_MISMATCH |
| 注入：噪声条目截断 | E_NOISE_ENTRY_TRUNCATED |
| 注入：补丁越界 | E_INJECTED_PATCH_OUT_OF_RANGE |

校验在任何 store 写入之前完成，失败时不产生新版本。注入为独立清单，每条带
唯一 `id`；命中后编号与业务码写入 `injection_hits` 与 `failures`。单阶段失败
只阻断其下游（记入 `blocked`），无依赖关系的阶段继续执行；失败阶段不写产物。

## 7. 增量与全量等价（需求7）

`tests/test_equivalence.py` 从同一基线出发，依次编辑提示词、种子、引导、
噪声表、初始图、img2img 替代步数、补丁、步数；每次在同一持久化 store 上
增量运行，并与“空白 store + 相同最终输入”的全量运行比较：

- 每个共享中间潜变量 `assert_array_equal`；
- 最终 PNG 逐字节相等；
- `compare_runs` 逐阶段比对内容指纹，不等价时给出 `first_mismatch_stage`。

## 8. 并发一致性与恢复（需求8）

- 每个阶段有 `(组, 步序)` 稳定排序键；结果列表在报告前按键排序，
  与线程调度顺序无关。
- 调度器等待依赖 Future 后再向线程池提交，worker 永不占用线程等待其他阶段，
  从而避免池饥饿/死锁；`BoundedSemaphore` 强制 `max_workers` 并发上限。
- `--jitter` 测试开关为任务加入随机调度延迟，验证串行/并发/抖动三态结果一致。
- 优雅退出：`SIGINT/SIGTERM` -> `request_shutdown()`，完成当前阶段原子替换后
  报告 `shutdown.graceful=true, exit_code=0`。
- 强制终止：`SIGKILL` 后重启，仅依据已 `os.replace` 完整落盘的产物与
  `progress.json` 恢复，报告 `recovery.recovered=true` 及恢复点版本号。

## 9. 复现实验命令

```bash
python -m dbench.sample_generators.generate_samples
python -m unittest discover -s tests -v
bash run_acceptance.sh
```
