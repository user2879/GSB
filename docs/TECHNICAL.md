# 技术说明

## 1. 模块边界

| 模块 | 职责 |
| --- | --- |
| `config.py` | YAML 唯一键加载、配置/波形/编辑/租约 schema 校验、有效配置指纹 |
| `waveform.py` | 固定种子合成、缺失掩码、批次追加、单窗替换、单段删除、参数编辑 |
| `windows.py` | 设备/通道稳定顺序切窗、窗口内容指纹 |
| `features.py` | 缺失填补、RMS/峭度/频带能量、当前训练集逐列标准化 |
| `fixedpoint.py` | 二进制定点量化、整数网格舍入 |
| `kmeans.py` | 定点 KMeans、并列决断、空簇接管、逐轮量化残差 |
| `kde.py` | 高斯核密度、固定求和顺序、定点阈值与边界判定 |
| `lease.py` | 真实单调时钟租约、代次、续期、到期回收、孤儿接管 |
| `snapshot.py` | 快照校验和、四元一致性、原子整体替换 |
| `cache.py` | 持久化内容指纹缓存、复用/重算账本 |
| `pipeline.py` | 编辑、缓存、特征、聚类、密度、租约、快照的联合状态机 |
| `cli.py` | `validate`、`run`、`verify`、`faults` 命令与规范对拍 |

## 2. 确定性输入

每个通道用稳定派生种子创建独立 NumPy PCG64 RNG：

```text
derive_seed(master_seed, channel_id, interval_start, interval_count)
```

波形由以下分量按固定顺序相加：

1. 常量偏置与线性趋势漂移；
2. 配置中的周期抖动项，按列表顺序；
3. 平稳高斯底噪；
4. 可选的非有限采样注入；
5. 缺失段掩码。

设备顺序、通道顺序、采样序号和编辑命令序号均保持配置/文件顺序。窗口 ID 为 `device:channel:start`，窗口序号在全部通道切窗后按该稳定顺序生成。

## 3. 逐窗特征

对窗口内 `present=True` 的采样：

1. 非有限值直接产生 `E_NONFINITE_SAMPLE`，该窗口不进入训练集合；
2. 无任何可见采样产生 `E_WINDOW_ALL_MISSING`；
3. 缺失位置填为该窗可见采样均值，来源记录为 `window_seen_mean`；
4. RMS 采用采样序号升序累加；
5. 超额峭度为 `m4 / m2^2 - 3`，中心矩按采样序号升序累加；
6. 对加窗序列做实数 FFT，对每个频带按频点序号升序累加 `|X[k]|^2 / N`。

观测向量维度为 `2 + len(bands)`。原始特征由内容键缓存，标准化统计量只从当前成功参与训练的窗口集合计算：

```text
mean[d] = sum_i x[i,d] / n
disp[d] = sqrt(sum_i (x[i,d] - mean[d])^2 / n)
z[i,d] = (x[i,d] - mean[d]) / disp[d]
```

零离散度维度记录在 `zero_dispersion_dimensions`，离散度以 `1.0` 代替。窗口集合、原始特征或标准化参数任一变化都会改变 standardization cache key，后续观测向量缓存随之失效；不会沿用上一轮统计量。

## 4. 定点 KMeans

定点值表示为：

```text
fixed_value = integer_units / (2 ^ fractional_bits)
```

输入标准化值先按 `quantization_units` 的整数网格做半远离零舍入。所有距离与累加均使用 Python 任意精度整数：

```text
distance^2 = sum_d (x[d] - c[d])^2
```

初始质心采用等距窗口索引 `floor(i * n / k)`。指派并列时选择较小质心 ID；更新时按质心 ID 升序、窗口序号升序累加。质心分量由整数总量直接舍入到量化网格，不经过浮点均值。

每轮记录：

- 新质心整数坐标；
- 各簇成员数；
- 质心更新量化残差（base units 的整数差）；
- 空簇接管动作及依据。

空簇选择“距其当前指派质心最远”的窗口接管；距离并列取最小窗口序号。接管记录包含源簇、窗口序号、距离与接管依据。最终 SSE 为整数 units 平方和。

## 5. 核密度与哨兵

KDE 使用定点量化并反量化后的标准化向量，核带宽取自配置枚举。对每个窗口计算：

```text
log p(x) = -log(n h^d (2*pi)^(d/2)) + logsumexp_j(-||x-x_j||^2/(2h^2))
```

内层距离按维度序号升序累加，核贡献按窗口序号升序求和。为稳定处理远端窗口，采用固定顺序的 log-sum-exp。

密度分值定义为当前训练集最大密度的千分比：

```text
relative_permille(x) = 1000 * p(x) / max_j p(x_j)
score_fixed = quantize(relative_permille)
```

`score_fixed` 与 `threshold_permille * 2^fractional_bits` 比较。边界条件为：

```text
abs(relative_permille - threshold_permille) <= half_quantization_step_permille
```

边界窗口同时输出 `fractional_bits + 1` 位宽的参考密度定点值、阈值差和参考密度真值。当前样本中 `threshold_permille=478`，命中两个边界窗口。

## 6. 租约状态机

每轮 KMeans 前按簇申请独立租约。租约保存：

- 签发与到期单调毫秒；
- TTL；
- 单调递增代次；
- 进程 PID 与 `/proc/<pid>/stat` 启动令牌；
- `active`、`superseded`、`committed`、`expired`、`reclaimed` 状态。

续期只在 `now < expires_ms` 时允许；续期创建更高代次的新租约并将旧租约标记为 superseded。到期回收将状态真实置为 expired 并释放记录。重启时，若原租约仍 active 但其 PID/启动令牌已不存活，则标记 reclaimed，不等 TTL 到期。

聚类与密度结果必须由当前活跃租约代次覆盖。若在提交前到期，簇中心与密度基线标记 invalidated，窗口不暴露簇/密度结果，运行不提交快照，并输出 `E_LEASE_EXPIRED`。

## 7. 快照四元一致性

快照包含：

```text
global_version + config_hash + data_fingerprint + lease_generation
```

同时保存 lease map、缓存摘要、标准化统计量、簇中心和密度基线。快照 JSON 带规范 checksum，通过临时文件、fsync 和 `os.replace` 整体替换。

恢复/提交前逐项比较四元字段；任一不一致产生 `E_SNAPSHOT_STALE`。旧快照只用于判定陈旧，不作为当前状态续跑；可复用的内容缓存仍按精确内容键独立读取。若快照租约代次高于持久化租约高水位，另产生 `E_LEASE_GENERATION_ROLLBACK` 并阻止聚类提交。

## 8. 缓存依赖与失效

缓存键由规范 JSON 的 SHA-256 生成：

| 产物 | 上游依赖 | 参数/租约依赖 |
| --- | --- | --- |
| 原始窗口特征 | 窗口内容指纹 | 窗长、窗移、频带 |
| 标准化统计量 | 当前窗口集合的原始缓存键 | 离散度参数 |
| 标准化观测 | 原始键、统计量键 | 量化参数字段 |
| 簇模型 | 全部观测键 | 模型参数、租约 map |
| 密度模型 | 全部观测键 | KDE 参数、量化参数、租约 map |
| 快照条目 | 分析产物 | 全局版本、配置哈希、数据指纹、租约代次 |

追加或替换会改变标准化统计量，因此标准化观测、簇模型和密度模型按真实依赖级联重算；未变化窗口的原始特征直接复用。缓存账本逐条记录 `reused`/`recomputed`、缓存键、依赖、窗口序号、阶段排序键与未命中原因。没有“全量算完后标成命中”的路径。

规范化对拍使用 `(产物类型, 窗口序号, 窗口路径)` 排序，保证任意增量历史与干净全量重建的产物顺序一致。命中/重算来源按定义在增量与全量之间不同，因此单独保留在缓存账本，不作为分析结果字段。

## 9. 并发与稳定排序

- 线程池最大并发严格来自 `runtime.max_workers`；
- 特征任务排序键为 `(window_index, "10-raw-window-feature")`；
- 指派任务排序键为 `(window_index, "kmeans-assign")`；
- KDE 任务排序键为 `(window_index, "kde-density")`；
- Future 结果按入队稳定键重新排序后才进入累加；
- 统计量、簇更新、边界列表和失败列表均按窗口序号/阶段名/字段路径规范排序。

## 10. 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_CONFIG_INVALID` | 配置、波形、编辑声明或租约声明字段非法，带具体字段路径 |
| `E_WINDOW_REF_OUT_OF_RANGE` | 编辑命令引用当前窗口集合中不存在的窗口 |
| `E_NONFINITE_SAMPLE` | 窗口参与计算的采样包含 NaN/Inf |
| `E_WINDOW_ALL_MISSING` | 窗口没有任何可见采样 |
| `E_LEASE_EXPIRED` | 租约到期、回收后继续提交或续期失败 |
| `E_LEASE_GENERATION_ROLLBACK` | 声明/快照代次低于持久化高水位 |
| `E_SNAPSHOT_STALE` | 快照四元字段任一不一致 |
| `E_SNAPSHOT_CORRUPT` | 快照 checksum 不匹配 |
| `E_CLUSTER_COUNT_EXCEEDS_WINDOWS` | 有效窗口数少于簇数 |

故障目录中的预期业务码：

| 样本 | 业务码 | 行为 |
| --- | --- | --- |
| `cross_window_reference` | `E_WINDOW_REF_OUT_OF_RANGE` | 跳过非法编辑，其余窗口继续并提交 |
| `invalid_kernel_bandwidth` | `E_CONFIG_INVALID` | 在字段 `$.kde.bandwidth` 拒绝带病运行 |
| `non_finite_sample` | `E_NONFINITE_SAMPLE` | 仅淘汰两个受影响窗口，其余八个窗口继续 |
| `lease_generation_rollback` | `E_LEASE_GENERATION_ROLLBACK` | 完成特征后阻止聚类与快照提交 |

## 11. 实际验证

参考命令：

```bash
bash scripts/run_acceptance.sh
```

该脚本写入：

- `artifacts/pytest-output.txt`：pytest 原始输出；
- `artifacts/config-validation.json`：有效配置摘要；
- `artifacts/example-report.json`：默认样本完整报告；
- `artifacts/verify-report.json`：7 个编辑前缀 + 重复增量运行与干净全量重建对拍；
- `artifacts/fault-report.json`：四类故障命中汇总；
- `artifacts/crash-report.json`：SIGKILL 后恢复判定；
- `artifacts/validation-summary.json`：机器可读验收汇总。

实际数值以生成文件为准。已验证的关键结果：

- pytest：27 passed；
- 默认样本：43 个窗口，4 个簇，SSE `49,829,460` units；
- 边界窗口：`imu-alpha:gy:000096`、`imu-alpha:gy:000128`；
- 二次运行：87 个缓存复用，3 个簇/密度/快照重算；
- 全量/增量：簇结果指纹相同 `0e16d92bd83ff593c739525a16f1963082140f889a149d9a1fdddc5ab847bda1`；
- 四故障：全部命中预期业务码；
- SIGKILL：返回 `-9/137`，重启回收 4 个孤儿租约，`resume_scope=cache_only_superseded`。

## 12. 已知边界

- 随机波形依赖当前 NumPy PCG64 实现；升级 NumPy 后若需跨版本比特级复现，应固定锁文件或替换为项目内 PRNG。
- 快照只保存分析基线与窗口索引，不复制原始浮点波形；原始波形仍由确定性描述和编辑流重建。
- 因标准化统计量依赖完整训练集合，单窗替换会真实级联改变全部标准化观测；精确复用发生在未变化窗口的原始特征层，而不是伪造旧标准化结果。
