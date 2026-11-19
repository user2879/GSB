# 本地确定性多阶段文生视频/图生视频渲染工作台

纯 Python 3.10+ / NumPy 实现，无 GPU、无外部网络、无容器。用于在反复调参与
强制终止恢复场景下，验证多阶段渲染的状态演进、精确失效传播与恢复一致性。

## 1. 目录结构

```
video_workbench/
├── vbench/
│   ├── errors.py        # 业务错误码（E_*）与带定位的工作台异常
│   ├── hashing.py       # sha256 内容指纹、参数指纹、缓存键
│   ├── stages.py        # 纯函数阶段：编码/初始化/去噪/补丁/解码
│   ├── config.py        # 配置校验、注入校验、交互命令应用
│   ├── persistence.py   # 暂存目录 + 原子整体替换的快照存储与恢复判定
│   ├── signals.py       # SIGTERM/SIGINT 优雅退出与 hard-kill 注入闸门
│   ├── engine.py        # DAG 线程池调度、缓存命中、失败隔离、诊断报告
│   ├── verify.py        # 增量 vs 全量逐帧/逐潜变量一致性
│   └── cli.py           # run/apply-commands/recover/validate/concurrency
├── samples/             # 固定种子样本（词表/提示词/噪声表/初始图）
├── config/              # 默认配置、交互命令流、异常注入配置
├── tests/test_pipeline.py
├── generate_samples.py  # 样本生成（固定主种子 20261120）
├── demo_recovery.py     # 真实子进程 SIGKILL/SIGTERM 恢复演示
├── run_demo.py          # 一键全量验收
└── requirements.txt
```

## 2. 渲染管线与确定性设计

阶段固定顺序，不得跳过或合并：

1. `encode`：受限词表 + 哈希词袋嵌入。词向量由模型常量种子按词表索引派生，
   与用户种子解耦，相同提示词永远字节级一致。
2. `image_encode`（仅 i2v）或 `latent_init`（t2v）：
   - t2v 初始潜变量由提示词内容指纹 + 种子派生独立 RNG；
   - i2v 初始图经均值池化编码为同形状图像潜变量。
3. `denoise_000 .. denoise_{S-1}`：每步读取上一步潜变量，叠加预置噪声表条目
   与提示词条件引导。两条路径在替代步之后调用同一个 `denoise_step`，
   共享完全相同的去噪与噪声引导语义。
4. `decode_000 .. decode_{F-1}`：最近邻上采样 + 线性投影 + 逐像素裁剪到
   0–255 的 uint8 帧；帧序由稳定排序键固定。

所有随机数都通过 `np.random.SeedSequence([seed, 阶段标签哈希])` 派生独立
`Generator`，无全局可变 RNG；全部浮点使用 float64，工作线程之间不共享
RNG 状态，因此与线程调度顺序、并发度无关。

图生视频分支：`i2v.replace_steps=K` 时，前 K 个去噪步以图像潜变量替代
（产物 `replaced=true`、`origin=image_latent_replace`），第 K 步起与
文生视频走同一去噪函数。初始图缺失或尺寸不匹配分辨率档位时返回
`E_INIT_IMAGE_INVALID` 并记录失败阶段 `image_encode`。

## 3. 缓存键与精确失效传播

每个阶段产物信封包含：`stage`、`content_fingerprint`、`cache_key`、
`dependency_fingerprints`、`param_fingerprints`、数组摘要（shape/dtype/min/
max/mean）与 `array_fingerprint`。

- 内容指纹只由阶段实际数据决定，不含缓存键/依赖清单等寻址信息，
  因此相同数据在增量链与全量链中指纹相同，不会污染下游。
- 缓存键 = 哈希(阶段名, 阶段参数指纹, 排序后的依赖内容指纹清单)。
- 单步去噪参数只含种子/引导/噪声表（含长度），步数变化通过噪声表长度与
  DAG 结构生效；改帧数只使解码失效；补丁只使该步及其下游失效，
  其上游（含 `encode`、`latent_init` 与更早去噪步）保持命中。
- 复用清单与重算清单均按阶段稳定排序键（阶段类型 → 步序 → 帧序）输出。

## 4. 快照持久化与恢复

- 每次运行在 `snapshots/tmp-<pid>/` 打开暂存区，上一版已提交产物以硬链接
  零拷贝纳入；单文件写入采用唯一临时名 + `os.replace` 原子落盘。
- 每个阶段完成后，暂存区被硬链接克隆为 `snapshots/v<版本号>/` 并原子切换
  `current` 符号链接（不支持符号链接时退化为原子目录替换），随后重建暂存区，
  保证已提交快照不可变、后续替换只断开暂存区 inode。
- 清单 `manifest.json` 记录全局版本号、配置哈希、已完成阶段、步序游标与
  帧序游标、`committed` 标记。
- 强制终止（`hard_kill` 注入在阶段产物落盘后 `os._exit(137)`，模拟 SIGKILL）
  只可能留下 `tmp-*` 暂存目录；重启时恢复逻辑仅读取 `current` 已完整替换的
  快照并校验产物完整性，忽略并清理暂存残骸，不承诺生成退出报告。
- 可处理的 SIGTERM/SIGINT 由 `SignalGate` 在阶段边界响应：完成当前阶段
  落盘与提交后以退出码 0 退出。

## 5. 并发模型与失败隔离

- DAG 波次调度：每一波把所有依赖已满足的任务交给 `ThreadPoolExecutor`
  （并发上限 `max_workers`），波次屏障等待全部结束后，再按稳定排序键
  统一落盘与阶段提交，提交瞬间无工作线程写盘。
- 每个任务携带由阶段名、步序号、帧序号派生的稳定排序键；最终帧顺序、
  中间潜变量清单顺序、失败清单顺序、产物摘要与单线程 `--serial` 逐字段一致。
- 单阶段异常被捕获为 failure 记录（阶段名、步/帧序号、业务码、异常类型、
  注入编号），下游标记 `UpstreamSkipped`，无关节点继续执行；诊断报告含
  各阶段耗时、复用/重算统计、注入命中编号。

## 6. 业务错误码

| 错误码 | 触发条件 |
| --- | --- |
| `E_CONFIG_INVALID` | 缺字段、类型错误、未知命令、非法模式/分辨率等 |
| `E_SEED_OUT_OF_RANGE` | 种子超出 `[0, 2^32-1]` |
| `E_GUIDANCE_INVALID` | 引导强度非有限数（NaN/Inf） |
| `E_STEP_ZERO` | 步数为 0 |
| `E_NOISE_TABLE_MISMATCH` | 噪声表缺失或长度与步数不符 |
| `E_INIT_IMAGE_INVALID` | 初始图缺失/维度错/尺寸与档位不符 |
| `E_PATCH_OUT_OF_RANGE` | 补丁矩形越界（写入前拦截，不产生新状态） |
| `E_STATE_DIVERGENCE` | 增量与全量渲染首个差异帧/阶段 |
| `E_STAGE_FAILED` | 单阶段运行期异常或上游失败隔离跳过 |

## 7. 使用方式

```bash
pip install -r requirements.txt
python generate_samples.py

# 单次增量渲染（报告为 JSON）
python -m vbench.cli run --config config/default_config.json \
    --workdir runs/t2v --report runs/t2v.json

# 交互命令流 + 空白工作区全量渲染逐帧比对
python -m vbench.cli apply-commands --config config/default_config.json \
    --commands config/commands.json --workdir runs/incr --report runs/apply.json

# 恢复点判定并续跑（--no-resume 仅输出恢复点）
python -m vbench.cli recover --config config/default_config.json \
    --workdir runs/incr --report runs/recover.json

# 配置/结构注入校验矩阵；并发/串行一致性
python -m vbench.cli validate-config --config config/default_config.json \
    --injections config/injections.json --report runs/validate.json
python -m vbench.cli concurrency-check --config config/default_config.json \
    --workdir runs/conc --report runs/concurrency.json

# 强制终止恢复演示（真实子进程 + 操作系统信号）
python demo_recovery.py
# 一键全量验收（含 10 个单元测试）
python run_demo.py
```

## 8. 实际验证结果

在 Python 3.11.0rc1 + NumPy 2.4.6（无 GPU、离线）环境实测：

- 文生视频 13 阶段（编码 1 + 初始化 1 + 去噪 6 + 解码 5）首次全量成功，
  二次运行 13/13 全部缓存命中、零重算。
- 18 条交互命令（提示词/种子/步数/引导/帧数/模式/替代步数/初始图/补丁增删）
  连续增量渲染后，与空白工作区一次性全量渲染的全部中间潜变量与 5–7 帧像素
  逐像素一致（`assert_snapshots_equal` 通过）。
- 补丁精确边界：对 step 3 加补丁时 `encode`、`latent_init`、
  `denoise_000..002` 全部命中，仅 step 3 及其下游去噪与解码重算；
  越界补丁在写入前返回 `E_PATCH_OUT_OF_RANGE`，工作目录不被创建。
- 7 个配置注入用例全部命中预期错误码；单阶段注入异常被隔离，
  下游 `UpstreamSkipped` 且注入编号写入报告。
- hard-kill（退出码 137）后重启从已提交快照恢复，已完成 6 阶段全部复用、
  剩余 7 阶段续算；SIGTERM 在阶段边界以退出码 0 优雅退出。
- 并发（`max_workers=4`）与单线程快照逐字段一致，清单排序完全相同。
- `python -m unittest tests.test_pipeline` 10/10 通过。
