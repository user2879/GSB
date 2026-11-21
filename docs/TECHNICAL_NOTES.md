# 确定性文生图 / 图生图渲染工作台 — 技术说明

## 1. 目标与约束

纯本地、确定性、无 GPU / 无网络 / 无模型权重的多阶段渲染工作台。运行依赖仅
`numpy` 与 `PyYAML`，PNG 编解码使用标准库 `zlib` / `struct` 自行实现，不引入
第三方图像库。所有浮点运算使用 `float64`，所有随机源都是由显式输入派生的独立
`numpy.random.RandomState`，不读取、不依赖任何运行时全局随机状态。

## 2. 目录结构

```
render_workbench/
  errors.py        业务错误码（E_CONFIG_INVALID 等）
  hashing.py       规范化序列化、内容指纹、派生种子与隔离 RNG、缓存键
  png_codec.py     确定性 RGB8 PNG 编码 / 解码（固定压缩级别，无时间戳）
  config.py        YAML 配置与 JSONL 提示词加载、分级校验、配置指纹
  encoder.py       受限词表哈希词袋嵌入（按词表顺序归一化 + L2 归一化）
  latent.py        潜变量初始化、去噪步、初始图编码、局部补丁
  decoder.py       线性投影解码、裁剪、银行家舍入、PNG 输出
  storage.py       原子快照产物存储、manifest、游标、恢复校验
  edits.py         分批编辑计划（事务化，失败整批回滚）
  pipeline.py      节点图、指纹缓存失效、线程池执行、共享段验证
  runner.py        批次编排、信号处理、JSON 诊断报告
  cli.py           render / full / baseline / inspect 命令行入口
tools/
  generate_samples.py   固定种子生成配置、提示词、编辑计划与初始图
  validate_workbench.py 端到端验证器（79 项断言）
  _kill_driver.py       恢复测试用的硬终止 / 优雅终止驱动
tests/               unittest 单元与集成测试（30 个）
samples/             本地确定性样本（两档分辨率、14 个编辑批次）
```

## 3. 管线节点图与确定性

每个 `(提示词, 分辨率档位)` 生成一张有向无环图：

```
embedding -> init_latent -> denoise:s0 -> ... -> denoise:sN -> frame:s*
                    \-> (img2img) img_encode -> denoise:sR -> ... -> frame
```

- **编码**：词项按受限词表顺序计数并归一化，每个词项的哈希词向量来自独立
  RNG，求和后做 L2 归一化。
- **初始化**：`RandomState(derive_seed("init-latent", 提示词指纹, 种子,
  作用域指纹, 档位, 形状))`，同输入逐元素一致。
- **去噪**：固定运算顺序
  `x <- (x + 噪声表项 * 噪声场 + (引导强度 * 引导表项) * 条件场) / (1 + 0.1*(s+1))`，
  噪声场与条件投影矩阵均由输入指纹派生。
- **图生图**：初始图经最近邻缩放、归一化到 [-1,1]、确定性线性投影得到潜变量，
  替代前 `img2img_replace_steps` 个早期去噪步；替代点之后两条路径调用同一个
  `denoise_step`，语义完全共享。
- **解码**：档位专属投影矩阵把潜变量映射到 RGB，裁剪到 [0,255] 后用
  `np.round`（银行家舍入）转 `uint8`，再以固定 zlib 级别写出无时间戳 PNG。

## 4. 指纹、缓存键与精确失效

三级指纹：

1. **内容指纹**：对数组按 dtype/shape/原始字节做 SHA-256；对 JSON 结构按
   键排序的规范化字节做 SHA-256。
2. **作用域指纹**：`global_semantic_payload`（词表、维度、种子、步数、引导
   强度、替代步数、容差等档位无关语义）+ 单档 `tier_payload`（宽高、噪声表、
   引导表）组合。新增/删除其他档位不会改变本档作用域指纹，因此新增档位只构建
   新档节点，旧档全部命中。
3. **缓存键**：`cache_key(节点类型, 节点参数, 上游内容指纹列表)`。上游任一
   内容变化才会改变键，实现“仅受影响阶段及其下游失效”。

`concurrency` 与 `snapshot_interval` 属于执行/持久化参数，不进入语义指纹，
因此任意并发度产出字节一致。

### 局部补丁的失效边界

补丁节点包裹某个去噪步输出：

- 补丁坐标越界 → `E_PATCH_OUT_OF_RANGE`，整批编辑回滚，保留批前状态。
- 补丁与目标区域现有潜变量的最大差异 ≤ `patch_tolerance` 时判定为 no-op：
  缓存键折叠为 `patch-noop` 并复用已验证旧节点，下游去噪/解码零重算。
- 真实补丁只使该补丁节点及其下游失效，编码与初始化保持命中。

## 5. 原子快照与恢复

提交协议：节点文件先写 `tmp/` 暂存，再 `os.replace` 原子移动到
`store/<node>/`；manifest 经临时文件原子替换发布。读者只会看到完整旧版或
完整新版，不会看到半写文件。manifest 记录全局版本号、配置哈希、每个节点的
缓存键/内容指纹/文件指纹，以及按 `(提示词, 档位)` 重建的步序与帧序游标。

- **优雅终止**：收到 SIGTERM/SIGINT 后完成当前波次产物落盘，写 `stopped`
  报告并以退出码 0 退出；重启后从已提交游标继续。
- **强制终止**：`os._exit` 模拟掉电。重启时 `recover()` 逐个校验 manifest
  节点的文件存在性与文件指纹，丢弃任何未完整提交的节点，仅依据完整产物与
  游标恢复；强制终止不承诺生成报告（测试断言无 `report.json`）。

## 6. 并发模型

线程池大小取自配置。每个任务携带由 `(节点类型, 提示词 ID, 档位, 步号,
帧号)` 派生的稳定排序键，波次内按该键排序提交、按该键收集结果。节点间只通过
已提交的不可变产物传递数据，无共享可变状态；包初始化时强制 BLAS/OpenBLAS/
MKL 单线程，保证并发度 1 与 N 的数值逐元素一致、输出顺序一致。

## 7. 失败处理

加载与编辑阶段的错误不抛异常中断，而是结构化为业务码进入失败清单：

- `E_CONFIG_INVALID`：缺字段、类型错误、种子越界、表长不符、档位非单调等。
- `E_TOKEN_OUT_OF_VOCAB`：词项不在受限词表。
- `E_PATCH_OUT_OF_RANGE`：补丁坐标超出档位图像范围。
- `E_INPUT_MISSING` / `E_INIT_IMAGE_INVALID`：初始图缺失或不可解码。

每个编辑批次作为事务：任一条目失败则整批回滚到批前配置与提示词状态，后续
批次继续处理。

## 8. 验证结果

运行：

```bash
pip install -r requirements.txt
python tools/generate_samples.py --out samples
python -m unittest discover -s tests          # 30 个单元/集成测试
python tools/validate_workbench.py            # 79 项端到端断言
```

端到端验证覆盖：增量 vs 空白全量逐像素/逐潜变量/PNG 字节一致、重复运行确定性、
并发 1 vs N 一致、单线程串行基准一致、精确失效边界、no-op 补丁零重算、档位
增删的真实维度变化、图生图共享段逐元素一致、三类失败码与不中断处理、硬终止
恢复与优雅终止（退出码 0）。当前样本下最终节点 118 个，全部断言通过。
