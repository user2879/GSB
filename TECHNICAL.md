# 技术说明

## 1. 管线与数据流

每个帧独立执行以下 DAG：

```text
text prompt ─┐
RGB PNG ─────┼─> encode ─> fusion ─> latent init ─> denoise step 0..N-1 ─> decode plan ─> block row 0..B-1 ─> frame PNG
motion table ┘
```

- 文本：按受控词表顺序归一化为词袋向量，记录命中词项位置。
- 图像：按 `grid_rows × grid_cols` 分块，每块输出 `mean_r, mean_g, mean_b, mean_abs_gx, mean_abs_gy`，所以图像维度必须为 `rows × cols × 5`。
- 运动：运动表维度必须为 `block_rows × block_cols × 2`。
- 文本词表长度、图像向量维度、运动向量维度均必须等于配置 `condition_dim`。
- 融合按条件类型名升序、再按维度序号升序处理。每次缩放和累加都量化到整数网格并记录量化步长、整数结果与残差。
- 潜变量由 `fusion digset + seed + 配置指纹` 派生 PCG64 随机源；不修改也不读取 numpy 全局随机状态。
- 去噪固定顺序为 `previous + noise_entry + scaled_condition`；输出摘要、舍入方向、上下舍入计数、总残差和最大残差均记录。
- 解码使用固定整数投影矩阵；`decode` 阶段生成行范围和逐行潜变量摘要，`block` 阶段只接收该分块行覆盖的 `latent[start:end]`，不接触其他行。
- 最终帧 PNG 由各分块行像素按行拼接后重新编码，不读取全图之外的潜变量。

## 2. 定点约定

允许枚举：

- `fraction_bits`: `6, 8, 10, 12`
- `storage_bits`: `16, 24, 32`
- `quant_level`: `0, 1, 2, 3`

缩放为 `scale = 2^fraction_bits`，整数量化步为 `2^quant_level`。除数为正时使用“半数远离零”的确定性舍入。存储范围是 `storage_bits` 对应有符号整数范围，越界截断残差进入记录。

量化档位、分数位宽和存储位宽全部进入缓存键。实际条件向量也随宽度量化，因此切换位宽或档位会改变中间整数、融合结果和后续缓存键，而非只改元数据。

## 3. v1 定宽行与 v2 JSON

v1 每行固定 169 字符，`|` 分隔，列宽如下：

| 列 | 宽度 | 对齐 |
|---|---:|---|
| stage | 20 | 左 |
| frame | 8 | 右 |
| step | 8 | 右 |
| block | 8 | 右 |
| guide_permille | 12 | 右 |
| noise_code | 8 | 右 |
| quant_bits | 8 | 右 |
| quant_level | 8 | 右 |
| residual | 16 | 右 |
| digest | 64 | 左 |

噪声码表：`0=gaussian, 1=uniform, 2=perlin, 3=none`。

### v1 -> v2

- `guide_permille / 1000.0` 精确转为浮点引导强度。
- 整数噪声码经码表映射为命名噪声方案。
- 生成 precision tag，例如 `permille=1235;noise_code=2;noise_scheme=perlin`。
- 保留 `_source_version: 1` 与 `_missing_fields`；v1 空列对应字段不会伪装成已提供。

### v2 -> v1

- 引导强度按 `round_half_away(value * 1000)` 还原为千分比。
- sidecar 保存：
  - `guide_rounding_residual` / `truncated_residual`；
  - 原 precision tag；
  - 被覆盖的 noise scheme、未知字段、缺失字段；
  - 完整原始 v2 记录与整份 v2 文档哈希；
  - 每条 v1 行的 SHA-256。
- 使用匹配 sidecar 重新升 v2 时，原文档字段逐值恢复。若 v1 被编辑，则只对行哈希仍匹配的记录使用旁路恢复。

## 4. 缓存键与失效

统一缓存键：

```text
SHA256(
  schema + stage
  + sorted(upstream: role, stage, cache_key, content_digest)
  + params_digest
  + fixed {fraction_bits, storage_bits, quant_level}
  + lease_generation
)
```

各阶段参数指纹只包含该阶段真正依赖的参数，所以：

- 提示词单帧替换：仅该帧 text 编码、融合及其下游失效；其他帧全部复用。
- 单帧图像替换/删除：仅该帧图像编码及下游失效。
- 融合权重改变：所有帧融合及下游失效，所有条件编码继续复用。
- 去噪步数减少：删除末尾步；保留步和更早产物复用。再次增加时，若原产物仍在持久缓存中可直接复用。
- 分块行数变化：去噪产物复用，`decode` 与所有 `block` 失效。
- 量化档位/位宽变化：所有定点阶段失效。
- 仅租约 TTL 变化：不改变内容键，全部复用；H 续期仍携带当前代次。

账本逐任务记录 `outcome=computed|reused`、缓存键、上游引用、artifact digest，并在 miss 时记录 `invalidation_basis`，例如：

```text
quantization_changed
lease_generation_changed
parameters_changed
upstream_changed:condition:text
cache_artifact_missing
```

## 5. 租约状态机

`LocalExecutor` 维护：

- 当前代次 `generation`；
- 活跃租约及其覆盖的 work unit；
- 签发时间 `issued_ms`、到期时间 `expires_ms`；
- 过期/释放事件。

规则：

1. `acquire` 先回收所有已过期租约，再令代次加一。
2. `renew/release/assert_active` 必须携带当前代次；不匹配返回 `E_LEASE_GENERATION`，代次不前进。
3. 任何租约操作检查到 `now >= expires_ms` 时，立即删除对应 work unit，产生 `E_LEASE_EXPIRED`。
4. 不存在 sleep 或“到期后继续等待”的路径。
5. 渲染会话覆盖 `0..max_denoise_steps-1` 的工作单元，并在每个调度批次前续期；过期时渲染抛错，账本记录错误和 executor 快照。
6. 同一会话内普通编辑不重新申请租约，因此租约代次稳定，缓存键可继续复用。

租约代次是缓存键的一部分。过期后重新申请会进入下一代，旧代次产物不会错误复用；内容本身仍由同一确定性内核产生。

## 6. 调度与预算

- `execution.max_workers` 是线程池上限，调度器不会创建超过该值的 worker。
- 稳定任务键格式：
  - `encode/frame=0000/text`
  - `fusion/frame=0000`
  - `latent/frame=0000`
  - `denoise/frame=0000/step=0000`
  - `decode/frame=0000`
  - `block/frame=0000/row=0000`
- 每步去噪消耗 `width × height × 3` 点；配置必须保证预算至少能推进一个步次。
- 单批按稳定顺序选择就绪任务，命中缓存的任务不消耗预算。
- 预算不足的已就绪 miss 进入 `deferred` 和 `pending`，批次标记 `E_POINT_BUDGET_PENDING`。
- 后续批次优先处理之前因预算未执行的待办（FIFO 待办序），新兴就绪任务排在旧待办之后，避免低预算下帧饿死。
- 未运行完不回写成功 manifest；已计算产物仍以原子文件持久化，后续批次可复用。

## 7. 异常隔离与原子性

- 配置、资源、命令均为显式校验；错误码稳定且 CLI 非零退出。
- `Workspace.apply_command` 在深拷贝上应用并完整校验，失败不会修改原 workspace。
- 缓存产物与输出文件使用同目录临时文件 + `os.replace` 原子写入。
- 缓存加载校验 JSON 结构、key、stage/stable key 和 artifact digest；损坏不静默跳过。
- 只有完整渲染成功后才写 `manifest.json` 和 `render_manifest.json`。

## 8. 确定性验证

自动化测试：

```bash
python3 -m pytest -q
# 39 passed
```

覆盖范围：

- 非法词项、坏 PNG 签名/像素模式/分辨率、坏运动表、非法定点枚举及 CLI 非零退出。
- v2 -> v1 -> v2 精确恢复，千分比舍入残差、噪声码映射、缺失字段。
- 文本词袋命中、图像块统计、融合排序、位宽/档位改变整数输出。
- 潜变量不依赖全局 numpy RNG；去噪摘要/舍入/残差稳定。
- 租约代次拒绝、过期真实释放、渲染过期错误与账本事件。
- 单帧提示词替换、权重、分块行、去噪步数、量化、租约 TTL 的精确失效。
- 分块只读自身潜变量区间；改变分块数不改变最终像素但改变分块键。
- 并发 1/3 结果与缓存键一致；低预算待办优先；预算恢复后与全量一致。
- 增量命令流与空白全量渲染的 PNG 字节、像素摘要、分块摘要、缓存键、调度顺序一致。
- 完整 demo 与租约 CLI。
