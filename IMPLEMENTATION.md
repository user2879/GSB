# 定点多模态仲裁渲染工作台

本目录是首版可运行工程：样本生成、整文件校验、定点内核、内容指纹缓存、loopback 租约执行器、
原子快照、预览摘要、Tkinter 账本和验收脚本均在本地完成。

## 快速复现

```bash
# Ubuntu/WSL；若 python3 -m venv 缺 ensurepip，可先使用已有 virtualenv
bash scripts/setup.sh

# 生成词表、两张固定种子 PNG、运动表、YAML 配置和编辑命令流
.venv/bin/python -m fixed_render.cli init-samples --root .

# 单次渲染与二次命中
.venv/bin/python -m fixed_render.cli render --root . --workspace runs/manual
.venv/bin/python -m fixed_render.cli render --root . --workspace runs/manual

# 九条编辑命令：提示词、图像替换/删除/恢复、运动、权重、步数、分块、量化
.venv/bin/python -m fixed_render.cli run-commands --root . \
  --workspace runs/incremental \
  --report artifacts/command_run.json

# 五类注入诊断 + 租约回收审计
.venv/bin/python -m fixed_render.cli diagnose --root . \
  --report artifacts/diagnostic_report.json

# 完整验收：随机命令序列、空白全量像素比对、并发、SIGKILL 恢复等
.venv/bin/python -m fixed_render.cli verify --root . \
  --report artifacts/verification_report.json

# 单元测试
.venv/bin/python -m unittest discover -s tests -v

# Tkinter 界面；无交互冒烟会写 artifacts/gui_smoke.json
TK_SMOKE_MS=800 bash scripts/run_gui.sh artifacts/diagnostic_report.json
```

首选系统依赖安装方式是 `sudo apt install python3-tk`。当前运行环境的 `python3-tk` 不可用，
因此已把 `libtcl8.6`、`libtk8.6`、`libXft`、`libXss` 的用户态解包放在 `.tk-runtime/`，
并在 `.venv` 中链接 Python 3.11 的 `tkinter` 与 `_tkinter`；GUI 会自动预加载这些 `.so`。

## 代码地图

- `fixed_render/config.py`：YAML 配置 schema、枚举、维度/整除校验和配置指纹。
- `fixed_render/miniyaml.py`：只用标准库实现的 YAML 子集；安装了 PyYAML 时优先用 PyYAML。
- `fixed_render/inputs.py`：受控词表、PNG 签名/IHDR/CRC/分辨率/模式、运动 CSV 和命令流校验。
- `fixed_render/samples.py`：固定整数种子生成词表、几何/纹理 PNG、运动矢量表和示例命令。
- `fixed_render/fixedpoint.py`：允许位宽 `4/8/12`，允许模式 `round_half_up/floor/round_half_even`，
  全部使用 `int64` 比率和舍入残差。
- `fixed_render/kernel.py`：词袋、图像分块均值/梯度、运动归一化、排序融合、潜变量、去噪步和分块解码。
- `fixed_render/engine.py`：阶段 DAG、指纹缓存、失效依据、复用账本、PNG/sidecar 和快照提交。
- `fixed_render/lease.py`：`127.0.0.1` 换行 JSON 租约服务/客户端、签发/续期/鉴权/回收。
- `fixed_render/storage.py`：内容制品库、临时目录后原子改名的快照和 JSON 账本。
- `fixed_render/workbench.py`：命令校验/应用、运动表可变副本、编辑后预览预算。
- `fixed_render/diagnostics.py`：五类独立注入清单和租约真实回收审计。
- `fixed_render/verification.py`：增量/全量一致性、随机命令序列、并发、SIGKILL 恢复。
- `fixed_render/gui.py`：展示诊断报告和其引用的同一份阶段账本。
- `fixed_render/worker_demo.py`：SIGKILL 恢复演示子进程，不捕获强杀、不写虚假退出报告。

## 校验与整文件拒绝

所有 schema 错误聚合为 `fields.<字段名>`，错误码为 `E_INPUT_INVALID`；加载器不会返回半装载对象。
覆盖项包括：非受控提示词、词表长度与 `text_dim` 不符、PNG 签名/IHDR CRC/8-bit RGB/64×64、
运动块数与 8×8 块网格不符、量化位宽/模式枚举、`tile_height | frame_height` 且为
`latent_scale` 倍数等。

## 指纹与精确失效

每个节点的内容身份为：

```text
content_key = SHA256(canonical({
  node, external_inputs,
  upstream: {id: {content_fingerprint}},
  params_fingerprint,
  quant: {frac_bits, mode, scale}
}))

cache_key = SHA256(canonical({content_key, lease_generation}))
```

制品以 `cache_key` 落盘，`content_index.json` 记录 `content_key -> cache_key/content_fingerprint`。
同一租约代次下编辑提示词只会使 `encode:text -> fuse -> latent -> denoise -> decode/tiles/output`
失效，图像和运动编码直接命中；改权重时三个编码器均命中；增加步数时已有去噪步命中，只计算新步；
改分块高度时编码器、融合、潜变量、去噪和帧参数节点命中，只重算新行段与帧装配。

若租约到期后以新代次重签，租约绑定的 `cache_key` 会变化，但内容索引可按 `content_key`
重新绑定并复用已持久化制品；这正是回收后续跑路径。

## 定点与分块边界

- 文本：`count / prompt_length` 直接量化为 Qf。
- 图像：4×4 块，每块记录 RGB 归一化均值和 x/y 梯度统计，共 `16*5=80` 维。
- 运动：8×8×2 个 Q8 矢量，按 32 像素包络归一化。
- 融合：按 `(condition_type, dimension)` 排序，确定性投影后做整数加权平均，逐步保存整数指纹和残差指纹。
- 解码：引擎先从最终潜变量切出 `latent[frame, latent_y0:latent_y1, :, :].copy()`，
  `decode_tile()` 只接收本行段；函数无法读取相邻分块潜变量。

## 租约语义

租约记录 `lease_id`、单调全局 `generation`、`issued_ms`、`expires_ms`、scope 和 slot。
续期必须提交当前公认代次；错误代次返回 `E_LEASE_STALE`。后台 reaper 每 5ms 扫描，
到期后真实把状态改为 `expired` 并减少 `used_slots`；旧令牌继续鉴权/续期得到
`E_LEASE_EXPIRED`。验收会等待 `used_slots == 0` 后再签发新租约，而不是用睡眠代替回收。

SIGKILL 演示中，子进程先写内容制品和 ready JSON，随后永久睡眠；父进程确认槽位被回收后
发送 `SIGKILL`，断言返回码为 `-9`、不存在 `exit_report.json`，再用持久化制品续跑并与空白全量比对。

## 快照与预览

每次提交创建 `snapshots/.vXXXXXX.tmp-*`，写入 `manifest.json`、`cursor.json`、
`pending.json`、`budget.json`，随后原子改名为 `vXXXXXX/`，再原子替换 `current.json`。
启动时比较版本、`config_fingerprint` 和 `condition_fingerprint`：相同则 `resume`，不同则
`replan_with_content_cache`；无快照则 `fresh_build`。

每条命令返回预览：完成帧数、受影响帧区间、各分块行完成状态、租约到期毫秒、代次和进度。
若命令加渲染耗时超过配置的 `preview_budget_ms`，返回 `E_PREVIEW_STALE`，并给出旧摘要的
`based_on_version`；预算充足时返回最新摘要，其中帧指纹与最终 PNG/sidecar 完全一致。

## 诊断与界面

`diagnostics/injection_manifest.yaml` 包含五个独立注入编号：

- `INJ-DIM-001` → `E_DIM_MISMATCH`
- `INJ-NAN-002` → `E_NONFINITE_LATENT`
- `INJ-TILE-003` → `E_TILE_OUT_OF_RANGE`
- `INJ-QUANT-004` → `E_QUANT_INVALID`
- `INJ-LEASE-005` → `E_LEASE_STALE`

单阶段失败后，无依赖关系的编码/其他分块仍会完成，下游标记为 `blocked`；报告保存阶段耗时、
租约轨迹、复用/重算顺序、pending 和业务码计数。Tkinter 左侧选择注入案例后，直接读取案例中的
`ledger_path`，因此界面和 JSON 报告展示的是同一份账本。
