# 技术说明与验收记录

## 1. 运行环境

- Python 3.11+，仅依赖 `numpy` 与 `pillow`；不使用 Docker、外部服务或账号。
- Windows PowerShell：

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 run_demo.py
py -3.11 -m unittest discover -s tests -v
py -3.11 scripts\check_tkinter.py
py -3.11 -m tkinter
```

也可以直接双击或执行 `run_demo.bat`，或运行 `scripts/verify_windows.ps1`。

实际验证环境是 WSL2/Linux 中的 Python `3.11.0rc1`、NumPy `1.26.4`、Pillow `10.3.0`。该精简 WSL 缺少 `libtk8.6.so`，所以这里的 `tkinter` 导入检查返回 `E_TKINTER_UNAVAILABLE`；代码没有把 Tk 作为核心链路依赖。Windows 官方 Python 安装器通常自带 Tcl/Tk，`python -m tkinter` 应弹出 Tk 示例窗口。

## 2. 层级与确定性约定

公开层级编号 `0...N-1` 按“最粗级 = 0，向细级递增”排列，因此调度器按层号升序时会先完成低频依赖。

样本为 `128x96`、4 层、下采样因子 `[2,2,2]`，公开层级尺寸（宽 x 高）为：

- L0：`16x12`，瓦片边 4，`3 x 4 = 12` 片
- L1：`32x24`，瓦片边 8，`3 x 4 = 12` 片
- L2：`64x48`，瓦片边 16，`3 x 4 = 12` 片
- L3：`128x96`，瓦片边 32，`3 x 4 = 12` 片

合计 48 片。几何校验要求每个下采样因子整除上一级尺寸，瓦片边整除该层宽高；否则分别返回 `E_CONFIG_INVALID` 或 `E_TILE_GEOMETRY`。

确定性规则：

- 样本合成只使用局部 `numpy.random.default_rng(seed)`，不调用全局随机状态。
- 下采样使用整数块均值，均值余数按 `ROUND_HALF_UP` 决断。
- 上采样使用中心映射双线性定点权重；等距邻点取等权，输出再半入整，边界钳制到顶/左样本。
- 决断规则写入每层金字塔元数据和快照 `pyramid_metadata.tie_break`。
- 层级指纹由整图内容指纹、公开层号和相对底图的累计下采样因子派生。

## 3. 修复与风格内核

- L0 最粗级先从已知像素开始做 8 邻域 BFS，按距离层由外向内计算权重均值（正交邻居权重 2，对角邻居权重 1），随后执行配置的定点扩散松弛轮数。
- L>0 级第 1 轮严格按 `(1-guidance)*本级初始像素 + guidance*上采样低频引导` 计算；后续轮先做邻域扩散再与引导加权，因此 `iterations` 是真实有效参数。
- 所有像素/权重计算使用 `int64` 定点数，小数位宽来自配置或 `set_quantization` 命令，允许枚举为 `4/8/12/16`。
- 风格 LUT 按层构建，输入为 0..255，条目为定点值；风格强度、三通道权重、量化档 `2/4/8/16/32/64/128/256` 均进入 LUT 指纹和瓦片缓存键。
- 风格只写掩码像素；掩码外像素从该层原始观测瓦片逐元素复制，基础层因此与原 PNG 掩码外字节完全一致。

## 4. 缓存键与精确失效

每片缓存键由以下组件再做 SHA-256 派生：

- `input_fingerprint`：本源瓦片像素 + 本掩码瓦片；
- `guide_fingerprint`：所有下级（更粗层）引用瓦片及其缓存键；
- `params_fingerprint`：迭代轮数、引导强度、风格强度、通道权重；
- `quant_fingerprint`：量化档与定点小数位宽；
- `layer_fingerprint` 与瓦片 ID。

清单中每片还保存 `dependencies`，例如 L1 瓦片会记录其双线性引导实际引用的 L0 瓦片。修改某层参数会改变该层全部瓦片键，并通过下级键嵌套使所有更细引用层失效；未变瓦片键相同，直接进入 `reused` 清单。每个 `reused_entries` / `recomputed_entries` 条目都带同一个 `cache_key`。

## 5. 预算、待办与续跑

调度顺序固定为：

1. 层号升序；
2. 瓦片行号升序；
3. 列号升序。

单批预算从当前所有键不匹配或无有效产物的瓦片中截取前 N 片，记入 `budget_charged`；剩余进入 `pending`，报告和快照状态为 `E_TILE_BUDGET_PENDING`。已成功瓦片立即进入快照，不会因预算耗尽回滚；待办瓦片状态为 `pending`，不会写成 `recomputed`，并在 `partial_pending.png` 中以琥珀色斜纹显式标记。

下一批先复用键仍匹配的瓦片，再按同一稳定顺序处理剩余失效项。无新编辑时，旧待办正好是失效队列的前缀续段。

## 6. 增量与全量一致性

自动化测试覆盖：

- 默认 8 条命令逐条增量运行后，各层修复/风格像素、最终 PNG、全部瓦片缓存键与空白全量重建一致；
- 先施加全部命令，再以预算 7 反复续跑至清空，最终仍与全量重建一致；
- 3 组固定种子随机增删掩码、改迭代/引导/权重/量化命令序列均一致；
- 同一预算调度下，两次空白重放的 `reused`、`recomputed`、`pending`、`budget_charged` 列表逐字相同。

“增量复用了哪些片”依赖历史，空白一次性全量必然全部重算；因此可移植的一致性契约是：最终像素、缓存键、空待办和规范顺序完全一致，而同一命令/预算历史的空白重放则逐字复现每份复用/重算清单。

## 7. 快照整体替换

快照目录默认是 `state/`：

- `tmp/staging/`：写 `products.npz`、每瓦片 `.bin`、`manifest.json`、`stages.json` 和 PNG；
- `snapshots/snapshot-XXXXXX-cmdYYYY-hash/`：由暂存目录 `os.replace` 整体改名提交；
- `commit_manifest.json`：临时文件 + `os.replace` 原子替换的当前提交指针。

清单包含 `global_version`、`config_hash`、`quantization_levels`、`command_cursor`（已清空待办的命令游标）、`applied_command_cursor`（已应用编辑游标）、`commit_seq`、五层阶段状态以及每瓦片版本/键/指纹。

启动时读取提交清单并比较 `global_version` 与 `config_hash`；不匹配返回 `E_SNAPSHOT_STALE`，不会逐文件覆盖旧快照。CLI 可加 `--recover-stale` 整体清空/重建并原子替换指针。

## 8. 并发与异常注入

- 同一层互不依赖的修复任务在线程池中并发；修复完成后并发风格任务。
- `max_workers=min(调用值, config.concurrency.max_workers)`，诊断中实际观测到 4 个 worker。
- 每个阶段任务都有稳定键 `repair:Lxx:Ryyyy:Czzzz` 或 `style:...`；失败清单按键排序，预算扣减按规范顺序提交。
- 独立注入配置支持 `nonfinite_pixels`、`invalid_mask_rectangle`、`quantization_out_of_range`。注入 ID 进入失败项和报告 `injection_hits`。
- 诊断验证串并行的像素、预算顺序、任务提交顺序、瓦片输出顺序和失败清单完全一致。

## 9. 实际验证结果

最近一次完整结果保存在：

- `artifacts/reports/verification-report.json`
- `artifacts/reports/test-run.txt`

已实际运行：

```text
python run_demo.py
# ok=true, consistency=true, injection_consistent=true, all_injections_hit=true

python -m unittest discover -s tests -v
# Ran 20 tests in about 34s ... OK

python -m workbench.cli snapshot-check
# status=compatible, command_cursor=8, pending=0
```

CLI 负路径也已实测：非法配置退出码 2 并输出 `E_CONFIG_INVALID`；修改全局版本后 `snapshot-check` 退出码 2 并输出 `E_SNAPSHOT_STALE`；`run-stream --recover-stale --drain --budget 48` 成功恢复到 8/8 命令完成。
