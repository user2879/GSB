# 配置分发结果验证与差异回归分析平台 — 技术说明

面向本地 Windows 环境、纯标准库 + NumPy 实现，不依赖容器或任何线上服务。
节点是独立 OS 子进程，绑定 `127.0.0.1` 高位（默认临时）端口，使用行分隔
JSON（JSON Lines）的回环 TCP 协议通信。

## 1. 模块结构

| 模块 | 职责 | 对应需求 |
| --- | --- | --- |
| `model.py` | 注册表/清单/会话参数数据类与强校验，确定性拓扑排序 | 1 |
| `differ.py` | 两版差异分类、派生链影响范围、受影响节点集合 | 2 |
| `fixedpoint.py` / `errorbound.py` | 整数定点量化、饱和、整数误差界传播与判定 | 3 |
| `node.py` | 模拟节点子进程、JSON Lines TCP、故障注入、确定性耗时 | 4 |
| `transport.py` / `scheduler.py` | 子进程生命周期、真实 socket 超时、双重并发配额、重试 | 4/6 |
| `compare.py` | 逐项逐节点摘要/耗时比较，E_REGRESSION 与 no_receipt 归类 | 4 |
| `attribution.py` | 派生链回溯、permille 贡献占比、主导层级、三类成因 | 5 |
| `ledger.py` | SQLite 账本/快照、单事务批次、整体替换、重放校验 | 7/8 |
| `pipeline.py` | 端到端编排、强制终止恢复、结论重放 | 4/7/8 |
| `canonical.py` | 规范化 JSON 与 SHA-256 内容摘要，保证可复核/逐字节一致 | 全部 |
| `cli.py` / `gui.py` | 命令行入口与惰性加载的 Tkinter 报告查看器 | 交付 |

## 2. 清单强校验（E_MANIFEST）

`load_registry` / `load_manifest` 收集全部违规后一次性抛出 `ManifestError`，
违规以 `(subject, reason)` 元组稳定排序输出，包含：

* 版本号必须为整数且 `version > predecessor_version`；候选版本必须大于基线。
* 键匹配 `^[A-Za-z_][A-Za-z0-9_.-]{0,63}$`，禁止重复键。
* 数值仅支持有限 `int`/`float`，显式拒绝布尔值与 NaN/Inf。
* 目标节点必须存在于注册表，禁止重复目标。
* 派生引用目标必须存在、禁止自依赖/重复父边，Kahn 算法检测环并报告涉事项。
* 分发策略限定 `full_mesh` / `canary` / `staged`。

## 3. 差异度量与影响链（顺序确定性）

所有集合在输出前排序，拓扑序采用带字典序 tie-break 的 Kahn 算法，
因此结论不依赖 dict/hash 遍历顺序。变更类型：`added`、`removed`、
`value_changed`、`targets_changed`、`coefficient_changed`（同项可多类型）。

影响链以“真正发生变更的项”为根，沿候选版 DAG 的显式子图做 DFS 求全部下游，
按候选拓扑序输出；未变更项及其纯下游不会成为根。删除项保留基线侧节点集合。

## 4. 定点量化与误差界（纯整数）

口径在报告 `precision_scheme` 中显式声明：

* `scale = 10 ** scale_exponent`，量化 `q = trunc_toward_zero(x * scale)`。
* 截断方向为**向零截断**，单值量化残差满足 `|x - q/scale| < 1/scale`。
* 所有运算用 Python 任意精度整数完成，再**饱和到有符号 64 位**
  （`INT64_MIN/MAX`），饱和在 bound 报告中以 `saturated` 标记。
* 定点乘：`mul(a,b)=sat(trunc_zero(a*b/scale))`，每次乘引入至多 1 个整数单位截断残差。

误差界全部以“量化整数单位”表达，沿拓扑序递归（整数运算）：

```
p_i = ceil(|coeff_q| / scale)                       # 整数上超界
propagated  = Σ (p_i * e_parent + e_parent + 2)    # +2 保证上超界仍 sound
truncation  = 父边数量
e_item      = sat(1 + propagated + truncation)
```

跨版本判定 `|Δq| <= e_base + e_cand + residual`：独立重量化的根值 `residual=2`，
由量化输入重算的派生整数 `residual=0`。判定输出 `abs_delta`、两侧 bound、
`cross_version_residual` 与 `limit` 全部分量。

## 5. 真实采集与回归判定

节点对 `(base_seed, node_id, item_key)` 做 FNV-1a 稳定哈希后用
`numpy.random.SeedSequence/Generator` 生成同一批确定性样本，再做
sum/mean/min/max 分组聚合。两版使用相同样本，仅配置缩放值不同。

比较按 `(node, key)` 排序输出：

* 摘要按分组键、指标逐项比较，超 `value_tolerance` 记 `summary_out_of_tolerance`，
  并记录两版摘要、差值、容差与**首个差异分组键**。
* 耗时差超 `elapsed_tolerance_ms` 记 `elapsed_out_of_tolerance`。
* 节点未回执/连接失败/重试耗尽记 `no_receipt`（绝不计为无差异），状态 `E_EXHAUST`。
* 目标集合变化记 `distribution_target_changed`。

## 6. 派生链归因（逐字节确定）

仅对 `E_REGRESSION` 项归因。新增/删除/直接变更项以自身为根，否则取可达该项的
最深变更祖先（链长最长、字典序 tie-break）。每层贡献用整数权重表示并归一化为
permille（总和恒为 1000），主导层级取 share 最大、层级顺序 tie-break。成因阈值
在会话参数中显式声明（默认 600/1000）：

* `node_execution_dispersion`：仅耗时回归且主导层 share ≥ dispersion 阈值。
* `derived_coefficient_amplification`：根含系数变更且主导层 share ≥ coefficient 阈值。
* `direct_value_change`：根为直接取值变更且主导层 share ≥ direct 阈值。
* 否则 `mixed`，并在 `cause_rule` 中写明所用阈值。

输入按 `(node,key)` 排序、全程整数与字符串，无浮点/回执顺序依赖，同输入多次运行
归因结论逐字节一致（测试 `test_attributions_byte_identical_across_runs` 守护）。

## 7. 并发调度与重试隔离

* 全局 `BoundedSemaphore(global_concurrency)` + 每节点
  `BoundedSemaphore(node_concurrency)`，任一时刻两层 in-flight 均不超限。
* 任务按 `(node, version, job_id)` 排序入队，记录单调时钟
  `wait_started/wait_ended/queue_wait_ms` 与每次 attempt 的起止毫秒。
* 每次采集使用真实 socket 超时；失败按固定 backoff 重试，**request_id 与 payload
  跨尝试完全不变**，保证重试不改变单次采集语义；尝试耗尽置 `E_EXHAUST`。

## 8. 账本、快照与恢复

* 每个节点批次在**单条 SQLite 事务**内写 `batches`（差异结论、误差界、findings、
  进度游标、内容摘要）+ `findings`，并清理对应租约。
* 表含整数 `ledger_format_version` 与 `precision_scheme_version`；每载荷带
  SHA-256 `content_digest`。
* 重开快照时版本或摘要不一致 → **整体替换**（清空 batches/findings/leases/snapshot
  后重建），不做增量合并，并稳定列出替换项。
* 强制终止后只依据已完整提交的批次恢复；仍处 `in_flight` 租约的项标记
  `unknown_rerun_required` 并要求重跑，平台不声称生成终止时刻退出报告。
* 恢复后按账本记录的版本与缩放因子重算差异、影响、误差界，并从账本内的原始摘要
  重新执行容差判定，与账本逐项比对；首个不一致定位到 `node.key.field` 并以
  `E_REPLAY` 报明依据（结构摘要不一致同样报 E_REPLAY）。

## 9. Windows 本地运行

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Tkinter 随 Windows 标准 python.org 安装自带；SQLite3 同为标准库
python -m config_regression.cli validate --baseline samples\baseline_manifest.json `
  --candidate samples\candidate_manifest.json --registry samples\registry.json `
  --session samples\session.json
python -m config_regression.cli run --baseline samples\baseline_manifest.json `
  --candidate samples\candidate_manifest.json --registry samples\registry.json `
  --session samples\session.json --database artifacts\session.db `
  --output artifacts\run_report.json
python -m config_regression.cli recover ...   # 同参数，强制终止后恢复+重放
python -m config_regression.cli gui          # Tkinter 报告查看器
```

节点默认 `port=0` 由 OS 分配未占用高位端口；如需固定端口可在注册表显式指定。
进程退出码：`0` 干净、`1` 存在回归/无回执、`2` 平台错误（E_MANIFEST 等）、
`3` Tkinter 不可用。
