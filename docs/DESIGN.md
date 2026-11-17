# 灰度路由与流量镜像中间件 — 定点误差传播技术说明

## 1. 范围与运行方式

纯本地、无网络、无容器。运行时只依赖 Python 3.11+ 标准库；样本生成使用
NumPy（固定随机种子 `20261118`，重复运行字节级可复现）；Tkinter 仅用于可选
报告查看器，缺失时优雅降级。

```
python -m graymirror.samplegen samples
python -m graymirror.cli run --rules samples/rules.json \
  --stream samples/stream.ndjson --scale-exponent 2 \
  --snapshot samples/snapshots/session-k2.json --report samples/report-k2.json
python -m graymirror.cli restore --rules samples/rules.json \
  --stream samples/stream.ndjson --snapshot samples/snapshots/session-k2.json
python -m graymirror.cli inject --kind weight_out_of_range|timestamp_regression|snapshot_scale_missing
python -m unittest discover -s tests
```

Windows PowerShell 一键演示：`scripts/run_demo.ps1`；Linux/macOS：`scripts/run_demo.sh`。

## 2. 定点量化口径

- 网格基数 `G = 10000`，规则中的灰度权重与镜像采样率都是 `0..G` 的整数万分比。
- 会话缩放因子是 10 的整数幂 `S = 10^k`，`k ∈ {0..6}`。
- 量化方向显式声明为向零截断；对非负整数即向下取整：

  `q(x) = floor(x·S/G)`，定点整数单位为“每满量程 S 份”。

- 映回万分比网格的有效比例：`w_q = q·G/S`，截断误差
  `e = x − w_q`，满足 `0 ≤ e < Δ`，其中单格步长（以万分比整数格表示）

  `Δ = G/S = 10000/10^k`。

  | k | S | Δ（万分比格） |
  |---|---|---|
  | 0 | 1 | 10000 |
  | 1 | 10 | 1000 |
  | 2 | 100 | 100 |
  | 4 | 10000 | 1（整数万分比精确） |
  | 6 | 1000000 | 1/100 |

  当 `k ≥ 4` 时任意整数万分比都能被精确表示，截断误差为 0。

### 2.1 分桶边界与权重之和的偏差

哈希分桶取 `0..G` 的整数桶 `b`。判定全程不使用浮点，采用整数交叉相乘：

`b 落入灰度区 ⇔ b·S < q_g·G`。

若对灰度与稳定两侧分别独立 `floor`，两侧定点数之和与满量程 `S` 至多相差 1 个
定点单位（每次取整只损失不足 1 个单位）。本实现不做独立量化，而是令
`q_stable = S − q_gray`，强制 `q_gray + q_stable = S`，把这至多 1 个单位的偏差
整体推移到灰度/稳定边界上：边界阈值从真实 `w_g` 平移到 `w_q`，平移量即
截断误差 `e < Δ`，因此任一侧占比的绝对误差上界都是 `1/S`（比例口径），
换算到万分比整数格为 `Δ = 10000/S`。

镜像采样使用独立域分隔哈希 `mirror|rule|request_id`，同样以
`b·S < q_m·G` 判定；镜像率的量化绝对误差上界同为 `1/S`。

## 3. 稳定哈希与顺序无关性

- 路由哈希材料 `route|rule_id|client_key`，镜像哈希材料
  `mirror|rule_id|request_id`，均取 SHA-256 后对 `G+1=10001` 取模得到
  `0..10000` 的整数桶。
- 不使用 Python 内建 `hash`（进程加盐）也不依赖字典遍历顺序；同一
  `(规则, 客户端键, 缩放因子)` 在任意机器、任意请求顺序下得到同一去向。
- 前缀匹配采用“最长前缀优先、等长按 rule_id 字典序”的确定性规则。
- 单元测试通过打乱请求流验证统计量、误差界与贡献排序完全一致。

## 4. 逐级误差传播与假设检验口径

设规则下请求数为 `n`，灰度定点边界与真实边界相差不足一个定点单位。

1. 第一级（灰度分流）：只有跨越边界的请求可能翻转通道，因此灰度/稳定
   占比的绝对误差上界 `B_route = 1/S`（比例），万分比口径 `10000/S`。
2. 第二级（镜像采样）：镜像占比绝对误差上界 `B_mirror = 1/S`。相对偏差上界
   `B_mirror / r = G/(r·S)`；`r=0` 时 floor 恒为 0、`r=G` 时恒为满量程，
   两种边界情形相对偏差严格为 0。
3. 第三级（成功率聚合）：通道占比的边界不确定性最多使 1 个请求在两个通道
   间翻转，故规则级主请求成功率误差上界 `B_sr = 1/n`（比例），万分比口径
   `10000/n`。

端到端合成采用保守的线性（L1）上界，避免对各级误差独立性作无依据假设：

`B_e2e = B_route + B_mirror = 2/S`（比例），万分比口径 `20000/S`。

全局指标按各规则请求量做加权平均；全局成功率误差上界为 `1/N`，
`N` 为请求总数。所有计算使用 `fractions.Fraction` 精确有理运算，报告中的
误差界以分数字符串给出，可无损复算。

### 4.1 贡献排序阈值规则

每个 `(规则, 级别)` 贡献 = 该规则请求量占比 × 该级网格步长 `1/S`。占总误差
上界比例 `≥ 1%` 标记为 `primary`，否则为 `minor`；排序键为
`(占比降序, rule_id, level)`，完全确定，与请求到达顺序无关。

### 4.2 实测收敛（200 条样本，固定种子）

| k | 灰度命中 | 稳定命中 | 镜像命中 | 单格 Δ（万分比） | 端到端界（万分比） |
|---|---|---|---|---|---|
| 0 | 0 | 200 | 0 | 10000 | 20000 |
| 1 | 28 | 172 | 77 | 1000 | 2000 |
| 2 | 41 | 159 | 86 | 100 | 200 |
| 4 | 41 | 159 | 86 | 1 | 2 |
| 6 | 41 | 159 | 86 | 1/100 | 1/50 |

`k≥4` 后分流与镜像命中不再变化，量化误差归零；误差界随 `10^k` 单调收敛，
与推导一致。

## 5. 镜像隔离

镜像判定写入独立的 `mirrors` 记录（含镜像目标、超时毫秒、状态、分桶与
判定依据），不修改主请求 `decisions` 的目标、通道、重试或状态，也不进入
主请求成功率统计。镜像状态由独立确定性摘要模拟（`SUCCESS/TIMEOUT`），
与主请求结果无耦合。

## 6. 版本整体替换与陈旧防护

- 任一权重、采样率或缩放因子变化都会构造新的 `RuleView`，版本号单调递增；
  历史视图 `stale=true`，不参与当前报告。
- 当前视图为整体替换而非增量合并；变更台账稳定列出受影响规则、字段
  旧值/新值、万分比变化量 `delta_per_10000`，以及误差界字段的旧值/新值。
- 规则指纹 = 规则与缩放因子规范化 JSON 的 SHA-256；请求流摘要 = 规范化
  NDJSON 的 SHA-256 + 行数 + 首尾请求标识。

## 7. 快照恢复与重放校验

快照持久化格式版本、规则版本、缩放因子、请求流摘要、逐规则量化权重与
采样率、路由与镜像统计、逐请求决策、镜像记录及误差结论。恢复时：

- 规则指纹或请求流摘要不一致 → 整体拒绝合并并报 `E_VERSION`，逐项列出
  `rule_fingerprint` / `stream_summary` 的快照值与当前值。
- 对齐后按快照版本与缩放因子重放，逐条比对路由与镜像字段，再比对聚合
  统计与量化规则；首个不一致定位到 `request_id.field` 并报 `E_REPLAY`。
- 快照缺失缩放因子属于网格契约失败，按 `E_GRID` 报告。

## 8. 业务码复现矩阵

| 业务码 | 触发方式 | 实测定位信息 |
|---|---|---|
| `E_RULE` | `inject --kind weight_out_of_range`（灰度权重 10001） | 行号 + `$.rules[0].gray_weight` |
| `E_STREAM` | `inject --kind timestamp_regression`（第 3 行时间戳回退） | 行号 + `$.timestamp_ms` |
| `E_GRID` | `inject --kind snapshot_scale_missing`（删除快照缩放因子） | 字段 `scale_exponent` |
| `E_VERSION` | 用改动后的规则恢复旧快照 | 差异项 `rule_fingerprint` |
| `E_REPLAY` | 篡改快照内某请求的分桶/通道后重放 | 首个 `request_id.field` |

所有错误同时写入结构化 JSONL 日志（默认 `samples/logs/events.jsonl`）。

## 9. 目录结构

```
graymirror/
  engine/errors.py        业务码与异常
  engine/fixedpoint.py    定点量化与整数交叉相乘判定
  engine/hashing.py       域分隔稳定哈希
  engine/loader.py        规则/请求流强校验（行号、字段路径）
  engine/router.py        灰度路由、镜像隔离、确定性模拟
  engine/statistics.py    统计与逐级误差界、贡献排序
  engine/session.py       版本整体替换、指纹、快照
  engine/replay.py        恢复后重放一致性校验
  injection.py            三类异常注入
  samplegen.py            可复现样本生成（NumPy 固定种子）
  viewer.py               可选 Tkinter 报告查看器（可降级）
  cli.py                  命令行入口
samples/                  规则、NDJSON、快照、日志、报告
tests/                    24 个 unittest 用例
docs/DESIGN.md            本说明
```
