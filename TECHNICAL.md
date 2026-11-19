# 技术说明

## 1. 解析与图校验

`narrative_analyzer.spec` 使用带重复键检测的 `yaml.SafeLoader`。解析后依次校验：

1. 顶层、节点、边、条件、数值与状态字段的类型和必填项。
2. 节点/边/条件 ID 唯一；起点存在且不是 ending；ending 无出边。
3. 条件中的分量、初值中的分量、效果表达式中的变量都属于显式状态分量表。
4. 边的 `when`/`preconditions` 引用已声明条件；缺失 `threshold` 单独触发 `E_SPEC_INVALID`。
5. 表达式通过 Python `ast` 构建自定义 `Expr`，只接受常量、状态变量、`Add`、`Mult` 与一元正负号。节点数上限 256，嵌套深度上限 32。
6. 用 Kahn 拓扑排序检查 DAG；未出队节点进入 `E_GRAPH_CYCLE.details.cycle_nodes`。

系数键采用 `<edge_id>/effect[<i>]/coef[<n>]`，由解析时的稳定前序遍历生成，不依赖对象地址或哈希。

## 2. 确定性重放

状态始终由 `spec.state.components` 的顺序驱动。每步流程固定为：

1. 按规格文件中的边声明顺序，评估当前节点所有出边的前置条件。
2. 优先按精确边 ID 匹配选择 token；否则按 `choice` 匹配。
3. 开启可达性约束时，在匹配候选中选第一条布尔可达边；分析模式可关闭该约束以保持原路径结构。
4. 按该边的 `effects` 列表顺序逐条执行效果。
5. 对表达式 AST 递归求值：先左后右，二元运算左结合；Python `float` 在 CPython 中即 IEEE-754 binary64，且不使用 FMA。

不得通过遍历映射、集合或 `set` 决定累加顺序。比较、差异报告等确需集合的场合均先排序。

每个状态值同时输出 `float.hex()` 与 `struct.pack('>d')` 位模式。`edge_sequence_hash` 是边 ID 列表的 SHA-256；每步内容哈希覆盖前后状态位模式、所有分支判定与效果轨迹，形成链式 `chain_hash`。

## 3. 条件判定与裕度

统一使用有符号裕度：

```text
signed_margin = current - threshold
absolute_margin = abs(signed_margin)
```

每条判定记录方向（`>`、`>=`、`<`、`<=`、`==`、`!=`）、当前值、阈值、阈值偏移、有符号/绝对裕度、阈值误差、状态误差和裕度区间。严格大于方向下，仅 `current > threshold` 为原始 `reachable=True`。

状态误差为 `Ev`、阈值表示误差为 `Et` 时：

```text
state_interval     = [current - Ev, current + Ev]
threshold_interval = [threshold - Et, threshold + Et]
margin_lower       = (current - Ev) - (threshold + Et)
margin_upper       = (current + Ev) - (threshold - Et)
```

比较器还计入一次阈值减法的舍入上界 `u * abs(signed_margin)`。若状态区间与阈值区间重叠，或等价地无法证明比较结果唯一，则 `uncertain=True`，但原始布尔值仍单独保留。`abs(signed_margin) < boundary_epsilon` 时 `boundary=True`。

## 4. 误差传播

每个表达式求值维护三类上界：

- `propagated`：来自当前状态已有误差；
- `coefficient`：来自数值常量的表示误差 `abs(c) * coefficient_relative_error`；
- `rounding`：来自舍入和跨项误差。

总上界为三者之和。单位舍入误差 `u = unit_roundoff`。对加法与乘法分别使用：

```text
fl(a+b): E <= Ea + Eb + u*abs(a+b)
fl(a*b): E <= abs(a)*Eb + abs(b)*Ea + Ea*Eb + u*abs(a*b)
```

乘法实现中的分类拆分为保守上界，允许略高估但不能低估；每一层运算按 AST 实际顺序叠加。`x += e` 等价于 `x = x + e`，`x *= e` 等价于 `x = x * e`。每个效果轨迹输出表达式值、传播/系数/舍入上界和更新后的目标误差；每条边输出汇总上界。

若值或误差非有限，或误差上界超过 `numeric.max_abs_error`，立即返回 `E_PRECISION_BOUND`，不继续产生看似有效的结局。

## 5. 扰动敏感度

扫描配置声明固定方向（`increase`/`decrease`）和严格递增的相对扰动量列表。对每个系数，按规格中的系数顺序、再按扰动列表顺序重放；只向 `ReplayOptions` 传入副本，不修改 `Spec`。每个判定键为：

```text
step[<i>]/<edge_id>/<condition_id>
```

当某判定的原始 `reachable` 首次与基线不同，记录该扰动量为临界值。所有 `(临界值, 系数声明序, 判定顺序)` 排序后输出排名；同一配置的两次运行结果完全一致。未翻转的系数也会列出，零系数标记为 `inert`。

## 6. 分支合并核对

从 `start` 到每个入度至少为 2 的合流节点做确定性 DFS，边按声明顺序展开。每条路径独立从初值重放，然后比较：

- 所有状态分量的值与误差上界；
- 合流节点所有出边判定的布尔值、不确定标记和裕度。

只要某分量 `abs(left-right) > path_tolerance` 就判定路径分歧，并按路径对顺序、分量声明顺序选择首个分歧。报告包含两侧完整边序列；结局相同不会覆盖状态差异。

## 7. 注入与首翻

注入配置在参数层标准化为 `ReplayOptions`：

- 系数相对扰动：`after = before * (1 + relative)`；
- 条件阈值微移：`after = before + delta`；
- 初值偏移：在初始化副本上执行 `before + delta`。

未知目标、重复目标、非有限结果或非法字段触发 `E_INJECT_CONFIG`。基线运行和注入运行都记录规范化配置及 SHA-256。按基线判定的遍历顺序（即边序）找到首个 `reachable` 改变的判定，附翻转前后当前值、阈值、有符号/绝对裕度、误差上界、边界/不确定标记、结局变化和全部生效偏移。

## 8. 快照与逐字节复现

`snapshot.json` 记录引擎版本、输入文件列表、每个输出文件的 SHA-256 和大小，以及快照内容哈希。检查时重新计算内容哈希和所有依赖文件；任一缺失、变化或版本不匹配即返回 `E_SNAPSHOT_STALE`。

演示构建不包含时间戳、主机名、随机 UUID 或绝对路径。核心产物先构建两次并在内存中逐字节比较，然后将比较清单写入 `reproducibility.json`；`verify.ps1` 还会在两个目录中分别运行 CLI 并比较整棵输出树。

