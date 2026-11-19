# 技术说明

## 1. 确定性与规范序列化

所有持久化 JSON 都经过 `spectral_diag.canonical`：

- 键按字典序排序；
- 使用紧凑分隔符；
- 浮点统一保留 12 位小数；
- 拒绝 NaN/Inf；
- 数组、numpy 标量和集合在序列化前递归规范化；
- SHA-256 对规范字节计算。

同一配置重复生成场景、重复积分或重复执行流水线，得到相同规范字节和哈希。场景的谱模型、材质、色样和光源全部由 `numpy.random.default_rng(seed)` 驱动，不读取网络或外部数据库。

## 2. 光谱与 XYZ 参考积分

基础波长网格为：

```text
lambda = 380, 385, ..., 780 nm
```

参考积分默认在 0.5 nm 网格上执行，共 801 点。输入 5 nm 谱与内置 5 nm 色匹配函数线性插值到参考网格；由解析连续模型生成真值时，直接在该网格上计算解析谱，避免把“对 5 nm 表插值”伪装成连续真值。

对光源 SPD `S(lambda)`、反射率 `R(lambda)`、观察者 `xbar/ybar/zbar`：

```text
Wx(lambda) = S(lambda) * xbar(lambda) * dlambda
Wy(lambda) = S(lambda) * ybar(lambda) * dlambda
Wz(lambda) = S(lambda) * zbar(lambda) * dlambda

k = 100 / sum(Wy)                    # y100_reflector
X = k * sum(R(lambda) * Wx(lambda))
Y = k * sum(R(lambda) * Wy(lambda))
Z = k * sum(R(lambda) * Wz(lambda))
```

`unity` 归一化取 `k=1`。光源/观察者项在乘以反射率之前形成，因此积分口径、白点和反射率项可分别审计。默认白光为完美漫反射体，白点 `Y=100`。

## 3. 场景真值

`spectral_diag.scene` 用固定种子的高斯混合生成：

- 12 个色样反射率；
- 8 个材质反射率；
- 冷色 source SPD；
- 暖色 camera SPD；
- 两组观察者/归一化口径下的白点与 XYZ。

每个色样对包含 source XYZ 和 target XYZ，二者都是同一反射率在对应 SPD 下的高密度参考积分。每个值附带真值来源、采样密度、观察者和归一化元数据。材质同时保留 target 真值，用于诊断迁移误差。

## 4. 矩阵反解与病态判定

给定 `A` 为 N x 3 source XYZ，`B` 为 N x 3 target XYZ，求：

```text
M = argmin ||A M - B||_2
```

实现使用 SVD 驱动的 `numpy.linalg.lstsq`。诊断字段包括：

- 设计矩阵秩；
- 奇异值；
- 设计矩阵条件数 `s0 / s2`；
- 矩阵自身条件数；
- 每样本 leverage；
- 共线样本对；
- 逐样本留一条件数；
- 逐样本 signed residual、L2 residual、RMSE 和最大绝对残差。

若秩小于 3 或条件数超过阈值，直接求解路径产生 `E_MATRIX_SINGULAR`。不构造、不输出、也不持久化单位矩阵作为替代解。

### 可复现剔除

对当前保留集合中的每个候选样本 `i`，计算删除它后的秩与条件数，选择：

1. 删除后秩最高；
2. 条件数最低；
3. 若仍相同，样本索引最小。

每一步记录删除索引、删除前后条件数、删除前后残差 RMSE 和剩余索引。若剩余不足 3 个样本或仍不满足阈值，返回 `E_MATRIX_SINGULAR`，并保留完整轨迹。

## 5. 量化与归因

量化步长：

```text
step = (range_high - range_low) / (2^bits - 1)
q(x) = round((x - low) / step) * step + low
```

默认 8 bit、范围 `[0, 150]`，步长 `0.5882352941176471`。任何 XYZ 分量越界立即返回 `E_QUANT_RANGE`，并报告首个样本索引与 XYZ 分量。

归因使用三个因子：

- `quantization`：source/target 色样 XYZ 量化；
- `distribution`：按 leverage 从高到低剔除声明数量的色样；
- `integration`：用 5 nm 输入和声明低密度口径重新积分，再重新求解。

残差定义统一为“用扰动矩阵预测参考 source 色样，与高密度参考 target 比较”的 RMSE，因此三种扰动对矩阵解的影响可比。三个因子的交互不可忽略，普通“逐个施加后相加”不等于全扰动结果，所以账本同时提供：

- 每个因子单独施加时的实测残差和相对基线增量；
- 三因子 Shapley 贡献；
- 交互分配量；
- 贡献总和与 `combined - baseline` 的严格对账。

对账容差为 `max(1e-10, |total_delta| * 1e-10 + 1e-12)`。任何不可加、内容哈希不符或贡献缺失都产生 `E_ATTRIB_MISMATCH`。

量化专项账本还输出：

- 3x3 矩阵逐元素偏移；
- 最大绝对偏移与 Frobenius 偏移；
- 条件数变化；
- 残差变化。

## 6. 观察者切换与失效闭包

依赖图位于 `spectral_diag.closure`：

```text
scene
  -> integration.source_weights / integration.camera_weights
  -> samples.source_xyz / samples.target_xyz
  -> whitepoints
  -> materials.source_xyz / materials.target_xyz
  -> matrix
  -> attribution
  -> materials.adapted_xyz
```

叶节点变化（观察者、归一化、参考密度、积分器、量化、归因参数、求解阈值）经反向依赖传播得到逻辑闭包。缓存键由依赖哈希递归形成。

切换观察者时，场景不重算；积分权重、色样 XYZ、白点、材质 XYZ、矩阵、归因和迁移材料重算。切回旧观察者时，所有依赖键命中缓存，`recomputed` 为空，中间结果哈希与首次一致。运行报告同时给出逻辑闭包、缓存命中和实际重算集合，并断言实际集合等于缓存未命中的失效闭包。

## 7. 异常注入与首个观测定位

支持的参数层注入：

| 注入 | 检出错误 |
|---|---|
| `collinear_sample` | `E_MATRIX_SINGULAR` |
| `zero_white_point` | `E_WHITE_POINT` |
| `quant_out_of_range` | `E_QUANT_RANGE` |
| `observer_invalid` | `E_OBSERVER_INVALID` |
| `low_integration_density` | `E_INTEGRATION_DENSITY` |

注入配置由规范 JSON 序列化并随运行写入 `injection.json`。检出后按“样本索引升序、XYZ 分量 X/Y/Z 顺序”定位首个受影响观测，输出：

- 样本索引、分量和分组；
- 观测值、预测值和残差；
- 行残差 L2；
- 误差区间；
- 是否为无界区间。

非奇异错误的默认误差区间是 `predicted +/- max(row_L2, quant_step/2)`。矩阵奇异时没有可提交矩阵，误差区间显式标记为无界，不用伪逆结果冒充正常解。

## 8. 原子持久化与终止语义

正常提交顺序：

1. 原子写 `runtime/active.json`；
2. 创建暂存目录；
3. 逐文件原子写入并 fsync；
4. 写 manifest、完整文件哈希和 `CLOSED.json`；
5. 暂存目录改名为最终快照目录；
6. 原子替换 `CURRENT.json`；
7. 追加已提交历史；
8. 删除 active 标记并写关闭日志。

`CURRENT.json` 之前任一步失败，快照都不算提交。恢复时：

- 只加载 `CURRENT.json` 指向、manifest 哈希和逐文件哈希均匹配的快照；
- 版本不支持时返回 `E_SNAPSHOT_VERSION`；
- 非 CURRENT 且未登记为历史提交的目录移入 `orphans/` 并记录；
- 存在 active 但无正常退出报告时，明确记录“没有退出报告”，绝不伪造退出状态。

强制终止测试使用子进程 `os._exit(86)`：它在快照目录已写入但 CURRENT 未提交时终止。重启后 CURRENT 仍指向前一个正常关闭快照，崩溃快照成为孤儿，active 被记录并清除。

## 9. 测试覆盖

`tests/` 共 25 项测试，覆盖：

- 场景和流水线确定性；
- 精确矩阵回解；
- 秩亏显式拒绝；
- 条件数剔除与轨迹；
- 量化越界定位；
- 归因可加性和哈希校验；
- 观察者/归一化闭包；
- 五类异常注入；
- 正常提交、关闭标记、真实强制终止、孤儿回收；
- 快照版本异常和文件篡改。

最近一次执行结果：`Ran 25 tests ... OK`，静态检查 `ruff check` 通过。
