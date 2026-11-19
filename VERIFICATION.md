# 实际验证记录

本记录在项目工作区本地执行，不使用 Docker，不访问真实或未授权动捕资产。

## 环境

- Python `3.11.0rc1`
- NumPy `2.4.6`
- 平台：本机 Linux/WSL 工作区；代码仅使用 Windows 兼容的标准库与 NumPy API，并提供 `run_demo.cmd`

## 自动化测试

命令：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

结果：`19` 项全部通过。

## 完整双跑

命令：

```bash
.venv/bin/python -m mocap_workbench demo --out workdir/run_final_c > workdir/run_final_c.stdout.json
.venv/bin/python -m mocap_workbench demo --out workdir/run_final_d > workdir/run_final_d.stdout.json
```

两次运行分别耗时 `35.64 s` 和 `35.15 s`，峰值 RSS 分别约 `458.2 MB` 和 `458.0 MB`，退出码均为 `0`。

两次输出目录各含 `62` 个文件。逐文件 SHA-256 比较结果：文件名集合完全一致，所有文件内容完全一致。标准输出与 `demo_report.json` 也逐字节一致。

最终运行摘要：

```text
demo_report.json SHA-256:
0929ffed31d64499621caac4fcf690bec58b75903346f0985c4a90d40a6fcc50

run_final_c.stdout.json SHA-256:
0929ffed31d64499621caac4fcf690bec58b75903346f0985c4a90d40a6fcc50
```

每套文件合计 `78,398,303` 字节；双份合计 `156,796,606` 字节。

## 三档结果

| 帧数 | 自动策略 | 场景摘要（前 16 位） | 最终姿态序列摘要（前 16 位） | 跨策略一致 |
|---:|---|---|---|---|
| 300 | `full_segment` | `1d4903fa384458a9` | `8533ce65711bbc90` | 是 |
| 3000 | `windowed` | `e2654a6d2d59f77a` | `74882793a5bf1e83` | 是 |
| 30000 | `sparse_anchor` | `0dad698fe2ed3bb2` | `d0643e4724e790e2` | 是 |

策略最大关节位置差：

- 300 帧：全段/分窗 `9.89e-08 m`；稀疏/分窗 `2.54396e-02 m`。
- 3000 帧：稀疏/分窗 `4.81505e-02 m`。
- 30000 帧：稀疏/分窗 `4.31428e-02 m`。

全部低于声明的 `5e-02 m` 一致性容差。3000/30000 帧的 `full_segment` 因预算超限被明确标记为不可用，返回 `E_STRATEGY_BUDGET`，未静默运行。

策略执行报告已写入每档 `sizes_report[].strategy_execution`，包含窗口/锚点边界帧、根接缝、最大相邻姿态跳变和连续性判定。三档 `boundary_continuity_ok` 均为真：

| 帧数 | 边界数 | 根接缝最大值 | 最大相邻姿态跳变 |
|---:|---:|---:|---:|
| 300 | 0 | `0 m` | `0 m` |
| 3000 | 20 | `1.90639e-02 m` | `5.78538e-02 m` |
| 30000 | 15000 | `1.42710e-02 m` | `8.56182e-02 m` |

孤立帧全相机不可见的关节补全已加入时间方向先验。修复前 30000 帧融合姿态曾出现约 `4.78e-01 m` 的单帧隐藏关节跳变；修复后三档融合姿态最大相邻跳变分别约 `6.29e-02 m`、`6.48e-02 m`、`8.12e-02 m`，隐藏关节骨长误差保持在 `1e-10 m` 以下。

## 增量闭包

300 帧增量演示结果：

| 编辑 | 闭包 | 复用 | 闭包=实算 |
|---|---|---|---|
| 单帧根高度约束 | `constraints: 1` | 其他 1199 个阶段帧 | 是 |
| 相机观测权重 | `fusion: 1, retarget: 1, strategy: 192, constraints: 192` | `fusion: 299, retarget: 299, strategy: 108, constraints: 108` | 是 |
| 目标左膝骨长 | `retarget: 300, strategy: 300, constraints: 300` | `fusion: 300` | 是 |

## 错误码链路

`demo_report.json` 与 `python -m mocap_workbench errors` 均实际触发：

- `E_OBS_MISSING` → 退出码 `10`
- `E_RETARGET_TOL` → 退出码 `11`
- `E_CONSTRAINT_UNSAT` → 退出码 `12`
- `E_STRATEGY_BUDGET` → 退出码 `13`
- `E_CLOSURE_MISMATCH` → 退出码 `14`

注：以上为最终代码版本的双跑摘要；若修改种子、预算或算法参数，摘要会按设计变化。
