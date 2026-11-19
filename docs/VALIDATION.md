# 验收记录

## 环境

- Python 3.11.0rc1（代码要求 Python >= 3.10）
- numpy 1.26.4、scipy 1.11.4、PyYAML 6.0.1、PyQt5 5.15.10、matplotlib 3.8.4
- Linux 容器内完成开发与自动测试；代码仅使用 Python 标准库的 WAV/路径/原子替换能力，并提供 Windows 的 `.cmd` 与 `.ps1` 入口。

## 自动测试

命令：`python3 -m pytest -q`

结果：`15 passed`。覆盖：

- 实球谐正交归一与 ACN 序号；
- Fibonacci 球面等面积与确定性；
- 48 kHz/16 bit/4 通道 WAV、真值方向与到达时间；
- 两次生成 WAV 与 manifest 逐字节一致；
- 低置信方向显式排除、事件计数守恒；
- 三档策略分化、内存预算与预算变化的局部影响；
- 跨档逐方向首个分歧及原因；
- 两次完整运行目录逐字节一致；
- 快照内容哈希复算、篡改检测、孤儿回收；
- 视图拒绝被修改的快照、PNG 完整性与 Qt 离屏导出。

## 默认三档实测

在默认配置下实测：

| 档位 | 策略 | 阶数 | 低置信方向 | 高置信方向 | 相对误差 p50 / p95 / max | 超差比例 |
|---|---|---:|---:|---:|---|---:|
| small | coarse_to_fine | 3 | 2 | 30 | 0.441974564229 / 1.905967167801 / 5.553854935368 | 12.84% |
| medium | full_least_squares | 6 | 35 | 63 | 0.304193788521 / 1.007887200674 / 4.777106044252 | 5.09% |
| large | clustered_representatives | 8 | 86 | 114 | 0.2902745879 / 1.0 / 3.476415565884 | 2.11% |

显著性口径为参考峰值 5%，逐方向相对容差为 1.0。小档因 32 点采样与 3 阶截断出现最高超差比例；大档在聚类后仍满足内存和 10 度分辨率预算。

跨档逐方向比较在默认容差 1.0、显著参考能量门槛 5% 下判定为不一致，报告准确定位到首个分歧方向、主导 ACN 分量与原因；small/medium、small/large、medium/large 三对档位的显著方向分歧比例分别为 `27.95%`、`30.93%`、`18.15%`。这是“不同源数/采样密度/阶数/策略”的真实可观测差异，没有用总体误差掩盖。

## 哈希与复算

当前默认运行版本：`sd1-41b50553ee9dd6ed`；快照内容哈希：
`41b50553ee9dd6ed10d0feacb503c704beae4c94fe699737cd2493c86afe6c5d`。
`verify` 实测复算 133 个产物，0 个不一致。相同输入在同一工作区第二次完整运行后，
141 个工作区文件逐一比较 SHA-256，结果完全一致。两个独立工作区生成的 140 个
产物文件也逐一相同（工作区中原有的 `workspace.json` 不属于生成产物）。

一次完整运行会输出：

- `artifacts/CURRENT.json`：最新已提交版本和内容哈希；
- `artifacts/runs/<version>/snapshot.json`：不可变快照；
- `artifacts/runs/<version>/reports/direction-errors.jsonl`：逐方向误差清单；
- `artifacts/runs/<version>/reports/step-hashes.json`：生成、投影、策略、核验、一致性、系数与清单哈希；
- `artifacts/views/<version>/{small,medium,large}.png`：等距柱状投影诊断图；
- `artifacts/latest-report.json`：汇总报告。

复算命令：`python -m spatial_diag verify --workspace artifacts`。该命令重新计算快照内容哈希及所有样本/清单/报告文件 SHA-256，并报告缺失或篡改项。
