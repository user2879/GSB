# 实际验证记录

验证日期：2026-11-20。验证环境为工作区提供的 Linux 环境：Python 3.11.0rc1、NumPy 2.4.6。实现仅使用 Python 标准库与 NumPy，阶段提交使用同文件系统临时文件 + `os.replace`，未使用容器、GPU、外部服务或网络访问。Windows 10/Server 上可使用同一入口和跨平台 API。

## 1. 固定样本

生成命令：

```bash
python run.py generate --output data --seed 20261120
```

实际结果：

- 有效物品：72；有效交互：507；用户：48；冷启动用户：3。
- 非法规格记录：11 条，均为 `E_RECORD_INVALID`；合法记录继续进入实验。
- 样本确定性摘要：`1505e11bca0dc2a0310b289eac9b134c5705eae5defa0dcade556838e480f898`。
- 配置哈希：`a8c2af126b9dc33e4ac7276f6b43b9890a3a6b5bb927c264034ab9f487fbad9f`。
- 样本包含 3 个无历史冷启动用户、52 号以后长尾物品，以及 4 个季节物品按固定 14 天周期产生的爆发事件。

两次独立生成得到相同的样本确定性摘要、配置哈希和各输入文件 SHA-256；自动化测试对此进行断言。

## 2. 自动化测试

执行：

```bash
python -m unittest discover -s tests -v
```

实际结果：`Ran 8 tests in 2.882s`，`OK`，8 项通过、0 项失败。覆盖：

- 固定样本可复现与非法记录保留；
- 配置非法字段生成 `E_CONFIG_INVALID`；
- 定点半偶舍入和整数量化；
- 三路召回排序、来源名次、同分规则与冷启动；
- 冷启动零候选权重重分配；
- 双位宽四环节误差、两类排序翻转归因、评估真实变化；
- 全量样例下并发/串行阶段载荷逐字段一致；
- 中断后从阶段快照续跑并得到一致结果；
- `os.replace` 提交后 JSON 完整、临时目录清理、配置/版本不匹配拒绝续跑。

## 3. 双位宽实验

执行：

```bash
python run.py run --input data --run-dir runs/verification --version 1 --report-dir reports/verification
```

配置为主位宽 5 位、参考位宽 9 位，量化步长分别为 `0.03125` 与 `0.001953125`，容差 `0.01`，`half_even` 舍入，最大并发 4，快照间隔 5。

实际执行结果：

- 48 个用户全部完成；召回、融合、评估每阶段提交 10 次。
- 冷启动用户 3 个，零候选权重重分配后热度通道有效权重为 `1.0`。
- 逐级误差：

| 环节 | 最大差值 | 超过容差候选数 | 首次超差计数 |
|---|---:|---:|---:|
| normalization | 0.015625 | 520 | 520 |
| channel_weighted_sum | 0.05078125 | 546 | 311 |
| feature_add | 0.0703125 | 820 | 308 |
| clipping | 0.056640625 | 744 | 0 |

- 47 个用户出现排序位置变化，共 518 个位置变化。
- `normalization_residual`：142 个；`feature_rounding_accumulation`：376 个。两类归因均由实际候选产生。
- 边界敏感候选清单共 1529 条，完整记录在报告 JSON。

真值评估使用各自位宽的实际排序结果，而不是替换显示字段：

| 位宽 | 命中率 | Precision@K | 平均位置偏差 | 平均 Top-K 重合 | 未命中真值候选 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.981771 | 0.981771 | 0.516857 | 7.854167 | 7 |
| 9 | 0.997396 | 0.997396 | 0.045078 | 7.979167 | 1 |

两档位评估摘要确不相同。稳定结果摘要：

```text
3288c9d3f797172789f58cee27246d1828c22e824da97fc9e99839f770bf5d0d
```

完整结果：`reports/verification/experiment_report.md`、`reports/verification/experiment_report.json`。

## 4. 并发与串行等价

串行命令：

```bash
python run.py run --input data --run-dir runs/verification-serial --version 1 --serial --report-dir reports/verification-serial
```

实际结果：

- 串行和线程池均以状态 `completed`、退出码 `0` 完成。
- 两者 `result_digest` 均为 `3288c9d3f797172789f58cee27246d1828c22e824da97fc9e99839f770bf5d0d`。
- `recall`、`fusion`、`evaluation` 三阶段完整载荷逐字段相等：`True`、`True`、`True`。
- 评估摘要逐字段相等：`True`。
- 并发运行每个阶段 `max_workers_used=4`，未超过配置上限。

## 5. 信号中断与快照恢复

实际运行中对进程发送 `SIGINT`：

- 进程打印 `received signal 2`，完成当前计算窗口及阶段提交后退出。
- 退出码：`0`。
- 运行状态：`interrupted`。
- 使用同一运行目录和相同配置哈希重新执行后状态为 `completed`。
- 恢复运行的结果摘要与全量基准相同：`3288c9d3f797172789f58cee27246d1828c22e824da97fc9e99839f770bf5d0d`。
- `recall`、`fusion`、`evaluation` 恢复后载荷与全量基准逐字段相等：`True`、`True`、`True`。

这说明信号没有直接杀死进程，而是到达可处理点完成当前阶段提交；重启只补算未提交用户。

## 6. 配置与版本一致性拒绝

修改输入配置 `seed` 后复用原运行目录：

```text
exit code = 2
code = E_CONFIG_MISMATCH
message = snapshot configuration hash differs from current configuration
```

配置不变、仅请求版本从 1 改为 2：

```text
exit code = 2
code = E_CONFIG_MISMATCH
message = snapshot run version differs from requested version
```

两次均未覆盖既有阶段产物，也未续跑任何用户。

## 7. 证据文件

- 主报告：`reports/verification/experiment_report.json`
- 人可读报告：`reports/verification/experiment_report.md`
- 机器可读验证摘要：`reports/verification/verification_evidence.json`
- 完整阶段快照示例：`runs/verification/`
- 样本清单：`data/sample_manifest.json`

验证摘要记录并发/串行/恢复摘要、阶段载荷等价性、测试数量、业务码拒绝结果和环境版本。
