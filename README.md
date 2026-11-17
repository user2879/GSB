# 确定性合成时序预测与诊断工作台（tsbench）

面向本地、无网络依赖的合成时序数据，完成 **预处理 → 滞后特征 → 确定性模型拟合 → 滚动回测**
全流程，并解决两个核心问题：

- 序列片段级增量更新（追加 / 尾部修正 / 历史区间替换）后的**精确失效判定**；
- 分批增量运行与空白工作区一次性全量运行的**逐字段结果一致性**。

## 环境准备

- Python 3.10+（本仓库在 `python3.10` 下验证；系统默认 `python3` 为 3.11 且未装 NumPy）。
- 依赖：`numpy`、`PyYAML`（见 `requirements.txt`）。本机 NumPy 2.2.6、PyYAML 5.4.1。
- 运行期只读写工作区内的 `inputs/` 与 `outputs/`，不访问网络。
- 为排除 BLAS 多线程浮点调度差异，脚本固定 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。

## 目录结构

```
inputs/
  experiment.yaml            # 实验/预测配置 + 3 条确定性合成序列描述
  updates.json               # 3 批更新计划（追加/尾部修正/历史区间替换）
  signal_large.yaml          # 信号与强制终止演练用的长序列配置
  signal_updates.json
  invalid/                   # 异常样本：坏配置、缺字段、未知序列、非有限值
src/tsbench/
  config.py                  # YAML 加载、字段校验、配置哈希
  synthetic.py               # 固定种子生成、时间戳/数值校验、更新操作落地
  updates.py                 # 更新计划 JSON 加载与分批应用
  transforms.py              # 季节分解、差分、滞后/滚动特征、精确依赖区间
  model.py                   # OLS、确定性批量梯度下降、滚动回测、误差指标
  pipeline.py                # 四阶段编排 + 内容指纹缓存键 + 复用/重算审计
  state.py                   # 原子写盘、清单 manifest、恢复判定
  runner.py                  # 线程池并发、批次调度、版本号、JSON 报告
  cli.py                     # run / full / full-final / verify / recovery 子命令
  util.py, errors.py         # 规范化哈希/原子写 与 业务错误码
tests/
  test_workbench.py          # 15 个单元/集成测试
  signal_demo.py, kill_demo.py
run_demo.sh                  # 一键端到端验收
```

## 快速运行

```bash
pip install -r requirements.txt          # 仅首次
./run_demo.sh                            # 增量 → 全量 → 逐字段比对 → 恢复点
```

单独命令：

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1
python3.10 -m tsbench.cli run        --config inputs/experiment.yaml --updates inputs/updates.json --output outputs/inc
python3.10 -m tsbench.cli full-final --config inputs/experiment.yaml --updates inputs/updates.json --output outputs/full
python3.10 -m tsbench.cli verify     --full-report outputs/full/reports/report-batch-09999.json \
                                     --incremental-report outputs/inc/reports/report-batch-00003.json
python3.10 -m tsbench.cli recovery   --config inputs/experiment.yaml --output outputs/inc
```

## 关键设计

**确定性生成与无随机拟合**

- 每条序列使用独立 `np.random.default_rng(seed)`，按“趋势 + 季节 → 噪声 → 异常点”的固定运算顺序生成。
- 模型为闭式最小二乘（`np.linalg.lstsq`，标准化特征）+ 确定性小批量梯度下降；批量始终按索引升序、
  零初始化、固定学习率/轮数，精度档位 `low/standard/high` 仅决定 GD 超参，全程无随机数。

**精确失效判定（不是时间启发式）**

- 预处理为每个时间索引维护真实的原始值依赖区间：季节项的有界回看源点、中心化趋势窗口、
  以及逐层差分扩张后的 `diff_deps`。
- 逐点缓存键 = `SHA256(阶段名, 参数指纹, 索引, 该点 diff_deps 覆盖的原始值窗口)`。
  历史区间替换只改变依赖窗口覆盖该区间的点；未覆盖的点键不变、直接复用。
- 特征矩阵、每个模型折、每个回测折的缓存键均由其上游内容指纹与参数指纹逐级派生，
  失效沿“预处理 → 特征 → 模型 → 回测”精确向下传播。

**全量与增量逐字段一致的两条保障**

1. 所有浮点在**写入缓存前**统一按固定精度舍入（`rounded_float`），内存实时值与 JSON 落盘值是同一个
   Python float，缓存命中与重算走完全相同的字节；求和统一为定序 Python 求和。
2. 回测重算直接复用已持久化的模型 `beta`（而非二次拟合），消除末位漂移。
3. `concurrency`、`snapshot_interval` 属于运行控制项，不进入配置哈希与任何缓存键。

**并发与稳定排序**

- 线程池最大并发数取配置；每个任务携带由序列 ID 与阶段名派生的稳定排序键，
  汇总前按序列 ID 重排，复用/重算/失败清单全部排序后输出，与单线程结果逐字段一致。

**原子快照与恢复**

- 每个产物先写同目录临时文件再 `os.replace` 整体替换；`manifest.json` 记录全局版本号、配置哈希、
  数据指纹、已完成序列与各阶段版本。
- 启动时配置哈希不一致即返回 `E_CONFIG_MISMATCH`（退出码 2）拒绝恢复。
- SIGTERM/SIGINT 在安全检查点抛出 `GracefulExit`，完成当前已替换快照后以退出码 0 退出；
  SIGKILL 后重跑只依据完整替换的产物恢复，临时残片不参与。

**业务错误码**

| 码 | 触发场景 | 处理 |
| --- | --- | --- |
| `E_CONFIG_INVALID` | 缺字段、滞后/周期非正、未知特征规则、坏 YAML/JSON | 致命，退出码 2 |
| `E_SERIES_INVALID` | 未知序列、时间戳断裂/非等间隔、非有限值、越界替换 | 进入失败清单，运行继续 |
| `E_CONFIG_MISMATCH` | 恢复时清单配置哈希与当前配置不符 | 拒绝恢复，退出码 2 |

## 实际验证结果

**单元/集成测试：`python3.10 -m unittest tests.test_workbench` → 15/15 通过**，覆盖
种子确定性、OLS/GD 确定性、全量=增量逐字段、两步预测覆盖全部序列、历史替换的精确失效、
并发 1 vs 4 一致、稳定排序、三类错误码、恢复点与配置不匹配保护。

**端到端演示（`run_demo.sh`，输出留存于 `outputs/demo_run.txt`）**

- 三批增量的复用/重算计数：批次1 `reused=0/recomputed=606`；批次2 `reused=415/recomputed=191`；
  批次3 `reused=308/recomputed=307`，体现未受影响产物被复用、不重复拟合。
- `verify` 结果：`"identical": true, "forecast_steps_covered": 2, "series_compared": ["S-A","S-B","S-C"]`。
- 报告含全局版本号、配置/数据指纹、四阶段耗时、模型系数摘要、逐折预测与 MAE/RMSE/MAPE、复用与重算清单、失败清单。

**异常业务码**：坏配置与缺字段均返回 `E_CONFIG_INVALID`（退出码 2）；未知序列与非有限值进入
`E_SERIES_INVALID` 失败清单；改动配置指向旧状态返回 `E_CONFIG_MISMATCH`。

**信号与强制终止**：`tests/signal_demo.py` 发送 SIGTERM 后进程以退出码 0 优雅退出并打印恢复点；
`tests/kill_demo.py` 发送 SIGKILL（returncode -9）后重跑成功，仅依据完整产物恢复，失败清单为空。
