# 多传感器时序预测 · 可重复演示工作台（tspworkbench）

面向多传感器等间隔时序的回测预测工作台，核心解决 **预测配置与结果在 v1（定宽文本）/
v2（JSON）两种交换版本之间流转时的互操作、精度损失与增量失效判定**，并保证任意编辑后
增量运行与从空白全量重建 **逐字段一致**。全部产物由固定种子在现场生成，无外部下载、无容器。

## 环境准备

```bash
python --version            # 需 Python 3.11
python -m pip install -r requirements.txt
python -c "import numpy, yaml; print(numpy.__version__)"
```

实测环境：Python 3.11、numpy 2.2.6、PyYAML 可用。Windows PowerShell / cmd 下命令等价
（`python -m tspworkbench.cli ...` 跨平台，路径用 `pathlib`，临时文件以 `os.replace` 原子替换）。

## 一分钟快速演示

```bash
python -m tspworkbench.cli generate --sample-dir data      # 固定种子生成样本/配置/注入
python -m tspworkbench.cli run --sample-dir data --work-dir work --fresh   # 空白全量
python -m tspworkbench.cli run --sample-dir data --work-dir work           # 增量（全 reused）
python -m tspworkbench.cli run --sample-dir data --work-dir work-inj --fresh --inject  # 故障注入
python -m tspworkbench.cli convert work/exchange/exchange.v1.txt work/exchange/up.v2.json
python -m tspworkbench.demo        # 9 步编辑序列 + 并发/串行 + 预算专项，全量一致性验证
```

## 目录结构

```
tspworkbench/
  config.py        配置加载/归一化/枚举与非负阶数校验/配置指纹
  sample.py        固定种子确定性样本生成与 JSONL 读写（SeedSequence 派生，无全局随机）
  windows.py       回测窗口切分、等间隔/非有限校验、确定性插补与位置标注
  models.py        naive / moving_average / autoregressive 三模型统一 fit/predict 接口
  quantize.py      浮点->定点整数量化与 MAE/RMSE/MAPE（近零绝对误差分支）
  cache.py         内容寻址缓存、分层缓存键与复用账本
  engine.py        逐窗口训练/预测/指标引擎（命中复用、未命中计算）
  runner.py        线程池调度、全局/单序列预算、失败清单、快照原子提交与续跑判定
  inject.py        独立故障注入配置与业务码映射
  report.py        诊断报告
  demo.py          编辑序列 + 增量/全量逐字段一致性编排
  cli.py           命令行入口
  exchange/
    ir.py          统一中间表示与硬校验（唯一键/非负阶数/模型枚举）
    v1_codec.py    v1 定宽文本解析与写入（E_FIELD_LAYOUT）
    v2_codec.py    v2 JSON 解析与写入（浮点权重/精度标签/命名模型/命名缺失策略）
    converter.py   v1<->v2 双向转换与版本指纹
    sidecar.py     降级旁路（截断残差/精度标签/命名缺失策略/双指纹）
tests/             14 个单元与集成测试
docs/TECHNICAL.md  设计与验证技术说明
```

## 关键设计如何对应业务要求

- **统一表示与解析校验**：v1 按 `constants.V1_LAYOUT` 定宽切片，长度不符/字段非整数即
  `E_FIELD_LAYOUT` 进失败清单；IR 校验 `(序列,窗口[,模型])` 唯一、滞后阶数非负、模型名枚举。
- **双向转换与旁路**：升版千分比 `/1000` 精确换算并打 `permille-exact` 标签、模型码映射命名模型；
  降版四舍五入回千分比，残差/精度标签/命名缺失策略写入 `*.sidecar.json`，旁路带配置指纹与
  版本指纹；再次升版据旁路还原原始浮点与命名信息，而非默认值。
- **确定性训练**：三模型同接口，显式记录训练窗口、拟合参数与残差序列；MA 窗长与 AR 滞后均取自
  IR/配置；插补只用当前窗口已见值并标注位置；不依赖运行时全局随机状态。
- **定点量化指标**：先把预测/真值真实量化为整数（`scale=2^(frac_bits+gear)`），再在整数域算
  MAE/RMSE/MAPE，累加顺序固定为窗口升序、步序升序；真值量化为 0 时走绝对误差分支并标注。
- **指纹失效与账本**：`window -> train -> predict -> metric` 分层键由上游内容指纹、参数指纹与
  交换版本号派生；指标记录其引用的窗口与模型集合；上游变更只令受影响窗口及下游失配重算，
  未失效项直接复用且不重算。
- **预算与快照**：线程池并发数取配置且不超过；全局剩余预算 + 按序列 ID 稳定顺序的单序列配额，
  超配额记 `E_SERIES_BUDGET_EXHAUSTED` 并保留待办游标；产物先写临时文件再整体替换，清单记录
  全局版本号、配置哈希、交换版本指纹与窗口游标，启动据此判定续跑。
- **稳定排序与注入隔离**：每任务带 `序列ID|窗口序号|阶段` 稳定排序键，最终顺序与并发度/调度时机
  无关；四类真实输入故障（时间戳非等间隔、观测非有限、窗口超长、量化档位越界）按注入编号映射到
  对应业务码并写入诊断报告。

## 验证结果

- `python -m unittest discover -s tests`：14/14 通过。
- `python -m tspworkbench.demo`：9 步编辑序列（版本切换×2、序列替换/增删、窗口与步长、模型超参、
  量化档位）每步增量与空白全量逐字段一致；并发/串行一致；预算耗尽业务码正确。完整结果见
  `demo_out/demo_report.json`，技术说明见 `docs/TECHNICAL.md`。
