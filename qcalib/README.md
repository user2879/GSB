# qcalib — 逐组量化标定与饱和截断领域组件

面向模型部署前的**逐列组**分位标定、零点推导、定点重缩放、饱和截断与反量化
误差归因。仅依赖 Python 3.10+ 标准库与 NumPy（NumPy 仅用于确定性样本生成），
核心算子全部使用 Python 整数/分数运算，保证跨平台、跨运行**逐字段一致**。

## 目录结构

```
qcalib/
├── qcalib/                 # 可集成领域组件
│   ├── errors.py           # 业务错误码与失败明细收集器
│   ├── enums.py            # 位宽/方案/舍入/饱和枚举、v1 定宽字段宽度
│   ├── fingerprint.py      # 内容指纹、缓存键、版本标识派生
│   ├── config.py           # 特征表 JSONL + 逐组配置加载校验
│   ├── exchange.py         # v1 定宽 / v2 JSON 解析、IR、双向转换、旁路
│   ├── calibration.py      # 分位标定、缩放因子与零点推导
│   ├── kernel.py           # 定点重缩放、饱和截断、越界累加
│   ├── attribution.py      # 反量化误差归因与分层敏感性
│   ├── store.py            # 内容寻址缓存与精确失效
│   ├── snapshot.py         # 临时目录写入 + 原子整体替换
│   ├── pipeline.py         # 预算调度、线程池、增量/全量统一管线
│   └── runner.py           # 端到端编排与诊断报告
├── examples/
│   ├── generate_samples.py # 固定种子确定性样本（规模可配）
│   └── run_demo.py         # 覆盖全部验收路径的端到端示例
├── tests/                  # unittest 套件（23 个用例）
└── docs/
    └── TECHNICAL.md        # 设计、定点语义、指纹/失效与验证记录
```

## 快速开始

```bash
# 仅首次需要
python -m pip install numpy

cd qcalib
python examples/generate_samples.py --out samples --rows 40 --groups 3 --cols 3
python examples/run_demo.py
python -m unittest discover -s tests -v
```

样本目录与输出目录均为本地可写目录；快照与旁路先写入独立临时目录，再
`os.replace` 原子替换，不访问网络、不依赖任何真实账号。

## 两种交换版本

- **v1 定宽文本**：字段固定宽度 `group(8) column(16) permille(12) zp(8) qbits(4)`，
  行尾附加 1 字符方案码（`S` 对称 / `A` 非对称）。宽度不足/错位 → `E_FIELD_TRUNCATED`。
- **v2 JSON**：浮点 `scale`、命名方案（`symmetric`/`asymmetric`）、命名舍入模式
  （`half_up`/`half_even`/`truncate`/`floor`）与精度标签。
- 统一中间表示 `CalibEntry` 校验缩放因子为有限正数（否则 `E_SCALE_INVALID`）、
  零点落在位宽允许区间（否则 `E_ZERO_OUT_OF_RANGE`）、方案/舍入在枚举内。
- v1→v2：`permille/1000` 精确换算并生成精度标签；v2→v1：`Decimal` 四舍五入还原
  permille，截断残差/精度标签/舍入模式/逐通道标记写入可再解析的旁路 JSON。
- 转换产物与旁路均带 `v<来源>-src<来源>-<内容指纹>` 版本标识。

## 定点语义

- 标定尺度按 `fbits` 决定的有效整数档数推导，裁剪区间随
  `qbits - fbits` 收缩，因此切换 `fbits` 或 `qbits` 都会**真实改变**中间整数、
  饱和元素数与反量化值（见 `tests/test_calibration_kernel.py`）。
- 量化 `q = round(x / scale) + zp`，反量化 `x' = (q - zp) * scale`；
  越界元素按 `clip`（裁剪并记录方向与截断前定点值）或 `reject`（跳过累加）处理。
- 累加器使用 Python 整数并按可配位宽判越界，输出 `E_ACC_OVERFLOW` 与组/列/位置。

## 增量与缓存

每阶段缓存键由上游内容指纹、标定参数指纹、位宽档位与交换版本号派生。单行修正、
组划分调整或位宽切换只使受影响列组及其下游失效；未失效产物直接复用，失效/复用
清单稳定排序。增量结果与从空白全量重建的产物及快照内容指纹**逐字段一致**。

## 业务错误码

| 代码 | 触发场景 |
| --- | --- |
| `E_CONFIG_INVALID` | 配置/取值非法、分位范围/位宽参数错误 |
| `E_GROUP_UNKNOWN` | 引用未在列组划分中声明的组 |
| `E_DUPLICATE_ROW` | 特征表行标识重复 |
| `E_MISSING_COLUMN` / `E_EMPTY_GROUP` | 列不存在 / 组内列为空 |
| `E_FIELD_TRUNCATED` | v1 定宽行宽度不足或错位 |
| `E_SCALE_INVALID` / `E_ZERO_OUT_OF_RANGE` | 缩放因子非有限正 / 零点越界 |
| `E_ENUM_INVALID` | 方案、舍入、位宽枚举非法 |
| `E_ACC_OVERFLOW` | 定点累加越界 |
| `E_CALIB_BUDGET_PENDING` | 单批元素预算不足，任务进入待办队列 |
