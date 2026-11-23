# 多光谱瓦片地物分类工作台 (MSTC)

面向本地确定性样本的逐瓦片多光谱地物分类与跨瓦片重叠带加权投票融合工作台。
纯本地、无网络、无容器；样本由固定种子程序生成，结果逐像素可重复。

## 环境准备

```bash
python --version            # 需要 >= 3.10
python -m pip install numpy  # 建议 >= 1.24（另需 PyYAML）
```

## 目录结构

```
mstc/        源码包（定点引擎、加载校验、分类、融合、后处理、缓存、调度、诊断、注入、CLI）
tests/       47 个 unittest 测试
configs/     YAML 配置（Windows 风格相对路径）与故障注入清单
demo/        可重复演示脚本
docs/        架构与验证说明
reports/     验收诊断报告与注入报告
```

## 快速开始

```bash
# 1) 生成确定性样本（在指定工作目录）
python -m mstc.cli generate --root <workspace>

# 2) 全流程运行并输出诊断报告
python -m mstc.cli run --root <workspace> --config config.yaml

# 3) 故障注入套件
python -m mstc.cli inject --root <workspace> --config config.yaml \
       --injections injections.json --report reports/inj.json

# 一键演示（增量/全量一致性、预算、快照、策略切换、注入）
python demo/run_demo.py

# 全量测试
python -m unittest discover -s tests -p 'test_*.py'
```

## 核心能力

- **定点确定性**：NDVI/NDWI/纹理/坡度与四类打分全部为 64 位整数定点，记录量化残差与舍入方向，
  量化档位切换真实改变内部整数与得分。
- **中心区域归属**：窗口仅拥有互不重叠的中心区域，边界像素不重复计数。
- **重叠带融合**：FV1（几何距离主导）/FV2（概率置信度主导）两套真实不同加权规则，平票字典序决断。
- **精确增量失效**：瓦片缓存键与融合依赖指纹驱动，仅重算受影响瓦片及其融合带。
- **三重预算调度**：全局像素预算、单瓦片配额、融合批次上限；线程池并发与串行逐像素一致。
- **原子快照恢复**：阶段临时目录整目录替换，版本/配置哈希陈旧判定 `E_SNAPSHOT_STALE`。
- **边界诊断与注入隔离**：双精度参考投票检测 `E_BOUNDARY_FLIP`，六类真实输入故障隔离并映射业务码。

详见 `docs/architecture.md` 与 `docs/verification.md`。
