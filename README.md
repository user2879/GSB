# 纹理与材质定点渲染数据管线

本地、离线、确定性的程序化纹理 / 材质处理与数值可信度诊断工具。所有样本、真值与
期望均由固定种子程序生成，不访问任何外部服务或网络。

## 环境要求

- Python 3.10+（实测 3.11）
- `numpy`、`Pillow`（见 `requirements.txt`）

```bash
python3 -m pip install -r requirements.txt
```

## 目录结构

```
texmat_pipeline/   管线源码（定点、噪声、加载、阶段、缓存、调度、快照、编排）
tools/             固定种子样本生成器与验收演示脚本
tests/             32 个 unittest 用例，覆盖全部 8 条业务要求
outputs/           演示运行产物与报告
workdata/          手工 CLI 运行的工作区
docs_architecture.md  架构与技术说明
```

## 快速开始

```bash
# 1. 生成多档分辨率、多噪声层的确定性样本
python3 tools/generate_samples.py --out workdata/sample --tiers 16 32 64 --layers 4

# 2. 校验并查看解析后的输入
python3 -m texmat_pipeline.cli inspect --input workdata/sample

# 3. 全量运行（线程池 + 预算 + 快照 + 双档误差报告）
python3 -m texmat_pipeline.cli run --input workdata/sample --work workdata/run --resolution 32

# 再次运行即增量续跑；--fresh 强制全量重建，--serial 强制单线程调度
python3 -m texmat_pipeline.cli run --input workdata/sample --work workdata/run --resolution 32

# 4. 一键验收演示（编辑序列、预算耗尽、快照恢复、运行报告）
python3 tools/run_demo.py --workspace outputs/demo --resolution 32
```

## 输入约定

- `*.png`：固定分辨率方形基础纹理，支持 `L/RGB/RGBA`，文件名含分辨率后缀；
- `materials.json`：每个材质的四通道层权重，通道权重和必须为正；
- `generation_config.json`：分辨率档、噪声层、混合模式、定点位宽与量化档位、
  舍入模式、线程数、单批点数预算、快照间隔。

## 输出产物

快照版本目录（原子整体替换）包含：

- `noise/layer_*.npy`：逐层定点噪声图像；
- `channels/<material>/*.npy|png`：四通道定点数据与 uint8 预览；
- `report.json`：通道均值、边界样本、误差归因、复用/重算计数、预算序列、失败码；
- `manifest.json`：全局版本、配置哈希、位宽档位、批次游标、产物指纹。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

详细设计与逐条业务要求映射见 `docs_architecture.md`，实测结果见
`outputs/VALIDATION_REPORT.md`。
