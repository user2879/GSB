# 生成式图像编辑可视化与诊断工作台

面向本地确定性样本的多阶段图像编辑管线，用于在数据规模连续变化时诊断算法路径
取舍、缓存复用范围与预算约束之间的相互作用。全部样本由固定种子程序生成，不
依赖外部下载或真实账号。

## 环境准备

要求 Python 3.11+（安装时勾选加入 PATH）：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy pillow pyyaml
```

Linux/macOS：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install numpy pillow pyyaml
```

## 运行

```bash
# 全量重建并附带独立快照的串行一致性复核
python -m workbench.cli --config config/workbench.yaml --reset --serial-check

# 增量续跑（默认并行，并发数取自配置 workers）
python -m workbench.cli --config config/workbench.yaml

# 串行模式
python -m workbench.cli --config config/workbench.yaml --mode serial

# 测试套件（仅使用标准库 unittest）
python -m unittest discover -s tests -v
```

产物位置：

- `samples/<tier>/`：确定性 PNG 与编辑指令 JSON、档位索引。
- `snapshots/`：内容寻址对象、临时目录与 `manifest.json`。
- `outputs/<run_id>/`：`report.json`、`dashboard.png`、`sample_strip.png`、
  `finals/` 与 `index.html`。

## 六阶段管线与可选路径

| 阶段 | 路径 A | 路径 B |
| --- | --- | --- |
| 掩码生成 | 阈值连通（颜色距离 + 连通域标记） | 边缘追踪（梯度闭合轮廓 + 填充） |
| 修复填充 | 邻域加权均值（定点、固定迭代） | 方向性插值（水平/垂直 1D 定点插值） |
| 风格映射 | 量化 LUT 逐像素查表 | 分通道定点线性变换 |

属性替换使用属性值与配置指纹派生的确定性重着色 LUT；合成按掩码保留策略在掩码内
取编辑结果、掩码外取源图。

## 关键保证

- 同种子样本字节级可复现；同路径同参数掩码逐像素一致。
- 阶段缓存键由上游内容/参数/量化指纹链式派生，指令修改、单图替换、量化切换与
  档位收窄/扩张只失效受影响阶段，其余精确复用。
- 增量运行与空白工作区全量重建逐像素一致；并行与串行逐字段、逐 PNG 字节一致。
- 快照原子替换、版本比对续跑、`E_CONFIG_MISMATCH` 拒绝不一致续跑。
- SIGINT/SIGTERM 在样本/阶段边界完成提交后以退出码 0 退出并保留进度。
- 业务错误码 `E_CONFIG_INVALID` / `E_SAMPLE_INVALID` /
  `E_REGION_OUT_OF_BOUND` / `E_CONFIG_MISMATCH`，单样本失败隔离不影响其余样本。

## 文档

详细设计、缓存键定义、定点规则、策略评分公式与验收结果见
[`docs/TECHNICAL.md`](docs/TECHNICAL.md)。
