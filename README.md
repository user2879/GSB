# 竖排古籍关键点定位诊断工作台

本地、确定性、可重复演示的竖排版面识别诊断系统。所有样本均由固定整数种子合成，PNG 编解码、滑窗、连通域、定点回归、阅读顺序、租约、缓存、调度、快照和 Tkinter 界面均在仓库内完成。

原始任务书保留在 `docs/original_requirement.md`，详细实现映射见 `docs/TECHNICAL_NOTES.md`，最新真实验证结果写入 `reports/acceptance_report.json` 与 `reports/verification.md`。

## 快速开始

标准 WSL Ubuntu 环境建议先执行：

```bash
sudo apt-get update
sudo apt-get install -y python3-tk python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pytest
pip install -e .
```

生成样本、运行定位、启动界面：

```bash
python -m vertical_workbench.cli init --root .
python -m vertical_workbench.cli run --root .
bash scripts/run_gui.sh
```

一键验收：

```bash
bash scripts/run_acceptance.sh
```

常用命令：

```bash
python scripts/check_environment.py
python -m vertical_workbench.cli inject --root .
.venv/bin/python -m pytest -q
```

## 目录

- `src/vertical_workbench/`：核心包。
- `samples/`：确定性 PNG 与 `specs.json`。
- `truth/`：整数关键点、bbox、栏区间和阅读顺序真值。
- `annotations/keypoint_versions.json`：可整体替换的标注员关键点版本。
- `configs/workbench.json`：滑窗、量化、并发、租约和预算配置。
- `configs/injections/`：四类独立注入配置。
- `ledger/ledger.jsonl`：编辑命令与运行账本。
- `snapshots/current`：指向当前阶段化整体快照的原子符号链接。
- `reports/`：验收、注入和 pytest 报告。
- `tests/`：一致性、租约、预算、注入、冲突和 GUI 冒烟测试。

## 核心能力

- 纯标准库 `zlib/struct` + NumPy 实现 8-bit 灰度 PNG 编码/解码和容器校验。
- 固定 `512x768` 版面，从右到左竖排栏，栏内自上而下生成字符块。
- 内置算法点阵/笔画字模，支持确定性轻微旋转和笔画抖动。
- 滑窗墨迹密度、竖直投影谷、纹理梯度和窗口指纹派生投影评分。
- 8-连通域、同色碎片合并依据、投影谷分裂、面积/得分边界相等标记。
- `Q4/Q8/Q16` 定点回归，`coarse/normal/fine` 量化档，确定性半入取整。
- 右到左、上到下阅读串联；`跨栏 / 缺号 / 错序` 按 `3>2>1` 优先级决断并留痕。
- 图像指纹、参数指纹、量化档、栏租约代次和标注摘要共同派生阶段缓存键。
- 标注版本号与内容摘要双重判定；仅内容摘要变化的文档使对应关键点、阅读顺序及下游失效。
- 线程池执行窗口和连通域阶段，结果按 `(文件名, 栏序号)` 稳定排序回收。
- 全局点数预算、单栏配额、每批连通域比较上限三重约束和待办优先队列。
- 真实 `time.monotonic_ns()` 租约时钟；过期回收真实释放栏区间和预留预算。
- Tkinter 界面展示租约、计数、冲突、失效/复用、待办和业务码，并支持参数/版本编辑。
- 快照先写临时目录，再原子发布目录并替换 `current` 符号链接。

## 一致性口径

验收中的“空白全量重放”会创建两个空目录，从同一种子生成样本，并逐条施加相同编辑命令流。两边使用相同的确定性逻辑时钟，逐字段比较完整结果，包括关键点坐标、定点中间值、阅读序号、冲突记录、业务码、缓存键、复用/重算清单、失败清单、待办队列、预算和租约代次。

另有测试覆盖 3 个随机种子的命令序列、`max_workers=1/2/4`、输入顺序聚合、租约过期后代次抬升，以及真实单调时钟回收。

## 本环境的 Tk 说明

当前交付环境的 sudo 需要密码，且系统缺少 `ensurepip`、`libtk8.6.so`、`libtcl8.6.so`、`libXft.so.2` 和 `libXss.so.1`。已实际采用免 root 回退：

```bash
python3 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/python -m pip install numpy pytest -e .

mkdir -p /tmp/tkdebs && cd /tmp/tkdebs
apt-get download libtk8.6 libtcl8.6 tcl8.6 tk8.6 libxft2 libxss1
for deb in *.deb; do dpkg-deb -x "$deb" "$HOME/.local/tk-local"; done
```

`scripts/run_gui.sh` 会自动设置 `LD_LIBRARY_PATH/TCL_LIBRARY/TK_LIBRARY`；`vertical_workbench.gui` 也会尝试用 `ctypes` 预加载本地库。标准 Ubuntu 上直接 `sudo apt-get install -y python3-tk` 即可，不需要此回退。
