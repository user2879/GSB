# 演奏表达时间轴诊断工作台（MWB）

本地桌面诊断工作台：把确定性生成的数字乐谱按演奏表达参数展开为整数 tick 时间轴，
支持编辑脚本逐步增量重演、依赖失效重算、分块持久化、中断恢复、并发资源上限，
以及只读的 tkinter 领域可视化。全程不依赖真实乐器、音频设备或外部资源。

## 环境准备

- Python 3.10+（开发验证于 Python 3.11）
- 安装依赖：

```bash
python -m pip install -r requirements.txt
```

  - NumPy 1.24+ 用于分段线性插值与轨道数组。
  - tkinter 随官方 Windows 安装包提供；Debian/Ubuntu 若缺失：`sudo apt install python3-tk`。
- 不使用 Docker 或容器。

## 三步上手

```bash
# 1. 生成确定性本地样本（samples/ 下三份 JSON）
python -m mwb.cli generate --samples-dir samples

# 2. 展开 + 增量重演 + 中断恢复 + 全量一致性比对，输出诊断报告
python -m mwb.cli run --state-dir state --output-dir output

# 3. 启动只读桌面视图
python -m mwb.cli gui --output-dir output
```

退出码约定：`0` 正常完成；`2` 可处理终止（已在块边界落盘中断标记，可直接重跑恢复）；
`1` 业务错误（stderr 输出带 `E_xxx` 业务码的 JSON，不产生持久化输出）。

## 输出物

- `state/<version_id>/manifest.json`：版本标识、步游标、块索引、依赖与摘要。
- `state/<version_id>/chunks/chunk-XXXX.json`：块表达状态、依赖键、内容摘要。
- `state/<version_id>/interrupted.json`：可处理终止标记。
- `output/timeline.json`：完整展开时间轴（逐 tick 速度/踏板/音符状态）。
- `output/diagnostic_report.json`：跨声部重叠、逐步失效范围、容差比对、摘要结论。

## 信号与恢复

```bash
# 外部发送可处理终止信号，任务在块边界退出（码 2）
kill -TERM <pid>
# 重跑同一命令即按版本标识复用有效块、只补算缺失/失效块
python -m mwb.cli run
```

## 测试

```bash
python -m pytest -q
```

覆盖解析校验与全部业务码、tick 展开确定性、表达状态逐 tick 查询、
增量与全量逐 tick 一致、并发度等价、内存预算峰值、版本隔离、
摘要损坏自愈、孤儿块隔离、可处理终止恢复，以及 GUI 只读数据流。

## 文档

- 设计与业务码细节见 `docs/DESIGN.md`。
- 本次实际验证记录见 `docs/VERIFICATION.md`。
