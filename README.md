# 遥感多波段辐射定标与跨传感器回归对齐组件

确定性、可增量、可快照恢复的遥感影像预处理组件。详细设计见 `docs_design.md`。

## 环境

- Python 3.11+，NumPy 1.26（可离线 wheel 安装）
- 无网络、无容器、无外部数据集

## 快速开始

```bash
pip install -r requirements.txt
bash scripts/run_demo.sh
```

或分步执行：

```bash
python3 -m rs_calib.cli generate --out sample_data --seed 20261122
python3 -m rs_calib.cli run --sample-dir sample_data --output-dir output/inc --mode incremental
python3 -m rs_calib.cli run --sample-dir sample_data --output-dir output/full --mode full
python3 -m rs_calib.cli compare output/inc/diagnostic_report_incremental.json output/full/diagnostic_report_full.json
python3 -m rs_calib.cli run --sample-dir sample_data --output-dir output/inj --mode incremental \
    --injection sample_data/injection_config.json
```

## 测试

```bash
python3 -m unittest discover -s tests -v
```

## 输入样本（固定种子生成于 `sample_data/`）

- `tiles_batch*.jsonl`：瓦片包（瓦片 ID、场景 ID、传感器、波段、像素计数、饱和标记、毫秒时间戳）
- `calibration_table.json`：逐波段增益/偏移、量化位宽、饱和上限、参考时刻
- `regression_config.yaml`：特征阶数、源白名单、初始权重、正则、定点位宽、预算、并发、快照间隔
- `edit_plan.json`：瓦片追加、计数回填、传感器系数与回归配置变更
- `injection_config.json`：五类异常注入样例

## 诊断报告字段

`diagnostic_report_*.json` 含定点物理量、回归模型、对齐值与残差、失效复用清单、预算账本、
失败清单、注入命中、快照事件与内容指纹集合。
