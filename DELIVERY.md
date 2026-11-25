# 候选截断与采样诊断组件

## 交付结构

- `candidate_diagnostics/config.py`：默认口径、配置合并与校验。
- `candidate_diagnostics/loader.py`：候选 JSONL、命令流逐行解析和失败清单。
- `candidate_diagnostics/pipeline.py`：定点量化、TopK、边界标记、分层配额、稳定种子抽样与打散。
- `candidate_diagnostics/formats.py`：v1 定宽行、v2 JSON、选择码表和旁路元数据往返。
- `candidate_diagnostics/fingerprinting.py`：规范 JSON、SHA-256 指纹、缓存键和稳定种子。
- `candidate_diagnostics/engine.py`：请求级、通道候选段级和序列化产物级增量账本。
- `candidate_diagnostics/concurrency.py`：有界线程池、稳定阶段键和并发度观测。
- `candidate_diagnostics/attribution.py`：高一位宽参考、四级误差和翻转原因归因。
- `candidate_diagnostics/stats_receipts.py`：旧管道文本、嵌套 JSON、CSV 三版统计回执归一与权威裁决。
- `candidate_diagnostics/scheduler.py`：打点预算、请求配额、临时目录原子提交、清单游标和续跑识别。
- `candidate_diagnostics/injections.py`：非有限分、重复 ID、比例越界、未知选择码等真实注入。
- `candidate_diagnostics/demo.py` / `demo.py`：端到端示例和诊断报告生成器。
- `samples/`：确定性候选、配置、编辑命令、注入计划、双版本结论和三份历史回执。
- `tests/`：29 个 pytest 用例。

## Windows 环境命令

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install numpy==1.26.4 pytest==8.2.2
python demo.py
python -m pytest -q
```

当前交付在 Linux 风格执行环境中验证。该镜像的 `python3.11 -m venv` 缺少 `ensurepip`，因此使用已安装的 `virtualenv -p python3.11 .venv` 创建同一虚拟环境；依赖仍严格安装为 `numpy==1.26.4` 与 `pytest==8.2.2`。

## 常用命令

```bash
.venv/bin/python scripts/generate_conclusion_samples.py
.venv/bin/python demo.py
.venv/bin/python -m pytest -q | tee artifacts/pytest-output.txt
```

Windows 下把 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`。

演示不需要启动 `127.0.0.1` 上的任何服务，所有输入和输出均为本地文件。

## 关键口径

- 归一化默认使用 `weighted_linear`：`1000 * slope * channel_weight * raw_score / raw_scale + bias`，并裁剪到 `[0,1000]`。
- 定点整数为 `round(normalized * 10**fixed_decimals / 10**gear_exponent) * 10**gear_exponent`；修改小数位或档位会改变真实整数和反量化值。
- 排序规则为定点分降序、候选 ID 升序。
- 层配额先按 `floor(layer_population * ratio)` 分配，总余数按通道名升序补齐。
- 抽样和打散使用本地 `random.Random(SHA256(business_config_fingerprint, request_id, stage, channel))`，不读取全局随机状态。
- v2 浮点得分使用 `[0,1]` 区间；v1 保存四舍五入后的千分比整数。
- 执行参数 `concurrency`、`budget` 不进入业务确定性指纹，因此调整线程数不改变候选结论。

## 失效粒度

缓存键由内容指纹、请求实际引用通道的权重指纹、采样指纹、量化指纹和版本指纹派生。账本记录：

- 每请求引用通道集合；
- 每通道候选段键和 `HIT/RECOMPUTED`；
- 每请求归一化、截断、抽样、打散键；
- v1/v2 产物键及其候选集合；
- 命中、重算和稳定请求顺序。

## 实际验证结果

- `29 passed`（完整输出保存在 `artifacts/pytest-output.txt`）。
- 演示生成 `artifacts/diagnostic-report.json` 与 `artifacts/diagnostic-report.md`。
- v2→v1→v2 往返 40 条记录，字段不匹配数为 0。
- 6 个异常注入全部命中预期业务码。
- 增量编辑终态与空白全量重建的业务字段和缓存键一致。
- 串行、1/2/3 并发度结果逐字段一致；屏障测试观测到真实 2 并发且未超过配置上限。
