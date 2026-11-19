# 数字乐谱演奏表达数据处理管线

本地确定性生成多声部乐谱与演奏表达事件，将逻辑拍演进为毫秒级演奏时间轴，
支持版本化增量重演、提交中断恢复与独立逐音符验证。无网络、无容器、无音频设备，
仅依赖 Python ≥ 3.10 与 NumPy（版本固定在 `requirements.txt`）。

## 快速开始

```bash
python -m pip install -r requirements.txt

# 一条命令：生成 -> 合成 -> 三次增量修改 -> 提交中途强制终止 ->
# 孤儿回收/恢复续写 -> 受控退出 -> 独立验证 -> 错误码链路 -> 双跑逐字节比对
python run_demo.py --workspace workspace_demo --repeat-check

# 单元测试（24 个，覆盖确定性、闭包、提交协议、恢复、错误码、验证器）
python -m unittest discover -s tests -v
```

退出码 0 且 JSON 中 `overall_ok=true`、`repeat_check.report_hashes_equal=true` 即验收通过。

## 工作区产物

```
workspace_demo/
  sample/
    score.json                 # 确定性乐谱与表达事件
    truth_P00.json ...         # 逐声部逻辑时间轴真值清单
    generation_report.json     # 声部数/事件总数/拍长口径/乐谱哈希
  state/
    manifest.json              # 当前稳定版本戳与分片哈希表
    operations.jsonl           # 版本化操作序列
    blobs/ closures/ orphans/ tmp/
    CLOSED, exit_report.json
  reports/
    demo_report.json           # 固定序列完整报告
    demo_report.sha256.json    # 报告内容哈希
    repeat_check.json          # 双跑逐字节比对结果
    recovery.json, worker_run.json, error_scenarios.json
workspace_demo_repeat/         # 第二次独立运行（用于哈希比对）
```

## 固定演示序列

| 版本 | 操作 | 期望闭包行为 |
| --- | --- | --- |
| v0 | 基线整段合成 | 全量分片 |
| v1 | 修改速度曲线两个关键点 | 首个分歧拍起全部音符重算，之前仅平移 |
| v2 | 修改力度区间渐弱终点 | 仅与区间闭区间相交的音符重算（提交写 blob 后强制终止） |
| v3 | 延长一个踏板区间 | 仅相交音符重算，踏板毫秒区间重渲染 |

崩溃后重启只恢复到 v1 稳定状态，v2 未提交 blob 进入 `orphans/`，
退出报告状态标注为 `missing_forced_termination`，随后续写 v2、v3，
最终与无中断参考线逐音符一致。

## 业务错误码

`E_BEAT_RANGE`、`E_VELOCITY_MONO`、`E_PEDAL_OVERLAP`、`E_VERSION_REGRESS`、
`E_SNAPSHOT_STALE`、`E_TIMELINE_HASH`，语义与链路见 `DESIGN.md` 第 7 节。
CLI 业务错误以 JSON 写 stderr 并返回退出码 2；强制终止返回 137。

## 独立验证

```python
from score_pipeline.validator import validate_workspace
report = validate_workspace("workspace_demo")  # 只读取文件，不使用引擎内存状态
```

验证器自带速度积分与力度包络实现，逐音符复核并在 `first_failure` 定位首个不一致音符。

## 文档

设计细节、哈希口径、提交协议、闭包语义与 Windows 说明见 `DESIGN.md`。
