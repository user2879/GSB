# 本地验收记录与命令

以下命令均在本工作区执行，不依赖容器或外部服务。

## 1. 单元与集成测试

```bash
python3 -m pytest -q
```

最近一次结果：`30 passed`。覆盖内容：

- 四类输入业务码；
- 对称/非对称量化、逐通道切换、退化回退；
- 重复运行逐元素一致；
- 温度与位宽对校准误差的影响；
- 软标签权重为零的关闭标记；
- 缓存复用与下游加权失效；
- 八类命令序列与空白全量重建逐字段一致；
- 并发度 1/4 的调度顺序与预算扣减一致；
- 点预算耗尽、交互预算中间帧；
- SIGTERM 阶段提交退出码 0；
- SIGKILL 后只恢复完整快照。

## 2. 八命令增量与全量一致性

```bash
python3 -m layerpack commands \
  --commands examples/commands-all-eight.jsonl \
  --weights samples/weights.jsonl \
  --calibration samples/calibration.json \
  --config samples/config.yaml \
  --state-dir /tmp/layerpack-acceptance \
  --reset --latency-budget-ms 0 --full-compare
```

观察到：

```json
{
  "status": "complete",
  "frame_count": 16,
  "persisted": true,
  "full_comparison": {
    "equal": true,
    "full_status": "complete"
  }
}
```

`frame_count=16` 来自八条命令各产生一个 interim 帧和一个 final 帧。增量与全量部署清单指纹相同，核心结果逐字段相同。
最近一次实测双端清单指纹均为 `941e4271c9afd7aad5f79af4e8982a461baf8c1ec7014b20619b36f3ff9229a1`。

## 3. 点预算耗尽

```bash
python3 -m layerpack run \
  --weights samples/weights.jsonl \
  --calibration samples/calibration.json \
  --config examples/config.point-budget.yaml \
  --state-dir /tmp/layerpack-budget
```

该示例把单批预算设为 `7`，严格小于样本中最小层的 8 个量化点。观察到：退出码 3、`status=pending`、业务码包含 `E_POINT_BUDGET_PENDING`，`pending_queue` 按层名稳定列出全部待算量化任务，部署清单为 `null`。预算恰好等于任务点数时允许执行，并有独立单元测试覆盖该边界。

## 4. 交互延迟预算

`examples/config.interact-budget.yaml` 将交互预算设为 0。CLI 会先输出 `interim` 帧（`persisted=false`），再等待并输出 `final` 帧（`persisted=true`）。API 用法：

```python
from layerpack.interaction import CompressionSession

session = CompressionSession(
    "samples/weights.jsonl",
    "samples/calibration.json",
    "examples/config.interact-budget.yaml",
    "/tmp/layerpack-interaction",
)
frame = session.submit_command({"type": "set_bits", "layer": "head", "bits": 8})
assert frame.payload["frame_type"] == "interim"
assert frame.payload["business_code"] == "E_INTERACT_BUDGET_PENDING"
final = session.wait_for_final(frame)
assert final.payload["frame_type"] == "final"
```

## 5. 信号与恢复

使用 `examples/config.signal.yaml`（阶段延迟 600ms）启动 `run`，约 800ms 后发送 `SIGTERM`，以确保 Python 已完成信号处理器安装且进程正处在阶段提交边界附近。观察到进程退出码 0、状态 `interrupted`、恢复点包含已完成阶段。

强制终止测试先完成一次正常快照，再启动带阶段延迟和新配置的进程并发送 `SIGKILL`。随后：

```bash
python3 -m layerpack status --state-dir <state-dir>
```

只返回强制终止前最后一个完整快照的版本与状态。

## 6. 诊断报告

每次终态运行写入：

- `<state-dir>/reports/diagnostics.json`
- `<state-dir>/reports/latest.json`

强制终止不保证报告；这是明确语义，恢复依赖原子阶段和快照清单。
