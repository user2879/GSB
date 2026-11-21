# 注入样例

- `commands-all-eight.jsonl`：覆盖八类编辑命令的完整序列。
- `config.point-budget.yaml`：预算小于任何单层点数，触发 `E_POINT_BUDGET_PENDING`。
- `config.interact-budget.yaml`：交互预算为 0，触发 `E_INTERACT_BUDGET_PENDING`。
- `config.zero-soft.yaml`：软标签权重为 0，验证关闭标记而非跳过样本。
- `config.invalid-bits.yaml`：非法位宽，验证 `E_CONFIG_INVALID`。
- `config.signal.yaml`：阶段延迟窗口，用于 SIGTERM/SIGKILL 验收。
- `inject-shape-mismatch.jsonl`：形状/元素数不匹配。
- `inject-nonfinite.jsonl`：非有限权重。
- `inject-calibration-missing.json`：缺少参考输出。

示例：

```bash
python3 -m layerpack validate \
  --weights examples/inject-nonfinite.jsonl \
  --calibration samples/calibration.json \
  --config samples/config.yaml --pretty
```
