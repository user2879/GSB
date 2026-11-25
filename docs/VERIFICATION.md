# 实际验证记录

验证时间：2026-11-26（Asia/Shanghai）

环境：WSL Ubuntu，Python 3.11.0rc1，NumPy 1.26.4，Pillow 10.3.0。核心验收不依赖 Tk；当前容器缺少 `libtk8.6.so`，可选 `vertical_ocr view` 会输出 `sudo apt install python3-tk` 提示并返回码 2。

## 可重复命令

```bash
bash scripts/run_acceptance.sh
```

本次实际执行退出码为 `0`，完整控制台日志保存在 `artifacts/verification-run.log`，机器报告为 `artifacts/acceptance-run/acceptance_report.json`。

## 结果摘要

- `python3 -m compileall`：通过。
- `python3 -m unittest discover -s tests`：13 个测试全部通过，最后一次实际耗时约 12.6 秒。
- `vertical_ocr acceptance`：25/25 项检查通过。
- 确定性样本：3 张 PNG、每张 3 栏 × 6 字，共 54 个字符块。
- 冷启动主报告单调时钟耗时：约 3705.862 ms（机器 JSON 为精确值）。
- 版面/真值校验：54/54 通过，无校验失败。
- 内置字模识别：54/54 与真值字符一致。
- 三版本回执：v1=54、v2=54、v3=54、旁路恢复=54。
- 同版本二次运行：缓存命中 123、重算 0；快照状态 `fresh`。
- 定点量化：所有字符的量化残差均不超过 `0.5 / quantization_level`。
- 关键点切换：54 个模板匹配全部命中复用，3 个图像的阅读串联重算，权威版本切换为 `version_no=99`。
- 分割罚项调整：普通非粘连样本的 123 个阶段结果全部复用；粘连 DP 上等代价取更靠上切点由单测覆盖。
- 替换 `page_002.png`：18 个该图字符重算，另外两图 36 个字符命中。
- 给 `page_001.png` 增栏：新栏 6 个字符重算，旧 54 个字符命中。
- 删除该栏后：剩余 54 个字符全部命中。
- 预算 `70000`：消耗 63339，完成 9 个字符，45 个进入 `E_MATCH_BUDGET_PENDING`；下一批首个处理项为 `page_001.png:p1-c1-b3`，随后待办清零。
- 强杀恢复：子进程在快照提交切换前收到 `SIGKILL`，返回码 `-9`；已提交指针仍停留在 v1；重启检测到 `global_version_mismatch` / `E_SNAPSHOT_STALE` 并重建 v2。

## 异常注入命中

| 注入编号 | 故障 | 观测业务码 |
|---|---|---|
| `INJ-PNG-001` | 截断 `page_002.png` PNG 数据块 | `E_INPUT_INVALID` |
| `INJ-KP-002` | 关键点 x 坐标改为 12.5 | `E_INPUT_INVALID` |
| `INJ-SEQ-003` | 同栏阅读序号加 5 形成跳号 | `E_INPUT_INVALID` |
| `INJ-WIN-004` | 滑窗宽 13 不能整除版面宽 720 | `E_INPUT_INVALID` |
| `INJ-FONT-005` | 字模版本改为未知 `dot-matrix-v999-unknown` | `E_TEMPLATE_VERSION_UNKNOWN` |

五项注入全部 `hit=true`。截断图进入失败清单后，其余图像仍继续处理；完整逐项结果见 `artifacts/acceptance-run/injections/injection_report.json`。

## 交换格式抽查

```bash
python3 -m vertical_ocr.cli convert \
  --from v3 --to v1 \
  --input demo_workspace/exchange/results.v3.txt \
  --output demo_workspace/exchange/acceptance.v1.txt \
  --ledger-dir demo_workspace/ledger

python3 -m vertical_ocr.cli convert \
  --from v1 --to v3 \
  --input demo_workspace/exchange/acceptance.v1.txt \
  --output demo_workspace/exchange/acceptance.restored.v3.txt \
  --ledger-dir demo_workspace/ledger
```

浮点 bbox `[10.2, 20.8, 110.25, 220.75]` 降级截断后再从旁路恢复，仍为原浮点值；置信度 `0.123456` 按四舍五入得到千分比 123，并可从旁路恢复原浮点值。字段过短、非法字符和缺键均返回 `E_FIELD_TRUNCATED`，错误中带 `version` 与 `field`。
