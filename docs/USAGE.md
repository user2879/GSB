# 使用指南与验证记录

## 1. 环境准备（WSL Ubuntu）

```bash
python3 --version            # 需 >= 3.10
pip install numpy pillow
sudo apt-get install -y python3-tk
python3 scripts/check_env.py # 依赖与目录可写性检查
```

本仓库实测环境：Python 3.11.0rc1、numpy 1.26.4、Pillow 10.3.0、Tk 8.6。

## 2. 常用命令

```bash
# 生成确定性本地样本工作区
python3 -m fixedpoint_render.cli init ./workspace --frames 5

# 仅校验输入（不进入渲染），非法输入返回 E_ 业务码与退出码 2
python3 -m fixedpoint_render.cli validate ./workspace

# 全量渲染
python3 -m fixedpoint_render.cli render ./workspace

# 单条增量编辑（示例：调整 2 号帧运动矢量）
python3 -m fixedpoint_render.cli edit ./workspace \
  '{"verb":"set_motion","frame":2,"motion":{...}}'

# 模拟在预算阶段后被强制终止，再依据已提交快照恢复
python3 -m fixedpoint_render.cli render ./workspace --crash-after 04_budget
python3 -m fixedpoint_render.cli render ./workspace --resume

# 查看诊断报告
python3 -m fixedpoint_render.cli report ./workspace
```

支持的编辑动词：`set_prompt`、`set_image`、`delete_image`、`set_motion`、
`set_weight`、`set_frame_budget`、`set_qmode`、`set_frac_bits`。

## 3. 测试与验收

```bash
python3 -m unittest discover -s tests -v   # 35 个用例
python3 scripts/acceptance.py              # 端到端验收，落盘 artifacts/
```

验收产物：

- `artifacts/acceptance_record.json`：机器可读的逐项结果。
- `artifacts/acceptance_summary.txt`：人类可读摘要。

## 4. 实际验证结果

最近一次 `python3 scripts/acceptance.py` 结果（全部 PASS）：

- 权重变更改变输出并使快照版本递增；量化档位切换同样升版本并改变整数结果。
- 预算仲裁满足 `总分配 <= 全局预算`、`逐帧分配 <= 逐帧需求`，并正确标记降级帧。
- 串行（1 worker）与并行（8 workers）5 帧 PNG 哈希完全一致。
- 含提示词、运动、权重、逐帧预算、量化档位、位宽、图像删除的完整命令序列后，
  增量渲染与干净全量渲染像素相等，且预算/降级/误差归因逐字段相等。
- 单帧运动编辑仅重算该帧编码与受帧间依赖影响的下游链，其它帧产物复用。
- 在 `04_budget` 后强制终止，恢复只承认已提交阶段，恢复结果与独立干净运行
  逐像素一致。

`python3 -m unittest discover -s tests`：35 个用例全部通过，无 ResourceWarning。

## 5. 关键复现性保证

- 所有指纹使用 BLAKE2b 规范化 JSON，排除墙钟时间、线程 ID 与目录遍历顺序。
- 随机源为每素材独立的计数器模式流，不读取 Python 全局随机状态。
- `max_workers` 不进入内容指纹，仅约束线程池上限。
- PNG 采用固定压缩参数；分块 PNG 字节在相同输入下逐字节可复现。
