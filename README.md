# 骨骼动画重定向与验证管线

纯 Python 3.11 + numpy 的确定性数据处理管线：程序化生成源/目标骨架与
源动作，按骨长比例与静止姿态差完成重定向、按目标帧率重采样，并以
自有 JSON 与本地 BTRJ 二进制两种格式往返落盘。帧块任务在线程池上
并行、按块键归并；资源预算不足时按固定规则降级且输出字节不变；
增量路径基于失效闭包只重算受影响块，并可与串行全量路径逐帧对账。

## 环境准备

- Windows 10+（代码仅使用标准库 + numpy，路径处理跨平台）
- Python 3.11，安装依赖：`py -m pip install -r requirements.txt`
- 无需网络与外部素材，样本由固定种子在本地生成

## 一键演示（推荐）

```bash
py -m animpipe.demo --output-dir artifacts/demo_run
```

Linux/macOS 将 `py` 换成 `python3`。运行结束后在输出目录得到：

- `source_motion.json` / `sample_trajectory.btrj`：生成的源动作与二进制样本
- `legacy_v1.json`：旧 schema 文档，用于演示 v1→v2 迁移
- `rt_source.json` / `rt_source.btrj`：双格式往返工件
- `retargeted.json` / `retargeted.btrj`：重定向+重采样结果
- `report.json`：各步内容哈希、残差、降级记录、闭包清单与七条错误码链路

## 运行测试

```bash
py -m unittest discover -s tests -v
```

## 关键验收点

- 双格式往返：JSON 容差 `1e-12`、BTRJ float32 容差 `2e-6`，二次编码字节一致
- schema 迁移幂等：重复迁移内容哈希不变；未知版本返回 `E_SCHEMA_VERSION`
- 并行/串行逐字节一致：任意合法线程数与块划分内容哈希、帧哈希相同
- 预算降级：线程钳制→串行→块缩小→拒绝（`E_BUDGET_DENIED`），输出不变且不丢帧
- 增量闭包：骨长/帧率/关键帧三类变更只重算闭包内块，闭包外哈希不变
- 逐帧对账：输出首个分歧帧、参与关节、残差与参数版本戳

技术细节见 `docs/DESIGN.md`，错误码定义见 `animpipe/errors.py`。
