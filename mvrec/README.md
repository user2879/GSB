# mvrec — 本地确定性多视图三维重建管线

面向本地固定种子样本的多视图三维几何重建实验环境，核心解决**视图分组并行调度**与
**双层资源预算（深度图槽位 / 全局重建点数）控制**问题。纯 Python 3.11 + NumPy，
无需 GPU、网络、容器或外部服务；PNG 编解码由标准库 `zlib` 自行实现。

## 目录结构

- `mvrec/pngio.py` — PNG 容器校验与灰度/彩色解码（含 CRC/签名/IEND 校验）
- `mvrec/config.py` — 重建配置与相机参数校验、结构哈希（续算一致性判定）
- `mvrec/dataset.py` — 视图扫描、序号唯一性/分辨率一致性校验、加载失败清单
- `mvrec/features.py` — 连通域特征检测与固定维描述子（支持非有限数注入）
- `mvrec/geometry.py` — 互最近邻匹配、DLT 三角化、视差/ Cheirality 约束、
  深度图量化与确定性平滑、体素级点云合并
- `mvrec/grouping.py` — 分组策略选择器（确定性评分，输出可审计决策记录）
- `mvrec/snapshot.py` — 临时目录整体替换的阶段快照与原子清单
- `mvrec/pipeline.py` — 线程池调度、槽位信号量、稳定顺序提交、挂起/续算、信号处理
- `mvrec/cli.py` — 命令行入口、注入配置加载、诊断报告与注入命中映射
- `mvrec/samplegen.py` — 固定种子合成样本（几何图元 + 每标记唯一纹理）
- `scripts/acceptance.py` — 端到端验收驱动（6 项检查）
- `scripts/signal_runner.py` — 信号处理测试辅助进程
- `tests/` — 21 项 unittest 测试
- `configs/recon_config.json`、`configs/injections.json` — 重建与注入配置

## 快速开始

```bash
pip install -r requirements.txt
python3 -m mvrec.cli generate /tmp/mvrec/sample --views 8
python3 -m mvrec.cli run \
  --input /tmp/mvrec/sample \
  --config configs/recon_config.json \
  --injections configs/injections.json \
  --snapshot /tmp/mvrec/snap \
  --report /tmp/mvrec/report.json
python3 scripts/acceptance.py
python3 -m unittest discover -s tests -v
```

## 确定性与并发一致性

- 工作线程只做**纯计算**（特征/匹配/三角化/深度图），不修改共享状态。
- 所有合并、预算扣减、快照写入、失败记录均在主线程按分组稳定键
  `(group_index, view_id, stage)` 的严格顺序执行。
- 匹配对、三维点、深度摘要、体素点序、预算扣减序列、待算队列、失败清单均带稳定键排序。
- `tests/test_concurrency_matrix.py` 在并发度 1/2/3/5/8 下逐字段比对全部产物。

## 双层预算语义

- **槽位预算（slot_budget）**：以深度图槽位计，通过有界信号量限制同时在算的分组；
  槽位为零时所有分组进入按稳定键排序的待算队列并记录 `E_SLOT_BUDGET_EXHAUSTED`。
- **点数预算（point_budget）**：全局累计重建点数上限，分组提交前判定；不足时该分组及
  其后未派发分组全部进入待算队列并记录 `E_POINT_BUDGET_EXHAUSTED`，不使用线程超时。
- **续算**：提高预算后对同一快照根目录再次运行，仅依据清单中已完整替换的产物恢复，
  已提交分组不重算，待算分组按原稳定顺序继续。

## 快照与恢复

- 特征、匹配三角化、深度图、点云合并四阶段均以「临时目录构建 → `os.replace` 整体替换」
  提交；清单 `manifest.json` 原子写入，记录全局版本、配置哈希、相机哈希、分组进度、
  已消耗槽位/点数、产物版本与待算队列。
- 收到 SIGTERM/SIGINT 后完成当前在途分组的原子替换与进度记录，以退出码 0 结束；
  强制终止后仅信任清单引用的产物目录，未引用的临时目录不参与恢复。
- 配置结构哈希（预算/线程/快照间隔等续算可覆盖字段除外）或相机哈希不匹配时返回
  `E_SNAPSHOT_INVALID`。

## 业务错误码

| 代码 | 触发条件 |
| --- | --- |
| `E_CONFIG_INVALID` | 重建配置字段缺失或取值非法 |
| `E_CAMERA_INVALID` | 相机缺失、内参不可逆、FOV 越界、旋转非正交 |
| `E_PNG_INVALID` | PNG 签名/CRC/IEND 损坏或格式不支持 |
| `E_VIEW_MISSING` | 视图文件缺失或分辨率不一致 |
| `E_MATCH_TOO_FEW_INLIERS` | 匹配对内点数低于阈值 |
| `E_PARALLAX_INVALID` | 无点满足最小三角化视差 |
| `E_FEATURE_NONFINITE` | 描述子含非有限数 |
| `E_VOXEL_INVALID` | 体素尺寸为零 |
| `E_SLOT_BUDGET_EXHAUSTED` | 槽位预算耗尽/为零，分组挂起 |
| `E_POINT_BUDGET_EXHAUSTED` | 点数预算耗尽/为零，分组挂起 |
| `E_SNAPSHOT_INVALID` | 快照哈希不匹配或清单损坏 |

单视图/单分组异常只记录 `view_id + stage + exception_type` 并隔离，其余视图继续执行；
诊断报告包含分组决策、预算消耗曲线、失败清单、注入命中编号与恢复点。
