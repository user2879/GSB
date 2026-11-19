# 实际验证记录

验证时间：2026-11-19（Asia/Shanghai）
运行环境：Ubuntu 22.04（WSL），Python 3.11.0rc1，NumPy 2.4.6，pytest 9.2.0。
说明：题目目标平台为 Windows + tkinter；当前容器无 sudo 无法安装系统包 python3-tk，
因此 GUI 真实窗口渲染通过 tkinter 桩件做数据流验证，并在无 tkinter 时验证优雅降级。
核心引擎、持久化与测试不依赖 tkinter，跨平台纯 Python 逻辑一致。

## 环境准备实测
- 系统 Python 无 pip/ensurepip，使用 get-pip.py 用户级安装 pip 26.3。
- `python3 -m pip install --user 'numpy>=1.24'` 安装 NumPy 2.4.6。
- `python3 -m pip install --user pytest` 安装 pytest 9.2.0。

## 单元/集成测试
命令：`python3 -m pytest -q`
结果：**27 passed**。覆盖：
1. 解析校验：音高/拍位/时值、速度严格递增、力度单调、算子白名单、引用目标、坏 JSON。
2. tick 展开：取整规则、重复展开逐音符一致、同声部重叠 E_OVERLAP_SAME_VOICE、跨声部重叠区间。
3. 表达演进：速度插值、踏板轨道、力度单调映射、断奏静音尾、逐 tick 状态可查询、五类依赖键。
4. 增量重演：增量结果与从头全量在容差 0 下逐 tick 时长差 0、力度差 0。
5. 并发等价：并发度 1 与 4 的块结果逐元素相同。
6. 内存预算：预算 1 时峰值在算块数为 1；预算 4 时出现真实并行（峰值≥2）。
7. 中断恢复：起始即中断、编辑中途 SIGTERM（退出码 2）后重跑收敛到 8 步完成。
8. 版本隔离：输入变更生成新版本目录，旧版本目录原样保留（两版本并存）。
9. 摘要损坏：篡改块内容触发 E_DIGEST，恢复流程剔除坏块补算后重新一致。
10. 强制终止：无 manifest 条目的孤儿块被移入 orphans/，不被误认为完成。
11. 新增踏板按 tick 区间失效（旧块无该依赖键场景）。
12. GUI 桩件验证只读取持久化 timeline/report，不重算、不改写。

## 端到端 CLI 实测
- `generate`：生成 samples 下 score/expression/edits 三份 JSON，退出码 0。
- `run`：9 个 tick 块全部完成，版本 `019f925560629364`，
  一致性 `max_duration_diff=0`、`max_velocity_diff=0`，摘要校验 passed，退出码 0。
- 非法音高样本：stderr 返回 `E_PITCH` JSON，退出码 1，state/output 目录均未生成。
- SIGTERM：块边界写 `interrupted.json`，退出码 2；重跑同命令恢复并完成，退出码 0。
- 字节级确定性：两次独立运行的状态块、timeline.json、diagnostic_report.json 逐字节相同。
- 原子性：业务错误后旧报告 SHA-256 不变，output 与 state 无 `.tmp-*`/`.mwb-commit-*` 残留。
- 选择性失效：单步 `stretch_set(broad)` 仅失效块 4..8，块 0..3 复用未重算。
- GUI 缺 tkinter：返回 E_IO JSON，退出码 1，不影响核心流程。

## 样本规模
- 乐谱：2 声部 × 8 音符，17 拍 × 480 tick = 8160 tick，块大小 960，共 9 块。
- 编辑脚本：8 个算子（速度增删、力度增、伸缩、连断、音符连断、踏板增删），并发 2、预算 2。
