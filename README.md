# MoCap Constraint Workbench（骨骼动捕约束推演本地工作台）

纯本地、无 GPU、无外部数据依赖的动捕后期工作台。输入由确定性规则生成的骨架/动捕曲线/约束/参数 JSON，
输出满足关节角度限值、末端 IK、骨长保持与地面接触约束的姿态序列、逐帧残差报告，以及可交互的 PySide6 预览窗口。

## 环境准备

- Windows 本地环境（代码同时可在 Linux/macOS 运行），Python 3.11+
- 依赖安装：`pip install -r requirements.txt`（仅 `numpy`、`PySide6`）
- 不使用 Docker 或任何容器；不访问网络目标

## 入口命令

所有命令均通过项目根目录的 `main.py` 调用（Windows 下把 `python3` 换成 `python`）。

```bash
# 1. 生成确定性样本（60 帧、14 骨、含缺失帧/非单位四元数/不可达 IK 目标）
python3 main.py generate --input samples/sample.json

# 2. 分块求解并原子导出报告与姿态
python3 main.py solve --input samples/sample.json --state-dir state/demo --output out/demo

# 3. 校验状态目录版本标识与逐块摘要
python3 main.py verify --input samples/sample.json --state-dir state/demo

# 4. 中断后从最近完成分块续算（版本标识不一致则自动从头重做）
python3 main.py resume --input samples/sample.json --state-dir state/demo --output out/demo

# 5. 串行逐帧求解 vs 并发分块求解的逐字段一致性
python3 main.py serial-check --input samples/sample.json --state-dir state/serial

# 6. 启动交互式窗口（时间轴拖动/逐帧步进）
python3 main.py gui --input samples/sample.json --state-dir state/workbench

# 7. 离屏界面自检（截图 + UI 状态导出，CI/无显示环境可用）
QT_QPA_PLATFORM=offscreen python3 main.py ui-selftest \
  --input samples/sample.json --state-dir state/ui \
  --state-export out/ui_state.json --screenshot out/ui_screenshot.png
```

Windows 一键入口：`run_workbench.bat`（启动界面）、`run_demo.bat`（生成+求解+校验）。
Linux/macOS：`./run_workbench.sh`。

进程可处理终止：`SIGINT`（Ctrl+C）退出码 130，`SIGTERM` 退出码 143；信号只在分块边界生效，
先写 `interrupt.flag`，未完成分块不会留下任何可被误认为已求解的帧文件。

## 输出与状态文件

- `out/<name>/report.json`：版本摘要、总帧数、补齐区间、通过/未收敛帧、每约束通过率、求解顺序、参数快照
- `out/<name>/poses.json`：版本摘要、骨序、每帧时间戳、局部四元数、根位置、世界坐标
- `state/<name>/manifest.json`：版本标识、帧布局、已完成分块清单及清单摘要
- `state/<name>/chunk_XXXXXX.json`：分块姿态、逐帧残差轨迹与分块内容摘要（临时文件 + `os.replace` 原子替换）
- 报告与姿态在临时目录同时写好后整体替换；版本不同整体替换，版本相同且帧数一致则保留旧版本

## 架构

```
mocbench/
  errors.py       业务错误码 E_xxx 与退出码映射
  quat.py         四元数运算（归一化、slerp、轴角、旋转对齐）
  skeleton.py     骨架拓扑校验、拓扑排序、静止偏移、正向运动学
  parser.py       输入解析与全部输入校验（失败即 E_xxx，不产生姿态输出）
  canonical.py    规范化 JSON 与 SHA-256 版本/分块摘要
  fill.py         缺失帧确定性补齐（slerp / 根骨静止兜底）与区间账本
  solver.py       有状态迭代求解器与逐约束残差轨迹
  persistence.py  分块原子持久化、摘要校验、失效标记、中断标记
  engine.py       有界线程池、内存预算信号量、分块边界中断、续算与失效重算
  report.py       残差报告与姿态的原子/版本化导出
  samplegen.py    确定性样本生成器（固定种子，缺失帧与未收敛帧可复现）
  app.py          PySide6 窗口：占位态、只读取已完成帧、后台补算不可变
  cli.py          命令行入口
tests/            unittest 验收套件（7 组，无第三方测试依赖）
```

## 求解器语义

每帧严格按声明顺序迭代 `angle_limit -> ik -> bone_length -> contact`，迭代上限与阻尼系数同时生效：

- 角度限值：从局部四元数分解绕约束轴的有符号转角，超出上下限时按 `阻尼 * 违约量` 投影回去
- 末端 IK：阻尼 CCD，沿末端到根的关节链求解，阻尼缩放每次旋转量，关节同时受角度限值投影
- 骨长保持：旋转只改变朝向，骨长由静止偏移与单位四元数天然保持，逐段记录长度残差
- 地面接触：接触帧取接触骨最低点，根骨垂直位置按阻尼抬升至地面高度
- 任一约束残差 ≤ 容差即通过；全部通过提前停止，否则跑到迭代上限并保留最后一次迭代的完整残差轨迹
- 求解仅依赖输入内容与帧索引，无随机量；并发与串行结果逐字段一致（实测最大绝对差为 0.0）

## 业务要求对照

1. 解析校验：唯一骨名/无环拓扑/正长度/帧数一致/四元数有限（非单位自动归一化并计数）/限值与目标索引校验，
   非法输入返回 `E_INPUT/E_BONE/E_HIERARCHY/E_LENGTH/E_CURVE/E_QUATERNION/E_LIMIT/E_TARGET/E_CONTACT/E_PARAM`
2. 缺失补齐：区间两端 slerp，根骨或越界区间用静止位姿兜底；时间戳 = 帧索引 / fps，总帧数不变，区间与方法入账
3. 多约束求解：固定求解顺序、阻尼与迭代上限同时生效、逐约束残差轨迹、未收敛帧附最后迭代明细与迭代次数
4. 预览一致性：未完成帧显示确定性 `PLACEHOLDER` 且不阻塞；后台只写入新完成帧，已查看帧快照逐字段比对不变
5. 可处理终止：信号仅在分块边界生效，落中断标记并以 130/143 退出；续算校验版本标识，不一致整体重做
6. 分块持久化：先临时文件再整体替换，续算/拼装逐块比对摘要，失败返回 `E_DIGEST` 并标记 `.invalid` 回退重算
7. 并发与资源：有界线程池 + 加权内存信号量，超预算按块索引序等待，不降低迭代上限或精度；首错取消其余待处理块
8. 原子导出：报告含全部规定统计字段，姿态导出读取旧版本/帧数，版本不同整体替换，两者要么同时更新要么都保留旧版

## 实测验证结果

命令：`python3 -m unittest tests.test_pipeline`（7 组全部 OK），另含 CLI/子进程级中断与界面离屏自检。

- 样本：60 帧、14 骨；补齐 6 个区间共 18 帧；自动归一化非单位四元数 10 次
- 收敛：59 帧通过，唯一未收敛帧为第 35 帧（不可达 IK 目标，残差约 10.59，90 次迭代轨迹完整保留）
- 每约束通过率：角度限值 60/60，骨长 60/60，接触 4/4，IK 3/4
- 串行 vs 并发：局部四元数、根位置、世界坐标最大绝对差 `0.0`
- 中断恢复：SIGTERM 退出码 143，仅完整分块落盘，续算结果与连续求解逐字段一致
- 摘要保护：篡改分块触发 `E_DIGEST`（退出码 12），自动标记失效并重算，恢复后与干净运行一致
- 版本切换：参数变化导致版本摘要变化时，报告与姿态整体替换；同版本重复运行保留旧输出
- 界面自检：占位态→已求解态切换正确，60 帧全部缓存，已查看帧在补算前后不可变，截图与状态导出可用
- 异常输入：13 类非法输入全部返回对应 `E_xxx` 业务码，且不产生任何姿态或状态文件
