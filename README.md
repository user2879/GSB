# 骨骼动画双格式互操作与动作重定向工具

本项目实现一个完全本地、确定性可复现的骨骼动画处理管线，覆盖：

- 自有版本化 JSON 动画格式（schema v1/v2/v3）与本地二进制轨迹格式（BTRJ）双向读写；
- JSON schema 固定迁移表、幂等迁移与缺失字段诊断；
- 二进制大端/小端、16/32 位索引宽度、`wxyz`/`xyzw` 四元数顺序兼容；
- 四元数归一化、符号对齐、关键帧插值与时轴重采样；
- 基于骨长比例和静止姿态差的动作重定向；
- 逐帧逐关节残差、全局姿态偏差、超差关节独立标记；
- 7 个业务错误码与首个分歧关节/帧定位；
- 固定种子样本集、完整验证入口、内容哈希与运行清单。

详细格式见 `docs/FORMAT.md`，重定向数学与容差定义见 `docs/RETARGETING.md`。

## 快速运行

环境要求：Python 3.10+、NumPy 1.24+。运行期不需要网络、GPU、Docker 或外部素材。

```bash
python3 -m pip install -r requirements.txt
python3 run_verification.py --out samples/generated
python3 -m unittest discover -s tests -v
```

等价 CLI 验证入口：

```bash
python3 -m bone_retarget verify --out samples/generated
```

也可安装为命令：

```bash
python3 -m pip install -e .
bone-retarget verify --out samples/generated
```

## 实际验证结果

本次工作区实际运行结果：

- Python 3.11.0rc1；NumPy 2.4.6；
- 18 个固定案例：3 套源/目标骨架 × 3 个 schema × 大小端、16/32 位索引宽度、两种四元数顺序、索引表重排；
- 11 个异常变体；
- 业务错误码覆盖 7/7：`E_SCHEMA_VERSION`、`E_FIELD_MISSING`、`E_INDEX_OVERFLOW`、`E_QUAT_INVALID`、`E_TIME_NONMONOTONIC`、`E_RETARGET_DIVERGE`、`E_FORMAT_ENDIAN`；
- 全部 JSON/二进制往返、迁移幂等、重采样后重定向一致性检查通过；
- 最大重定向残差约 `6.26e-16 rad`，最大全局姿态偏差约 `6.77e-16 rad`，无超差关节；
- 两次完整运行输出的 `report.json`、`hashes.sha256` 与清单哈希逐字节一致；
- 最终清单哈希：`c308d1537af1d64af7c4a27a841af69b949041d02b9c6356939e31673e5265c9`；
- 标准库测试：12 项全部通过。

实际报告入口：

- `samples/generated/report.json`：机器可读总报告；
- `samples/generated/report.txt`：可读摘要及逐关节残差；
- `samples/generated/hashes.sha256`：全部产物文件哈希；
- `samples/generated/manifest.sha256`：排除最终报告文件后的产物清单；
- `samples/generated/cases/<case>/retarget_report.json`：每帧每关节残差、全局旋转与局部/全局矩阵；
- `samples/generated/cases/<case>/target_transforms.json`：目标骨架逐帧关节变换；
- `samples/generated/anomalies/anomaly_report.json`：异常输入与业务码核验结果。

## CLI

```bash
# JSON -> 二进制
python3 -m bone_retarget convert \
  --input source.json --output source.bin \
  --endian big --index-width 2 --quat-order xyzw

# 二进制 -> JSON v3
python3 -m bone_retarget convert \
  --input source.bin --output source_v3.json \
  --schema-version 3 --quat-order wxyz

# 旧 schema 迁移到 v3，并报告幂等哈希
python3 -m bone_retarget migrate \
  --input source_v1.json --output migrated_v3.json

# 重采样到 30 FPS 后重定向，输出目标 JSON、二进制、残差报告和逐帧变换
python3 -m bone_retarget retarget \
  --source-animation source.json \
  --source-skeleton source_skeleton.json \
  --target-skeleton target_skeleton.json \
  --target-fps 30 --output-dir output
```

所有命令只读取本地文件。业务异常以结构化 JSON 输出到 stderr，并返回非零退出码；不会用空姿态继续。

## 项目结构

```text
bone_retarget/
  errors.py          业务错误码与携带 joint/frame 的异常
  math3d.py          四元数、矩阵、符号对齐与插值
  models.py          Skeleton/Joint/Animation 数据模型
  deterministic.py   固定种子骨架与动作生成器
  json_format.py     JSON schema、迁移、严格解析与写出
  binary_format.py   BTRJ 端序/索引/四元数顺序兼容读写
  resample.py        无外插时轴重采样
  retarget.py        静止姿态重差重定向与残差报告
  consistency.py     容差比较与首个分歧定位
  verify.py          固定样本集与端到端验证
  cli.py             命令行入口
docs/
  FORMAT.md          JSON/BTRJ 格式与迁移表
  RETARGETING.md     重定向数学、残差和一致性规则
tests/
  test_pipeline.py   12 项标准库 unittest
samples/generated/   本地确定性样本和本次运行报告
run_verification.py  可直接执行的验收入口
```

## 关键实现约定

- 内部四元数一律为 `wxyz`；外部 `xyzw` 仅在格式边界重排。
- 归一化后首关键帧采用确定性半球，后续关键帧与上一帧点积非负。
- 零范数、NaN、Inf 返回 `E_QUAT_INVALID`，并携带关节名与帧号。
- 二进制解析先由标记独立探测实际端序，再校验声明，避免“按声明错误读取”。
- 索引表允许任意关节排列，解析后按规范关节下标重排，不丢精度。
- 重定向旋转采用 `D = A_world * inverse(S_rest_world)`，再求
  `D * T_rest_world` 并换回目标局部空间。
- 根位移按 `target_root.length / source_root.length` 标量缩放。
- 重采样时间序列从源起点开始，以 `1/fps` 为步长，仅保留不超过源终点的时刻。
- 超差只标记具体关节和帧，不整体丢弃动作。
- 验证报告不包含时间戳或绝对路径，固定输入下产物逐字节稳定。
