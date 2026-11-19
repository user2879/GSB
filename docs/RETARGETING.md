# 重定向与一致性判定

## 1. 旋转映射

对每个源关节，内部记录按父到子组合：

```text
S_world  = S_parent_world * S_local     # 源静止姿态世界旋转
A_world  = A_parent_world * A_local     # 源动画世界旋转
D        = A_world * inverse(S_world)   # 源相对静止姿态的全局旋转增量
T_desired_world = D * T_rest_world      # 目标静止姿态上加同一全局增量
T_local = inverse(T_parent_world) * T_desired_world
```

因此静止姿态差异通过 `D * T_rest_world` 显式重基，而不是把源局部旋转直接复制到
目标骨架。目标骨架额外存在、源骨架不存在的关节保持目标静止局部旋转。

## 2. 根位移

```text
root_translation_scale = target_root.length / source_root.length
target_root_translation = source_root_translation * root_translation_scale
```

标量缩放保持根位移方向语义。骨长与静止姿态均由固定种子生成器产生。

## 3. 残差与标记

对每一帧、每个目标关节输出：

- `residual_rad`：由目标局部旋转重新组合出的世界姿态，与按 `D * T_rest_world`
  计算出的期望世界姿态之间的角距离，用于检测重定向重建误差。
- `global_deviation_rad`：目标全局旋转增量 `T_world * inverse(T_rest_world)` 与
  源全局增量 `D` 的角距离，用于检测源姿态语义是否保留。
- `target_local_rotation_wxyz`、`target_global_rotation_wxyz`、
  `target_local_matrix`、`target_global_matrix`。

默认容差 `1e-6 rad`。任一指标超限只会把对应关节标记为 `flagged`，并记录首个
超限帧；不会因此把整段动作判定为空或直接丢弃。验证脚本会汇总每个关节、每帧和
全部逐帧逐关节样本。

## 4. 一致性比较

`compare_animations` 比较：

- 关节名布局；
- 帧数；
- 时间戳绝对误差；
- 四元数姿态角距离（与四元数符号无关）；
- 根位移欧氏距离。

一旦超差，`E_RETARGET_DIVERGE` 携带首个分歧的 `frame` 与 `joint`。排序规则为
帧优先，同帧内依次优先旋转、时间戳、根位移、帧数。

## 5. 验证路径

每个固定案例执行：

1. JSON 源文档解析与 schema 迁移。
2. 源 JSON → 二进制 → 解析，核对语义哈希和逐关节姿态。
3. 二进制 → JSON v3 → 再解析，核对往返一致性。
4. 源动作重采样到 30 FPS。
5. 重采样结果分别经 JSON、二进制往返，再分别重定向。
6. 与直接重定向结果逐帧逐关节比较，并生成第一分歧定位。
7. 目标动作写成 JSON 与二进制，再次核验读写后的目标姿态。
