---
name: casbot-02-robot-retargeter-setup
description: "How to configure a new Casbot_02 humanoid robot in the robot_retargeter motion retargeting pipeline, including axis_map fixes based on GMR framework comparison."
metadata: 
  node_type: memory
  type: project
  originSessionId: f7c1e032-7b03-4f2d-9116-f464384a0182
---

# Casbot_02 动作重定向仓库配置

## 仓库路径
- robot_retargeter: `/home/csabot/amp_lab/robot_retargeter`
- GMR 框架 (参考): `/home/csabot/amp_lab/GMR-master`

## 新增/修改的文件

### 1. MJCF XML: `asset/robot/casbot_02/xmls/casbot_skeleton_25dof.xml`
基于原始 `casbot_skeleton_25dof.xml` 添加：
- `hips_sphere` (base_link 子body, 髋部参考点)
- `neck_sphere` + `head_sphere` (waist_yaw_link 子body, 肩部/头部参考点)
- `left_foot_end_link` / `left_toe_link` (足跟/足尖接触检测体)
- `right_foot_end_link` / `right_toe_link` (同上)
- `<actuator>` 段，25 个 motor（原 XML 缺少执行器定义）
- 肘关节轴**保持原始方向** `axis="0 1 0"`, range="-1.8675 0"（负方向=屈曲）

### 2. 机器人配置: `config/robot/casbot_02.yaml`

**ik_match_table 手臂权重**（适配 5-DOF 手臂）:
```yaml
"left_arm": ["left_elbow_pitch_link", 10.0, 3],   # rot=3 (不是G1的1)
"right_arm": ["right_elbow_pitch_link", 10.0, 3],
"left_fore_arm": ["left_wrist_yaw_link", 10, 3],
"right_fore_arm": ["right_wrist_yaw_link", 10, 3],
```

**key_frame_config 手臂 axis_map**（来自 GMR 框架对比分析）:
```yaml
# left_fore_arm 和 left_hand: -90° Y 旋转
x: [0, 0, 1],  y: [0, 1, 0],  z: [-1, 0, 0]

# right_fore_arm 和 right_hand: +90° Y 旋转（镜像）
x: [0, 0, 1],  y: [0, -1, 0],  z: [1, 0, 0]
```

**关节限制**（无肘部偏置）:
```yaml
joints_limit_offset_degrees:
  knee: [10.0, 0.0]
```

### 3. 未修改的核心代码
`scripts/smpl_replay.py` 和 `scripts/robot_retarget.py` **保持原始版本**。所有 Casbot_02 适配仅通过 YAML + XML 完成。

## 关键发现

### 手臂方向问题根因
robot_retargeter 使用 `key_frame_config.axis_map_cols` 进行坐标系映射。每个机器人链接的 axis_map 通过**骨架子body名称**查找：
- 机器人 `left_arm` → 骨架子体 `left_fore_arm` → 查 key_frame_config `left_fore_arm`
- 机器人 `left_fore_arm` → 骨架子体 `left_hand` → 查 key_frame_config `left_hand`

Casbot_02 与 G1 的手臂坐标系方向不同。GMR 框架中 Casbot_02 肘部有四元数偏移 `[0.707, 0, -0.707, 0]` (-90°Y)，而 G1 是恒等。通过矩阵分解得到对应的 axis_map，修正后肘部正常弯曲。

### 肘部冗余问题
Casbot_02 shoulder_pitch 和 elbow_pitch **都绕Y轴旋转**（Jacobian 均为1.0），IK 总是优先用肩部代偿。这是 5-DOF 手臂的固有特性，通过正确的 axis_map 可缓解。

## 使用方法
```bash
cd /home/csabot/amp_lab/robot_retargeter
VIS_ROBOTS="casbot_02" SMPL_MOTION_FILE="dataset/ACCAD/Form_1_stageii.npz" ./bash/retarget_from_smplx.sh
```

## 项目原理要点
- **自动比例缩放**: 从 MJCF 测量机器人骨骼长度，与 SMPL 骨骼逐帧比较，自动计算 scale factor
- **axis_map vs GMR 四元数偏移**: axis_map 定义"SMPL轴→机器人轴"的一次性映射；GMR 需要每关节手调四元数
- **接触感知 IK**: 自动检测足部着地（速度<0.5 + 高度<0.05），锁定接触点

## 相关文件路径
- G1 参考配置: `config/robot/g1.yaml`
- G1 MJCF: `asset/robot/g1_description/mjcf/g1.xml`
- GMR G1 配置: `/home/csabot/amp_lab/GMR-master/general_motion_retargeting/ik_configs/smplx_to_g1.json`
- GMR Casbot_02 配置: `/home/csabot/amp_lab/GMR-master/general_motion_retargeting/ik_configs/smplx_to_casbot_02.json`
- 骨架定义: `config/skeleton/skeleton.yaml`
- Casbot_02 网格: `asset/robot/casbot_02/meshes/`
