#!/usr/bin/env python3
"""
将重定向后的 CSV 文件转换回 NPZ 格式
根据目标机器人的 XML 文件获取关节名称和数量
"""

import numpy as np
import os
import sys
import argparse
import yaml
import mujoco
from pathlib import Path


def find_robot_config(robot_name: str) -> Path | None:
    """根据机器人名称查找配置文件"""
    # 获取脚本所在目录的上级目录（项目根目录）
    script_dir = Path(__file__).parent.parent
    config_path = script_dir / "config" / "robot" / f"{robot_name}.yaml"
    if config_path.exists():
        return config_path
    return None


def load_robot_joints_from_xml(xml_path: Path) -> list[str]:
    """从 MuJoCo XML 文件加载关节名称列表（排除 freejoint）"""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    joint_names = []
    for i in range(model.njnt):
        jnt_type = model.jnt_type[i]
        # 排除 freejoint（浮动基座关节）
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_names.append(name)
    return joint_names


def csv_to_npz(csv_path, reference_npz_path, output_npz_path=None, fps=None, robot_name=None):
    """
    将 CSV 转换为与参考 NPZ 相同格式的 NPZ
    
    CSV 格式: pos(3) + quat(4, xyzw) + joints(N)
    """
    # 1. 读取参考 NPZ 的结构
    print(f"=== 读取参考 NPZ: {reference_npz_path} ===")
    ref_data = np.load(reference_npz_path, allow_pickle=True)
    
    ref_keys = ref_data.files
    ref_shapes = {k: ref_data[k].shape for k in ref_keys}
    ref_dtypes = {k: ref_data[k].dtype for k in ref_keys}
    
    print(f"Keys: {ref_keys}")
    for k in ref_keys:
        print(f"  {k}: shape={ref_shapes[k]}, dtype={ref_dtypes[k]}")
    
    # 2. 读取 CSV
    print(f"\n=== 读取 CSV: {csv_path} ===")
    csv_data = np.loadtxt(csv_path, delimiter=",")
    num_frames = csv_data.shape[0]
    num_joints = csv_data.shape[1] - 7
    
    print(f"Frames: {num_frames}, Joints: {num_joints}")
    
    # 3. 从机器人 XML 加载关节名称
    robot_joint_names = None
    if robot_name:
        config_path = find_robot_config(robot_name)
        if config_path:
            print(f"\n=== 找到机器人配置: {config_path} ===")
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                xml_path_str = config.get('robot_xml_path', '')
                if xml_path_str:
                    xml_path = Path(xml_path_str)
                    if not xml_path.is_absolute():
                        xml_path = config_path.parent.parent.parent / xml_path
                    if xml_path.exists():
                        robot_joint_names = load_robot_joints_from_xml(xml_path)
                        print(f"=== 从 XML 加载关节: {xml_path} ===")
                        print(f"关节数量: {len(robot_joint_names)}")
                        print(f"关节名称: {robot_joint_names}")
                    else:
                        print(f"警告: XML 文件不存在: {xml_path}")
            except Exception as e:
                print(f"警告: 无法从配置文件加载关节名称: {e}")
        else:
            print(f"警告: 未找到机器人配置文件: {robot_name}")
    
    # 4. 提取 CSV 数据
    root_pos = csv_data[:, 0:3]
    quat_xyzw = csv_data[:, 3:7]
    joint_pos = csv_data[:, 7:]

    quat_wxyz = np.zeros_like(quat_xyzw)
    quat_wxyz[:, 0] = quat_xyzw[:, 3]  # w
    quat_wxyz[:, 1] = quat_xyzw[:, 0]  # x
    quat_wxyz[:, 2] = quat_xyzw[:, 1]  # y
    quat_wxyz[:, 3] = quat_xyzw[:, 2]  # z
    
    # 5. 构建新 NPZ（根据 CSV 关节数重新构建）
    output = {}
    for key in ref_keys:
        if key == 'joint_pos' or key == 'joint_vel':
            # 关节数据：使用 CSV 的关节数
            new_shape = (num_frames, num_joints)
            output[key] = joint_pos.reshape(new_shape).astype(ref_dtypes[key])
            print(f"  {key}: {new_shape} (from joint_pos, num_joints={num_joints})")
        elif 'pos' in key.lower() and 'joint' not in key.lower() and 'key_body' not in key.lower():
            # 根节点位置
            new_shape = (num_frames,) + ref_shapes[key][1:]
            output[key] = root_pos.reshape(new_shape).astype(ref_dtypes[key])
            print(f"  {key}: {new_shape} (from root_pos)")
        elif 'quat' in key.lower() or 'rot' in key.lower():
            # 四元数
            if ref_shapes[key][-1] == 4:
                if ref_shapes[key][1] == 1:
                    new_shape = (num_frames,) + ref_shapes[key][1:]
                    output[key] = quat_wxyz.reshape(new_shape).astype(ref_dtypes[key])
                    print(f"  {key}: {new_shape} (from quat)")
                else:
                    # 多个关键体的四元数，跳过
                    output[key] = ref_data[key]
                    print(f"  {key}: skipped (multi-body quat)")
            else:
                # 其他旋转数据，直接复制
                output[key] = ref_data[key]
        elif 'fps' in key.lower() or 'framerate' in key.lower() or 'frame_rate' in key.lower():
            # 帧率
            new_fps = fps if fps is not None else float(ref_data[key])
            output[key] = np.array(new_fps, dtype=ref_dtypes[key])
            print(f"  {key}: {new_fps}")
        elif 'names' in key.lower():
            # 名称列表
            if 'joint' in key.lower():
                # 关节名称优先使用 XML 中的名称
                if robot_joint_names and len(robot_joint_names) > 0:
                    output[key] = np.array(robot_joint_names)
                    print(f"  {key}: {len(output[key])} names (from XML)")
                else:
                    output[key] = ref_data[key]
                    print(f"  {key}: {len(output[key])} names (from ref)")
            else:
                output[key] = ref_data[key]
        else:
            # 其他数据直接复制
            output[key] = ref_data[key]
            print(f"  {key}: copied from reference")
    
    # 6. 保存
    if output_npz_path is None:
        output_npz_path = os.path.splitext(csv_path)[0] + '.npz'
    
    np.savez(output_npz_path, **output)
    ref_data.close()
    
    print(f"\n=== 转换结果 ===")
    print(f"Saved NPZ to {output_npz_path}")
    print(f"Frames: {num_frames}")
    print(f"Joints: {num_joints}")
    if robot_joint_names:
        print(f"Joint names (from XML): {robot_joint_names}")
    print(f"Keys: {list(output.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 CSV 转换为 NPZ 格式")
    parser.add_argument("--csv", required=True, help="输入的 CSV 文件路径")
    parser.add_argument("--ref", required=True, help="参考的 NPZ 文件路径")
    parser.add_argument("--output", default=None, help="输出的 NPZ 文件路径（默认与 CSV 同名）")
    parser.add_argument("--fps", type=float, default=None, help="帧率（默认使用参考 NPZ 的帧率）")
    parser.add_argument("--robot", default=None, help="目标机器人名称（自动查找 config/robot/ 下的配置）")
    
    args = parser.parse_args()
    
    csv_to_npz(args.csv, args.ref, args.output, args.fps, args.robot)