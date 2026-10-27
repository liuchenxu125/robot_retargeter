import numpy as np
import os
import yaml
import mujoco
from pathlib import Path


def get_mjcf_joint_order(robot_name, config_dir="config/robot"):
    """从机器人配置文件中读取 MJCF，获取关节顺序"""
    config_path = Path(config_dir) / f"{robot_name}.yaml"
    if not config_path.exists():
        print(f"警告: 配置文件 {config_path} 不存在，跳过关节重排")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mjcf_path = config.get('robot_xml_path')
    if mjcf_path is None:
        print(f"警告: 配置文件中未找到 robot_xml_path，跳过关节重排")
        return None
    
    # 如果是相对路径，相对于项目根目录
    mjcf_path = Path(mjcf_path)
    if not mjcf_path.is_absolute():
        mjcf_path = Path(__file__).parent.parent / mjcf_path
    
    if not mjcf_path.exists():
        print(f"警告: MJCF 文件 {mjcf_path} 不存在，跳过关节重排")
        return None
    
    # 使用 MuJoCo 解析 MJCF，获取关节顺序
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    joint_names = []
    for jnt_id in range(model.njnt):
        jnt_type = model.jnt_type[jnt_id]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        joint_names.append(name)
    
    print(f"MJCF 关节顺序 ({len(joint_names)}): {joint_names}")
    return joint_names


def show_npz_structure(npz_file_path):
    """展示 npz 文件结构并返回键名信息"""
    data = np.load(npz_file_path, allow_pickle=True)
    
    print(f"=== NPZ 文件结构 ===")
    print(f"Keys: {data.files}")
    
    pos_key = None
    quat_key = None
    dof_key = None
    fps_key = None
    
    for key in data.files:
        arr = data[key]
        print(f"\n--- {key} ---")
        print(f"shape: {arr.shape}, dtype: {arr.dtype}")
        
        if np.issubdtype(arr.dtype, np.number):
            print(f"min: {arr.min():.4f}, max: {arr.max():.4f}")
        
        # 自动检测位置键
        if pos_key is None and arr.shape[-1] in [3, 6, 7]:
            if any(k in key.lower() for k in ['pos', 'position']):
                pos_key = key
        
        # 自动检测四元数键
        if quat_key is None and arr.shape[-1] == 4:
            if any(k in key.lower() for k in ['rot', 'quat', 'orientation']):
                quat_key = key
                if arr.ndim >= 2:
                    print(f"第一帧四元数模长: {np.linalg.norm(arr.reshape(-1, 4)[0]):.6f}")
        
        # 自动检测关节角度键
        if dof_key is None and arr.ndim == 2 and arr.shape[1] > 4:
            if any(k in key.lower() for k in ['joint', 'dof', 'qpos', 'angle']):
                dof_key = key
        
        # 自动检测帧率键
        if fps_key is None and arr.ndim == 0:
            if any(k in key.lower() for k in ['fps', 'framerate', 'rate']):
                fps_key = key
                print(f"FPS value: {arr}")
    
    data.close()
    
    return {
        'pos_key': pos_key,
        'quat_key': quat_key,
        'dof_key': dof_key,
        'fps_key': fps_key,
    }


def npz_to_csv(npz_file_path, output_csv_path=None, target_fps=None, height_offset=0.0, auto_detect=True, robot_name=None):
    """
    将npz文件转换为CSV格式，适配任意机器人
    无表头，无时间列
    """
    data = np.load(npz_file_path, allow_pickle=True)
    
    if auto_detect:
        keys_info = show_npz_structure(npz_file_path)
        pos_key = keys_info['pos_key']
        quat_key = keys_info['quat_key']
        dof_key = keys_info['dof_key']
        fps_key = keys_info['fps_key']
        
        if pos_key is None:
            raise KeyError("No position key found")
        if quat_key is None:
            raise KeyError("No quaternion key found")
    else:
        pos_key = 'body_pos_w'
        quat_key = 'body_quat_w'
        dof_key = 'joint_pos'
        fps_key = 'fps'

    root_pos = data[pos_key]
    root_rot = data[quat_key]
    dof_pos = data[dof_key] if dof_key is not None else None
    original_fps = float(data[fps_key]) if fps_key is not None else 50.0
    
    # 如果未指定目标帧率，使用原始帧率
    if target_fps is None:
        target_fps = original_fps
        print(f"使用原始帧率: {original_fps}fps")
    
    # Reshape (N, 1, D) -> (N, D)
    if root_pos.ndim == 3:
        root_pos = root_pos.reshape(-1, root_pos.shape[-1])
    if root_rot.ndim == 3:
        root_rot = root_rot.reshape(-1, root_rot.shape[-1])
    
    if height_offset != 0.0:
        print(f"Applying height offset: {height_offset:.3f}m")
        root_pos[:, 2] += height_offset

    num_frames = root_pos.shape[0]
    original_duration = num_frames / original_fps
    new_num_frames = int(original_duration * target_fps)
    
    original_times = np.linspace(0, original_duration, num_frames)
    new_times = np.linspace(0, original_duration, new_num_frames)
    
    pos_dim = root_pos.shape[1]
    root_pos_interp = np.zeros((new_num_frames, pos_dim))
    for i in range(pos_dim):
        root_pos_interp[:, i] = np.interp(new_times, original_times, root_pos[:, i])
    
    quat_dim = root_rot.shape[1]
    root_rot_interp = np.zeros((new_num_frames, quat_dim))
    for i in range(new_num_frames):
        idx = np.searchsorted(original_times, new_times[i])
        if idx == 0:
            root_rot_interp[i] = root_rot[0]
        elif idx >= num_frames:
            root_rot_interp[i] = root_rot[-1]
        else:
            t0, t1 = original_times[idx-1], original_times[idx]
            alpha = (new_times[i] - t0) / (t1 - t0)
            q0, q1 = root_rot[idx-1], root_rot[idx]
            dot = np.dot(q0, q1)
            if dot < 0:
                q1 = -q1
                dot = -dot
            if dot > 0.9995:
                root_rot_interp[i] = q0
            else:
                theta = np.arccos(dot)
                sin_theta = np.sin(theta)
                root_rot_interp[i] = (np.sin((1-alpha)*theta) * q0 + np.sin(alpha*theta) * q1) / sin_theta
    
    if dof_pos is not None:
        # 关节重排：如果指定了机器人名称，按 MJCF 顺序重排
        mjcf_joint_order = None
        if robot_name is not None:
            mjcf_joint_order = get_mjcf_joint_order(robot_name)
        
        if mjcf_joint_order is not None and dof_key is not None:
            npz_joint_names = [str(n) for n in data['joint_names']] if 'joint_names' in data.files else None
            if npz_joint_names is not None:
                missing = [name for name in mjcf_joint_order if name not in npz_joint_names]
                if missing:
                    print(f"警告: NPZ 中缺少关节 {missing}，跳过重排")
                else:
                    index = [npz_joint_names.index(name) for name in mjcf_joint_order]
                    dof_pos = dof_pos[:, index]
                    print(f"关节已按 MJCF 顺序重排")
        
        # 使用 dof_pos 自己的帧数进行插值
        dof_num_frames = dof_pos.shape[0]
        dof_original_duration = dof_num_frames / original_fps
        dof_new_num_frames = int(dof_original_duration * target_fps)
        
        dof_original_times = np.linspace(0, dof_original_duration, dof_num_frames)
        dof_new_times = np.linspace(0, dof_original_duration, dof_new_num_frames)
        
        dof_pos_interp = np.zeros((dof_new_num_frames, dof_pos.shape[1]))
        for i in range(dof_pos.shape[1]):
            dof_pos_interp[:, i] = np.interp(dof_new_times, dof_original_times, dof_pos[:, i])
    else:
        dof_pos_interp = None
    
    if output_csv_path is None:
        output_csv_path = os.path.splitext(npz_file_path)[0] + '.csv'
    
    num_joints = dof_pos_interp.shape[1] if dof_pos_interp is not None else 0
    
    with open(output_csv_path, 'w') as f:
        for i in range(new_num_frames):
            row = []
            
            for j in range(pos_dim):
                row.append(f'{root_pos_interp[i][j]:.6f}')
            
            # 四元数：假设输入是 wxyz 格式，CSV 需要 xyzw 格式
            q = root_rot_interp[i]
            row.extend([f'{q[1]:.6f}', f'{q[2]:.6f}', f'{q[3]:.6f}', f'{q[0]:.6f}'])
            
            if dof_pos_interp is not None:
                dof_vals = dof_pos_interp[i][:num_joints]
                row.extend([f'{v:.6f}' for v in dof_vals])
            
            f.write(','.join(row) + '\n')
    
    data.close()
    
    print(f"\n=== 转换结果 ===")
    print(f"Saved CSV to {output_csv_path}")
    print(f"Shape: ({new_num_frames}, {pos_dim + quat_dim + num_joints})")


def batch_convert_npz_to_csv(input_folder, output_folder, target_fps=30.0, robot_name=None):
    os.makedirs(output_folder, exist_ok=True)
    npz_files = [f for f in os.listdir(input_folder) if f.endswith('.npz')]
    
    if not npz_files:
        print(f"No npz files found in {input_folder}")
        return
    
    print(f"Batch converting {len(npz_files)} files...")
    for npz_file in npz_files:
        npz_path = os.path.join(input_folder, npz_file)
        csv_filename = os.path.splitext(npz_file)[0] + '.csv'
        output_csv_path = os.path.join(output_folder, csv_filename)
        print(f"\nProcessing: {npz_file}")
        try:
            npz_to_csv(npz_path, output_csv_path, target_fps, robot_name=robot_name)
        except Exception as e:
            print(f"Failed: {e}")
    
    print("\nBatch conversion complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将 NPZ 文件转换为 CSV 格式")
    parser.add_argument("--npz", required=True, help="原始 NPZ 文件路径")
    parser.add_argument("--robot", required=True, help="机器人名称")
    parser.add_argument("--output", default=None, help="输出 CSV 文件路径（默认与 NPZ 同路径同名）")
    
    args = parser.parse_args()
    
    # 如果未指定输出路径，默认与 NPZ 同路径同名
    if args.output is None:
        args.output = os.path.splitext(args.npz)[0] + '.csv'
    
    npz_to_csv(args.npz, args.output, robot_name=args.robot)