#!/usr/bin/env python3
"""
播放 NPZ 格式的运动数据文件

支持两种格式：
1. 机器人运动格式（包含 body_pos_w, body_quat_w, joint_pos, fps）
2. AMASS/SMPL 格式（包含 trans, root_orient, pose_body）

使用方法：
    # 播放机器人运动 npz
    python tools/play_motion.py --npz dataset/e1_21dof/amp/walk.npz --robot e1_21dof
    
    # 指定帧率和循环播放
    python tools/play_motion.py --npz motion.npz --robot g1 --fps 30 --loop
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation


def detect_npz_type(npz_path):
    """检测 npz 文件的类型"""
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    data.close()
    
    # 机器人运动格式
    if 'body_pos_w' in keys and 'body_quat_w' in keys:
        return 'robot'
    
    # AMASS/SMPL 格式
    if 'trans' in keys and 'root_orient' in keys and 'pose_body' in keys:
        return 'amass'
    
    return 'unknown'


def load_robot_npz(npz_path):
    """加载机器人运动格式的 npz 文件"""
    data = np.load(npz_path, allow_pickle=True)
    
    pos = data['body_pos_w']
    quat = data['body_quat_w']
    joint_pos = data['joint_pos'] if 'joint_pos' in data.files else None
    # fps = float(data['fps']) if 'fps' in data.files else 30.0
    fps = float(np.asarray(data['fps']).reshape(-1)[0]) if 'fps' in data.files else 30.0

    # 处理形状
    if pos.ndim == 3:
        pos = pos.reshape(-1, pos.shape[-1])
    if quat.ndim == 3:
        quat = quat.reshape(-1, quat.shape[-1])
    
    result = {
        'type': 'robot',
        'pos': pos,
        'quat': quat,
        'joint_pos': joint_pos,
        'fps': fps,
        'num_frames': pos.shape[0],
    }
    
    if 'joint_names' in data.files:
        result['joint_names'] = [str(n) for n in data['joint_names']]
    
    data.close()
    return result


def load_amass_npz(npz_path, smpl_model_path=None):
    """加载 AMASS/SMPL 格式的 npz 文件"""
    data = np.load(npz_path, allow_pickle=True)
    
    trans = np.asarray(data['trans'], dtype=np.float32)
    root_orient = np.asarray(data['root_orient'], dtype=np.float32)
    pose_body = np.asarray(data['pose_body'], dtype=np.float32)
    fps = float(np.asarray(data['mocap_frame_rate']).reshape(-1)[0])
    
    result = {
        'type': 'amass',
        'trans': trans,
        'root_orient': root_orient,
        'pose_body': pose_body,
        'fps': fps,
        'num_frames': trans.shape[0],
    }
    
    if 'betas' in data.files:
        result['betas'] = np.asarray(data['betas'], dtype=np.float32).reshape(-1)
    if 'pose_hand' in data.files:
        result['pose_hand'] = np.asarray(data['pose_hand'], dtype=np.float32)
    
    data.close()
    return result


def get_mjcf_path(robot_name: str) -> Path:
    """获取机器人 MJCF 文件路径"""
    config_path = Path(f"config/robot/{robot_name}.yaml")
    if not config_path.exists():
        print(f"错误：找不到机器人配置文件 {config_path}")
        sys.exit(1)
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mjcf_path = Path(config['robot_xml_path'])
    if not mjcf_path.is_absolute():
        mjcf_path = Path(__file__).parent.parent / mjcf_path
    
    if not mjcf_path.exists():
        print(f"错误：找不到 MJCF 文件 {mjcf_path}")
        sys.exit(1)
    
    return mjcf_path


def load_model_with_ground(xml_file: Path) -> mujoco.MjModel:
    """加载机器人 MJCF 并添加地面、天空盒和灯光"""
    spec = mujoco.MjSpec.from_file(str(xml_file))
    existing_tex = {texture.name for texture in spec.textures}
    existing_mat = {material.name for material in spec.materials}

    existing_geom_names = {geom.name for geom in spec.worldbody.geoms}
    existing_light_names = {light.name for light in spec.worldbody.lights}


    if "skybox" not in existing_tex:
        spec.add_texture(
            name="skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.3, 0.5, 0.7],
            rgb2=[0.0, 0.0, 0.0],
            width=512,
            height=512,
        )
    if "groundplane" not in existing_tex:
        spec.add_texture(
            name="groundplane",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.2, 0.3, 0.4],
            rgb2=[0.1, 0.2, 0.3],
            width=300,
            height=300,
        )
    if "groundplane" not in existing_mat:
        spec.add_material(
            name="groundplane",
            textures=["", "groundplane"],
            texrepeat=[5, 5],
            texuniform=True,
            reflectance=0.2,
        )

    if "sun" not in existing_light_names:
        spec.worldbody.add_light(
            name="sun",
            pos=[0, 0, 20.0],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.7, 0.7, 0.7],
            specular=[0.3, 0.3, 0.3],
        )
    if "fill_light" not in existing_light_names:
        spec.worldbody.add_light(
            name="fill_light",
            pos=[4, 4, 6.0],
            dir=[-0.5, -0.5, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.4, 0.4, 0.4],
            specular=[0.1, 0.1, 0.1],
        )

    if "ground" not in existing_geom_names:
        ground = spec.worldbody.add_geom(
            name="ground",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            material="groundplane",
            pos=[0, 0, 0],
        )
        ground.contype = 0
        ground.conaffinity = 0
    return spec.compile()


def format_progress_line(cursor: int, num_frames: int, fps: float, paused: bool) -> str:
    """格式化终端进度显示"""
    progress = cursor / max(num_frames - 1, 1)
    bar_len = 30
    filled = int(bar_len * progress)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = progress * 100
    status = "暂停" if paused else "播放"
    time_str = f"{cursor/fps:.1f}s/{num_frames/fps:.1f}s"
    return f"\r{status} [{bar}] {pct:5.1f}% {cursor}/{num_frames} {time_str}   "


def play_robot_motion(model, data, motion_data, loop=False):
    """播放单个机器人运动数据"""
    pos = motion_data['pos']
    quat = motion_data['quat']
    joint_pos = motion_data['joint_pos']
    fps = motion_data['fps']
    num_frames = motion_data['num_frames']
    
    joint_addrs = []
    for jnt_id in range(model.njnt):
        jnt_type = model.jnt_type[jnt_id]
        if jnt_type != mujoco.mjtJoint.mjJNT_FREE:
            joint_addrs.append(int(model.jnt_qposadr[jnt_id]))
    
    frame_time = 1.0 / fps
    
    state = {
        "frame": 0,
        "paused": False,
        "step_delta": 0,
        "resync_clock": False,
    }
    state_lock = threading.Lock()
    
    def key_callback(keycode: int) -> None:
        with state_lock:
            if keycode == glfw.KEY_SPACE:
                state["paused"] = not state["paused"]
                state["resync_clock"] = True
            elif keycode == glfw.KEY_COMMA:
                state["paused"] = True
                state["step_delta"] -= 1
                state["resync_clock"] = True
            elif keycode == glfw.KEY_PERIOD:
                state["paused"] = True
                state["step_delta"] += 1
                state["resync_clock"] = True
            elif keycode == glfw.KEY_LEFT_BRACKET:
                state["paused"] = True
                state["step_delta"] -= 10
                state["resync_clock"] = True
            elif keycode == glfw.KEY_RIGHT_BRACKET:
                state["paused"] = True
                state["step_delta"] += 10
                state["resync_clock"] = True
            elif keycode == glfw.KEY_R:
                state["frame"] = 0
                state["step_delta"] = 0
                state["resync_clock"] = True
                state["paused"] = False
    
    def render_frame(frame_idx: int) -> None:
        data.qpos[:3] = pos[frame_idx]
        data.qpos[3:7] = quat[frame_idx]
        
        if joint_pos is not None and len(joint_addrs) > 0:
            for i, addr in enumerate(joint_addrs):
                if i < joint_pos.shape[1]:
                    data.qpos[addr] = joint_pos[frame_idx, i]
        
        mujoco.mj_forward(model, data)
    
    print("控制：[Space] 暂停/继续  [,] 后退1帧  [.] 前进1帧  [[] 后退10帧  []] 前进10帧  [R] 重播")
    print(f"总帧数：{num_frames}，帧率：{fps:.1f} FPS，时长：{num_frames/fps:.1f}s")
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        last_advance_time = time.perf_counter()
        last_progress_line = ""
        
        while viewer.is_running():
            frame_start = time.perf_counter()
            
            with state_lock:
                paused = bool(state["paused"])
                frame = int(state["frame"])
                step_delta = int(state["step_delta"])
                resync_clock = bool(state["resync_clock"])
                state["step_delta"] = 0
                state["resync_clock"] = False
            
            if resync_clock:
                last_advance_time = time.perf_counter()
            
            if step_delta != 0:
                frame = max(0, min(frame + step_delta, num_frames - 1))
                paused = True
                with state_lock:
                    state["paused"] = True
                    state["frame"] = frame
                last_advance_time = time.perf_counter()
            elif not paused:
                now = time.perf_counter()
                if now - last_advance_time >= frame_time:
                    if frame == num_frames - 1 and not loop:
                        render_frame(frame)
                        viewer.sync()
                        with state_lock:
                            state["paused"] = True
                        continue
                    frame = (frame + 1) % num_frames if loop else frame + 1
                    with state_lock:
                        state["frame"] = frame
                    last_advance_time = now
            
            render_frame(frame)
            
            progress_line = format_progress_line(frame, num_frames, fps, paused)
            if progress_line != last_progress_line:
                sys.stdout.write(progress_line)
                sys.stdout.flush()
                last_progress_line = progress_line
            
            viewer.sync()
            
            elapsed = time.perf_counter() - frame_start
            if paused or step_delta != 0:
                time.sleep(0.01)
            else:
                remaining = frame_time - elapsed
                if remaining > 0.0:
                    time.sleep(remaining)
        
        sys.stdout.write("\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="播放 NPZ 运动数据文件")
    parser.add_argument("--npz", required=True, help="NPZ 文件路径")
    parser.add_argument("--robot", help="机器人名称（用于机器人运动格式）")
    parser.add_argument("--fps", type=float, help="播放帧率（默认使用文件中的帧率）")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--end-frame", type=int, default=-1, help="结束帧（-1 表示到末尾）")
    
    args = parser.parse_args()
    
    npz_path = Path(args.npz)
    if not npz_path.exists():
        print(f"错误：文件不存在 {npz_path}")
        sys.exit(1)
    
    # 检测文件类型
    npz_type = detect_npz_type(npz_path)
    print(f"检测到 NPZ 类型：{npz_type}")
    
    if npz_type == 'robot':
        if not args.robot:
            print("错误：机器人运动格式需要指定 --robot 参数")
            sys.exit(1)
        
        motion_data = load_robot_npz(npz_path)
        mjcf_path = get_mjcf_path(args.robot)
        model = load_model_with_ground(mjcf_path)
        data = mujoco.MjData(model)
        
        # 覆盖帧率
        if args.fps:
            motion_data['fps'] = args.fps
        
        # 处理帧范围
        if args.end_frame > 0:
            end = min(args.end_frame, motion_data['num_frames'])
            motion_data['pos'] = motion_data['pos'][args.start_frame:end]
            motion_data['quat'] = motion_data['quat'][args.start_frame:end]
            if motion_data['joint_pos'] is not None:
                motion_data['joint_pos'] = motion_data['joint_pos'][args.start_frame:end]
            motion_data['num_frames'] = end - args.start_frame
        
        play_robot_motion(model, data, motion_data, loop=args.loop)
    
    elif npz_type == 'amass':
        print("AMASS/SMPL 格式需要使用 smpl_replay.py 脚本播放")
        print("使用命令：python scripts/smpl_replay.py --motion_file", npz_path)
        sys.exit(1)
    
    else:
        print("错误：无法识别的 NPZ 文件格式")
        print("支持的格式：")
        print("  - 机器人运动格式（包含 body_pos_w, body_quat_w）")
        print("  - AMASS/SMPL 格式（包含 trans, root_orient, pose_body）")
        sys.exit(1)


if __name__ == "__main__":
    main()