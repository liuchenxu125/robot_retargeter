#!/usr/bin/env python3
"""在 MuJoCo 中按帧播放机器人 CSV，默认 30 Hz。

CSV 无表头，每行：x,y,z,qx,qy,qz,qw,joint_0,...（米、弧度）。
关节顺序须与模型 qpos 一致，模型须包含位于 qpos 开头的浮动基座。

示例（在项目根目录执行）：
    python tools/play_csv.py --csv dataset/lafan1_g1/walk1_subject1.csv --robot g1
    python tools/play_csv.py --csv motion.csv --robot CASBOT02 --fps 60 --loop
    python tools/play_csv.py --csv motion.csv --xml path/to/robot.xml
"""

import argparse
from pathlib import Path
import threading
import time

import mujoco
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_model_path(robot: str | None, xml: Path | None) -> Path:
    if xml is not None:
        return xml.expanduser().resolve()
    config_path = PROJECT_ROOT / "config" / "robot" / f"{robot}.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    path = Path(config["robot_xml_path"]).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model(xml_path: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms):
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05], rgba=[0.25, 0.3, 0.35, 1],
            contype=0, conaffinity=0,
        )
    if not list(spec.worldbody.lights):
        spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1])
    return spec.compile()


def load_csv(csv_path: Path, model: mujoco.MjModel, quat_order: str) -> np.ndarray:
    if (model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) != 1):
        raise ValueError("模型必须只有一个浮动基座，且位于 qpos 开头")
    frames = np.loadtxt(csv_path, delimiter=",", ndmin=2)
    if frames.shape[0] == 0 or frames.shape[1] != model.nq:
        raise ValueError(f"CSV 形状为 {frames.shape}，需要非空数据且列数等于模型 nq={model.nq}")
    if not np.isfinite(frames).all():
        raise ValueError("CSV 包含 NaN 或无穷大")
    if quat_order == "xyzw":
        frames[:, 3:7] = frames[:, [6, 3, 4, 5]]
    norms = np.linalg.norm(frames[:, 3:7], axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("CSV 包含零四元数")
    frames[:, 3:7] /= norms[:, None]
    # 同时规范化可能存在的球关节四元数。
    for frame in frames:
        mujoco.mj_normalizeQuat(model, frame)
    return frames


def play(model: mujoco.MjModel, frames: np.ndarray, fps: float, loop: bool) -> None:
    import glfw
    import mujoco.viewer

    data = mujoco.MjData(model)
    lock = threading.Lock()
    state = {"paused": False, "frame": 0, "reset_clock": False}

    def on_key(key: int) -> None:
        with lock:
            if key == glfw.KEY_SPACE:
                state["paused"] = not state["paused"]
            elif key in (glfw.KEY_COMMA, glfw.KEY_PERIOD):
                delta = -1 if key == glfw.KEY_COMMA else 1
                state["frame"] = max(0, min(len(frames) - 1, state["frame"] + delta))
                state["paused"] = True
            elif key == glfw.KEY_R:
                state["frame"] = 0
                state["paused"] = False
            else:
                return
            state["reset_clock"] = True

    data.qpos[:] = frames[0]
    mujoco.mj_forward(model, data)
    print("控制：Space 暂停/继续，逗号/句号 后退/前进一帧，R 重播；关闭窗口退出")
    interval = 1.0 / fps
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = frames[0, :3]
        viewer.cam.distance = max(2.0, model.stat.extent * 1.5)
        deadline = time.perf_counter() + interval
        while viewer.is_running():
            now = time.perf_counter()
            with lock:
                if state["reset_clock"]:
                    deadline = now + interval
                    state["reset_clock"] = False
                if not state["paused"] and now >= deadline:
                    # 使用累计时钟，避免渲染耗时导致播放速度逐渐变慢。
                    steps = int((now - deadline) / interval) + 1
                    cursor = state["frame"] + steps
                    if loop:
                        cursor %= len(frames)
                    elif cursor >= len(frames) - 1:
                        cursor = len(frames) - 1
                        state["paused"] = True
                    state["frame"] = cursor
                    deadline += steps * interval
                cursor = state["frame"]
            # 运动学回放，不进行动力学积分，以保留 CSV 原始姿态。
            with viewer.lock():
                data.qpos[:] = frames[cursor]
                data.time = cursor / fps
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(min(0.01, max(0.001, deadline - time.perf_counter())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True, help="无表头 CSV 路径")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--robot", help="config/robot 下的配置名称，例如 g1、CASBOT02")
    source.add_argument("--xml", type=Path, help="直接指定 MuJoCo XML 路径")
    parser.add_argument("--fps", type=float, default=30.0, help="播放频率 Hz（默认 30）")
    parser.add_argument("--loop", action="store_true", help="循环播放，默认播完后停在最后一帧")
    parser.add_argument("--quat-order", choices=("xyzw", "wxyz"), default="xyzw", help="CSV 根四元数顺序（默认 xyzw）")
    parser.add_argument("--no-viewer", action="store_true", help="只校验模型和 CSV，不打开窗口")
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps 必须是大于 0 的有限数")
    try:
        model = load_model(resolve_model_path(args.robot, args.xml))
        frames = load_csv(args.csv.expanduser(), model, args.quat_order)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    print(f"帧数：{len(frames)}，频率：{args.fps:g} Hz，时长：{len(frames) / args.fps:.2f} 秒")
    if not args.no_viewer:
        play(model, frames, args.fps, args.loop)


if __name__ == "__main__":
    main()
