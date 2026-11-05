"""Visualize motion trajectories of multiple robots simultaneously with MuJoCo.

File naming convention: <motion_name>_<robot_name>.csv, for example Form_1_stageii_g1.csv.
So you only need to provide one motion name and a list of robot names; the script will
construct the file paths automatically.

Note: MuJoCo free-joint qpos ordering is [pos(xyz), quat(wxyz)].
After loading CSV files, quaternions must be converted from xyzw to wxyz.

Usage example:
    python scripts/multi_robot_visualize.py \
        --motion body_check_001__A548_M_from_g1 \
        --robots h2 \
        --source_fps 120 \
        --render_fps 30
"""

import argparse
import csv
import math
import os
import re
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from tqdm import tqdm

from convert_bones_to_lafan1 import euler_xyz_deg_to_quat_xyzw, safe_float

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Robot config directory: each robot has a <robot_name>.yaml that contains a
# "robot_xml_path" field (relative to the project root) pointing to its MJCF.
ROBOT_CONFIG_DIR = os.path.join(PROJECT_DIR, "config", "robot")

MOTION_DIR = os.path.join(SCRIPT_DIR, "..", "output_data", "robot_motion")
# Original G1 clips. Seed output keeps these basenames but stores casbot qpos,
# so the G1 side is loaded from the source dataset when it is not already converted.
G1_SOURCE_DIRS = [
    "/home/casbot/Desktop/SuperMimic-Reproduced/data/seed/g1/csv",
    "/home/casbot/Desktop/SuperMimic-Reproduced/data/raw/lafan1/g1",
]
_CSV_INDEX: dict[str, dict[str, list[str]]] = {}
_ROBOT_NQ: dict[str, int] = {}


def available_robots() -> list[str]:
    """List robot names by scanning config/robot/*.yaml (filename = robot name)."""
    if not os.path.isdir(ROBOT_CONFIG_DIR):
        return []
    names = [
        os.path.splitext(f)[0]
        for f in os.listdir(ROBOT_CONFIG_DIR)
        if f.endswith(".yaml")
    ]
    return sorted(names)


def get_robot_xml(robot: str) -> str:
    """Resolve a robot's MJCF path by reading robot_xml_path from its yaml config.

    The config file is config/robot/<robot>.yaml and robot_xml_path is stored
    relative to the project root.
    """
    config_path = os.path.join(ROBOT_CONFIG_DIR, f"{robot}.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Robot config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    xml_path = config.get("robot_xml_path")
    if not xml_path:
        raise KeyError(f"'robot_xml_path' missing in config: {config_path}")
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(PROJECT_DIR, xml_path)
    return xml_path


def _norm_motion_name(name: str) -> str:
    """Treat one or more underscores as the same, so A464 matches __A464."""
    return re.sub(r"_+", "_", name)


def _csv_index(root: str) -> dict[str, list[str]]:
    root = os.path.abspath(root)
    cached = _CSV_INDEX.get(root)
    if cached is not None:
        return cached
    index: dict[str, list[str]] = {}
    if os.path.isdir(root):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if not filename.endswith(".csv"):
                    continue
                stem = filename[:-4]
                index.setdefault(_norm_motion_name(stem), []).append(os.path.join(dirpath, filename))
    _CSV_INDEX[root] = index
    return index


def _robot_nq(robot: str) -> int:
    cached = _ROBOT_NQ.get(robot)
    if cached is not None:
        return cached
    model = mujoco.MjModel.from_xml_path(get_robot_xml(robot))
    _ROBOT_NQ[robot] = int(model.nq)
    return int(model.nq)


def _first_line_is_header(csv_path: str) -> bool:
    with open(csv_path, "r", encoding="utf-8") as f:
        token = f.readline().split(",", 1)[0].strip()
    if token == "":
        return True
    try:
        float(token)
    except ValueError:
        return True
    return False


def _numeric_ncols(csv_path: str) -> int:
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.split(",", 1)[0].strip()
            try:
                float(token)
            except ValueError:
                continue
            return len(line.strip().split(","))
    raise ValueError(f"Motion file has no numeric rows: {csv_path}")


def _lookup_paths(root: str, motion: str, robot: str) -> list[str]:
    index = _csv_index(root)
    found: list[str] = []
    for key in (_norm_motion_name(motion), _norm_motion_name(f"{motion}_{robot}")):
        found.extend(index.get(key, []))
    return found


def resolve_motion_csv(motion: str, robot: str, motion_dir: str) -> tuple[str, str]:
    """Find a motion CSV for one robot.

    Search order:
    1. output_data/robot_motion/<motion>_<robot>.csv
    2. The same folder and its subfolders (seed/, lafan1/), matching the original
       filename. Extra underscores are ignored, and the column count must match
       that robot.
    3. For g1 only, the original Bones or LAFAN1 dataset.

    Returns (path, kind) where kind is "lafan" or "bones".
    """
    legacy = os.path.join(motion_dir, f"{motion}_{robot}.csv")
    if os.path.isfile(legacy):
        kind = "bones" if _first_line_is_header(legacy) else "lafan"
        return legacy, kind

    nq = _robot_nq(robot)
    lafan_hits = [
        path
        for path in _lookup_paths(motion_dir, motion, robot)
        if not _first_line_is_header(path) and _numeric_ncols(path) == nq
    ]
    if robot == "g1":
        for source_dir in G1_SOURCE_DIRS:
            for path in _lookup_paths(source_dir, motion, robot):
                if _first_line_is_header(path):
                    continue
                if _numeric_ncols(path) == nq:
                    lafan_hits.append(path)

    if lafan_hits:
        return _pick_motion_path(lafan_hits, motion, robot), "lafan"

    if robot == "g1":
        bones_hits: list[str] = []
        for source_dir in G1_SOURCE_DIRS:
            for path in _lookup_paths(source_dir, motion, robot):
                if _first_line_is_header(path):
                    bones_hits.append(path)
        if bones_hits:
            return _pick_motion_path(bones_hits, motion, robot), "bones"

    searched = [os.path.abspath(motion_dir)]
    if robot == "g1":
        searched.extend(G1_SOURCE_DIRS)
    raise FileNotFoundError(
        f"Motion file not found for '{motion}' on {robot}. Looked in: " + ", ".join(searched)
    )


def _pick_motion_path(paths: list[str], motion: str, robot: str) -> str:
    def score(path: str) -> tuple[int, int]:
        stem = os.path.splitext(os.path.basename(path))[0]
        rank = 0
        if stem == f"{motion}_{robot}":
            rank += 100
        if stem == motion:
            rank += 80
        if stem.endswith("_" + robot):
            rank += 40
        return rank, -path.count(os.sep)

    return max(paths, key=score)


def load_bones_motion(csv_path: str, root_scale: float = 0.01) -> np.ndarray:
    """Load a Bones Seed CSV (header, Euler degrees, centimeters) as MuJoCo qpos."""
    rows: list[list[float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        joint_cols = [c for c in reader.fieldnames if c.endswith("_dof")]
        for row in reader:
            qx, qy, qz, qw = euler_xyz_deg_to_quat_xyzw(
                safe_float(row, "root_rotateX"),
                safe_float(row, "root_rotateY"),
                safe_float(row, "root_rotateZ"),
            )
            rows.append(
                [
                    safe_float(row, "root_translateX") * root_scale,
                    safe_float(row, "root_translateY") * root_scale,
                    safe_float(row, "root_translateZ") * root_scale,
                    qw,
                    qx,
                    qy,
                    qz,
                    *[math.radians(safe_float(row, joint)) for joint in joint_cols],
                ]
            )
    if not rows:
        raise ValueError(f"CSV has no frames: {csv_path}")
    return np.asarray(rows, dtype=np.float64)


def load_motion(csv_path: str) -> np.ndarray:
    """Load a motion CSV and convert the base quaternion from xyzw to wxyz.

    Each returned row can be written directly to MuJoCo data.qpos:
        [pos(xyz), quat(wxyz), joint_0, joint_1, ...]
    """
    data = np.loadtxt(csv_path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 7:
        raise ValueError(
            f"Motion file {csv_path} must have at least 7 columns (pos3 + quat4), got {data.shape[1]}"
        )

    qpos = data.copy()
    # CSV: first 3 columns are position, columns 4-7 are quaternion (xyzw)
    quat_xyzw = data[:, 3:7]
    # MuJoCo free joint requires quaternion in wxyz order
    qpos[:, 3] = quat_xyzw[:, 3]  # w
    qpos[:, 4] = quat_xyzw[:, 0]  # x
    qpos[:, 5] = quat_xyzw[:, 1]  # y
    qpos[:, 6] = quat_xyzw[:, 2]  # z
    return qpos


def build_combined_spec(robots: list[str]) -> mujoco.MjSpec:
    """Merge multiple robot MJCF models into one scene.

    Use a prefix (robot_name + "_") to avoid body/joint/geom name conflicts across models.
    Each robot will have a free joint named "<robot>_floating_base_joint".
    """
    spec = mujoco.MjSpec()
    spec.modelname = "multi_robot"

    # Increase global headlight illumination. A non-reflective ground can make
    # the scene too dark under a single directional light, so raise ambient/diffuse.
    spec.visual.headlight.ambient = [0.5, 0.5, 0.5]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.specular = [0.3, 0.3, 0.3]

    # Skybox texture
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=512,
    )
    # White-ish grid texture: cell interior light gray, grid lines dark gray.
    grid_res = 256          # Pixel resolution per tile
    line_px = 3             # Grid line width (pixels)
    grid_rgba = np.zeros((grid_res, grid_res, 4), dtype=np.uint8)
    bg_color = (100, 100, 100)    # Cell background color
    line_color = (50, 50, 50)  # Grid line color
    # Fill background first, then draw border lines on all four sides
    grid_rgba[:, :, :3] = bg_color
    grid_rgba[:, :, 3] = 255
    grid_rgba[:line_px, :, :3] = line_color
    grid_rgba[-line_px:, :, :3] = line_color
    grid_rgba[:, :line_px, :3] = line_color
    grid_rgba[:, -line_px:, :3] = line_color

    grid_tex = spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=grid_res,
        height=grid_res,
        nchannel=4,
    )
    grid_tex.data = grid_rgba.reshape(-1).tobytes()

    spec.add_material(
        name="groundplane",
        textures=["", "groundplane"],
        texrepeat=[10, 10],
        texuniform=True,
        reflectance=0.0,
    )

    spec.worldbody.add_light(
        pos=[0, 0, 20.0],
        dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.7, 0.7, 0.7],
        specular=[0.3, 0.3, 0.3],
    )

    # Add a ground plane using the grid material defined above
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],  # For planes, the first two size values at 0 mean infinite extent
        material="groundplane",
        pos=[0, 0, 0],
    )

    for robot in robots:
        robot_spec = mujoco.MjSpec.from_file(get_robot_xml(robot))
        for j in robot_spec.joints:
            if j.type == mujoco.mjtJoint.mjJNT_FREE:
                j.name = "floating_base_joint"
                break
        frame = spec.worldbody.add_frame()
        spec.attach(robot_spec, prefix=f"{robot}_", frame=frame)

    for g in spec.geoms:
        g.contype = 0
        g.conaffinity = 0

    return spec


def get_qpos_start(model: mujoco.MjModel, freejoint_name: str) -> int:
    """Return the starting qpos index of a robot free joint in the merged model."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name)
    if jid < 0:
        raise ValueError(f"Joint not found in model: {freejoint_name}")
    return model.jnt_qposadr[jid]


def main() -> None:
    parser = argparse.ArgumentParser(description="MuJoCo multi-robot motion visualization")
    parser.add_argument(
        "--motion",
        default="body_check_001__A548_M_from_g1",
        help="Motion name. Matches <motion>_<robot>.csv, or the original filename under seed/ and lafan1/",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        default=[
                # "unitree_a2w",
                # "agibot_x2",
                #  "g1", 
                 "h2", 
                #  "r1", 
                #  "t800",
                #  "pm01", 
                #  "unitree_a2",
                #  "limx_oli",
                #  "pnd_adam",
                #  "booster_t1",
                #  "jaka_pi",
                #  "hightorque_hi",
                 
                 ],
        choices=available_robots(),
        help="List of robot names to visualize, choices: " + ", ".join(available_robots()),
    )
    parser.add_argument(
        "--motion_dir",
        default=MOTION_DIR,
        help="Directory containing motion CSV files",
    )
    parser.add_argument("--source_fps", type=float, default=30.0, help="Source data frame rate (original capture FPS)")
    parser.add_argument(
        "--render_fps",
        type=float,
        default=60.0,
        help="Actual rendering FPS, e.g. 60/30; if lower than source FPS, frames are skipped to reduce load while preserving playback speed",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args()

    robots = args.robots

    # Build file paths and load each robot's motion data
    robot_qpos: dict[str, np.ndarray] = {}
    for robot in robots:
        csv_path, kind = resolve_motion_csv(args.motion, robot, args.motion_dir)
        print(f"{robot}: {csv_path}")
        robot_qpos[robot] = load_bones_motion(csv_path) if kind == "bones" else load_motion(csv_path)

    n_frames = min(q.shape[0] for q in robot_qpos.values())
    print(
        "  ".join(f"{r} frames={robot_qpos[r].shape[0]} (qpos {robot_qpos[r].shape[1]})" for r in robots)
        + f"  ->  playing {n_frames} frames"
    )

    # Build the merged model
    spec = build_combined_spec(robots)
    model = spec.compile()
    data = mujoco.MjData(model)

    # Locate each robot's starting qpos index (via free joint)
    robot_start: dict[str, int] = {}
    robot_dim: dict[str, int] = {}
    for robot in robots:
        start = get_qpos_start(model, f"{robot}_floating_base_joint")
        dim = robot_qpos[robot].shape[1]
        if start + dim > model.nq:
            raise ValueError(
                f"qpos dimension mismatch: {robot}[{start}:{start + dim}] exceeds model.nq={model.nq}"
            )
        robot_start[robot] = start
        robot_dim[robot] = dim

    # Precompute XY placement offsets on the ground in an outward grid from the origin.
    # Use a near-square grid layout so added robots spread out naturally.
    spacing = 2.0  # Robot spacing (meters)
    n = len(robots)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    robot_offset: dict[str, tuple[float, float]] = {}
    for idx, robot in enumerate(robots):
        col = idx % cols
        row = idx // cols
        # Center the grid so its center lies at the origin
        dx = (col - (cols - 1) / 2.0) * spacing
        dy = (row - (rows - 1) / 2.0) * spacing
        robot_offset[robot] = (dx, dy)

    def apply_frame(i: int) -> None:
        for idx, robot in enumerate(robots):
            start = robot_start[robot]
            dim = robot_dim[robot]
            data.qpos[start : start + dim] = robot_qpos[robot][i]
            # Spread robots on the XY plane by grid offsets
            # Free-joint qpos order is [x, y, z, qw, qx, qy, qz], so x=start and y=start+1
            dx, dy = robot_offset[robot]
            data.qpos[start] += dx
            data.qpos[start + 1] += dy
        # Visualization only: forward kinematics is enough for global body poses.
        # Skip heavier mj_forward stages like collision, constraints, and dynamics.
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

    # Frame skipping: source data at args.source_fps, render every step frames for render_fps.
    # Use dt = step / source_fps to preserve real-time playback speed.
    step = max(1, round(args.source_fps / args.render_fps))
    dt = step / args.source_fps
    print(f"source_fps={args.source_fps:g}, render_fps≈{args.source_fps / step:g}, render_every={step} frame(s)")
    apply_frame(0)

    # Playback control state, updated by keyboard callback
    state = {"frame": 0, "paused": False, "replay": False}

    def key_callback(keycode: int) -> None:
        # Space: pause/resume; R: replay from start
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
            print("Paused" if state["paused"] else "Resumed")
        elif keycode in (ord("r"), ord("R")):
            state["replay"] = True
            print("Replay")

    print("Controls: [Space] Pause/Resume  [R] Replay")

    # Progress bar in rendered frames (total is after step-based frame skipping)
    total_render_frames = math.ceil(n_frames / step)
    pbar = tqdm(total=total_render_frames, desc="Playback", unit="frame", dynamic_ncols=True)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            t0 = time.time()

            # Handle replay request
            if state["replay"]:
                state["frame"] = 0
                state["replay"] = False
                state["paused"] = False
                pbar.reset(total=total_render_frames)

            apply_frame(state["frame"])
            viewer.sync()

            # Sync progress bar to current frame (counted in rendered frames)
            pbar.n = min(state["frame"] // step, total_render_frames)
            pbar.refresh()

            if not state["paused"]:
                state["frame"] += step
                if state["frame"] >= n_frames:
                    if args.loop:
                        state["frame"] = 0
                        pbar.reset(total=total_render_frames)
                    else:
                        # Stay at the last frame; wait for R replay or window close
                        state["frame"] = n_frames - 1
                        state["paused"] = True
                        pbar.n = total_render_frames
                        pbar.refresh()

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            else:
                print(
                    f"Warning: Processing one frame took {elapsed:.3f}s, exceeding the target frame interval {dt:.3f}s"
                )

    pbar.close()


if __name__ == "__main__":
    main()
