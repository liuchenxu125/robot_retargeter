#!/usr/bin/env python3
"""Retarget every Bones Seed G1 CSV onto casbot02_primitives.

Output files keep the source filename exactly and are written to
output_data/robot_motion/seed/. Finished files are skipped, so the run
can be stopped and started again.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
import tempfile
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

DEFAULT_SRC = Path("/home/casbot/Desktop/SuperMimic-Reproduced/data/seed/g1/csv")
DEFAULT_OUT = PROJECT_ROOT / "output_data" / "robot_motion" / "seed"
SOURCE_CONFIG = PROJECT_ROOT / "config" / "robot" / "g1.yaml"
TARGET_CONFIG = PROJECT_ROOT / "config" / "robot" / "casbot02_primitives.yaml"
FPS = 120.0
FORMAT = "bones"
ROOT_SCALE = 0.01

_WORKER = None


def list_source_csvs(src_dir: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename.endswith(".csv"):
                files.append(Path(dirpath) / filename)
    files.sort()
    return files


def bones_csv_to_qpos(path: Path):
    import numpy as np
    from convert_bones_to_lafan1 import euler_xyz_deg_to_quat_xyzw, safe_float

    rows: list[list[float]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        joint_cols = [name for name in reader.fieldnames if name.endswith("_dof")]
        for row in reader:
            tx = safe_float(row, "root_translateX") * ROOT_SCALE
            ty = safe_float(row, "root_translateY") * ROOT_SCALE
            tz = safe_float(row, "root_translateZ") * ROOT_SCALE
            qx, qy, qz, qw = euler_xyz_deg_to_quat_xyzw(
                safe_float(row, "root_rotateX"),
                safe_float(row, "root_rotateY"),
                safe_float(row, "root_rotateZ"),
            )
            values = [tx, ty, tz, qx, qy, qz, qw]
            values.extend(math.radians(safe_float(row, name)) for name in joint_cols)
            rows.append([float(f"{value:.6f}") for value in values])
    if not rows:
        raise ValueError(f"CSV has no frames: {path}")

    data = np.asarray(rows, dtype=np.float64).astype(np.float32)
    quat_xyzw = data[:, 3:7].copy()
    data[:, 3] = quat_xyzw[:, 3]
    data[:, 4] = quat_xyzw[:, 0]
    data[:, 5] = quat_xyzw[:, 1]
    data[:, 6] = quat_xyzw[:, 2]
    return data


def lafan_csv_to_qpos(path: Path):
    """Load a headerless LAFAN1 CSV: xyz meters, quaternion xyzw, joints in radians."""
    import numpy as np

    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 7:
        raise ValueError(f"LAFAN1 CSV needs at least 7 columns: {path}")
    qpos = data.astype(np.float32, copy=True)
    quat_xyzw = qpos[:, 3:7].copy()
    qpos[:, 3] = quat_xyzw[:, 3]
    qpos[:, 4] = quat_xyzw[:, 0]
    qpos[:, 5] = quat_xyzw[:, 1]
    qpos[:, 6] = quat_xyzw[:, 2]
    return qpos


class RetargetWorker:
    def __init__(self, out_dir: Path) -> None:
        import mujoco
        import numpy as np
        import smpl_replay
        import yaml
        from robot_replay import (
            append_target_contact_keypoints,
            build_robot_replay_buffers,
            build_robot_retarget_keypoints,
            canonicalize_contact_name,
            compute_leg_displacement_scale,
            compute_robot_contact_sequence,
            compute_robot_link_lengths,
            load_effective_knee_angle_offset_degrees,
            load_path_config,
            load_robot_links_config,
            load_scalar_float_config,
            load_scalar_int_config,
            load_yaml_body_list_config,
            offset_keypoints_by_contact_height,
            scale_keypoint_frame_displacements,
        )
        from robot_retarget import RobotRetarget

        # One progress bar per frame would bury a 142k-file run.
        smpl_replay.tqdm = lambda iterable, **_kwargs: iterable
        import robot_retarget
        robot_retarget.tqdm = lambda iterable, **_kwargs: iterable

        original_from_xml = mujoco.MjModel.from_xml_path
        model_cache: dict[str, object] = {}

        def cached_from_xml(path):
            key = str(path)
            model = model_cache.get(key)
            if model is None:
                model = original_from_xml(key)
                model_cache[key] = model
            return model

        mujoco.MjModel.from_xml_path = cached_from_xml

        self.out_dir = out_dir
        self.tmp_dir = out_dir.parent / f".{out_dir.name}_tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.np = np
        self.RobotRetarget = RobotRetarget
        self.build_robot_replay_buffers = build_robot_replay_buffers
        self.build_robot_retarget_keypoints = build_robot_retarget_keypoints
        self.append_target_contact_keypoints = append_target_contact_keypoints
        self.compute_robot_contact_sequence = compute_robot_contact_sequence
        self.compute_leg_displacement_scale = compute_leg_displacement_scale
        self.scale_keypoint_frame_displacements = scale_keypoint_frame_displacements
        self.offset_keypoints_by_contact_height = offset_keypoints_by_contact_height

        self.source_config = SOURCE_CONFIG
        self.target_config = TARGET_CONFIG
        self.source_mjcf = load_path_config(self.source_config, "robot_xml_path").expanduser().resolve()
        self.target_mjcf = load_path_config(self.target_config, "robot_xml_path").expanduser().resolve()
        self.source_links = load_robot_links_config(self.source_config)
        self.target_links = load_robot_links_config(self.target_config)
        self.source_contacts = load_yaml_body_list_config(self.source_config, "contact_links")
        self.contact_names = tuple(canonicalize_contact_name(name) for name in self.source_contacts)
        self.contact_vel_window = load_scalar_int_config(
            self.source_config, "contact_vel_calculate_window", default=6
        )
        self.contact_vel_threshold = load_scalar_float_config(
            self.source_config, "contact_vel_threshold", default=0.5
        )
        self.contact_height_threshold = load_scalar_float_config(
            self.source_config, "contact_height_threshold", default=0.025
        )
        self.contact_height_lpf_alpha = load_scalar_float_config(
            self.source_config, "contact_height_lpf_alpha", default=0.2
        )
        self.link_lengths = compute_robot_link_lengths(
            config_path=self.target_config,
            robot_mjcf_path=self.target_mjcf,
        )
        self.knee_offset = load_effective_knee_angle_offset_degrees(self.target_config)

        with self.target_config.open("r", encoding="utf-8") as handle:
            target_yaml = yaml.safe_load(handle)
        self.ik_match_table = target_yaml.get("ik_match_table", {})
        self.joints_limit_offset = target_yaml.get("joints_limit_offset_degrees", {})
        self.contact_position_cost = float(target_yaml.get("contact_pos_fixed_factor", 10.0))
        self.collision_avoidance = target_yaml.get("collision_avoidance")
        self.plant_feet = bool(target_yaml.get("plant_feet_on_ground", False))
        self.plant_clearance = float(target_yaml.get("plant_feet_clearance", 0.0))
        robot_xml = target_yaml.get("robot_xml_path", "")
        if robot_xml and not os.path.isabs(robot_xml):
            robot_xml = str(PROJECT_ROOT / robot_xml)
        self.robot_xml = robot_xml
        self.contact_body_names = target_yaml.get("contact_links")
        self.retargeter = None
        self.home_qpos = None

    def build_payload(self, motion_file: Path, qpos):
        positions, quaternions, contact_positions = self.build_robot_replay_buffers(
            qpos_frames=qpos,
            source_robot_mjcf_path=self.source_mjcf,
            source_robot_links=self.source_links,
            contact_body_names=self.source_contacts,
        )
        _speeds, contact_states = self.compute_robot_contact_sequence(
            contact_positions=contact_positions,
            fps=FPS,
            vel_window=self.contact_vel_window,
            vel_threshold=self.contact_vel_threshold,
            height_threshold=self.contact_height_threshold,
        )
        (
            keypoints,
            keypoint_quaternions,
            raw_quaternions,
            source_link_vectors,
            _link_scales,
            _link_scale_is_static,
        ) = self.build_robot_retarget_keypoints(
            source_positions=positions,
            source_quaternions=quaternions,
            target_robot_link_lengths=self.link_lengths,
            source_robot_config_path=self.source_config,
            target_robot_config_path=self.target_config,
        )
        keypoints, keypoint_quaternions, extra_names = self.append_target_contact_keypoints(
            keypoints=keypoints,
            keypoint_quaternions=keypoint_quaternions,
            anchor_raw_quaternions=raw_quaternions,
            target_robot_links=self.target_links,
            target_robot_config_path=self.target_config,
            target_robot_mjcf_path=self.target_mjcf,
        )
        displacement_scale, _robot_leg, _source_leg = self.compute_leg_displacement_scale(
            robot_link_lengths=self.link_lengths,
            skeleton_link_vectors=source_link_vectors,
            knee_angle_offset_degrees=self.knee_offset,
        )
        keypoints = self.scale_keypoint_frame_displacements(
            keypoints=keypoints,
            displacement_scale=displacement_scale,
            root_keypoint_idx=0,
        )
        keypoint_names = ["hips_mean", *list(self.target_links.keys()), *extra_names]
        keypoints, _height_offsets = self.offset_keypoints_by_contact_height(
            keypoints=keypoints,
            keypoint_names=keypoint_names,
            contact_names=self.contact_names,
            contact_positions=contact_positions,
            contact_states=contact_states,
            height_lpf_alpha=self.contact_height_lpf_alpha,
        )
        return {
            "keypoint_names": keypoint_names,
            "positions": keypoints.astype(self.np.float32),
            "quaternions": keypoint_quaternions.astype(self.np.float32),
            "fps": float(FPS),
            "contact_names": list(self.contact_names),
            "contact_states": contact_states.astype(self.np.bool_),
            "motion_file": motion_file,
        }

    def ensure_retargeter(self, payload) -> None:
        if self.retargeter is not None:
            return
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as handle:
            pickle.dump(payload, handle)
            keypoint_path = handle.name
        try:
            self.retargeter = self.RobotRetarget(
                model_path=self.robot_xml,
                keypoint_path=keypoint_path,
                ik_match_table=self.ik_match_table,
                solver="daqp",
                verbose=False,
                render_debug=False,
                joints_limit_offset_degrees=self.joints_limit_offset,
                contact_body_names=self.contact_body_names,
                contact_position_cost=self.contact_position_cost,
                collision_avoidance=self.collision_avoidance,
            )
        finally:
            os.remove(keypoint_path)
        self.home_qpos = self.retargeter.configuration.data.qpos.copy()

    def apply_payload(self, payload) -> None:
        retargeter = self.retargeter
        retargeter.keypoint_names = payload["keypoint_names"]
        retargeter.keypoint_name_to_idx = {
            name: idx for idx, name in enumerate(retargeter.keypoint_names)
        }
        retargeter.keypoints_pos = payload["positions"]
        retargeter.keypoints_quat = payload["quaternions"]
        retargeter.contact_names = payload["contact_names"]
        retargeter.contact_state_name_to_idx = {
            name: idx for idx, name in enumerate(retargeter.contact_names)
        }
        retargeter.contact_seq = payload["contact_states"]
        retargeter.fps = float(payload["fps"])
        retargeter.time_step = 1.0 / retargeter.fps
        retargeter.num_frames = retargeter.keypoints_pos.shape[0]
        retargeter.num_keypoints = retargeter.keypoints_pos.shape[1]
        retargeter.contact_targets = []
        retargeter.setup_contact_targets()
        retargeter.result_pos = []
        retargeter.configuration.data.qpos[:] = self.home_qpos
        retargeter.configuration.data.qvel[:] = 0
        import mujoco
        mujoco.mj_forward(retargeter.model, retargeter.configuration.data)

    def run(self, src: str) -> tuple[str, str, int, float, str]:
        source = Path(src)
        started = time.perf_counter()
        try:
            if FORMAT == "lafan":
                qpos = lafan_csv_to_qpos(source)
            else:
                qpos = bones_csv_to_qpos(source)
            payload = self.build_payload(source, qpos)
            self.ensure_retargeter(payload)
            self.apply_payload(payload)
            self.retargeter.retarget()
            if self.plant_feet:
                self.retargeter.plant_feet_on_ground(clearance=self.plant_clearance)
            self.retargeter.apply_shoulder_roll_outward_offset(degrees=3.0)
            temporary = self.tmp_dir / source.name
            final_path = self.out_dir / source.name
            self.retargeter.save_results_as_csv(str(temporary))
            os.replace(temporary, final_path)
            audit = Path(str(temporary) + ".collision.json")
            if audit.exists():
                os.replace(audit, str(final_path) + ".collision.json")
            frames = int(qpos.shape[0])
            return ("ok", source.name, frames, time.perf_counter() - started, "")
        except Exception as exc:
            traceback.print_exc()
            return ("fail", source.name, 0, time.perf_counter() - started, str(exc))


def _init_worker(out_dir: str, fps: float, source_format: str) -> None:
    global DEFAULT_OUT, FPS, FORMAT
    DEFAULT_OUT = Path(out_dir)
    FPS = float(fps)
    FORMAT = source_format


def _worker_run(src: str) -> tuple[str, str, int, float, str]:
    global _WORKER
    if _WORKER is None:
        _WORKER = RetargetWorker(DEFAULT_OUT)
    return _WORKER.run(src)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-retarget Seed G1 CSVs onto casbot02")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many pending files")
    parser.add_argument("--only", default="", help="Process only this source filename")
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument(
        "--format",
        choices=("bones", "lafan"),
        default=FORMAT,
        help="bones: Seed Euler/cm CSV. lafan: headerless xyzw quaternion CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = args.src.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    global DEFAULT_OUT, FPS, FORMAT
    DEFAULT_OUT = out_dir
    FPS = float(args.fps)
    FORMAT = args.format

    sources = list_source_csvs(src_dir)
    if args.only:
        sources = [path for path in sources if path.name == args.only]
        if not sources:
            raise FileNotFoundError(f"No source CSV named {args.only}")
    already = [path for path in sources if (out_dir / path.name).is_file()]
    pending = [path for path in sources if not (out_dir / path.name).is_file()]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        f"[info] source={len(sources)} pending={len(pending)} "
        f"already={len(already)} workers={args.workers} fps={FPS} format={FORMAT} out={out_dir}",
        flush=True,
    )
    if not pending:
        return

    log_path = out_dir.parent / f"{out_dir.name}_batch.log"
    fail_path = out_dir.parent / f"{out_dir.name}_batch_failures.txt"
    workers = 1 if args.only or args.limit == 1 else max(1, args.workers)
    ctx = get_context("spawn")
    ok = 0
    failed = 0
    frames = 0
    started = time.perf_counter()
    sources_pending = [str(path) for path in pending]

    def consume(results, log, fail_log) -> None:
        nonlocal ok, failed, frames
        for index, (status, name, file_frames, seconds, message) in enumerate(results, start=1):
            if status == "ok":
                ok += 1
                frames += file_frames
            else:
                failed += 1
                fail_log.write(f"{name}\t{message}\n")
                fail_log.flush()
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed > 0 else 0.0
            remaining = (len(pending) - index) / rate if rate > 0 else 0.0
            line = (
                f"[{index}/{len(pending)}] {status} {name} "
                f"frames={file_frames} {seconds:.1f}s eta={remaining / 3600:.1f}h"
            )
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

    with log_path.open("a", encoding="utf-8") as log, fail_path.open("a", encoding="utf-8") as fail_log:
        # One file runs in this process. A spawn pool deadlocks on a single worker here.
        if workers == 1:
            _init_worker(str(out_dir), FPS, FORMAT)
            consume((_worker_run(path) for path in sources_pending), log, fail_log)
        else:
            with ctx.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(str(out_dir), FPS, FORMAT),
            ) as pool:
                consume(pool.imap_unordered(_worker_run, sources_pending, chunksize=1), log, fail_log)

    elapsed = time.perf_counter() - started
    print(
        f"[done] ok={ok} failed={failed} frames={frames} "
        f"elapsed={elapsed / 3600:.2f}h out={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
