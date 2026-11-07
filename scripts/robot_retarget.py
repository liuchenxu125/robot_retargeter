"""Retarget keypoint motions to a robot and export MuJoCo qpos CSV.

This script loads robot and keypoint settings from a robot YAML config,
runs IK-based retargeting frame by frame, and saves the resulting motion
as a CSV under output_data/robot_motion.

Usage:
    # Run with a specific robot config from terminal
    python scripts/robot_retarget.py --config config/robot/h2.yaml

    # Override render_debug from terminal (highest priority)
    python scripts/robot_retarget.py --config config/robot/agibot_x2.yaml --render-debug

    # Override keypoints_path from terminal by motion stem only
    python scripts/robot_retarget.py --config config/robot/h2.yaml --keypoints-name body_check_001__A548_M_from_g1
"""

import argparse
import math
import mink
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import pickle
import time
import csv
import copy
import json
import os
import glfw
from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot

if __package__:
    from .collision_safety import (CollisionProjector, RecoveringCollisionAvoidanceLimit,
                                   clearance_deficit, distance_rows)
else:
    from collision_safety import (CollisionProjector, RecoveringCollisionAvoidanceLimit,
                                  clearance_deficit, distance_rows)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global pause state and keyboard callback
_PAUSED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retarget keypoint motions to a robot from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("config", "robot", "h2.yaml"),
        help="Robot YAML config path (default: config/robot/h2.yaml)",
    )
    parser.add_argument(
        "--keypoints-name",
        type=str,
        default=None,
        help=(
            "Override keypoints_path by motion stem only. "
            "Example: --keypoints-name body_check_001__A548_M_from_g1 -> "
            "output_data/keypoints/<config_stem>/<name>_keypoints.pkl"
        ),
    )
    render_debug_group = parser.add_mutually_exclusive_group()
    render_debug_group.add_argument(
        "--render-debug",
        dest="render_debug",
        action="store_true",
        help="Force enable MuJoCo debug viewer (overrides YAML render_debug).",
    )
    render_debug_group.add_argument(
        "--no-render-debug",
        dest="render_debug",
        action="store_false",
        help="Force disable MuJoCo debug viewer (overrides YAML render_debug).",
    )
    parser.set_defaults(render_debug=None)
    return parser.parse_args()


def key_callback(keycode: int) -> None:
    """Toggle pause/play with the space key."""
    global _PAUSED
    if keycode == glfw.KEY_SPACE:
        _PAUSED = not _PAUSED


def _geom_lowest_z(model, data, geom_id, mesh_verts) -> float:
    """World z of the lowest point on a geom."""
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        pos = data.geom_xpos[geom_id]
        mat = data.geom_xmat[geom_id].reshape(3, 3)
        return float((mesh_verts[geom_id] @ mat.T + pos)[:, 2].min())
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = float(model.geom_size[geom_id, 0])
        half = float(model.geom_size[geom_id, 1])
        mat = data.geom_xmat[geom_id].reshape(3, 3)
        axis = mat[:, 2]
        center = data.geom_xpos[geom_id]
        return float(min((center + axis * half)[2], (center - axis * half)[2]) - radius)
    return float(data.geom_xpos[geom_id, 2] - model.geom_rbound[geom_id])


def foot_ground_lifts(model, qpos, clearance: float = 0.0) -> np.ndarray:
    """Per-frame world-z lift that puts the lowest foot point on ``clearance``.

    ``qpos`` uses MuJoCo order: xyz, quaternion wxyz, then joints. The returned
    lift is added to the root z only, so joint angles stay unchanged.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    visual_ids = []
    collision_ids = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.endswith("_foot_visual"):
            visual_ids.append(geom_id)
        elif "foot" in name and "collision" in name:
            collision_ids.append(geom_id)
    foot_ids = visual_ids or collision_ids
    if qpos.size == 0 or not foot_ids:
        return np.zeros(qpos.shape[0], dtype=np.float64)

    mesh_verts = {}
    for geom_id in foot_ids:
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        adr = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        mesh_verts[geom_id] = np.asarray(model.mesh_vert[adr:adr + count], dtype=np.float64)

    data = mujoco.MjData(model)
    lifts = np.zeros(qpos.shape[0], dtype=np.float64)
    for frame_idx, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        lowest = min(_geom_lowest_z(model, data, geom_id, mesh_verts) for geom_id in foot_ids)
        lifts[frame_idx] = max(0.0, float(clearance) - lowest)
    return lifts


class RobotRetarget:

    LEGACY_CONTACT_CONFIG_TO_NAMES = {
        "foot_end_link": ("left_foot_end", "right_foot_end"),
        "toe_link": ("left_toe", "right_toe"),
        "hand_link": ("left_hand", "right_hand"),
    }

    def __init__(
        self,
        model_path: str,
        keypoint_path: str,
        ik_match_table: dict,
        solver: str = "daqp",
        verbose: bool = False,
        damping: float = 1.0,
        render_debug: bool = False,
        joints_limit_offset_degrees: dict | None = None,
        contact_body_names: list | tuple | dict | None = None,
        contact_position_cost: float = 10.0,
        collision_avoidance: dict | None = None,
    ):
        self.xml_file = model_path
        self.keypoint_path = keypoint_path
        self.ik_match_table = ik_match_table
        self.model = self._load_model_with_ground(self.xml_file)
        self.data = mujoco.MjData(self.model)
        self.max_iter = 50
        self.damping = damping
        self.render_debug = render_debug
        self.joints_limit_offset_degrees = joints_limit_offset_degrees or {}
        self.contact_body_names = self._normalize_contact_body_names(contact_body_names)
        self.contact_position_cost = float(contact_position_cost)
        self.collision_avoidance = collision_avoidance or {}
        self.verbose = verbose
        self.solver = solver
        self.human_body_to_task = {}
        self.task_errors = {}
        self.result_pos = []
        self.contact_state_name_to_idx = {}
        self.keypoint_name_to_idx = {}
        self.body_name_to_source_keypoint = {}
        self.body_name_to_contact_task = {}
        self.contact_targets = []
        self.current_contact_points = []

        self.robot_motor_names = {}

        self.setup_retarget_configuration()
        self.load_keypoints()
        self.setup_contact_targets()

    def _normalize_contact_body_names(self, contact_body_names):
        if contact_body_names is None:
            return []
        if isinstance(contact_body_names, dict):
            flattened_body_names = []
            for field_name, contact_names in self.LEGACY_CONTACT_CONFIG_TO_NAMES.items():
                body_names = contact_body_names.get(field_name, [])
                if len(body_names) != len(contact_names):
                    continue
                flattened_body_names.extend(body_names)
            return flattened_body_names
        if isinstance(contact_body_names, (list, tuple)):
            return list(contact_body_names)
        raise TypeError(
            "contact_body_names must be a list/tuple of body names or a legacy contact-body mapping"
        )

    def _load_model_with_ground(self, xml_file: str) -> mujoco.MjModel:
        """Load the robot XML and add a MuJoCo default-style ground (checker plane + skybox).

        The ground geom has collision disabled (contype/conaffinity=0) and serves only as a
        visual reference; it does not affect mink IK's pure kinematic solving.
        """
        spec = mujoco.MjSpec.from_file(xml_file)

        # Skip if ground/skybox assets already exist to avoid name conflicts
        existing_tex = {t.name for t in spec.textures}
        existing_mat = {m.name for m in spec.materials}

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

        # Lighting: main directional light + angled fill light to avoid an overly dark scene
        spec.worldbody.add_light(
            pos=[0, 0, 20.0],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.7, 0.7, 0.7],
            specular=[0.3, 0.3, 0.3],
        )
        spec.worldbody.add_light(
            pos=[4, 4, 6.0],
            dir=[-0.5, -0.5, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.4, 0.4, 0.4],
            specular=[0.1, 0.1, 0.1],
        )

        # Add the (infinite) ground plane, collision disabled, for visualization only
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

    def load_keypoints(self):
        with open(self.keypoint_path, "rb") as f:
                keypoints_data = pickle.load(f)
        self.keypoint_names = keypoints_data["keypoint_names"]
        self.keypoint_name_to_idx = {
            keypoint_name: idx for idx, keypoint_name in enumerate(self.keypoint_names)
        }
        self.keypoints_pos = keypoints_data["positions"] 
        self.keypoints_quat = keypoints_data["quaternions"]  
        self.contact_names = keypoints_data.get("contact_names", [])
        self.contact_state_name_to_idx = {
            contact_name: idx for idx, contact_name in enumerate(self.contact_names)
        }
        self.contact_seq = keypoints_data["contact_states"]
        self.fps = keypoints_data.get("fps", 30)
        self.time_step = 1.0 / self.fps
        self.num_frames = self.keypoints_pos.shape[0]
        self.num_keypoints = self.keypoints_pos.shape[1]

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        self.tasks = []
        self.robot_frame_names = []  # Robot body name corresponding to each task

        # Apply offsets to the configured joint limits: raise the lower bound
        self._apply_joints_limit_offset()
        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        self._setup_collision_avoidance_limits()
        # VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
        # self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 

        for keypoint_name, entry in self.ik_match_table.items():
            robot_frame, pos_weight, rot_weight = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=robot_frame,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                # Use the keypoint name as the key for indexing keypoint data in update_targets
                self.human_body_to_task[keypoint_name] = task
                self.body_name_to_source_keypoint[robot_frame] = keypoint_name

                self.tasks.append(task)
                self.robot_frame_names.append(robot_frame)
                self.task_errors[task] = []
        pass

    def _setup_collision_avoidance_limits(self):
        """Register mink.CollisionAvoidanceLimit from the collision_avoidance config.

        Config schema:
            enabled: bool
            gain / minimum_distance / detection_distance: shared self-collision params
            geom_groups: {group_name: [geom_name, ...]}  (plain geom names also work in pairs)
            geom_padding: {geom_name: extra_clearance_in_meters}
            pairs: [[groupA, groupB], ...]  self-collision group pairs
            floor: {enabled, geom, geoms, gain, minimum_distance, detection_distance}
        """
        self.self_collision_limits = []
        self.ground_collision_limits = []
        cfg = self.collision_avoidance
        if not cfg or not cfg.get("enabled", False):
            return

        geom_groups = cfg.get("geom_groups", {})

        def resolve_group(name):
            geoms = geom_groups.get(name, [name])
            for geom_name in geoms:
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                if geom_id < 0:
                    raise ValueError(f"[collision avoidance] geom not found in model: {geom_name}")
            return geoms

        padding = cfg.get("geom_padding", {})
        if not isinstance(padding, dict):
            raise ValueError("[collision avoidance] geom_padding must be a geom-to-distance mapping")
        geom_padding = {}
        for geom_name, value in padding.items():
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id < 0:
                raise ValueError(f"[collision avoidance] geom not found in model: {geom_name}")
            try:
                distance = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"[collision avoidance] padding for {geom_name} must be finite and non-negative"
                ) from None
            if isinstance(value, bool) or not np.isfinite(distance) or distance < 0:
                raise ValueError(
                    f"[collision avoidance] padding for {geom_name} must be finite and non-negative"
                )
            geom_padding[geom_name] = distance

        excluded_geom_pairs = []
        for name_a, name_b in cfg.get("exclude_pairs", []):
            for a in resolve_group(name_a):
                for b in resolve_group(name_b):
                    excluded_geom_pairs.append((self.model.geom(a).id, self.model.geom(b).id))
        recovery = cfg.get("recovery", {})

        def add_limits(pairs, params, default_gain, default_minimum, default_detection,
                       self_collision=False):
            # Partition by total padding so each pair keeps its own clearance.
            # Mink still handles body/parent and collision-mask filtering.
            pairs_by_padding = {}
            for group_a, group_b in pairs:
                for geom_a in group_a:
                    for geom_b in group_b:
                        name_a = self.model.geom(geom_a).name
                        name_b = self.model.geom(geom_b).name
                        extra = geom_padding.get(name_a, 0.0) + geom_padding.get(name_b, 0.0)
                        pairs_by_padding.setdefault(extra, []).append(([geom_a], [geom_b]))
            for extra, padded_pairs in pairs_by_padding.items():
                limit = RecoveringCollisionAvoidanceLimit(
                        self.model,
                        padded_pairs,
                        excluded_geom_pairs=excluded_geom_pairs,
                        displacement_bounds=bool(params.get("full_body", False)),
                        recovery_gain=(float(recovery.get("gain", 0.25))
                                       if self_collision and recovery.get("enabled", False) else 0.0),
                        max_recovery_distance=float(recovery.get("max_distance", 0.002)),
                        gain=float(params.get("gain", default_gain)),
                        minimum_distance_from_collisions=float(
                            params.get("minimum_distance", default_minimum)
                        ) + extra,
                        collision_detection_distance=float(
                            params.get("detection_distance", default_detection)
                        ) + extra,
                    )
                self.ik_limits.append(limit)
                if self_collision:
                    self.self_collision_limits.append(limit)
                else:
                    self.ground_collision_limits.append(limit)

        geom_pairs = []
        for pair in cfg.get("pairs", []):
            if len(pair) != 2:
                raise ValueError(f"[collision avoidance] pair must have 2 entries, got {pair}")
            geom_pairs.append((resolve_group(pair[0]), resolve_group(pair[1])))

        if cfg.get("auto_pairs", False):
            robot_geoms = [geom_id for geom_id in range(self.model.ngeom)
                           if self.model.geom_bodyid[geom_id] != 0
                           and (self.model.geom_contype[geom_id]
                                or self.model.geom_conaffinity[geom_id])]
            geom_pairs.append((robot_geoms, robot_geoms))

        if geom_pairs:
            add_limits(geom_pairs, cfg, 0.85, 0.003, 0.05, self_collision=True)
            if self.verbose:
                print(f"[collision avoidance] self-collision pairs: {cfg.get('pairs', [])}")

        floor_cfg = cfg.get("floor", {})
        if floor_cfg.get("enabled", False):
            floor_geom = floor_cfg.get("geom", "floor")
            contact_geoms = floor_cfg.get("geoms", [])
            if floor_cfg.get("full_body", False):
                contact_geoms = [self.model.geom(i).name for i in range(self.model.ngeom)
                                 if self.model.geom_bodyid[i] != 0
                                 and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])]
            if contact_geoms:
                resolve_group(floor_geom)
                for geom_name in contact_geoms:
                    resolve_group(geom_name)
                add_limits([(contact_geoms, [floor_geom])], floor_cfg, 0.7, 0.0, 0.02)
                if self.verbose:
                    print(f"[collision avoidance] floor pairs: {contact_geoms} vs {floor_geom}")

    def setup_contact_targets(self):
        if not self.contact_body_names or not self.contact_state_name_to_idx:
            return

        if len(self.contact_body_names) != len(self.contact_names):
            if self.verbose:
                print(
                    "[contact target] skip: robot contact_links count "
                    f"{len(self.contact_body_names)} != keypoint contact_names count {len(self.contact_names)}"
                )
            return

        for body_name, contact_name in zip(self.contact_body_names, self.contact_names):
            contact_state_idx = self.contact_state_name_to_idx.get(contact_name)
            if contact_state_idx is None:
                continue

            source_keypoint_name = self._resolve_body_source_keypoint(body_name)
            if source_keypoint_name is None:
                if self.verbose:
                    print(f"[contact target] could not find a source keypoint for {body_name}")
                continue

            task = self._ensure_contact_task(body_name)
            self.contact_targets.append(
                {
                    "body_name": body_name,
                    "contact_name": contact_name,
                    "contact_state_idx": contact_state_idx,
                    "source_keypoint_name": source_keypoint_name,
                    "mean_positions": self._build_contact_mean_positions(
                        contact_state_idx=contact_state_idx,
                        source_keypoint_name=source_keypoint_name,
                    ),
                    "task": task,
                }
            )

        if self.verbose and self.contact_targets:
            summary = [
                f"{item['contact_name']}->{item['body_name']} (src={item['source_keypoint_name']})"
                for item in self.contact_targets
            ]
            print(f"[contact target] enabled: {summary}")

    def _resolve_body_source_keypoint(self, body_name):
        if body_name in self.keypoint_name_to_idx:
            return body_name
        return self.body_name_to_source_keypoint.get(body_name)

    def _ensure_contact_task(self, body_name):
        existing_task = self.body_name_to_contact_task.get(body_name)
        if existing_task is not None:
            return existing_task

        task = mink.FrameTask(
            frame_name=body_name,
            frame_type="body",
            position_cost=self.contact_position_cost,
            orientation_cost=0.0,
            lm_damping=1,
        )
        self.body_name_to_contact_task[body_name] = task
        self.tasks.append(task)
        self.robot_frame_names.append(body_name)
        self.task_errors[task] = []
        return task

    def _get_target_pose(self, frame_idx, keypoint_name):
        keypoint_idx = self.keypoint_name_to_idx[keypoint_name]
        target_pos = self.keypoints_pos[frame_idx, keypoint_idx]
        target_quat = self.keypoints_quat[frame_idx, keypoint_idx]
        return target_pos, target_quat

    def _interpolate_target_pose(self, frame_idx, keypoint_name, alpha):
        """Interpolate source poses between frames for collision-aware IK substeps."""
        prev_idx = max(frame_idx - 1, 0)
        pos0, quat0 = self._get_target_pose(prev_idx, keypoint_name)
        pos1, quat1 = self._get_target_pose(frame_idx, keypoint_name)
        pos = (1.0 - alpha) * np.asarray(pos0) + alpha * np.asarray(pos1)
        quat0 = np.asarray(quat0, dtype=np.float64)
        quat1 = np.asarray(quat1, dtype=np.float64)
        # q and -q represent the same rotation; choose the short interpolation arc.
        if np.dot(quat0, quat1) < 0:
            quat1 = -quat1
        quat = (1.0 - alpha) * quat0 + alpha * quat1
        quat /= max(np.linalg.norm(quat), 1e-12)
        return pos, quat

    def _get_body_pose(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body not found: {body_name}")
        return self.configuration.data.xpos[body_id], self.configuration.data.xquat[body_id]

    def _build_contact_mean_positions(self, contact_state_idx, source_keypoint_name):
        mean_positions = np.zeros((self.num_frames, 3), dtype=np.float64)
        contact_states = self.contact_seq[:, contact_state_idx].astype(bool)
        if not np.any(contact_states):
            return mean_positions

        source_keypoint_idx = self.keypoint_name_to_idx[source_keypoint_name]
        source_positions = np.asarray(
            self.keypoints_pos[:, source_keypoint_idx, :], dtype=np.float64
        )

        frame_idx = 0
        while frame_idx < self.num_frames:
            if not contact_states[frame_idx]:
                frame_idx += 1
                continue

            interval_end = frame_idx + 1
            while interval_end < self.num_frames and contact_states[interval_end]:
                interval_end += 1

            interval_mean = np.mean(source_positions[frame_idx:interval_end], axis=0)
            mean_positions[frame_idx:interval_end] = interval_mean
            frame_idx = interval_end

        return mean_positions

    def _get_contact_locked_position(self, frame_idx, contact_target):
        contact_state_idx = contact_target["contact_state_idx"]
        source_keypoint_name = contact_target["source_keypoint_name"]

        target_pos, _ = self._get_target_pose(frame_idx, source_keypoint_name)
        in_contact = bool(self.contact_seq[frame_idx, contact_state_idx])
        if in_contact:
            return contact_target["mean_positions"][frame_idx], True
        return target_pos, False
    
    def _apply_joints_limit_offset(self):
        """Add offsets to the lower/upper limits of matching joints per joints_limit_offset_degrees.

        The key is a joint-name substring (e.g. knee_joint / elbow_joint) and matches all joints
        in the model whose name contains that substring (e.g. left_knee_joint, right_knee_joint).
        The value is [lower_offset_deg, upper_offset_deg]: the lower offset is added to the lower
        bound and the upper offset to the upper bound (positive raises, negative lowers). A single
        scalar value is also supported (applied to the lower bound only).
        """
        if not self.joints_limit_offset_degrees:
            return

        # Collect all joint names in the model
        all_joint_names = []
        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is not None:
                all_joint_names.append((joint_id, name))

        for key, offset_deg in self.joints_limit_offset_degrees.items():
            # Support [lower_offset, upper_offset] or a single scalar (lower bound only)
            if isinstance(offset_deg, (list, tuple)):
                if len(offset_deg) != 2:
                    raise ValueError(
                        f"joints_limit_offset_degrees['{key}'] must be two values [lower, upper], "
                        f"got {offset_deg}"
                    )
                lower_offset_rad = np.radians(float(offset_deg[0]))
                upper_offset_rad = np.radians(float(offset_deg[1]))
            else:
                lower_offset_rad = np.radians(float(offset_deg))
                upper_offset_rad = 0.0

            if lower_offset_rad == 0.0 and upper_offset_rad == 0.0:
                continue
            matched = [(jid, name) for jid, name in all_joint_names if key in name]
            if not matched:
                if self.verbose:
                    print(f"[joint limit] no matching joint found: {key}")
                continue
            for joint_id, joint_name in matched:
                lower, upper = self.model.jnt_range[joint_id]
                new_lower = lower + lower_offset_rad
                new_upper = upper + upper_offset_rad
                # Ensure the lower bound does not exceed the upper bound
                if new_lower > new_upper:
                    new_lower, new_upper = new_upper, new_lower
                self.model.jnt_range[joint_id, 0] = new_lower
                self.model.jnt_range[joint_id, 1] = new_upper
                if self.verbose:
                    print(
                        f"[joint limit] {joint_name} range: "
                        f"[{lower:.4f}, {upper:.4f}] -> [{new_lower:.4f}, {new_upper:.4f}] rad"
                    )
 
    def update_targets(self, frame_idx, interpolation_alpha=None):
        # Record the keypoint targets (pos, quat_wxyz) mapped to tasks for the current frame, for visualization
        self.current_targets = []
        self.current_contact_points = []
        for keypoint_name, task in self.human_body_to_task.items():
            if interpolation_alpha is None:
                target_pos, target_quat = self._get_target_pose(frame_idx, keypoint_name)
            else:
                target_pos, target_quat = self._interpolate_target_pose(
                    frame_idx, keypoint_name, interpolation_alpha
                )
            task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos))
            robot_frame, pos_weight, rot_weight = self.ik_match_table[keypoint_name]
            self.current_targets.append(
                {
                    "pos": np.asarray(target_pos, dtype=np.float64),
                    "quat": np.asarray(target_quat, dtype=np.float64),
                    "robot_frame": robot_frame,
                    "pos_weight": float(pos_weight),
                    "rot_weight": float(rot_weight),
                }
            )

        for contact_target in self.contact_targets:
            source_keypoint_name = contact_target["source_keypoint_name"]
            target_pos, is_active = self._get_contact_locked_position(frame_idx, contact_target)
            if interpolation_alpha is not None and not is_active:
                target_pos, _ = self._interpolate_target_pose(
                    frame_idx, source_keypoint_name, interpolation_alpha
                )
            if is_active:
                _, target_quat = self._get_target_pose(frame_idx, source_keypoint_name)
            else:
                target_pos, target_quat = self._get_body_pose(contact_target["body_name"])
            contact_target["task"].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_pos)
            )
            if is_active:
                self.current_contact_points.append(np.asarray(target_pos, dtype=np.float64))
    
    def error(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks]
            )
        )
    
    def _draw_pose(self, scene, pos, quat_wxyz, point_rgba, axis_alpha=1.0,
                   point_radius=0.035, axis_radius=0.005, axis_length=0.08):
        """Draw a pose in the scene: one small sphere + red/green/blue (XYZ) axes. Returns whether it succeeded."""
        max_geoms = len(scene.geoms)
        if scene.ngeom + 4 > max_geoms:
            return False

        pos = np.asarray(pos, dtype=np.float64)
        identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        axis_colors = (
            np.array([1.0, 0.0, 0.0, axis_alpha], dtype=np.float32),  # X red
            np.array([0.0, 1.0, 0.0, axis_alpha], dtype=np.float32),  # Y green
            np.array([0.0, 0.0, 1.0, axis_alpha], dtype=np.float32),  # Z blue
        )

        # Position sphere
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([point_radius, point_radius, point_radius], dtype=np.float64),
            pos,
            identity_mat,
            np.asarray(point_rgba, dtype=np.float32),
        )
        scene.ngeom += 1

        # Derive the rotation matrix from the quaternion (wxyz); its columns are the axis directions
        rot_mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_mat, np.asarray(quat_wxyz, dtype=np.float64))
        rot_mat = rot_mat.reshape(3, 3)
        # A cylinder defaults along its local z axis, so rotate each axis to its target direction
        base_to_axis = (
            Rot.from_euler("y", 90, degrees=True).as_matrix(),   # z->x
            Rot.from_euler("x", -90, degrees=True).as_matrix(),  # z->y
            np.eye(3),                                           # z->z
        )
        for axis_idx in range(3):
            axis_world_rot = rot_mat @ base_to_axis[axis_idx]
            axis_dir = rot_mat[:, axis_idx]
            center = pos + axis_dir * (0.5 * axis_length)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([axis_radius, 0.5 * axis_length, 0.0], dtype=np.float64),
                center,
                axis_world_rot.reshape(-1),
                axis_colors[axis_idx],
            )
            scene.ngeom += 1
        return True

    def _draw_point(self, scene, pos, point_rgba, point_radius=0.04):
        """Draw only a single spherical point in the scene."""
        max_geoms = len(scene.geoms)
        if scene.ngeom + 1 > max_geoms:
            return False

        pos = np.asarray(pos, dtype=np.float64)
        identity_mat = np.eye(3, dtype=np.float64).reshape(-1)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([point_radius, point_radius, point_radius], dtype=np.float64),
            pos,
            identity_mat,
            np.asarray(point_rgba, dtype=np.float32),
        )
        scene.ngeom += 1
        return True

    def draw_target_keypoints(self, viewer, point_radius=0.035, axis_radius=0.005, axis_length=0.08, targets=None):
        """Draw the positions and axes of the target keypoints and the corresponding robot bodies.

        - Target keypoint: yellow sphere + full-brightness axes.
        - Robot body: cyan sphere + semi-transparent axes.
        """
        if targets is None:
            targets = getattr(self, "current_targets", None)
        scene = viewer.user_scn
        scene.ngeom = 0

        target_rgba = np.array([1.0, 1.0, 0.0, 0.9], dtype=np.float32)   # yellow
        robot_rgba = np.array([0.0, 1.0, 1.0, 0.9], dtype=np.float32)    # cyan
        contact_rgba = np.array([1.0, 0.0, 0.0, 0.95], dtype=np.float32) # red

        # 1) Target keypoints
        if targets:
            for target in targets:
                if isinstance(target, dict):
                    target_pos = target["pos"]
                    target_quat_wxyz = target["quat"]
                    robot_frame = target.get("robot_frame")
                    pos_weight = target.get("pos_weight", 1.0)
                    rot_weight = target.get("rot_weight", 1.0)
                else:
                    # Backward compatibility with the old (pos, quat) tuple format
                    target_pos, target_quat_wxyz = target
                    robot_frame = None
                    pos_weight = 1.0
                    rot_weight = 1.0

                # Zero position weight: draw the sphere at the corresponding body's position (overlapping)
                if pos_weight == 0 and robot_frame is not None:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_frame)
                    if body_id >= 0:
                        target_pos = self.configuration.data.xpos[body_id]

                if rot_weight == 0:
                    # Zero orientation weight: show only the sphere, no axes
                    if not self._draw_point(
                        scene, target_pos, target_rgba,
                        point_radius=point_radius,
                    ):
                        break
                else:
                    if not self._draw_pose(
                        scene, target_pos, target_quat_wxyz, target_rgba,
                        axis_alpha=1.0,
                        point_radius=point_radius, axis_radius=axis_radius, axis_length=axis_length,
                    ):
                        break

        # 2) Current position and orientation of the corresponding robot bodies
        data = self.configuration.data
        for robot_frame in self.robot_frame_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_frame)
            if body_id < 0:
                continue
            body_pos = data.xpos[body_id]
            body_quat = data.xquat[body_id]  # wxyz
            if not self._draw_pose(
                scene, body_pos, body_quat, robot_rgba,
                axis_alpha=0.5,
                point_radius=point_radius, axis_radius=axis_radius, axis_length=axis_length,
            ):
                break

        # 3) Currently active contact points, drawn as red spheres
        for contact_pos in self.current_contact_points:
            if not self._draw_point(
                scene,
                contact_pos,
                contact_rgba,
                point_radius=max(point_radius * 1.5, 0.02),
            ):
                break
    
    def _integrate_ik_velocity(self, velocity: np.ndarray, dt: float, frame_idx: int) -> bool:
        """Integrate one IK velocity. Reject non-finite or exploded solutions.

        A tight collision limit can make daqp return a velocity that breaks
        MuJoCo's quaternion integration. Keep the last valid pose instead.
        """
        if velocity is None or not np.isfinite(velocity).all():
            print(f"[ik] frame {frame_idx}: non-finite velocity, keep pose", flush=True)
            return False
        peak = float(np.max(np.abs(velocity)))
        if self.collision_avoidance.get("recovery", {}).get("enabled", False):
            max_step = float(self.collision_avoidance.get("max_ik_step", 0.05))
            step_peak = peak * dt
            if step_peak > max_step:
                velocity = velocity * (max_step / step_peak)
                peak = float(np.max(np.abs(velocity)))
        # 200 rad/s or m/s over the 1 ms model step is 0.2 rad / 0.2 m.
        if peak > 200.0:
            print(f"[ik] frame {frame_idx}: velocity peak {peak:.1f}, keep pose", flush=True)
            return False
        qpos = self.configuration.data.qpos.copy()
        mujoco.mj_integratePos(self.model, qpos, velocity, dt)
        quat_norm = float(np.linalg.norm(qpos[3:7]))
        if not np.isfinite(qpos).all() or quat_norm < 1e-8:
            print(f"[ik] frame {frame_idx}: invalid qpos after integrate, keep pose", flush=True)
            return False
        self.configuration.data.qpos[:] = qpos
        self.configuration.update()
        return True

    def retarget(self):
        global _PAUSED
        viewer = None
        _PAUSED = False

        if self.render_debug:
            viewer = mujoco.viewer.launch_passive(
                self.model, self.configuration.data, key_callback=key_callback
            )
            print("Controls: Space play/pause")

        try:
            collision_substeps = max(1, int(self.collision_avoidance.get("ik_substeps", 4)))
            for frame_idx in tqdm(range(self.num_frames), desc="Retargeting", unit="frame"):
                start_time = time.time()
                # Solve along the source-frame interval, so collision constraints
                # see intermediate poses instead of only the destination pose.
                for substep in range(1, collision_substeps + 1):
                    alpha = substep / collision_substeps
                    self.update_targets(frame_idx, interpolation_alpha=alpha)

                    curr_error = self.error()
                    dt = self.configuration.model.opt.timestep
                    vel = mink.solve_ik(
                        self.configuration, self.tasks, dt, self.solver, self.damping, limits=self.ik_limits
                    )
                    if not self._integrate_ik_velocity(vel, dt, frame_idx):
                        break
                    next_error = self.error()
                    num_iter = 0
                    while num_iter < self.max_iter:
                        if curr_error - next_error <= 0.001:
                            recovery = self.collision_avoidance.get("recovery", {})
                            if not recovery.get("enabled", False):
                                break
                            tolerance = float(self.collision_avoidance.get(
                                "export_correction", {}).get("tolerance", 0.00005))
                            rows = distance_rows(self.model, self.configuration.data,
                                                 self.self_collision_limits)
                            if clearance_deficit(rows) <= tolerance:
                                break
                        curr_error = next_error
                        vel = mink.solve_ik(
                            self.configuration, self.tasks, dt, self.solver, self.damping, limits=self.ik_limits
                        )
                        if not self._integrate_ik_velocity(vel, dt, frame_idx):
                            break
                        next_error = self.error()
                        num_iter += 1
                curr_pos = self.configuration.data.qpos.copy()
                self.result_pos.append(curr_pos)

                if viewer is not None:
                    if not viewer.is_running():
                        break
                    mujoco.mj_forward(self.model, self.configuration.data)
                    self.draw_target_keypoints(viewer)
                    viewer.sync()
                    # Stay on the current frame while paused, until unpaused or the window is closed
                    while _PAUSED and viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                    if not viewer.is_running():
                        break
                    end_time = time.time()
                    elapsed = end_time - start_time
                    time.sleep(max(0, self.time_step - elapsed))
                    # time.sleep(0.02)  # reduce CPU usage
                    
        finally:
            if viewer is not None:
                viewer.close()
    
    def plant_feet_on_ground(self, clearance: float = 0.0) -> None:
        """Raise root height so the lowest foot point is not below the ground.

        Joint angles, root xy, and root orientation are left unchanged.
        """
        if len(self.result_pos) == 0:
            return
        lifts = foot_ground_lifts(self.model, np.asarray(self.result_pos), clearance=clearance)
        for frame, lift in zip(self.result_pos, lifts):
            frame[2] += lift
        if self.verbose:
            print(
                "[plant feet] root z lift m: "
                f"mean={lifts.mean():.4f} max={lifts.max():.4f}"
            )

    def apply_shoulder_roll_outward_offset(self, degrees: float = 3.0) -> None:
        """Hardcoded outward offset on both shoulder roll joints.

        Left shoulder roll opens outward with positive angle; right opens
        outward with negative angle. Applied to every frame in result_pos.
        """
        if len(self.result_pos) == 0 or degrees == 0.0:
            return
        rad = math.radians(degrees)
        offsets = {
            "upper_left_2_joint": +rad,
            "upper_right_2_joint": -rad,
        }
        applied = []
        for joint_name, delta in offsets.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                continue
            adr = int(self.model.jnt_qposadr[jid])
            lo = float(self.model.jnt_range[jid, 0])
            hi = float(self.model.jnt_range[jid, 1])
            for frame in self.result_pos:
                frame[adr] = min(max(frame[adr] + delta, lo), hi)
            applied.append(f"{joint_name}{'+' if delta>0 else ''}{delta:.4f}rad")
        if self.verbose and applied:
            print("[shoulder roll] outward offset: " + ", ".join(applied))

    def save_results_as_csv(self, output_path):
        if len(self.result_pos) == 0:
            raise ValueError("No retarget results to save. Run retarget() first.")

        self.correct_export_collisions()

        # [num_frame, num_joint], the first 7 dims are posXYZ + quat(wxyz)
        result = np.asarray(self.result_pos, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] < 7:
            raise ValueError(
                f"result_pos shape must be [num_frame, num_joint>=7], got {result.shape}"
            )

        # Reorder the first-7-column quaternion from wxyz (cols 3..6) to xyzw
        result_xyzw = result.copy()
        result_xyzw[:, 3:7] = result[:, [4, 5, 6, 3]]

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(result_xyzw.tolist())

        if getattr(self, "last_collision_report", None) is not None:
            with open(str(output_path) + ".collision.json", "w", encoding="utf-8") as handle:
                json.dump(self.last_collision_report, handle, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"Saved retarget results to: {output_path} (shape={result_xyzw.shape})")

    def correct_export_collisions(self):
        """Repair all postprocessed poses and refuse unresolved collision exports."""
        options = self.collision_avoidance.get("export_correction", {})
        limits = getattr(self, "self_collision_limits", [])
        self.last_collision_report = None
        if not options.get("enabled", False) or not limits:
            return
        settings = {name: options[name] for name in (
            "max_iterations", "max_joint_step", "tolerance", "slack_cost") if name in options}
        projector = CollisionProjector(self.model, limits, solver=self.solver, **settings)
        original = np.asarray(self.result_pos, dtype=np.float64)
        corrected = original.copy()
        report = {
            "frames": len(original), "fps": float(self.fps),
            "checked_pairs": sum(len(limit.geom_id_pairs) for limit in limits),
            "minimum_distance_m": float(self.collision_avoidance.get("minimum_distance", 0.003)),
            "tolerance_m": projector.tolerance,
            "penetrating_frames_before": 0, "penetrating_frames_after": 0,
            "below_clearance_frames_before": 0, "below_clearance_frames_after": 0,
            "corrected_frames": 0, "max_joint_correction_rad": 0.0,
            "minimum_distance_before_m": None, "minimum_distance_after_m": None,
            "excluded_geom_pairs": self.collision_avoidance.get("exclude_pairs", []),
            "unresolved": [],
        }
        for index, pose in enumerate(original):
            corrected[index], before, after, iterations = projector.project(pose)
            for label, rows in (("before", before), ("after", after)):
                minimum = min((row.distance for row in rows), default=None)
                if minimum is not None:
                    key = "minimum_distance_" + label + "_m"
                    report[key] = minimum if report[key] is None else min(report[key], minimum)
                report["penetrating_frames_" + label] += int(
                    any(row.distance < -1e-6 for row in rows))
                report["below_clearance_frames_" + label] += int(
                    clearance_deficit(rows) > projector.tolerance)
            if not np.array_equal(corrected[index], pose):
                report["corrected_frames"] += 1
                delta = np.empty(self.model.nv)
                mujoco.mj_differentiatePos(self.model, delta, 1.0, pose, corrected[index])
                report["max_joint_correction_rad"] = max(
                    report["max_joint_correction_rad"], float(np.max(np.abs(delta))))
            if clearance_deficit(after) > projector.tolerance:
                worst = max(after, key=lambda row: row.margin - row.distance)
                report["unresolved"].append({
                    "frame_1based": index + 1, "time_s": index / self.fps,
                    "pair": [self.model.geom(g).name for g in worst.pair],
                    "distance_m": worst.distance, "required_m": worst.margin,
                    "iterations": iterations,
                })
            if (index + 1) % 500 == 0:
                print(f"[export collision] checked {index + 1}/{len(original)}, "
                      f"corrected={report['corrected_frames']}, "
                      f"unresolved={len(report['unresolved'])}", flush=True)
        self.last_collision_report = report
        if report["unresolved"]:
            raise RuntimeError(
                f"Export collision correction failed on {len(report['unresolved'])} frames; "
                f"CSV not written. First failures: {report['unresolved'][:3]}"
            )
        floor = self.collision_avoidance.get("floor", {})
        if floor.get("enabled", False):
            corrected[:, 2] += foot_ground_lifts(
                self.model, corrected, clearance=float(floor.get("minimum_distance", 0.0)))
        self.result_pos = list(corrected)
        print(f"[export collision] penetration frames "
              f"{report['penetrating_frames_before']} -> {report['penetrating_frames_after']}, "
              f"below clearance {report['below_clearance_frames_before']} -> "
              f"{report['below_clearance_frames_after']}, "
              f"corrected={report['corrected_frames']}", flush=True)



if  __name__ == "__main__":
    args = parse_args()
    workspace_root = os.path.abspath(os.path.join(SCRIPT_DIR,"..", "robot", ".."))
    config_path = os.path.expanduser(args.config)
    if not os.path.isabs(config_path):
        config_path = os.path.join(workspace_root, config_path)

    print("mujoco version: ", mujoco.__version__)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    ik_match_table = config.get("ik_match_table", {})
    robot_xml_path = config.get("robot_xml_path", "")
    keypoints_path = config.get("keypoints_path", "")
    keypoints_idx = config.get("keypoints_idx","")

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    if args.keypoints_name:
        keypoints_path = os.path.join(
            "output_data",
            "keypoints",
            config_name,
            f"{args.keypoints_name}_keypoints.pkl",
        )

    if robot_xml_path and not os.path.isabs(robot_xml_path):
        robot_xml_path = os.path.join(workspace_root, robot_xml_path)
    if keypoints_path and not os.path.isabs(keypoints_path):
        keypoints_path = os.path.join(workspace_root, keypoints_path)

    verbose_mode = config.get("verbose", False)
    config_render_debug = bool(config.get("render_debug", False))
    render_debug = config_render_debug if args.render_debug is None else bool(args.render_debug)
    joints_limit_offset_degrees = config.get("joints_limit_offset_degrees", {})
    contact_body_names = config.get("contact_links")
    if contact_body_names is None:
        contact_body_names = {
            field_name: config.get(field_name, [])
            for field_name in RobotRetarget.LEGACY_CONTACT_CONFIG_TO_NAMES
        }
    contact_position_cost = config.get("contact_pos_fixed_factor", 10.0)
    collision_avoidance = config.get("collision_avoidance")

    robot_retarget = RobotRetarget(
        model_path=robot_xml_path,
        keypoint_path=keypoints_path,
        ik_match_table=ik_match_table,
        solver="daqp",
        verbose=verbose_mode,
        render_debug=render_debug,
        joints_limit_offset_degrees=joints_limit_offset_degrees,
        contact_body_names=contact_body_names,
        contact_position_cost=contact_position_cost,
        collision_avoidance=collision_avoidance,
    )

    robot_retarget.retarget()
    if config.get("plant_feet_on_ground", False):
        robot_retarget.plant_feet_on_ground(
            clearance=float(config.get("plant_feet_clearance", 0.0))
        )
    robot_retarget.apply_shoulder_roll_outward_offset(degrees=3.0)

    keypoint_stem = os.path.splitext(os.path.basename(keypoints_path))[0]
    if keypoint_stem.endswith("_keypoints"):
        keypoint_stem = keypoint_stem[: -len("_keypoints")]
    output_csv = os.path.join(
        workspace_root,
        "output_data/robot_motion",
        f"{keypoint_stem}_{config_name}.csv",
    )
    robot_retarget.save_results_as_csv(output_csv)
