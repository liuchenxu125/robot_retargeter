"""mjlab configuration for the E1 V3 21-DOF humanoid."""

from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


E1_V3_21DOF_XML = Path(__file__).resolve().parent / "mjcf" / "E1_21dof.xml"
assert E1_V3_21DOF_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """Load the robot-only portion of the standalone E1 MJCF.

  The source XML intentionally remains usable by the conversion and replay
  tools, so it contains a floor, explicit foot-floor pairs, torque motors and
  actuator-force sensors. An mjlab Scene supplies its own terrain and the
  position actuators below, therefore those standalone elements are removed
  from this fresh spec before it is attached to the scene.
  """
  spec = mujoco.MjSpec.from_file(str(E1_V3_21DOF_XML))

  for pair in list(spec.pairs):
    spec.delete(pair)
  for sensor in list(spec.sensors):
    if sensor.type == mujoco.mjtSensor.mjSENS_ACTUATORFRC:
      spec.delete(sensor)
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for geom in list(spec.geoms):
    if geom.name == "floor":
      spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)

  return spec


# Motor limits copied from the Isaac Lab E1 definition.
E1_HIP_PITCH_EFFORT = 120.0
E1_HIP_PITCH_VELOCITY = 12.04
E1_HIP_ROLL_EFFORT = 120.0
E1_HIP_ROLL_VELOCITY = 12.04
E1_HIP_YAW_EFFORT = 36.0
E1_HIP_YAW_VELOCITY = 13.61
E1_KNEE_EFFORT = 120.0
E1_KNEE_VELOCITY = 12.04
E1_ANKLE_PITCH_EFFORT = 36.0
E1_ANKLE_PITCH_VELOCITY = 13.61
E1_ANKLE_ROLL_EFFORT = 30.0
E1_ANKLE_ROLL_VELOCITY = 15.71
E1_WAIST_YAW_EFFORT = 60.0
E1_WAIST_YAW_VELOCITY = 13.09
E1_SHOULDER_PITCH_EFFORT = 60.0
E1_SHOULDER_PITCH_VELOCITY = 13.09
E1_SHOULDER_ROLL_EFFORT = 36.0
E1_SHOULDER_ROLL_VELOCITY = 13.61
E1_SHOULDER_YAW_EFFORT = 15.0
E1_SHOULDER_YAW_VELOCITY = 15.71
E1_ELBOW_EFFORT = 60.0
E1_ELBOW_VELOCITY = 13.09
E1_ARMATURE = 0.01


def _position_actuator(
  target_names_expr: tuple[str, ...],
  stiffness: float,
  damping: float,
  effort_limit: float,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=stiffness,
    damping=damping,
    effort_limit=effort_limit,
    armature=E1_ARMATURE,
  )


E1_HIP_PITCH_ACTUATOR = _position_actuator(
  (".*_hip_pitch_joint",), 200.0, 5.0, E1_HIP_PITCH_EFFORT
)
E1_HIP_ROLL_ACTUATOR = _position_actuator(
  (".*_hip_roll_joint",), 200.0, 5.0, E1_HIP_ROLL_EFFORT
)
E1_HIP_YAW_ACTUATOR = _position_actuator(
  (".*_hip_yaw_joint",), 80.0, 3.0, E1_HIP_YAW_EFFORT
)
E1_KNEE_ACTUATOR = _position_actuator(
  (".*_knee_joint",), 200.0, 5.0, E1_KNEE_EFFORT
)
E1_ANKLE_PITCH_ACTUATOR = _position_actuator(
  (".*_ankle_pitch_joint",), 80.0, 3.0, E1_ANKLE_PITCH_EFFORT
)
E1_ANKLE_ROLL_ACTUATOR = _position_actuator(
  (".*_ankle_roll_joint",), 60.0, 2.0, E1_ANKLE_ROLL_EFFORT
)
E1_WAIST_YAW_ACTUATOR = _position_actuator(
  ("waist_yaw_joint",), 150.0, 4.0, E1_WAIST_YAW_EFFORT
)
E1_SHOULDER_PITCH_AND_ELBOW_ACTUATOR = _position_actuator(
  (".*_shoulder_pitch_joint", ".*_elbow_joint"),
  30.0,
  2.0,
  E1_SHOULDER_PITCH_EFFORT,
)
E1_SHOULDER_ROLL_ACTUATOR = _position_actuator(
  (".*_shoulder_roll_joint",), 30.0, 2.0, E1_SHOULDER_ROLL_EFFORT
)
E1_SHOULDER_YAW_ACTUATOR = _position_actuator(
  (".*_shoulder_yaw_joint",), 30.0, 2.0, E1_SHOULDER_YAW_EFFORT
)


E1_21DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    E1_HIP_PITCH_ACTUATOR,
    E1_HIP_ROLL_ACTUATOR,
    E1_HIP_YAW_ACTUATOR,
    E1_KNEE_ACTUATOR,
    E1_ANKLE_PITCH_ACTUATOR,
    E1_ANKLE_ROLL_ACTUATOR,
    E1_WAIST_YAW_ACTUATOR,
    E1_SHOULDER_PITCH_AND_ELBOW_ACTUATOR,
    E1_SHOULDER_ROLL_ACTUATOR,
    E1_SHOULDER_YAW_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


E1_21DOF_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.75),
  joint_pos={
    ".*_hip_pitch_joint": -0.10,
    ".*_knee_joint": 0.23,
    ".*_ankle_pitch_joint": -0.13,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_elbow_joint": 1.0,
  },
  joint_vel={".*": 0.0},
)


E1_21DOF_FOOT_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=1,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.8,),
)


def get_e1_v3_21dof_robot_cfg(
  action_delay_range: tuple[int, int] = (0, 0),
) -> EntityCfg:
  """Return a fresh mjlab entity configuration for E1 V3 21-DOF."""
  delay_min_lag, delay_max_lag = action_delay_range
  if delay_min_lag < 0 or delay_max_lag < delay_min_lag:
    raise ValueError(f"Invalid action_delay_range={action_delay_range}")

  articulation = deepcopy(E1_21DOF_ARTICULATION)
  for actuator in articulation.actuators:
    actuator.delay_min_lag = delay_min_lag
    actuator.delay_max_lag = delay_max_lag
    actuator.delay_hold_prob = 0.9 if delay_max_lag > 0 else 0.0
    actuator.delay_update_period = 4 if delay_max_lag > 0 else 0

  return EntityCfg(
    init_state=E1_21DOF_HOME,
    collisions=(E1_21DOF_FOOT_COLLISION,),
    spec_fn=get_spec,
    articulation=articulation,
  )


E1_21DOF_ACTION_SCALE: dict[str, float] = {}
for actuator in E1_21DOF_ARTICULATION.actuators:
  assert isinstance(actuator, BuiltinPositionActuatorCfg)
  assert actuator.effort_limit is not None
  for name_expr in actuator.target_names_expr:
    E1_21DOF_ACTION_SCALE[name_expr] = (
      0.25 * actuator.effort_limit / actuator.stiffness
    )


get_e1_21dof_robot_cfg = get_e1_v3_21dof_robot_cfg


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity import Entity

  robot = Entity(get_e1_v3_21dof_robot_cfg())
  viewer.launch(robot.spec.compile())
