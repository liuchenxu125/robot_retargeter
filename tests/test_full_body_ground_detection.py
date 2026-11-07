"""Whole-body detection without new contact or free-root export objectives."""

from pathlib import Path
import unittest

import mink
import mujoco
import numpy as np
import yaml

from scripts.collision_safety import CollisionProjector, RecoveringCollisionAvoidanceLimit
from scripts.robot_retarget import RobotRetarget


class FullBodyGroundDetectionTest(unittest.TestCase):
    def test_all_physical_geoms_include_pelvis_and_exclude_removed_palms(self):
        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((root/'config/robot/casbot02_primitives.yaml').read_text())
        model = mujoco.MjModel.from_xml_path(str(root/cfg['robot_xml_path']))
        robot = RobotRetarget.__new__(RobotRetarget)
        robot.model = model; robot.collision_avoidance = cfg['collision_avoidance']
        robot.ik_limits = []; robot.verbose = False
        robot._setup_collision_avoidance_limits()
        physical = {i for i in range(model.ngeom) if model.geom_bodyid[i] != 0
                    and (model.geom_contype[i] or model.geom_conaffinity[i])}
        floor = model.geom('floor').id
        checked = {g for limit in robot.ground_collision_limits
                   for pair in limit.geom_id_pairs for g in pair if g != floor}
        self.assertEqual(len(physical), 30)
        self.assertEqual(physical, checked)
        self.assertIn(model.geom('pelvis_collision').id, checked)
        self.assertTrue(all(l.minimum_distance_from_collisions == 0.0
                            for l in robot.ground_collision_limits))
        self.assertTrue(all(l.recovery_gain == 0.0 for l in robot.ground_collision_limits))
        self.assertTrue(all(l.minimum_distance_from_collisions == .003
                            for l in robot.self_collision_limits))

    def test_ground_bounds_use_displacements_and_do_not_force_recovery(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="1 1 .1"/>
          <body pos="0 0 .025"><freejoint/><geom name="body" size=".01"/></body>
        </worldbody></mujoco>''')
        limit = RecoveringCollisionAvoidanceLimit(model, [(['body'], ['floor'])],
            minimum_distance_from_collisions=0.0, collision_detection_distance=.05,
            displacement_bounds=True, recovery_gain=0.0)
        configuration = mink.Configuration(model)
        first = limit.compute_qp_inequalities(configuration, .001)
        second = limit.compute_qp_inequalities(configuration, .01)
        self.assertAlmostEqual(first.h[0], second.h[0])
        self.assertAlmostEqual(first.h[0], .85*.015)
        pose = configuration.data.qpos.copy(); pose[2] = .005
        configuration.update(pose)
        inside = limit.compute_qp_inequalities(configuration, .001)
        self.assertEqual(inside.h[0], 0.0)
        self.assertAlmostEqual(inside.G[0, 2], -1.0)

    def test_export_projector_keeps_root_fixed(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body><freejoint/><geom size=".01"/></body>
        </worldbody></mujoco>''')
        projector = CollisionProjector(model, [])
        lower, upper = projector.step_bounds(model.qpos0)
        np.testing.assert_array_equal(lower[:6], 0.0)
        np.testing.assert_array_equal(upper[:6], 0.0)


if __name__ == '__main__':
    unittest.main()
