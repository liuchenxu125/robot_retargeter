"""Numerical regression tests for approach limits, recovery and export refusal."""

import tempfile
import unittest
from pathlib import Path

import mink
import mujoco
import numpy as np

from scripts.collision_safety import (CollisionProjector, RecoveringCollisionAvoidanceLimit,
                                      clearance_deficit, distance_rows)
from scripts.robot_retarget import RobotRetarget


def sphere_model(separation=0.025, axis="1 0 0"):
    return mujoco.MjModel.from_xml_string(f"""
    <mujoco><worldbody>
      <body name="body_a"><joint type="slide" axis="{axis}"/>
        <geom name="a" size=".01" mass="1"/></body>
      <body name="body_b" pos="{separation} 0 0"><joint type="slide" axis="{axis}"/>
        <geom name="b" size=".01" mass="1"/></body>
    </worldbody></mujoco>""")


def recovery_limit(model):
    return RecoveringCollisionAvoidanceLimit(
        model, [(["a"], ["b"])], recovery_gain=0.25,
        minimum_distance_from_collisions=0.003,
        collision_detection_distance=0.05,
    )


class CollisionSafetyTest(unittest.TestCase):
    def test_ik_step_does_not_cross_clearance_at_small_dt(self):
        for dt in (0.001, 0.01):
            with self.subTest(dt=dt):
                model = sphere_model()
                configuration = mink.Configuration(model)
                task = mink.PostureTask(model, cost=1.0)
                task.set_target(np.array([0.01, -0.01]))
                limit = recovery_limit(model)
                velocity = mink.solve_ik(configuration, [task], dt, "daqp", limits=[limit])
                configuration.integrate_inplace(velocity, dt)
                distance = mujoco.mj_geomDistance(model, configuration.data, 0, 1, 0.05, None)
                self.assertGreaterEqual(distance, 0.003 - 1e-8)
                self.assertLess(distance, 0.005)

    def test_penetration_requests_recovery_independent_of_dt(self):
        model = sphere_model(separation=0.018)
        configuration = mink.Configuration(model)
        limit = recovery_limit(model)
        bounds = [limit.compute_qp_inequalities(configuration, dt).h[0]
                  for dt in (0.001, 0.01)]
        self.assertAlmostEqual(bounds[0], -0.00125)
        self.assertAlmostEqual(bounds[0], bounds[1])

    def test_penetration_jacobian_matches_finite_difference(self):
        model = sphere_model(separation=0.018)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        row = distance_rows(model, data, [recovery_limit(model)], with_jacobians=True)[0]
        step = 1e-7
        data.qpos[0] += step
        mujoco.mj_forward(model, data)
        shifted = mujoco.mj_geomDistance(model, data, 0, 1, 0.05, None)
        self.assertAlmostEqual((shifted - row.distance) / step, row.jacobian[0], places=5)

    def test_projector_recovers_overlap_with_small_change(self):
        model = sphere_model(separation=0.018)
        reference = np.zeros(model.nq)
        projector = CollisionProjector(model, [recovery_limit(model)])
        corrected, before, after, _ = projector.project(reference)
        self.assertGreater(clearance_deficit(before), 0.004)
        self.assertLessEqual(clearance_deficit(after), projector.tolerance)
        self.assertLess(np.max(np.abs(corrected - reference)), 0.003)

    def test_explicit_and_xml_exclusions(self):
        model = sphere_model()
        limit = RecoveringCollisionAvoidanceLimit(
            model, [(["a"], ["b"])], excluded_geom_pairs=[(0, 1)])
        self.assertEqual(limit.geom_id_pairs, [])
        xml = """<mujoco><worldbody>
        <body name="A"><joint type="slide"/><geom name="a" size=".01"/></body>
        <body name="B"><joint type="slide"/><geom name="b" size=".01"/></body>
        </worldbody><contact><exclude body1="A" body2="B"/></contact></mujoco>"""
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual(RecoveringCollisionAvoidanceLimit(
            model, [(["a"], ["b"])]).geom_id_pairs, [])

    def test_auto_pairs_covers_unlisted_pair(self):
        robot = RobotRetarget.__new__(RobotRetarget)
        robot.model = sphere_model()
        robot.verbose = False
        robot.ik_limits = []
        robot.collision_avoidance = {"enabled": True, "auto_pairs": True}
        robot._setup_collision_avoidance_limits()
        self.assertEqual(robot.self_collision_limits[0].geom_id_pairs, [(0, 1)])

    def test_large_finite_ik_velocity_is_bounded_instead_of_freezing(self):
        robot = RobotRetarget.__new__(RobotRetarget)
        robot.model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <body><freejoint/><geom size=".01"/></body>
        </worldbody></mujoco>""")
        robot.configuration = mink.Configuration(robot.model)
        robot.collision_avoidance = {"recovery": {"enabled": True}, "max_ik_step": 0.05}
        velocity = np.zeros(robot.model.nv)
        velocity[0] = 250.0
        self.assertTrue(robot._integrate_ik_velocity(velocity, 0.001, 0))
        self.assertAlmostEqual(robot.configuration.data.qpos[0], 0.05)
        self.assertFalse(robot._integrate_ik_velocity(velocity * np.nan, 0.001, 0))

    def test_failed_correction_does_not_overwrite_existing_csv(self):
        model = sphere_model(separation=0.018, axis="0 0 1")
        robot = RobotRetarget.__new__(RobotRetarget)
        robot.model = model
        robot.solver = "daqp"
        robot.fps = 30
        robot.result_pos = [np.zeros(model.nq)]
        robot.self_collision_limits = [recovery_limit(model)]
        robot.collision_avoidance = {
            "minimum_distance": 0.003,
            "export_correction": {"enabled": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.csv"
            path.write_text("original motion")
            with self.assertRaisesRegex(RuntimeError, "CSV not written"):
                robot.save_results_as_csv(path)
            self.assertEqual(path.read_text(), "original motion")


if __name__ == "__main__":
    unittest.main()
