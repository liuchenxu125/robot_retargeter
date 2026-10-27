"""Run with: python -m unittest discover -s tests."""

import copy
import unittest
from pathlib import Path

import mink
import mujoco
import numpy as np
import yaml

from scripts.robot_retarget import RobotRetarget


class CollisionPaddingTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <geom name="floor" type="plane" size="1 1 .1"/>
          <body pos="0 0 .1"><joint type="slide"/>
            <geom name="a" size=".01" mass="1"/></body>
          <body pos=".025 0 .1"><joint type="slide"/>
            <geom name="b" size=".01" mass="1"/></body>
          <body pos=".1 0 .1"><joint type="slide"/>
            <geom name="c" size=".01" mass="1"/></body>
        </worldbody></mujoco>""")
        self.cfg = {
            "enabled": True,
            "geom_groups": {"others": ["b", "c"]},
            "pairs": [["a", "others"], ["b", "a"]],
        }

    def limits(self, cfg, model=None):
        robot = RobotRetarget.__new__(RobotRetarget)
        robot.model = self.model if model is None else model
        robot.collision_avoidance = cfg
        robot.ik_limits = []
        robot.verbose = False
        robot._setup_collision_avoidance_limits()
        return robot.ik_limits

    def test_padding_adds_and_tightens_actual_constraint(self):
        original = self.limits(self.cfg)[0]
        self.cfg["geom_padding"] = {"a": .002, "b": .003}
        padded = self.limits(self.cfg)
        ab = next(limit for limit in padded
                  if np.isclose(limit.minimum_distance_from_collisions, .008))
        self.assertEqual(len(ab.geom_id_pairs), 1)  # overlapping/reversed pairs deduplicated
        self.assertAlmostEqual(ab.collision_detection_distance, .055)
        ac = next(limit for limit in padded if limit is not ab)
        self.assertAlmostEqual(ac.minimum_distance_from_collisions, .005)
        configuration = mink.Configuration(self.model)
        before = original.compute_qp_inequalities(configuration, .01)
        after = ab.compute_qp_inequalities(configuration, .01)
        idx = original.geom_id_pairs.index(ab.geom_id_pairs[0])
        self.assertGreater(before.h[idx], 0)
        self.assertEqual(after.h[0], 0)  # no further approach inside padded clearance

    def test_floor_inherits_padding(self):
        self.cfg["pairs"] = []
        self.cfg["geom_padding"] = {"a": .005}
        self.cfg["floor"] = {"enabled": True, "geoms": ["a", "b"]}
        limits = self.limits(self.cfg)
        self.assertEqual(len(limits), 2)
        self.assertAlmostEqual(limits[0].minimum_distance_from_collisions, .005)
        self.assertAlmostEqual(limits[0].collision_detection_distance, .025)
        self.assertEqual(limits[1].minimum_distance_from_collisions, 0)

    def test_invalid_padding(self):
        for value in [-.001, float("nan"), float("inf"), True, None, "bad"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.limits({**self.cfg, "geom_padding": {"a": value}})
        with self.assertRaisesRegex(ValueError, "geom not found"):
            self.limits({**self.cfg, "geom_padding": {"typo": .005}})
        with self.assertRaises(ValueError):
            self.limits({**self.cfg, "geom_padding": []})

    def test_real_mesh_model_preserves_pairs(self):
        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((root / "config/robot/CASBOT02.yaml").read_text())
        model = mujoco.MjModel.from_xml_path(str(root / cfg["robot_xml_path"]))
        original = self.limits(cfg["collision_avoidance"], model)
        padded_cfg = copy.deepcopy(cfg["collision_avoidance"])
        padded_cfg["geom_padding"] = {"coll_l_elbow": .005, "coll_r_elbow": .005}
        padded = self.limits(padded_cfg, model)
        original_pairs = {pair for limit in original for pair in limit.geom_id_pairs}
        padded_pairs = [pair for limit in padded for pair in limit.geom_id_pairs]
        self.assertEqual(original_pairs, set(padded_pairs))
        self.assertEqual(len(padded_pairs), len(set(padded_pairs)))


if __name__ == "__main__":
    unittest.main()
