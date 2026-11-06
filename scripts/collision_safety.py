"""Collision recovery and nonlinear validation for exported robot poses."""

from dataclasses import dataclass

import mink
import mujoco
import numpy as np
from qpsolvers import solve_qp


class RecoveringCollisionAvoidanceLimit(mink.CollisionAvoidanceLimit):
    """Keep Mink's approach limits and actively recover violated clearances.

    Recovery distances are capped per IK step. XML body exclusions and explicit
    geom exclusions are honored before Mink initializes its broadphase arrays.
    """

    def __init__(self, model, geom_pairs, *, excluded_geom_pairs=(),
                 recovery_gain=0.0, max_recovery_distance=0.002, **kwargs):
        self.excluded_geom_pairs = {tuple(sorted(pair)) for pair in excluded_geom_pairs}
        self.recovery_gain = float(recovery_gain)
        self.max_recovery_distance = float(max_recovery_distance)
        if not 0.0 <= self.recovery_gain <= 1.0:
            raise ValueError("collision recovery gain must be in [0, 1]")
        if not np.isfinite(self.max_recovery_distance) or self.max_recovery_distance <= 0:
            raise ValueError("max_recovery_distance must be positive and finite")
        super().__init__(model, geom_pairs, **kwargs)

    def _construct_geom_id_pairs(self, geom_pairs):
        pairs = super()._construct_geom_id_pairs(geom_pairs)
        excluded_bodies = {int(signature) for signature in self.model.exclude_signature}
        result = []
        for a, b in pairs:
            if (a, b) in self.excluded_geom_pairs:
                continue
            body_a, body_b = sorted((int(self.model.geom_bodyid[a]),
                                     int(self.model.geom_bodyid[b])))
            if (body_a << 16) + body_b not in excluded_bodies:
                result.append((a, b))
        return sorted(result)

    def compute_qp_inequalities(self, configuration, dt):
        constraint = super().compute_qp_inequalities(configuration, dt)
        if self.recovery_gain == 0.0:
            return constraint
        # Mink's QP solves for delta-q (solve_ik divides it by dt afterwards).
        # Build displacement bounds from geometry instead of inheriting the
        # installed collision limit's velocity-scaled upper bounds.
        for index in np.flatnonzero(np.isfinite(constraint.h)):
            a, b = self.geom_id_pairs[index]
            distance = mujoco.mj_geomDistance(
                self.model, configuration.data, a, b,
                self.collision_detection_distance, None,
            )
            deficit = self.minimum_distance_from_collisions - distance
            constraint.h[index] = (self.gain * max(0.0, -deficit)
                                   + self.bound_relaxation * dt)
            if deficit > 0.0:
                recovery = min(self.recovery_gain * deficit, self.max_recovery_distance)
                constraint.h[index] = self.bound_relaxation * dt - recovery
        return constraint


@dataclass
class DistanceRow:
    pair: tuple
    distance: float
    margin: float
    jacobian: np.ndarray | None = None


def distance_rows(model, data, limits, *, with_jacobians=False, buffer=0.001):
    """Signed distances and signed-distance derivatives for the checked pairs."""
    rows = []
    fromto = np.empty(6)
    jac_a = np.zeros((3, model.nv))
    jac_b = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    for limit in limits:
        margin = limit.minimum_distance_from_collisions
        cutoff = max(limit.collision_detection_distance, margin + buffer)
        for a, b in limit.geom_id_pairs:
            if (np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b])
                    - model.geom_rbound[a] - model.geom_rbound[b] > cutoff):
                continue
            distance = float(mujoco.mj_geomDistance(model, data, a, b, cutoff, fromto))
            if distance >= cutoff - 1e-12:
                continue
            jacobian = None
            if with_jacobians and distance < margin + buffer:
                delta = fromto[:3] - fromto[3:]
                norm = np.linalg.norm(delta)
                if norm > 1e-12:
                    normal = delta / norm * (1.0 if distance >= 0.0 else -1.0)
                    mujoco.mj_jac(model, data, jac_a, jac_rot, fromto[:3],
                                  int(model.geom_bodyid[a]))
                    mujoco.mj_jac(model, data, jac_b, jac_rot, fromto[3:],
                                  int(model.geom_bodyid[b]))
                    jacobian = normal @ (jac_a - jac_b)
                else:
                    # A degenerate nearest-point normal cannot justify a safe step.
                    jacobian = np.zeros(model.nv)
            rows.append(DistanceRow((a, b), distance, margin, jacobian))
    return rows


def clearance_deficit(rows):
    return max((max(0.0, row.margin - row.distance) for row in rows), default=0.0)


class CollisionProjector:
    """Correct an exported pose near its reference and verify actual distances.

    The QP uses penalized slack so an initial deep penetration does not make a
    small trust-region step infeasible. Slack never bypasses final validation.
    Free-root motion is fixed: rigid translation cannot repair self collision.
    """

    def __init__(self, model, limits, *, solver="daqp", max_iterations=80,
                 max_joint_step=0.05, tolerance=0.00005, slack_cost=1e6):
        self.model = model
        self.data = mujoco.MjData(model)
        self.limits = limits
        self.solver = solver
        self.max_iterations = int(max_iterations)
        self.max_joint_step = float(max_joint_step)
        self.tolerance = float(tolerance)
        self.slack_cost = float(slack_cost)
        if (self.max_iterations < 1 or not 0 < self.max_joint_step < np.inf
                or not 0 < self.tolerance < np.inf or not 0 < self.slack_cost < np.inf):
            raise ValueError("export correction settings must be positive and finite")

    def rows(self, qpos, *, jacobians=False):
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return distance_rows(self.model, self.data, self.limits,
                             with_jacobians=jacobians,
                             buffer=max(0.001, self.tolerance * 4))

    def step_bounds(self, qpos):
        lo = np.full(self.model.nv, -self.max_joint_step)
        hi = np.full(self.model.nv, self.max_joint_step)
        for jid in range(self.model.njnt):
            kind = self.model.jnt_type[jid]
            adr = int(self.model.jnt_dofadr[jid])
            if kind == mujoco.mjtJoint.mjJNT_FREE:
                lo[adr:adr + 6] = hi[adr:adr + 6] = 0.0
            elif kind in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                if self.model.jnt_limited[jid]:
                    value = qpos[self.model.jnt_qposadr[jid]]
                    lo[adr] = max(lo[adr], self.model.jnt_range[jid, 0] - value)
                    hi[adr] = min(hi[adr], self.model.jnt_range[jid, 1] - value)
        return lo, hi

    def project(self, reference):
        reference = np.asarray(reference, dtype=np.float64)
        pose = reference.copy()
        initial_rows = self.rows(pose)
        before = clearance_deficit(initial_rows)
        if before <= self.tolerance:
            return pose, initial_rows, initial_rows, 0
        rows = initial_rows
        for iteration in range(self.max_iterations):
            rows = self.rows(pose, jacobians=True)
            if clearance_deficit(rows) <= self.tolerance:
                return pose, initial_rows, rows, iteration
            active = [row for row in rows if row.jacobian is not None]
            nv, count = self.model.nv, len(active)
            # Minimize deviation from the unmodified pose, with expensive slack.
            toward_reference = np.empty(nv)
            mujoco.mj_differentiatePos(self.model, toward_reference, 1.0, pose, reference)
            P = np.diag(np.r_[np.ones(nv), np.full(count, self.slack_cost)])
            linear = np.r_[-toward_reference, np.zeros(count)]
            G = np.zeros((count, nv + count))
            h = np.empty(count)
            for i, row in enumerate(active):
                G[i, :nv] = -row.jacobian
                G[i, nv + i] = -1.0
                # A small buffer avoids accepting poses just below the target.
                h[i] = row.distance - row.margin - 2 * self.tolerance
            lower, upper = self.step_bounds(pose)
            solution = solve_qp(P, linear, G, h,
                                lb=np.r_[lower, np.zeros(count)],
                                ub=np.r_[upper, np.full(count, np.inf)],
                                solver=self.solver)
            if solution is None or not np.isfinite(solution).all():
                break
            step = solution[:nv]
            old_merit = sum(max(0.0, row.margin - row.distance) ** 2 for row in rows)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
                candidate = pose.copy()
                mujoco.mj_integratePos(self.model, candidate, step, scale)
                candidate_rows = self.rows(candidate)
                merit = sum(max(0.0, row.margin - row.distance) ** 2
                            for row in candidate_rows)
                if (clearance_deficit(candidate_rows) <= self.tolerance
                        or merit < old_merit * (1.0 - 1e-6)):
                    pose, rows = candidate, candidate_rows
                    accepted = True
                    break
            if not accepted:
                break
        return pose, initial_rows, self.rows(pose), iteration + 1
