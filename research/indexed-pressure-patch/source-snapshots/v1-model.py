"""Frozen, planar compliant patch: unilateral normal foundation + dynamic slip.

An isolated candidate/reference, not the production directional contact law.
Local normal is z. Planar mechanics restrict y=0, vy=wx=wz=0.
Compression is affine: d + y*tilt_x - x*tilt_y. Its rate is -local normal speed.
Weights are nonnegative integrated pressure-shape weights, summing to one.
K and C are total foundation stiffness/damping, NOT assumed Young's moduli.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Patch:
    points: np.ndarray
    weights: np.ndarray
    basis: np.ndarray = field(init=False)
    gram: np.ndarray = field(init=False)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if (self.points.ndim != 2 or self.points.shape[1] != 2
                or self.weights.shape != (len(self.points),) or not len(self.points)
                or not np.all(np.isfinite(self.points))
                or not np.all(np.isfinite(self.weights)) or np.any(self.weights < 0)
                or not np.isclose(self.weights.sum(), 1., rtol=1e-13, atol=1e-15)):
            raise ValueError('finite x/y sites and nonnegative normalized weights required')
        self.basis = np.column_stack((np.ones(len(self.points)), self.points[:, 1], -self.points[:, 0]))
        self.gram = self.basis.T @ (self.weights[:, None] * self.basis)


def evaluate(patch, compression, velocity, omega, stiffness, damping, mu_d):
    """Instantaneous wrench and energy audit. No time-stepping or static closure.

    At local zero tangential speed the kinetic-friction contribution is zero;
    this is not a claim that static friction is zero or has been solved.
    """
    compression, velocity, omega = [np.asarray(v, dtype=float) for v in (compression, velocity, omega)]
    if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in (compression, velocity, omega)):
        raise ValueError('three finite compression/velocity/omega components required')
    if not np.all(np.isfinite([stiffness, damping, mu_d])) or stiffness <= 0 or min(damping, mu_d) < 0:
        raise ValueError('positive stiffness and nonnegative damping/dynamic friction required')
    q = np.array([velocity[2], omega[0], omega[1]])
    delta = patch.basis @ compression
    un = patch.basis @ q
    elastic = stiffness * np.maximum(delta, 0)
    # Never exert attractive force, nor force across an open site.
    fn = patch.weights * np.where(delta > 0, np.maximum(0, elastic - damping * un), 0)
    x, y = patch.points.T
    slip = np.column_stack((velocity[0] - omega[2] * y, velocity[1] + omega[2] * x))
    speed = np.linalg.norm(slip, axis=1)
    traction = -mu_d * fn[:, None] * np.divide(slip, speed[:, None], out=np.zeros_like(slip), where=speed[:, None] > 0)
    forces = np.column_stack((traction, fn))
    offsets = np.column_stack((patch.points, np.zeros(len(x))))
    moment = np.cross(offsets, forces).sum(axis=0)
    force = forces.sum(axis=0)
    stored = .5 * stiffness * np.sum(patch.weights * np.maximum(delta, 0)**2)
    stored_rate = -np.sum(patch.weights * elastic * un)
    normal_loss = np.sum((patch.weights * elastic - fn) * un)
    sliding_loss = mu_d * np.dot(fn, speed)
    power = force @ velocity + moment @ omega
    return dict(force=force, moment=moment, stored_energy=stored, stored_rate=stored_rate,
                normal_dissipation=normal_loss, sliding_dissipation=sliding_loss,
                power=power, energy_rate_residual=power + stored_rate + normal_loss + sliding_loss,
                active_sites=int(np.count_nonzero(fn)), all_loaded=bool(np.all(delta > 0) and np.all(elastic - damping*un > 0)))


def directional_residual(force, moment, velocity, omega):
    """Audit user's n/t and s/n spans without replacing either definition."""
    n = np.array([0., 0., 1.])
    t_defined, s_defined = np.linalg.norm(velocity) > 0, np.linalg.norm(omega) > 0
    force_map = np.column_stack((n, velocity / np.linalg.norm(velocity))) if t_defined else n[:, None]
    moment_map = np.column_stack((omega / np.linalg.norm(omega), n)) if s_defined else n[:, None]
    linear = np.linalg.lstsq(force_map, force, rcond=None)[0]
    angular = np.linalg.lstsq(moment_map, moment, rcond=None)[0]
    return dict(linear_residual=float(np.linalg.norm(force - force_map @ linear)),
                angular_residual=float(np.linalg.norm(moment - moment_map @ angular)),
                t_defined=bool(t_defined), s_defined=bool(s_defined),
                linear_components=linear.tolist(), angular_components=angular.tolist())


def gather_motion(body_velocity, body_omega, owners, signs, arms, frame):
    """One contact's incidence range; owner is the sole body index k."""
    v, w = body_velocity[owners], body_omega[owners]
    u = np.sum(signs[:, None] * (v + np.cross(w, arms)), axis=0)
    omega = np.sum(signs[:, None] * w, axis=0)
    return frame.T @ u, frame.T @ omega


def scatter_wrench(force, moment, owners, signs, arms, frame, body_force, body_moment):
    """Ordered indexed accumulation, including lever moment + independent couple."""
    f, m = frame @ force, frame @ moment
    np.add.at(body_force, owners, signs[:, None] * f)
    np.add.at(body_moment, owners, signs[:, None] * (np.cross(arms, f) + m))
