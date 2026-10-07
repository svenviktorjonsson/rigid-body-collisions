"""Restricted viscoelastic reduction, not a full many-body collision model.

Normal impacts: spherical Hertz geometry, small deformation and frozen effective
mass, with a repulsive force-zero release and explicit remaining strain energy.
Rolling: supported pure rolling, no axial spin or sliding; static capacity is
checked instantaneously at the start, rather than only on integrated impulses.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


def normal_reference(beta, rtol=2e-10):
    """Dimensionless x''=-x**1.5-beta*sqrt(x)*x'; x(0)=0,x'(0)=1.

    beta=1.5*A*(k/m_eff)**(2/5)*v_in**(1/5). In physical units
    x_scale=(m_eff*v_in**2/k)**(2/5), t_scale=x_scale/v_in.
    Force-zero release precedes geometric recovery when beta>0.
    """
    if not math.isfinite(beta) or beta < 0:
        raise ValueError('finite nonnegative damping required')
    def rhs(t, y):
        x, v = y[:2]; root = math.sqrt(max(x, 0.))
        force = root*(max(x, 0.)+beta*v)
        return [v, -force, beta*root*v*v]
    def release(t, y):
        return y[0]+beta*y[1]
    release.terminal = True; release.direction = -1
    sol = solve_ivp(rhs, (0., 100.), [0., 1., 0.], events=release,
                    rtol=rtol, atol=rtol*.01, max_step=.06)
    if not sol.success or len(sol.t_events[0]) != 1:
        raise RuntimeError('normal release not resolved')
    x, v, loss = sol.y[:, -1]
    if v >= 0 or x < -1e-8:
        raise RuntimeError('invalid unloading state')
    stored = .4*max(x, 0.)**2.5
    energy = .5*v*v + stored + loss
    forces = np.sqrt(np.maximum(sol.y[0], 0.))*(np.maximum(sol.y[0], 0.)+beta*sol.y[1])
    return dict(beta=float(beta), restitution=float(-v), duration=float(sol.t[-1]),
                release_compression=float(max(x, 0.)), stored_energy_at_release=float(stored),
                dissipated_energy=float(loss), energy_balance_residual=float(energy-.5),
                minimum_sampled_force=float(np.min(forces)))


@dataclass
class NormalTable:
    beta: np.ndarray
    restitution: np.ndarray

    def __post_init__(self):
        self.beta = np.asarray(self.beta, float)
        self.restitution = np.asarray(self.restitution, float)
        if self.beta.ndim != 1 or self.beta.shape != self.restitution.shape or len(self.beta) < 3:
            raise ValueError('one-dimensional equal-sized table required')
        if not np.all(np.isfinite(np.r_[self.beta, self.restitution])):
            raise ValueError('finite table required')
        if self.beta[0] != 0 or np.any(np.diff(self.beta) <= 0):
            raise ValueError('increasing nonnegative grid required')
        if np.any(self.restitution < 0) or np.any(self.restitution > 1+1e-8) or np.any(np.diff(self.restitution) > 1e-8):
            raise ValueError('passive monotone restitution required')
        self.interpolant = PchipInterpolator(self.beta, np.minimum(self.restitution, 1.), extrapolate=False)

    def evaluate(self, beta):
        x = np.asarray(beta, float)
        if not np.all(np.isfinite(x)) or np.any(x < self.beta[0]) or np.any(x > self.beta[-1]):
            raise ValueError('table range exceeded; no extrapolation')
        return self.interpolant(x)


def rolling_advance(mass, radius, alpha, normal, axis, omega, dt, relaxation,
                    mu_s, ell, gravity=9.81):
    """Exact sustained rolling decay with a free couple and static reaction.

    This is a 3D sphere (or its planar motion restriction), not a plane-strain
    contact law for a 2D material. Effective alpha may describe other inertia;
    transfer of the viscoelastic sphere constitutive law is not then established.
    """
    vals = np.array([mass, radius, alpha, dt, ell, gravity], float)
    if not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        raise ValueError('positive finite mechanical inputs required')
    if not np.all(np.isfinite([omega, relaxation, mu_s])) or min(omega, relaxation, mu_s) < 0:
        raise ValueError('finite nonnegative speed/relaxation/static friction required')
    n, s = np.asarray(normal, float), np.asarray(axis, float)
    if n.shape != (3,) or s.shape != (3,) or not np.allclose([n@n,s@s,n@s],[1.,1.,0.],atol=1e-12,rtol=0):
        raise ValueError('unit normal and pure rolling axis required')
    # Necessary and sufficient throughout this supported decay: speed and the
    # required force both decrease, so the initial capacity is the largest.
    if relaxation*omega/(1+alpha) > mu_s*(1+1e-12):
        raise ValueError('instantaneous static capacity insufficient; sliding required')
    N = mass*gravity; I = alpha*mass*radius**2
    rate = relaxation*gravity/((1+alpha)*radius)
    change = omega*math.expm1(-rate*dt)
    outgoing_omega = omega+change
    direction = np.cross(s,n)
    p = mass*radius*change*direction
    moment = (I+mass*radius**2)*change*s
    r = -radius*n
    body_L = np.cross(r,p)+moment
    # Combined quantities carry uppercase Delta; small delta refers to p,L.
    scaled_V = np.r_[radius*outgoing_omega*direction,ell*outgoing_omega*s]
    scaled_Delta_P = np.r_[p,body_L/ell]
    energy_before = .5*(I+mass*radius**2)*omega**2
    loss = -energy_before*math.expm1(-2*rate*dt)
    return dict(velocity=scaled_V[:3], angular_velocity=scaled_V[3:]/ell,
                linear_impulse=p, angular_impulse=moment, body_angular_change=body_L,
                combined_change=scaled_Delta_P, energy_before=energy_before,
                energy_after=energy_before-loss, dissipation=loss,
                initial_mu_r=relaxation*omega, final_mu_r=relaxation*outgoing_omega,
                decay_rate=rate)
