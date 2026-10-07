"""Isolated sustained-contact rolling branch, not the production impact solver.

mu_r is dimensionless with physical moment length R. Normal support balances
gravity; the horizontal/angular update includes the static reaction and free
couple. No-slip is required initially. A failed static capacity is rejected;
sliding cannot be resolved by this restricted branch. No axial torsion here.
"""
import numpy as np


def skew(r):
    x, y, z = r
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def advance(mass, radius, alpha, normal, axis, omega, dt, mu_r, mu_s, ell, gravity=9.81):
    values = np.array([mass, radius, alpha, dt, ell, gravity], float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('positive finite mechanical inputs required')
    if not np.isfinite(mu_r) or not np.isfinite(mu_s) or min(mu_r, mu_s, omega) < 0:
        raise ValueError('finite nonnegative coefficients and rolling speed required')
    n, s = np.asarray(normal, float), np.asarray(axis, float)
    if n.shape != (3,) or s.shape != (3,) or not np.allclose([n@n, s@s, n@s], [1, 1, 0], atol=1e-12):
        raise ValueError('orthonormal normal and pure rolling axis required')
    e = np.cross(s, n)  # Translation direction; zero-slip reaction is constraint-defined.
    r = -radius*n
    inertia = alpha*mass*radius**2
    B = np.block([[np.eye(3), -skew(r)/ell], [np.zeros((3, 3)), np.eye(3)]])
    mobility = np.diag([1/mass]*3 + [ell**2/inertia]*3)
    D = np.zeros((6, 2)); D[:3, 0] = e; D[3:, 1] = s
    W = B @ mobility @ B.T
    reduced = D.T @ W @ D
    normal_impulse = mass*gravity*dt
    # No-slip rolling effective inertia limits a purely resistive impulse at rest.
    couple = -min(mu_r*radius*normal_impulse, (inertia+mass*radius**2)*omega)
    scalar_tangent = -reduced[0, 1]*(couple/ell)/reduced[0, 0]
    if abs(scalar_tangent) > mu_s*normal_impulse + 1e-14*normal_impulse:
        raise ValueError('static capacity insufficient: sliding branch required')
    V = np.r_[radius*omega*e, ell*omega*s]
    contact_impulse = D @ np.array([scalar_tangent, couple/ell])
    body_impulse = B.T @ contact_impulse
    outgoing = V + mobility @ body_impulse
    before = .5*V @ np.linalg.solve(mobility, V)
    after = .5*outgoing @ np.linalg.solve(mobility, outgoing)
    work = contact_impulse @ (B @ V) + .5*contact_impulse @ W @ contact_impulse
    return dict(velocity=outgoing[:3], angular_velocity=outgoing[3:]/ell,
                tangent_impulse=scalar_tangent, angular_impulse=couple,
                normal_impulse=normal_impulse, energy_before=before,
                energy_after=after, impulse_energy_change=work,
                slip=(B @ outgoing)[:3], physical_body_impulse=np.r_[body_impulse[:3], ell*body_impulse[3:]])


def audit(seed=20261006):
    rng = np.random.default_rng(seed)
    largest = {'relative_motion_error': 0., 'relative_energy_error': 0., 'relative_scale_error': 0.}
    cases = 0
    for dim in (2, 3):
        for _ in range(100):
            R = 10**rng.uniform(-3, -.3); m = 10**rng.uniform(-3, 1)
            alpha = rng.choice([.4, .5, 2/3]); om = rng.uniform(.1, 100)
            mu_r = rng.uniform(.0001, .2); dt = 10**rng.uniform(-5, 2)
            if dim == 2:
                n, s = np.array([0., 1, 0]), np.array([0., 0, 1])
            else:
                q, _ = np.linalg.qr(rng.normal(size=(3, 3))); n, s = q[:, 1], q[:, 2]
            e = np.cross(s, n)
            # This is a synthetic static capacity, not an inferred material value.
            mu_s = 2*mu_r/(1+alpha)
            expected_om = max(0., om-mu_r*9.81*dt/((1+alpha)*R))
            physical = []
            for ell in (R*1e-2, R, R*1e2):
                out = advance(m, R, alpha, n, s, om, dt, mu_r, mu_s, ell)
                motion_err = max(np.linalg.norm(out['velocity']-R*expected_om*e)/(R*om),
                                 np.linalg.norm(out['angular_velocity']-expected_om*s)/om,
                                 np.linalg.norm(out['slip'])/(R*om))
                energy_err = abs(out['energy_after']-out['energy_before']-out['impulse_energy_change'])/out['energy_before']
                assert motion_err < 1e-10 and energy_err < 1e-10
                assert out['energy_after'] <= out['energy_before']*(1+1e-12)
                largest['relative_motion_error'] = max(largest['relative_motion_error'], float(motion_err))
                largest['relative_energy_error'] = max(largest['relative_energy_error'], float(energy_err))
                physical.append(np.r_[out['velocity']/(R*om), out['angular_velocity']/om])
            scale_err = float(np.max(np.abs(np.array(physical)-physical[0])))
            largest['relative_scale_error'] = max(largest['relative_scale_error'], scale_err)
            assert scale_err < 1e-10
            cases += 1
    # Independent planar Newton/Euler closure: m dv = f, I dw = -R f + couple.
    m, R, alpha, om, mu_r, dt = 1., .03, .4, 20., .02, .001
    out = advance(m, R, alpha, [0, 1, 0], [0, 0, 1], om, dt, mu_r, .1, .007)
    I = alpha*m*R**2
    f, delta_om = np.linalg.solve(np.array([[1/m, -R], [-R, -I]]), np.array([0., -out['angular_impulse']]))
    assert abs(f-out['tangent_impulse']) < 1e-12
    assert abs(om+delta_om-out['angular_velocity'][2]) < 1e-10
    frictionless = advance(m, R, alpha, [0, 1, 0], [0, 0, 1], om, dt, 0., 0., R)
    assert np.allclose(frictionless['angular_velocity'], [0, 0, om])
    try:
        advance(m, R, alpha, [0, 1, 0], [0, 0, 1], om, dt, mu_r, 0., R)
    except ValueError:
        rejected = True
    else:
        raise AssertionError('failed to reject impossible no-slip state')
    return dict(case_count=cases, length_scales_per_case=3, pass_count=cases,
                errors=largest, independent_planar_balance_pass=True,
                zero_rolling_torque_preserves_motion=True, insufficient_static_capacity_rejected=rejected,
                scope='Synthetic isolated rolling controls, not empirical collision or native group validation')
