"""Sparse frozen-contact mechanics, with verified zero-restitution normal solves.

No collision discovery, integration, compliance or rolling law is supplied here.
The active-set algorithm is established convex QP machinery, not novel physics.
"""
from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, lsmr, spsolve


@dataclass(frozen=True)
class PlanarSystem:
    inverse_mass: np.ndarray
    contact_map: sparse.csr_matrix
    _mobility: dict = field(default_factory=dict, init=False, repr=False)

    def mobility(self, channels=(0,)):
        """Return selected contact map and mobility, caching this fixed geometry."""
        channels = tuple(channels)
        if channels not in self._mobility:
            indices = (3*np.arange(self.contact_map.shape[0] // 3)[:, None] + np.asarray(channels)).ravel().astype(int)
            G = self.contact_map[indices].tocsr()
            K = (G.multiply(self.inverse_mass) @ G.T).tocsr()
            self._mobility[channels] = (G, K)
        return self._mobility[channels]


def assemble_sparse(centers, masses, inertias, contacts, ell=1.):
    """Same contact/wedge/scaled-unit convention as assemble_planar, without dense matrices."""
    centers = np.asarray(centers, dtype=float)
    masses, inertias = np.asarray(masses, dtype=float), np.asarray(inertias, dtype=float)
    n = len(centers)
    if centers.shape != (n, 2) or masses.shape != (n,) or inertias.shape != (n,):
        raise ValueError('One 2D centre, mass and inertia per body required')
    if not np.isfinite(ell) or ell <= 0 or not np.isfinite(centers).all() or np.isnan(masses).any() or np.isnan(inertias).any() or np.any(masses <= 0) or np.any(inertias <= 0):
        raise ValueError('Finite centres, positive length, and positive masses/inertias required')
    weights = np.column_stack((1 / masses, 1 / masses, ell**2 / inertias)).ravel()
    contacts = list(contacts)
    if contacts:
        bodies = np.asarray([c[:2] for c in contacts])
        points = np.asarray([c[2] for c in contacts], dtype=float)
        normals = np.asarray([c[3] for c in contacts], dtype=float)
        if bodies.shape != (len(contacts), 2) or points.shape != (len(contacts), 2) or normals.shape != (len(contacts), 2):
            raise ValueError('Contact body pairs, 2D points and 2D normals required')
        if not np.isfinite(bodies).all() or np.any(bodies != np.floor(bodies)) or np.any(bodies < 0) or np.any(bodies >= n):
            raise ValueError('Integer contact body indices in range required')
        bodies = bodies.astype(int)
        lengths = np.linalg.norm(normals, axis=1)
        if not np.isfinite(points).all() or not np.isfinite(normals).all() or np.any(lengths == 0):
            raise ValueError('Finite contact point and nonzero normal required')
        normals = normals / lengths[:, None]
        tangents = np.column_stack((-normals[:, 1], normals[:, 0]))
        lever = points[:, None, :] - centers[bodies]
        # Contact, endpoint body, response channel, body degree of freedom.
        coefficient = np.zeros((len(contacts), 2, 3, 3))
        for channel, direction in enumerate((normals, tangents)):
            coefficient[:, :, channel, :2] = direction[:, None, :]
            coefficient[:, :, channel, 2] = (lever[:, :, 0]*direction[:, None, 1] - lever[:, :, 1]*direction[:, None, 0]) / ell
        coefficient[:, :, 2, 2] = 1.
        coefficient *= np.array([1., -1.])[None, :, None, None]
        if not np.isfinite(coefficient).all() or not np.isfinite(weights).all():
            raise ValueError('Contact/mass scaling overflows supported floating representation')
        mask = coefficient != 0
        rows = np.broadcast_to(3*np.arange(len(contacts))[:, None, None, None] + np.arange(3)[None, None, :, None], coefficient.shape)[mask]
        cols = np.broadcast_to(3*bodies[:, :, None, None] + np.arange(3)[None, None, None, :], coefficient.shape)[mask]
        G = sparse.coo_matrix((coefficient[mask], (rows, cols)), shape=(3*len(contacts), 3*n)).tocsr()
    else:
        G = sparse.csr_matrix((0, 3*n))
    G.eliminate_zeros()
    # This object owns immutable geometry/mass data; rebuild when either changes.
    weights.flags.writeable = False
    G.data.flags.writeable = False; G.indices.flags.writeable = False; G.indptr.flags.writeable = False
    return PlanarSystem(weights, G)


def box_qp(K, q, lower=0., upper=np.inf, tolerance=1e-9, max_iterations=128):
    """Verified block active-set box QP; accepts sparse or dense PSD mobility.

    Singular free blocks use least squares, without adding material softness.
    Degenerate/cycling/infeasible cases raise rather than returning an unchecked
    iterate. This is not a universal convergence guarantee for block pivots.
    """
    q = np.asarray(q, dtype=float)
    lower, upper = np.broadcast_to(lower, q.shape), np.broadcast_to(upper, q.shape)
    if K.shape != (len(q), len(q)) or not np.isfinite(q).all() or np.any(lower > upper) or np.isnan(lower).any() or np.isnan(upper).any() or np.isposinf(lower).any() or np.isneginf(upper).any():
        raise ValueError('Compatible mobility, finite linear term, and feasible bounds required')
    if tolerance <= 0 or not np.isfinite(tolerance): raise ValueError('Positive finite tolerance required')
    x = np.clip(np.zeros(len(q)), lower, upper)
    gradient = np.asarray(K @ x + q)
    mode = np.zeros(len(q), dtype=np.int8)  # -1 lower, 0 free, +1 upper
    mode[(x == lower) & (gradient > tolerance)] = -1
    mode[(x == upper) & (gradient < -tolerance)] = 1
    mode[lower == upper] = 2
    initial_violation = np.where(mode == -1, np.maximum(-gradient, 0),
                        np.where(mode == 1, np.maximum(gradient, 0),
                        np.where(mode == 2, 0, np.abs(gradient))))
    if np.max(initial_violation, initial=0) <= tolerance:
        return x, {'iterations': 0, 'singular_solves': 0,
                   'stationarity_residual': float(np.max(initial_violation, initial=0))}
    visited = set(); singular_solves = 0
    for iteration in range(max_iterations):
        identity = mode.tobytes()
        if identity in visited: raise RuntimeError('Block active-set cycle; no verified solution')
        visited.add(identity)
        free, fixed = np.flatnonzero(mode == 0), np.flatnonzero(mode != 0)
        x[mode == -1] = lower[mode == -1]; x[mode == 1] = upper[mode == 1]
        x[mode == 2] = lower[mode == 2]
        previous = x.copy()
        if len(free):
            A = K[free][:, free] if sparse.issparse(K) else K[np.ix_(free, free)]
            cross = K[free][:, fixed] if sparse.issparse(K) else K[np.ix_(free, fixed)]
            rhs = -q[free] - cross @ x[fixed]
            if sparse.issparse(K):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', MatrixRankWarning)
                        solution = spsolve(A.tocsc(), rhs)
                    if not np.isfinite(solution).all(): raise MatrixRankWarning('Nonfinite sparse factorisation')
                except MatrixRankWarning:
                    singular_solves += 1
                    solution = lsmr(A, rhs, atol=1e-13, btol=1e-13, maxiter=max(100, 5*len(free)))[0]
            else:
                try: solution = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    singular_solves += 1; solution = np.linalg.lstsq(A, rhs, rcond=1e-13)[0]
            x[free] = solution
        below, above = x < lower - tolerance, x > upper + tolerance
        if np.any(below | above):
            # Move from the feasible iterate to the first bound, rather than
            # simultaneously clamping every bad component (which can cycle).
            candidate = x.copy()
            x = previous
            direction = candidate-x
            ratio = np.full(len(x), np.inf)
            ratio[below] = (lower[below]-x[below])/direction[below]
            ratio[above] = (upper[above]-x[above])/direction[above]
            alpha = float(np.clip(np.min(ratio), 0, 1))
            x = np.clip(x + alpha*direction, lower, upper)
            hit = ratio <= alpha + 1e-12
            mode[hit & below] = -1; mode[hit & above] = 1
            continue
        x = np.clip(x, lower, upper)
        gradient = np.asarray(K @ x + q)
        release_low = (mode == -1) & (gradient < -tolerance)
        release_high = (mode == 1) & (gradient > tolerance)
        if np.any(release_low | release_high):
            mode[release_low | release_high] = 0
            continue
        violation = np.where(mode == -1, np.maximum(-gradient, 0),
                    np.where(mode == 1, np.maximum(gradient, 0),
                    np.where(mode == 2, 0, np.abs(gradient))))
        residual = float(np.max(violation, initial=0))
        if residual > tolerance or not np.isfinite(x).all():
            raise RuntimeError(f'Box-QP stationarity residual {residual:g} exceeds {tolerance:g}')
        return x, {'iterations': iteration+1, 'singular_solves': singular_solves,
                   'stationarity_residual': residual}
    raise RuntimeError('Active-set work limit; no verified solution')


def normal_solve(system, velocity, tolerance=1e-8, dense=False, strategy='sparse'):
    """Coupled normal solve at fixed contacts. Rejects failed complementarity."""
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != system.inverse_mass.shape or not np.isfinite(velocity).all():
        raise ValueError('Finite scaled body velocity per degree of freedom required')
    N, K = system.mobility((0,)); u = np.asarray(N @ velocity)
    if strategy not in ('sparse', 'dense', 'auto'): raise ValueError('Choose sparse, dense or auto strategy')
    effective_dense = dense or strategy == 'dense' or (strategy == 'auto' and len(u) <= 128)
    p, stats = box_qp(K.toarray() if effective_dense else K, u, tolerance=tolerance)
    post = velocity + system.inverse_mass * (N.T @ p)
    w = np.asarray(N @ post)
    feasibility = float(max(0, -np.min(w, initial=0)))
    complementarity = float(np.max(np.abs(p*w), initial=0))
    active_error = float(np.max(np.abs(w[p > tolerance]), initial=0))
    if max(feasibility, active_error) > tolerance:
        raise RuntimeError('Normal velocity/complementarity acceptance failed')
    impulses = np.zeros(system.contact_map.shape[0]); impulses[::3] = p
    stats.update({'normal_velocity': w.tolist(), 'normal_feasibility_m_s': feasibility,
                  'active_normal_velocity_m_s': active_error,
                  'complementarity_residual': complementarity,
                  'normal_mobility_nnz': K.nnz, 'contact_map_nnz': system.contact_map.nnz,
                  'method': 'dense_active_set' if effective_dense else 'sparse_active_set'})
    return post, impulses, stats


def friction_solve(system, velocity, friction, tolerance=1e-8, max_iterations=128, strategy='auto'):
    """Sparse semismooth solve of implicit inelastic 2D Coulomb contact.

    Normal complementarity; sticking or saturated friction opposing post-slip.
    One coefficient, no tangential elasticity, restitution or patch couples.
    A merit line search prevents full-step mode oscillation. A disclosed sparse
    least-squares fallback may be slower and is not a physical-law substitution.
    Nonunique, infeasible or unconverged systems remain possible; acceptance
    requires normal, cone, slip and mechanical-energy residuals, not optimizer
    success. No universal convergence or uniqueness is claimed.
    """
    from scipy.optimize import least_squares
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != system.inverse_mass.shape or not np.isfinite(velocity).all():
        raise ValueError('Finite scaled body velocity per degree of freedom required')
    G, K = system.mobility((0, 1)); u = np.asarray(G @ velocity)
    count = len(u) // 2
    mu = np.broadcast_to(np.asarray(friction, dtype=float), (count,))
    if not np.isfinite(mu).all() or np.any(mu < 0): raise ValueError('Finite nonnegative friction required')
    if strategy not in ('auto', 'general'): raise ValueError('Choose auto or general friction strategy')
    _, normal_impulses, normal_stats = normal_solve(system, velocity, tolerance, strategy='auto')
    p = np.zeros(2*count); p[::2] = normal_impulses[::3]
    rho = 1 / np.maximum(K.diagonal(), 1e-15)

    def residual(impulse, jacobian=False):
        w = u + K @ impulse
        pn, pt = impulse[::2], impulse[1::2]
        zn, zt = (impulse-rho*w)[::2], (impulse-rho*w)[1::2]
        cap = mu*np.maximum(pn, 0)
        F = np.empty_like(impulse)
        F[::2] = pn-np.maximum(zn, 0)
        F[1::2] = pt-np.clip(zt, -cap, cap)
        if not jacobian: return F
        contact = zn > 0
        stick = (cap > 0) & (np.abs(zt) <= cap)
        equation = np.zeros(2*count, dtype=bool)
        equation[::2] = contact; equation[1::2] = stick
        J = K.multiply((rho*equation)[:, None]) + sparse.diags((~equation).astype(float))
        slip = np.flatnonzero((~stick) & (pn > 0) & (mu > 0))
        derivative = -mu[slip]*np.sign(zt[slip])
        J += sparse.coo_matrix((derivative, (2*slip+1, 2*slip)), shape=K.shape).tocsr()
        return F, J.tocsr()

    singular_solves = 0; line_search_steps = 0; fallback_evaluations = 0
    method = 'semismooth'; iteration = 0
    # Exact structural zero is required. No small normal/tangent coupling is
    # silently dropped to make the solver faster.
    cross = K[::2, 1::2]; cross.eliminate_zeros()
    decoupled = strategy == 'auto' and cross.nnz == 0
    if decoupled:
        cap = mu*np.maximum(p[::2], 0)
        tangent_K = K[1::2, 1::2]
        tangent, tangent_stats = box_qp(tangent_K.toarray() if count <= 128 else tangent_K,
            u[1::2], lower=-cap, upper=cap, tolerance=tolerance*.1)
        p[1::2] = tangent; method = 'decoupled_box'; iteration = tangent_stats['iterations']
        singular_solves = tangent_stats['singular_solves']
    for iteration in range(max_iterations if not decoupled else 0):
        F, J = residual(p, True)
        if np.max(np.abs(F), initial=0) <= tolerance*.1: break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', MatrixRankWarning)
                step = spsolve(J.tocsc(), -F)
            if not np.isfinite(step).all(): raise MatrixRankWarning('Nonfinite Newton factorisation')
        except MatrixRankWarning:
            singular_solves += 1
            step = lsmr(J, -F, atol=1e-13, btol=1e-13, maxiter=max(100, 10*count))[0]
        merit = float(F @ F)
        for line in range(25):
            alpha = .5**line
            trial = p+alpha*step; trial_F = residual(trial)
            if float(trial_F @ trial_F) <= (1-1e-4*alpha)*merit:
                p = trial; line_search_steps += line
                break
        else: break
    if np.max(np.abs(residual(p)), initial=0) > tolerance*.1:
        method = 'least_squares_fallback'
        candidate = least_squares(residual, p, jac=lambda x: residual(x, True)[1],
            tr_solver='lsmr', ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=512)
        p = candidate.x; fallback_evaluations = candidate.nfev
    w = u+K @ p; pn, pt = p[::2], p[1::2]; wn, wt = w[::2], w[1::2]
    normal_violation = max(float(np.max(-wn, initial=0)), float(np.max(np.abs(wn[pn > tolerance]), initial=0)))
    cone_violation = float(np.max(np.abs(pt)-mu*pn, initial=0))
    slip = np.abs(wt) > tolerance
    slip_violation = float(np.max(np.abs(pt[slip]+mu[slip]*pn[slip]*np.sign(wt[slip])), initial=0))
    equation_residual = float(np.max(np.abs(residual(p)), initial=0))
    passive_change = float(u @ p + .5*p @ (K @ p))
    energy_tolerance = tolerance*max(1., abs(float(p @ (K @ p))))
    if np.min(pn, initial=0) < -tolerance or max(normal_violation, cone_violation, slip_violation, equation_residual) > tolerance or passive_change > energy_tolerance:
        raise RuntimeError(f'Coulomb acceptance failed: normal={normal_violation:g}, cone={cone_violation:g}, slip={slip_violation:g}, equation={equation_residual:g}, passive={passive_change:g}')
    post = velocity + system.inverse_mass * (G.T @ p)
    full = np.zeros(system.contact_map.shape[0]); full[::3] = pn; full[1::3] = pt
    return post, full, {'method': method, 'iterations': iteration+1,
        'normal_initialization_iterations': normal_stats['iterations'],
        'singular_solves': singular_solves, 'line_search_halvings': line_search_steps,
        'fallback_evaluations': fallback_evaluations,
        'normal_residual_m_s': normal_violation,
        'friction_capacity_residual_kg_m_s': cone_violation,
        'slip_law_residual_kg_m_s': slip_violation, 'equation_residual': equation_residual,
        'contact_energy_change_minus_boundary_work_J': passive_change,
        'active_contacts': int(np.count_nonzero(pn > tolerance)),
        'sticking_contacts': int(np.count_nonzero((pn > tolerance) & (np.abs(wt) <= tolerance))),
        'sliding_contacts': int(np.count_nonzero((pn > tolerance) & (np.abs(wt) > tolerance))),
        'mobility_nnz': K.nnz}
