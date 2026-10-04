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
    rows, cols, values = [], [], []
    for edge, (one, two, point, normal) in enumerate(contacts):
        if not (0 <= one < n and 0 <= two < n): raise ValueError('Contact body index out of range')
        point, normal = np.asarray(point, dtype=float), np.asarray(normal, dtype=float)
        if point.shape != (2,) or normal.shape != (2,) or not np.isfinite(point).all() or not np.isfinite(normal).all() or np.linalg.norm(normal) == 0:
            raise ValueError('Finite contact point and nonzero normal required')
        normal = normal / np.linalg.norm(normal); tangent = np.array([-normal[1], normal[0]])
        for body, sign in ((one, 1), (two, -1)):
            r = point - centers[body]
            for channel, direction in enumerate((normal, tangent)):
                coefficient = [direction[0], direction[1], (-r[1]*direction[0] + r[0]*direction[1]) / ell]
                for component, value in enumerate(coefficient):
                    if value:
                        rows.append(3*edge + channel); cols.append(3*body + component); values.append(sign*value)
            rows.append(3*edge+2); cols.append(3*body+2); values.append(sign)
    G = sparse.coo_matrix((values, (rows, cols)), shape=(3*len(contacts), 3*n)).tocsr()
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
            mode[below] = -1; mode[above] = 1
            x = np.clip(x, lower, upper)
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


def normal_solve(system, velocity, tolerance=1e-8, dense=False):
    """Coupled normal solve at fixed contacts. Rejects failed complementarity."""
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != system.inverse_mass.shape or not np.isfinite(velocity).all():
        raise ValueError('Finite scaled body velocity per degree of freedom required')
    N, K = system.mobility((0,)); u = np.asarray(N @ velocity)
    p, stats = box_qp(K.toarray() if dense else K, u, tolerance=tolerance)
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
                  'normal_mobility_nnz': K.nnz, 'contact_map_nnz': system.contact_map.nnz})
    return post, impulses, stats
