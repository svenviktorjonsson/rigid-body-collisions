"""Continuous collision detection for smooth disks in a fixed unit square."""

import numpy as np


def advance_disks(x, v, r, m, dt, restitution=1.0, gravity=0.0):
    """Advance position/velocity arrays in place through one time step.

    Free motion and impacts are exact when gravity is zero. Uniform gravity
    uses symmetric half-step velocity kicks around the collision drift.
    Collisions are frictionless: only velocity along the contact normal changes.
    Inputs must be finite float arrays describing nonoverlapping disks in the box,
    with positive masses and radii. Simulation.add_object validates new bodies.
    """
    if not np.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")
    if not np.isfinite(restitution) or not 0 <= restitution <= 1:
        raise ValueError("restitution must be between 0 and 1")
    if not np.isfinite(gravity) or gravity < 0:
        raise ValueError("gravity must be finite and nonnegative")
    if dt == 0 or len(x) == 0:
        return

    v[:, 1] -= gravity * dt / 2
    remaining = dt
    i, j = np.triu_indices(len(x), k=1)

    # Every impulse changes future trajectories, so recompute after each event.
    for _ in range(10000):
        delta = x[i] - x[j]
        relative = v[i] - v[j]
        speed_sq = np.einsum("ij,ij->i", relative, relative)
        approach = np.einsum("ij,ij->i", delta, relative)
        gap = np.einsum("ij,ij->i", delta, delta) - (r[i] + r[j]) ** 2
        discriminant = approach ** 2 - speed_sq * gap
        pair_times = np.full(len(i), np.inf)
        approaching = (approach < 0) & (speed_sq > 0) & (discriminant > 0)
        # Avoid cancellation for disks already close to contact.
        pair_times[approaching] = np.maximum(
            0.0,
            gap[approaching]
            / (-approach[approaching] + np.sqrt(discriminant[approaching])),
        )
        pair_index = int(np.argmin(pair_times)) if len(i) else None
        pair_time = pair_times[pair_index] if pair_index is not None else np.inf

        wall_times = np.full_like(x, np.inf)
        target = np.where(v > 0, 1 - r[:, None], r[:, None])
        moving = v != 0
        np.divide(target - x, v, out=wall_times, where=moving)
        wall_times[moving] = np.maximum(0.0, wall_times[moving])
        wall_flat = int(np.argmin(wall_times))
        wall_time = wall_times.flat[wall_flat]
        event_time = min(pair_time, wall_time)

        if not np.isfinite(event_time) or event_time > remaining:
            x += v * remaining
            break

        x += v * event_time
        remaining -= event_time
        if wall_time <= pair_time:
            body, axis = np.unravel_index(wall_flat, wall_times.shape)
            x[body, axis] = target[body, axis]
            v[body, axis] *= -restitution
        else:
            first, second = i[pair_index], j[pair_index]
            normal = x[first] - x[second]
            normal /= np.linalg.norm(normal)
            normal_speed = np.dot(v[first] - v[second], normal)
            impulse = -(1 + restitution) * normal_speed / (1 / m[first] + 1 / m[second])
            v[first] += impulse * normal / m[first]
            v[second] -= impulse * normal / m[second]
        # Loop again even at the end of the step to resolve corner contacts.
    else:
        raise RuntimeError("Too many collision events in one step; reduce dt")

    v[:, 1] -= gravity * dt / 2
