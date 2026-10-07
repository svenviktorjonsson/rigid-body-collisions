"""Explicit normal/tangential endpoint restitution coefficients.

Convention: u_normal_after = -e_normal*u_normal_before and, when friction
capacity permits, u_tangent_after = -e_tangent*u_tangent_before. Tangential
restitution -1 means unchanged slip; 0 sticking; positive values slip reversal.
"""
import math


def coefficients(normal, tangential):
    if normal is None and tangential is None:
        return None
    if normal is None or tangential is None:
        raise ValueError('Specify both normal_restitution and tangential_restitution')
    if isinstance(normal, bool) or isinstance(tangential, bool):
        raise ValueError('Restitution coefficients must be finite numbers')
    normal, tangential = float(normal), float(tangential)
    if not math.isfinite(normal) or not 0 <= normal <= 1:
        raise ValueError('normal_restitution must be in [0,1]')
    if not math.isfinite(tangential) or not -1 <= tangential <= 1:
        raise ValueError('tangential_restitution must be in [-1,1]')
    return dict(normal=normal, tangential=tangential)
