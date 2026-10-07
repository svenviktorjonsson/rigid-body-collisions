# Consistent gyroscopic velocity in the tangent contact equations

The adapter now includes both bodies' implicit gyroscopic angular-velocity
increments in each tangent RHS. Upstream normal equations and final body
writeback already include them. The signed torque Jacobian for body B supplies
its own minus sign; an extra sign must not be inserted.

A native two-body regression chooses each COM velocity so that the free contact
point velocity, including gyro, is zero. In upstream tangents the two spurious
RHS components are approximately 0.481 and -1.742 m/s; the corrected components
are below 1e-12 m/s. The normal component is already zero. This regression
exercises anisotropic inertia and gyroscopic increments on both bodies; it is
included in `spatial_friction_checks` and hosted builds.

The correction fixes the mechanics mismatch. It **does not resolve every
nonlinear contact solve**. A separately frozen follow-up replays the two rejected
random-hull scenes with identical authored geometry, mass, material coefficients,
gravity, prescribed wall motion, timestep ladder and acceptance gates. Every one
of the six attempts still rejects the strict 1e-8 m/s contact residual gate:

| Case | Travel fraction | Final rejected residual (m/s) |
|---|---:|---:|
| Eight shaking hulls, seed 42 | 0.06 | 1.77895e-8 |
| Eight shaking hulls, seed 42 | 0.03 | 1.08727e-8 |
| Eight shaking hulls, seed 42 | 0.015 | 3.47786e-5 |
| 27 rotating/shaking hulls, seed 7301 | 0.06 | 2.30410e-5 |
| 27 rotating/shaking hulls, seed 7301 | 0.03 | 1.71763e-7 |
| 27 rotating/shaking hulls, seed 7301 | 0.015 | 1.97331e-5 |

There is no trajectory history or verified fast setting for a rejected run.
Reasons, timings and complete inputs are retained; no friction-pyramid fallback
is accepted. Unchanged gates make this a disclosed remaining solver/integration
limit, not a claimed successful random-shape engine. Further active-set handling
of redundant contacts and a verified compliant contact branch remain necessary.

Execution source: `78e63313b5d13dd80170d03c54bcaffccbbf47e8`.
Baseline source: `7c279768f6518e66b1ee620d44bd9b37309b1b5f`.
The exact source snapshot, six rejection receipts and scenes are SHA-256-pinned
in `results/summary.json`. `python -m research.audit_spatial_gyro` independently
checks source integrity, equality of authored physics and controls, every retained
attempt and the absence of any qualification. The original 76-attempt study and
its three qualified references remain unchanged.
