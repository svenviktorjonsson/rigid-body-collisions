# Shared contact-point hull follow-up

All **six** predeclared trajectory attempts reject. There are **zero** completed
histories and **zero** qualified reference ladders. The unchanged circular-law
residual threshold remains **1e-8 m/s**; no failed refinement level is skipped.

| Scene | Travel fraction | Rows | Rejected residual (m/s) |
| --- | ---: | ---: | ---: |
| fast_shake8_hulls42 | 0.06 | 48 | 1.1475683e-06 |
| fast_shake8_hulls42 | 0.03 | 48 | 1.2099446e-08 |
| fast_shake8_hulls42 | 0.015 | 30 | 1.7766608e-06 |
| fast_rotate_shake27_hulls7301 | 0.06 | 216 | 4.4907071e-07 |
| fast_rotate_shake27_hulls7301 | 0.03 | 192 | 1.9651417e-05 |
| fast_rotate_shake27_hulls7301 | 0.015 | 123 | 0.0059708811 |

Execution source: `38f407d12208654075c07315e96b9fc213612b91`. Both three-level ladders use the original authored
hulls, density-derived mass/inertia, coefficient mixing, zero restitution,
gravity, 20 m/s shaking, optional 10 rad/s container spin, and travel fractions
0.06/0.03/0.015. Trajectory budgets and physical gates remain unchanged.

The contact discretization is explicitly changed: both finite dynamic bodies
use the midpoint of their surface endpoints; a finite body against a static or
kinematic body uses the finite-body endpoint. Both lever arms, angular mobility,
free-velocity and split RHS, transported warm angular impulses, and boundary
work use the same world point. This corrects wrench geometry, but these results
show that the correction plus current bounded recovery is insufficient to
complete either challenging hull scene.

Every rejection has a matrix snapshot and exact reason in `results/rejections/`.
Source, authored geometry, partial checkpoints, final traces, hashes and binary
identity are retained. Independent audit verifies source provenance, paired
controls/geometry, all rejection snapshots and both failed reference receipts.
It recomputes trajectory/physical gates independently for any completed history;
there are none in this study, so no trajectory-error estimate is available.

Costs are descriptive: the controlled timing study finished before these runs;
independent archive/documentation work could run concurrently. No speed ranking,
mechanical infeasibility, convergence order, or material authenticity follows
from these six rejections. Earlier archives remain unchanged.

Audit: `python -m research.audit_shared_hulls`.
