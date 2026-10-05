# Shared-contact hull ladder with bounded Jacobian rank retry

The six predeclared attempts yield **0 completed histories**,
**6 retained rejections**, and **0 qualified references**.
All original contact, passivity, physical and quarter-budget trajectory gates remain.

| Scene | Travel fraction | Rows | Outcome | Residual (m/s) |
| --- | ---: | ---: | --- | ---: |
| fast_shake8_hulls42 | 0.06 | 48 | Rejected | 1.1475683e-06 |
| fast_shake8_hulls42 | 0.03 | 51 | Rejected | 0.00062897858 |
| fast_shake8_hulls42 | 0.015 | 30 | Rejected | 1.7766608e-06 |
| fast_rotate_shake27_hulls7301 | 0.06 | 216 | Rejected | 4.4907071e-07 |
| fast_rotate_shake27_hulls7301 | 0.03 | 192 | Rejected | 1.9651417e-05 |
| fast_rotate_shake27_hulls7301 | 0.015 | 123 | Rejected | 0.0059708811 |

Execution source: `0862e30736af34b3736e53154f0679a790113f9f`. Geometry, material parameters, shared contact-point
construction, wall schedules, spin, effort levels and thresholds are unchanged
from the shared-contact source38 study. The numerical change is one relative
singular cutoff1e-10 retry on the projected-equation Jacobian after the ordinary
1e-12 Newton search stalls. The physical mobility matrix A is unchanged, and
recovery remains subject to the same global256-SVD-call ceiling and full original
contact/passivity gate. This is numerical search, not physical compliance.

The frozen earlier48-row seed42 system independently and natively passes after
this retry. The seed42 fraction0.03 trajectory progresses beyond that particular
failure, then encounters a later rejection. A captured-system success therefore
does not imply completion or accurate refinement of the whole trajectory.

Every attempt, rejected matrix, exact reason, source, authored scene, binary
identity and checkpoint remains archived. The independent auditor checks source
identity, paired controls and scenes, snapshot hashes, all rejection gates and
reference receipts. It independently derives mass/inertia, trajectory errors,
energy and authored surface containment for completed histories. No accuracy
claim is possible for an incomplete history. Earlier failed studies remain intact.

The reported timings are descriptive under recorded collaborative workload;
no new speed ranking or mechanical infeasibility claim follows from this study.

Audit: `python -m research.audit_shared_hulls --directory research/shared-hull-rank-followup --source-commit 0862e30736af34b3736e53154f0679a790113f9f`.
