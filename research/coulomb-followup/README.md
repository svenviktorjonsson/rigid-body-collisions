# Second rejected random-hull contact systems

This diagnostic follow-up preserves the original contact matrix, velocity targets,
normal complementarity, isotropic Coulomb circle, coefficient 0.4, and strict
1e-8 m/s residual. It does not add compliance or change restitution. A successful
frozen contact solve does **not** qualify a complete shaking trajectory.

| Capture | Rows / rank(A) | Captured residual (m/s) | Verified residual (m/s) | Diagnosis / numerical remedy |
| --- | --- | --- | --- | --- |
| hull42-second | 45 / 34 | 1.0603230e-5 | 2.7104377e-11 | Local nonzero minimum of projected-equation merit on an inconsistent sticking face; one opposing-slip face restart |
| hull7301-second | 267 / 159 | 9.9020023e-8 | 1.9743628e-12 | Exceeded native rare-recovery 256-row eligibility limit; unchanged published mechanical-nullspace recovery solves it |

## 45-row solution and limits

The stalled Jacobian has rank 42. Its three null directions also lie in the
mechanical nullspace. Simultaneously imposing the active normals and three
sticking tangent pairs gives 14 equations of rank 11 and a target range error
about 8.33e-6 m/s. The active normal equations alone are consistent. No global
normal infeasibility witness was found, and this is not an infeasibility proof.

Neither the published single-direction gauge search, an 80-attempt lattice /
breadth-first combination search, nor targeted feasible neutral relocations
produced acceptance. The broader mobility nullspace has 11 directions (six
remain after freezing opening-contact impulses); the exploratory constrained
restarts also failed. These failures are retained, rather than dropped from the
study. Numerical friction continuation also failed; its optimizer status was
never treated as a physical acceptance certificate.

The successful restart is deliberately **not mechanically neutral**. Starting
from the two-step stalled Newton result, select the contact with largest
nonzero tangential projection residual and replace that contact's tangent
initialization with `-mu * p_n * w_t / norm(w_t)`. Leave every other initialization
coordinate alone. This retains its existing pressure and circular capacity,
and is invariant under a rotation of the tangent basis. It changes numerical
initialization, not any physically applied intermediate impulse. The initial
contact-velocity change is 0.6381 m/s and residual is 0.6857 m/s; these are not
accepted physical states. The subsequent 24 Newton increments solve the same
original equations, with total 26 Newton calls including the initial warm solve.
The implementation has a global 256-call ceiling and at most eight restarts.

At the accepted endpoint, contact 0 sticks; contacts 1 and 2 slide at approximately
2.10e-5 and 6.71e-6 m/s. Its passive work bound is -1.4327651933 J. Its endpoint
contact velocities differ from the original stalled state by only 1.37e-5 m/s,
even though the restart traverses a different merit basin. This is evidence for
a missing mode escape, not a reason to relax the acceptance threshold.

A separate derivative-choice experiment used the original residual with a
saturated projection derivative at/near the circle boundary. Forty bounded
starts (warm, stalled, six neutral gauges; five boundary tolerances) all failed.
The best residual was 8.32844e-6 m/s. The largest boundary tolerances are
quasi-Newton experiments, not exact generalized derivatives away from the
boundary. None changed the final law or gate.

## Replay and audit

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python -m research.hull_recovery_followup research/coulomb-followup/hull42-second-rejected.json --output /tmp/hull42-followup.json
OPENBLAS_NUM_THREADS=1 python -m research.hull_recovery_followup research/coulomb-followup/hull7301-second-rejected.json --mode baseline --output /tmp/hull7301-followup.json
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 python research/coulomb-followup/verify.py
```

The independent verifier recomputes normal/circular projection residuals and
finite passive work from the archived matrix and impulse. It checks both
captures, rejection under a four-call effort ceiling, and tangent-basis
covariance. It records contact cone excess and positive tangent work separately.
`manifest.json` pins each diagnostic input/output/script and the unchanged
published diagnostic source. Floating-point nullspace basis choices can select
a different equivalent pressure gauge on different LAPACK builds; acceptance
requires the same strict physical gate in every replay.

All results here concern frozen matrices. Native implementation and fresh,
source-pinned full-trajectory studies are required before claiming that the
random-hull shaking cases are resolved. The earlier rejected six trajectories
remain evidence of a limitation of their frozen implementation.
