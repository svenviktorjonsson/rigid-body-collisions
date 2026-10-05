# Independent diagnostics of later full-run rejections

These captures come from the frozen 95d224f integration and its six prescribed
full hull trajectories. Passing earlier captured systems did not prevent new
trajectory failures. All four later captures examined here have independently
verified passive solutions under their original matrices, coefficients and
1e-8 m/s residual gates.

| Captured failure | Rows | Independent final residual, m/s | Passive change, J |
|---|---:|---:|---:|
| seed42 reference 0, velocity | 60 | 4.06e-15 | -16.946597374 |
| seed42 reference 1, velocity | 39 | 2.23e-9 | -0.489904899 |
| seed42 reference 2, velocity | 51 | 8.41e-15 | -0.079535882 |
| seed7301 reference 0, position | 291 | 4.21e-16 | -0.006402157 |

The 51-row warm Fischer–Burmeister/TRF solve completes in 238 evaluations. The
60-row and 39-row continuation searches require opposing-slip boundary restarts
at contact 2. The records retain all attempted intermediate problems, impulses
and failed final gates. Intermediate continuation coefficients belong only to
the numerical search and are never physically applied. The diagnostics allow
more work than production recovery and do not establish equivalent native speed
or successful full trajectories.

For the 291-row position failure, normal pressure feasibility is independently
confirmed by linear programming. Ordinary quadratic minimization and its first
FB polish fail around 3.18e-8 m/s. Multiplying only the numerical quadratic
objective by 398.09550657 changes its stopping behavior without changing its
minimizer or the physical matrix. A second quadratic search followed by three
FB/TRF evaluations satisfies the complete original law. This is a position
projection diagnostic; its passive quadratic objective is not an assertion that
angular split pose correction preserves physical rotational kinetic energy.

All final gates independently recompute every original normal and circular
tangent projection, finite energy, passivity, nonnegative normal impulses and
normal upper bounds. The auditor explicitly records negligible negative-normal
roundoff clamps and recomputes acceptance after those clamps. No physical matrix
regularization, material change or tolerance relaxation is used.

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.study_coulomb_new_failures research/hull-active-completion/results/rejections/fast_shake8_hulls42/reference_1.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.study_new_position_failure
```

The first command also accepts reference 0 or 2. Reproduction rewrites its
selected diagnostic output. Preserve the frozen archive before experimenting.

The separate `research.audit_hull_active_completion` auditor verifies native
progress output, whose quaternions refer to principal inertia axes. It recovers
the constant basis from initial authored and backend orientations, checks that
basis against independently integrated full inertia and every shape, then
converts orientations back to authored axes for energy and containment. Partial
progress is a prefix observation, never a completed or qualified trajectory.
Full qualification requires all prescribed physical gates, strict contact
residuals and both adjacent refinement edges.
