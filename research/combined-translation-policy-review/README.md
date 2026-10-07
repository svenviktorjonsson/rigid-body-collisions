This is a prospective research helper for separately declared
`split_translation_combined`, reviewed against frozen source108. It is not
included in the running world solver and does not qualify any trajectory.

The helper copies the existing post-transport solver-body pool and applies
accepted physical impulse **deltas** in the exact upstream row/A/B order. It
preserves existing warm starts and includes external force and gyro velocity
increments when evaluating the full signed normal Jacobian. Its all-row pose
target subtracts that final physical rate from the desired combined geometric
rate. Fake push/turn velocities never enter the predictor.

The proposed C++ controls use the actual compiled upstream MLCP writeback as
their parity oracle. They exercise warm normal/two-tangent impulses, a fixed
slot, rotated anisotropic inertia, nonunit factors, external force/gyro, angular
normal motion, branch boundaries and failure output preservation. The opposing
face example should require zero pose impulse under the all-row combined rule.
Root owns the separate integrated sphere/wall regression and any production
glue. A first-order target does not prove continuous containment or an accuracy
order for changing geometry.

After root commits and publishes the plan/helper/controls and authorizes their
execution, run with that exact published full SHA:

```sh
python -m research.combined-translation-policy-review.run_controls --source-commit FULL_SHA
```

This builds a separate research executable against existing read-only Bullet
libraries. It does not invoke CMake or mutate production sources, the running
spatial executable, libraries or headers. Source and library/header hashes,
compiler command/version, all compile/test output and failure outcomes are
saved. Existing result/build directories cannot be replaced. A guard checks
the current production study source, executable and runtime libraries before
and after both operations. Compilation and unit-control CPU contention is
disclosed and supplies no controlled performance comparison.

Root may separately declare `early_component_recovery=true` in a future full
study, and has identified a margin-before-AABB-cache geometry correction to
declare there as well. Those additional changes are not implemented by this
helper. Combined changes require explicit prospective declarations and cannot
support attribution to one change without a separately controlled comparison.
Original material, strict physical and refinement gates remain necessary for
any promotion. The current source108 study and its failed accuracy edges stay
visible and unchanged.
