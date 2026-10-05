# Independent live integration review

No blocking issue found in the snapshotted integration of the optional early recovery schedule, combined physical/translation target, hull margin/cache ordering, Python API, and CMake targets. Root is responsible for production edits, full-world tests, source freeze/publication, and commits. This review performed no production source, build, executable, or library mutation.

## Scope and provenance

`source-manifest.json` records the live source snapshot before compilation. This snapshot is distinct from the previously published prospective e2 candidate. `compile-plan.json` preceded isolated compilation; `compile-receipt.json` records exact commands, exits, executable hashes, and compiler/library guards. All three unique executables were compiled **without** `SPATIAL_LAPACK_RECOVERY`, with double precision and contraction disabled, against the existing read-only Bullet libraries. A separate unique CMake configuration used `SPATIAL_LAPACK_RECOVERY=OFF`; this verified target creation and generated compile definitions, rather than rebuilding the shared Bullet production dependency. `verify.py` and its hash preceded execution. All four numerical thread environment variables were one.

## Results

All 17 checks in `verification.json` passed, with no compiler or numerical failure:

- Both newly integrated native check executables pass without LAPACK.
- Native opt-in rejects unavailable LAPACK recovery, incompatible solver, disabled recovery, and a nonboolean JSON flag.
- Native combined translation rejects incompatible solver.
- Omitted versus explicitly false early recovery yields identical native JSON receipts except the measured `step_s`, on the simple no-contact control.
- Python rejects nonboolean early flags, incompatible solvers, disabled recovery, and combined translation on an incompatible solver. It accepts the new combined mode with matching wire and numerical-model metadata in the no-LAPACK executable.
- One-frame combined contact with a rotating prescribed wall preserves the exact physical endpoint (orientation/linear velocity/angular velocity) of the velocity-only control and satisfies the split gate.
- No-LAPACK CMake includes all three targets with double precision and no LAPACK compile definition.
- Snapshot, compiler/library, and compiled executable hash guards pass.

The live early branch retains default false, applies only to the physical velocity solve, seeds its helper from the actual first rejected phase, uses fresh per-call caps, records both early and aggregate support work, restores the original next-lane seed on decline, and writes accepted impulses only after the original-law full-system gate. The default ordering remains unchanged by this option. The earlier 22-capture independent parity/decline audit establishes the deeper schedule behavior; the present small controls complement rather than replace it.

The combined helper applies accepted-minus-warm impulse increments in upstream row order, includes force/gyro and angular physical motion, excludes numerical push/turn rates, and calculates signed remaining clearance. Its targets are a declared linearized numerical position policy. This is not a material compliance model, exact finite-rotation nonpenetration theorem, full trajectory convergence guarantee, or globally passive physical-contact claim.

The hull cache correction intentionally changes the effective cached hull AABBs/contact discovery. Therefore early-option default solver parity must not be described as proving identity of legacy default hull-world trajectories across this geometry fix. The API metadata correctly records the new margin/cache ordering and the optional early recovery caps/order.

Costs are not ranked. The standalone no-LAPACK checks do not demonstrate real-time performance, 2D/3D world refinement, convergence of the upcoming full study, or all-world default equivalence.
