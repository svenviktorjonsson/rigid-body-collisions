# Independent saved-production receipt audit

`audit.py` reads the saved 9e97be07 production receipt and every original captured
system. It never executes a physics solver. All 22 full-row circular projections,
normal bounds and finite quadratic passivity gates pass, together with independently
recomputed native contact-law bookkeeping. The maximum original projection
residual is 9.92288212167501e-9m/s against the unchanged 1e-8 gate. The previous
twenty Float64 impulse arrays are bitwise identical, including signed zero.

The saved source archive has 24 sources matching git9e97be07 exactly. The extra
position_geometry.h fingerprint is explicitly classified as archived, uncommitted,
uncompiled collateral. Its exact saved bytes are verified, its absence from the
commit is verified, and no committed recorded source references it. Its presence
does not identify a serializer in the tested model.

Default checking is portable and does not read the current host binary/libraries.
`--check-current-runtime` additionally checks current binary and all seven recorded
shared-library byte hashes. Both modes passed; the latter local attestation is
saved separately. This records the declared linked dependency set, not the kernel,
dynamic loader or machine isolation. Recorded source, capture plans, capture hashes,
original budgets, native response and all exported search caps are verified.
Pressure-attempt counts are not exported; pressure SVD and pivot-guide call caps are
checked. Internal guide pivots have no exposed hard bound.

Reproduce without overwriting retained receipts:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/component-recovery-integration/audit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/component-recovery-integration/checks.py
```

Use `--save` with a new destination for another audit receipt. Existing destinations
are refused. `final-independent-audit.json` is the final portable receipt;
`final-current-runtime-audit.json` is the current-host byte check. The earlier
`independent-audit.json` and its exact producer, `initial-auditor-source.py`, are
retained rather than overwritten after adding saved-summary/counter checks.

Five adversarial/portability controls pass. They forbid access to current runtime
bytes in portable mode, mutate an impulse while recomputing its raw response,
rehash changed committed source, exceed a declared SVD cap, and remove a mandatory
counter. The first controls attempt used an impulse perturbation large enough to
fail passivity before the intended projection check; the harness expected the
wrong error message. Its source/log are retained. A smaller perturbation reaches
the intended projection failure, with the auditor itself unchanged.

These are captured algebraic-system checks. For captured position-target systems,
the quadratic bound belongs to the numerical projection objective; it is not a
physical kinetic/potential energy ledger for pose repair. These results neither
qualify a full trajectory nor establish material calibration or performance
superiority. The independent frozen six-run refinement failures remain visible.
