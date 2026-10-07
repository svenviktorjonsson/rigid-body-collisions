# Prospective code review — no numerical execution

The proposal changes scheduling, not the contact equations. It is optional and is not promoted to production. Root is still running the full frozen study; this preparation neither changes its files nor invokes a compiler.

| Requirement | Source pointer and assessment |
|---|---|
| Frozen default lane order | `frozen/spatial_backend/coulomb.h:199`; candidate changes only add a false-default branch and optional argument. Reverse transformations exactly reconstruct frozen header/runner bytes. This is a static source claim; compiled default behavior remains to be checked. |
| Actual failed first256 seed | `candidate/spatial_backend/coulomb.h:206` retains existing first phase; `:215` creates `candidate` from that phase's `rejected` vector; `:219` attempts early recovery only if its budget was exactly256. No cold or historical capture seed is substituted. |
| Full original gate | Unchanged `frozen/spatial_backend/support_restart.h:28` recomputes all original rows, finite bounds and passivity; `:89` is the helper's only assignment to its caller's impulse vector. A reduced component optimizer flag is never enough. Original `lo`/dependencies are already validated by the preceding unchanged iteration phase. |
| Decline immutability | Early trial acts only on local `candidate`; the caller's `x` is assigned only on true helper acceptance. Candidate `coulomb.h:256` explicitly restores the rejected first-phase vector before existing normal/active and subsequent lanes. The original `rejected` vector itself is unchanged. |
| Exact components and caps | Unchanged support `:59` declines structurally inspected components exceeding64 rows before numerical helper calls. No coupling threshold, material change, diagonal regularization or compliance is introduced. |
| Fresh per-call budget | Candidate `coulomb.h:220` declares a new local `Stats early`; existing late support declaration remains new per call. Aggregated statistics are receipts, never a cap input. |
| Work accountability | New fields at candidate `coulomb.h:33` record all early attempts, declines, helper/pressure/pivot work. Existing support totals aggregate early plus tail. Native output at candidate `runner.cpp:214` exposes extra-budget and different-root limitations only when the flag is explicitly enabled. |
| Velocity versus position | Only the velocity `CoulombMLCP` call passes the option. Existing position projection and translation/gap policy code do not opt in; their behavior and numerical target choices remain separate. |
| Native default/API | Candidate native parser `runner.cpp:40` defaults `early_component_recovery` to false and rejects enablement without Coulomb/compiled-enabled recovery. No Python production API is edited; eventual `spatial_engine.run(..., early_component_recovery=False)` validation and metadata must be integrated explicitly by root. |

## Prepared checks, not passed tests

`policy_checks.cpp:13` prepares a known immediately successful first-phase contact and verifies no early helper call or extra sweeps. `:31` prepares an exactly connected72-row system whose69-row postrelease support exceeds the unchanged64-row component cap, requiring decline without impulse writes or numerical helper/SVD calls. It repeats with a new cap object and an accumulated separate receipt, and tests an asymmetric matrix's early decline. These sources have not been compiled or run.

`replay_policy.cpp` reuses the frozen independent normal/complementarity, upper-bound, friction support-function and finite contact-energy-bound checks. Its diagnostic first phase at `:61` begins from the exact supplied capture array and archives that rejection. The candidate build can enable the extra argument; a frozen build rejects an early request. It checks rejection preserves `x`, reconciles attempts = accepts + declines, and enforces original per-early and disclosed two-call maximum receipts, including `:144`. This driver also remains uncompiled/unexecuted.

The replay diagnostic call adds work outside the solve and contains no timing comparison. Historical captures are final rejected arrays, not original world warm starts. Their contact RHS-system quadratic is not a stand-alone physical kinetic energy/work ledger for arbitrary targets. Full zero-restitution trajectories still need their unchanged body, boundary-work and numerical position ledgers.

## Remaining review risks

1. Default compiled impulses/counters must be matched to the frozen baseline, including absent versus false native JSON flags. Textual reconstruction is useful but not a substitute.
2. Historical22-capture results must be rerun for this insertion. The old prior diagnostic restarted a whole baseline after decline; this proposal continues existing lanes without repeating256 sweeps. Its measured costs cannot be inherited from that different driver.
3. Early support release can choose another accepted nonassociated Coulomb root, leading to a different contact manifold and trajectory. Original-law passivity is a necessary acceptance gate, not an accuracy equivalence certificate. A lower work count cannot erase a failed full-history comparison.
4. A declined early call adds structure scans, full gates and possibly a complete extra search budget before the original pipeline. Count it. Work counters do not bound wall time, and upstream normal-pivot internal work is not exposed.
5. Native thresholds below256 sweeps do not trigger the option. This is intentional; do not silently increase a user's initial sweep budget to force it.
6. Explicit compiler clearance is still required. No claims of compilation, numerical test success, full-engine speed, convergence, or hard real-time behavior are made for this candidate.

The static checks confirm all31 frozen source hashes and exact restoration of the two changed candidate copies. Production guarded source/header and `spatial_engine.py` status were checked and remain unchanged by this agent. Root alone chooses later integration, executes the prospectively declared comparisons and commits/pushes.
