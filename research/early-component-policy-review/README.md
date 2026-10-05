# Optional early exact-component schedule — prepared research candidate

Prepared against frozen production source `108a9bb4c7899f75d760b27b179cc56557904a08`. All preparation is contained here. **No compiler, build, native replay or full trajectory study has been run for this candidate.** Root must first confirm completion of the ongoing frozen full study. This directory is an optional performance proposal, not an applied production repair.

The prospective plan is [plan.json](plan.json), with its original SHA256 receipt in [plan.sha256](plan.sha256). The source snapshots are under `frozen/`; candidate copies are under `candidate/`. [optional-schedule.patch](optional-schedule.patch) is the proposed integration diff. [prepare_candidate.py](prepare_candidate.py) records reversible transformations; removing precisely those changes reconstructs frozen files byte for byte. This proves source preservation outside the opt-in additions, not numeric runtime equivalence or compilation success.

## Proposed behavior

The existing initial `coulombIterate` phase remains unchanged. Only when it rejects after 256 sweeps, recovery is enabled, the optional flag is true, LAPACK recovery is compiled and the original system has at most 4096 rows, try the unchanged exact-component V3 helper using **that actual rejected impulse array**. The helper uses complete contact triples and exact nonzero couplings and must pass the unchanged **full original system** projection, normal bounds, circular friction and finite passivity gate. Reduced-system success is insufficient.

On acceptance, apply only the gated candidate. On decline, preserve `x` and the rejected array and explicitly restore the original next-lane candidate. Continue the original normal/active/continuation/polish/remaining-sweeps/supplemental/tail ordering. The old tail call remains present with its own fresh statistics and caps. Unlike the earlier captured replay experiment, this insertion does **not** repeat the initial 256 sweeps after decline.

The default option is false. Default velocity ordering and position paths remain unchanged. The native candidate parser reads `early_component_recovery` only as an explicit boolean option and rejects enablement without Coulomb and compiled/enabled recovery. The ordinary Python wrapper has **not** been modified or copied: eventual public API integration, validation and numerical-model provenance are root's separate decision. Do not silently enable the option by building these copies.

## Code-review pointers

- Frozen `spatial_backend/coulomb.h`: `coulombIterate` writes `x` only after its accepted gate; `coulombSolve` creates the first rejected seed and the original fallback ordering. The new branch immediately follows the existing `candidate` construction.
- Frozen `spatial_backend/support_restart.h`: `fullGate` checks every original row and bounds, including inactive rows; `x` is written only after `fullGate` succeeds. Exact component grouping and immutable size/call/SVD caps remain unchanged.
- Candidate `spatial_backend/coulomb.h`: `early_component_attempts` and associated receipts aggregate across solves; fresh local `support_restart_v3::Stats early` controls each attempt. The existing tail's fresh object remains untouched. Aggregate support receipts count early and tail work, including declined attempts.
- Candidate `spatial_backend/runner.cpp`: default false parser and conditional `early_component_policy` receipts. An off run keeps the original output fields. Enabled runs expose the extra work and response/trajectory limitation.
- [policy_checks.cpp](policy_checks.cpp) is an **uncompiled** local test harness. It prepares immediate-success/default comparisons, failed component and asymmetric-input immutability checks, and separate per-call receipt checks. These assertions are not reported as passed.

The prospective transformation check has run as text processing only. All 31 copied source hashes match the frozen commit; only candidate `coulomb.h` and `runner.cpp` change. No guarded production source or build output is altered.

## Work budgets and performance limits

Each V3 call retains maximum 4096 full rows, 64 rows per actually searched exact component, 8 passes, 8 helper calls, 1024 nonlinear/projector SVD calls, 1024 pressure SVD calls and 8 normal-pivot guide calls. A structurally inspected over-cap component can be recorded without being numerically searched. Upstream internal pivot work is not exposed. Exact-component scans and full-row validation also cost work not summarized by an SVD count.

Early decline followed by the original tail can spend **two independent V3 budgets**: up to 16 passes/helper calls, 2048 nonlinear/projector SVD calls, 2048 pressure SVD calls and 16 pivot guides, in addition to the original fallback pipeline. Every attempt must count, even if it declines. There is no hard real-time guarantee, universal wall-clock bound or claim that the policy is always faster.

The earlier `translation-first256-review` results motivate trying this schedule: 51 of 66 early attempts succeeded and all 132 final captured replies passed original gates. Those are historical rejected-system starts, not original world warm starts. They use a different decline driver and contain timing contamination. Their accepted roots can differ. **Their counts are not new execution results for this candidate, and do not prove full-engine trajectory accuracy or speed.**

## Tests required after root's explicit execution clearance

1. Compile only isolated research targets against the copied sources and pinned Float64/LAPACK dependencies. Compile baseline frozen and candidate using the same compiler flags. Do not rebuild or replace the production executable used by the ongoing study.
2. Require default candidate versus frozen equality of acceptance, impulses and work counters from identical starts on all 22 frozen captures; excluded timing fields may differ. Also compare omitted flag versus explicit false in the native input. No runtime-equivalence result is claimed until this runs.
3. Record an independently recomputed first256 result and actual rejected array. Test opt-in from the original identical start; retain every early decline, all final rejected inputs and all counter receipts. Recompute the original equations/bounds/passivity outside helper numerical merits for every accepted reply. A proxy cold or old-tail seed is not acceptable.
4. Verify no early attempt after first-phase success, below 256 sweeps, with recovery disabled or without compiled LAPACK; ensure failed structural/physical gates do not write `x`. Verify fresh early/tail call caps despite accumulated receipts from previous solves. An early and later tail count must reconcile with the aggregate counters.
5. Run held-out full histories with identical geometry, matrix/material laws, targets, tolerances and all energy/work ledgers. Freeze any new comparison plan before their execution. Preserve reference failures and accepted-root trajectory differences. Do not change accuracy gates, relax laws or retrofit ensemble metrics to label a failed path successful.

Original-law acceptance is a necessary safety condition for applying an impulse, **not proof that alternative accepted roots give the same subsequent history**. Keep the option experimental until its complete trajectory and cost evidence supports promotion. Root alone decides integration, commits and publication.
