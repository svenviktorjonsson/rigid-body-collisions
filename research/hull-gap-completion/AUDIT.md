The independent auditor is `research/audit_hull_gap_completion.py`. Its default
archive check reads saved metadata and source bytes; it does not require the
current machine's binary or libraries to match the execution host. The optional
local byte attestation checks the executable and every resolved recorded shared
library explicitly. Every receipt destination must be new.

Before execution, validate the prospective controls without running native code:

```sh
python -m research.audit_hull_gap_completion --plan-only
python -m research.hull-gap-completion.auditor_checks
```

Once the frozen runner has created provenance and its source archive, capture a
local runtime receipt while the frozen execution binary is still available:

```sh
python -m research.audit_hull_gap_completion --source-commit FULL_SHA --runtime-only
```

After all six attempts finish, use a fresh prefix for the complete independent
archive, principal-frame progress, runtime metadata and numerical pose ledger:

```sh
python -m research.audit_hull_gap_completion --source-commit FULL_SHA --receipt-prefix final-independent
```

During a running study, `--progress-only --receipt-prefix NEW_PREFIX` audits
available accepted output-frame prefixes. It never qualifies a trajectory.
Complete qualification requires all six terminal outcomes and both original
adjacent refinement edges within the original quarter budgets; a rejection
cannot be skipped. Normal-only position snapshots require the explicitly
declared gap policy and the new schema; the historical translation-only
missing-snapshot exception remains restricted to its old policy.

The new position residual must be finite, nonnegative and at most the unchanged
contact tolerance in each accepted prefix and final result. The six pose-ledger
fields must be finite, their signed changes bounded by their accumulated
absolute changes, and their final values identical to the last native prefix.
Saved per-lane ledger receipt hashes are checked at completion.

These ledger checks disclose numerical displacement, gravity-potential change
and orbital-angular-momentum change. Per-update repair impulses are not archived,
so the audit does not independently reconstruct each repair. The original
full-tensor kinetic-plus-gravitational energy minus boundary work remains the
physical gate; the numerical ledger is not subtracted from it.

Controls rerun the two historical complete archives and require byte-identical
independent receipts. Malformed ledger values, a changed original gate or
budget, and qualification from incomplete bookkeeping are rejected. The first
control harness run selected a nonexistent historical receipt filename; that
attempt is retained in `auditor-checks-initial-failure.json`. The corrected
harness changed only its filename selection.
