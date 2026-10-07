# Minimum-norm QR contact search, prospective v2

This is a research-only numerical search experiment against the exact production
headers at `95d224f1cfb36ec5e34bf4e5bd1a1e8484f9e8eb`. It changes how a Newton
direction is proposed. It does not change the physical mobility matrix, circular
Coulomb law, material coefficients, iteration budgets, or final acceptance gates.
The prospective plan was committed at
`f1aadaccc37d286163b301b9056c5622375cb4af` before any v2 solver execution.

The earlier basic-solution QR experiment in `../qr-contact-review` is retained
unchanged. It regressed one baseline-accepted capture and was not timed. This
separate version completes the minimum-norm solution in rank-deficient pivoted
coordinates, rather than setting all discarded coordinates to zero.

## Numerical method and limits

Column-pivoted Householder QR operates on a uniformly scaled numerical Jacobian
and right-hand side. The rank threshold is relative to the largest initial
column norm, at 1e-12. It is a numerical column-rank decision, not a physical
regularization or a claim to identify exact rank. A small Cholesky solve completes
the minimum-norm direction using whichever of the retained/free-coordinate Gram
systems is smaller. The identity in that search Gram is part of the mathematical
minimum-norm correction; no identity is added to the physical contact matrix.

The candidate must pass a null-coordinate orthogonality certificate and predict
at least 0.001 relative squared-merit reduction in the original Jacobian. The
original Armijo line search must also yield at least 0.001 actual reduction for
this QR proposal. Failed rank, finite-value, Cholesky, orthogonality, model, or
line-search checks fall back to the original SVD direction. This progress guard
addresses the very small accepted steps that consumed the first version's budget.

QR is limited to 384 reduced rows and 1024 calls per captured solve. Original
trust step limits and pressure/nullspace SVD remain unchanged. Dense QR and the
Gram solve still have cubic worst-case cost; fewer SVD calls alone do not prove
lower cost. Overflow in the unscaled model-merit certificate causes a safe
fallback, and extreme conditioning can reject the Gram candidate. These bounds
limit this trial, not the solver's complete wall-clock runtime.

## Frozen validation

The numerical fixture binary passes. All 40 untimed captured-system attempts
completed: baseline and v2 each accept 17 of 20 captures, with zero cases where a
baseline-accepted capture fails in v2. Every accepted impulse independently passes
the original full-row projection, nonnegative-normal, finite, and passivity gates.
The three later seed42 velocity captures fail in both variants; they remain
functional evidence and cannot contribute successful-cost ratios.

The 291-row position capture passes this general captured-system replay in both
variants. That does not erase its earlier rejection in the full world's distinct
position-projection path. Captured solves also do not qualify any full trajectory.

## Timing protocol

Following validation, the committed protocol executes one warmup pair and five
randomized paired repetitions for every one of the twenty fixed captures. Both
variants use the same compiler flags, captured budget, single-thread environment,
and CPU4 affinity. Native steady-clock measurements surround only the solve;
parsing and process startup are excluded. Source, executable, and input hashes are
checked before every attempt. All failed and slower attempts are retained.

Only captures accepted by both variants in all five timed repetitions receive a
successful-cost ratio. The predeclared statistic is the median of the five paired
baseline/QR ratios, with per-variant medians and ranges also recorded. These are
descriptive measurements in a shared collaborative container. Affinity does not
establish CPU, memory, or machine isolation. Per-attempt load averages and a
separate 30-second live-process observation ledger disclose concurrent workload.
The sampler is a lightweight observation process rather than a solver change.

All 240 scheduled timing-stage attempts completed and independently re-audited:
40 warmups and 200 timed solves, with zero baseline-accepted regressions. The same
17 of 20 captures pass in both variants. Twelve eligible captures have median
paired baseline/QR cost ratios above one, and five below one. The larger 267-row
and 321-row captures have observed paired ratios 2.143 and 2.919 respectively;
the earlier 45-row v1 regression now passes with an observed ratio 1.930.
[The complete table](cost-results.md) includes every capture, both cost ranges,
the slower cases, and all failed attempts. Ratios are descriptive under the
recorded shared workload and apply to this frozen control source, preceding
subsequent production repairs.

The supplementary analytic fixtures cover both Gram branches and pass. Across
all seventeen accepted baseline/QR validation pairs, the largest difference in
physical relative contact velocity is 2.3914e-11m/s; none exceeds ten original
tolerances. This provides no evidence for distinct physical branches among
these retained roots, without proving uniqueness.

No production adoption or universal performance superiority is claimed.
Numerical cost improvements do not resolve the published full-world
refinement failure: the completed spinning/shaking ref1/ref2 pair differed by
0.1086m in RMS position, 6.122m/s in velocity, 74.26rad/s in angular velocity, and
1.334rad in orientation, all above the prescribed quarter budgets.

## Reproduction and frozen receipts

`build_prototype.py` rebuilds the two native variants from the declared control
commit and records source, compiler and binary hashes. Use a separate checkout
to reproduce the experiment without overwriting these retained receipts.
`run_experiment.py --phase validation --plan-commit f1aadac` precedes the timing
phase with the same arguments and `--phase timing`. Thread environment values
are forced to one by the driver. `audit_experiment.py --phase validation` and
`--phase timing` verify the completed phases against the committed plan, all
input/source/binary hashes, randomized pair order, original budgets, raw velocity
writeback and independently recomputed full gates. `compare_accepted_roots.py`
and `write_cost_report.py` reproduce the supplemental root comparison and table.

`results/timing-process-observations.jsonl` contains 30-second workload snapshots
and a final observation that the experiment process has ended. Load and live
process observations retain executable names and CPU/process metadata, with
command arguments omitted from the public ledger. They cannot prove isolation.
A brief supplementary fixture
compile and independent audit execution occurred during timing and are disclosed
as part of the shared workload. `manifest.json` freezes all published research
sources and receipts; native executables are excluded but their hashes are
retained in build provenance.
