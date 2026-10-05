# Rigid-body research package

This package is maintained in `svenviktorjonsson/rigid-body-collisions` under
`research/`. Read [VALIDATION.md](VALIDATION.md) for runnable source audits,
fast/reference comparisons, refinement checks and the engine-adapter contract.
The [catalog](adaptive-benchmarks/benchmark-catalog.json) and
[protocol](adaptive-benchmarks/benchmark-plan.txt) record the public benchmark
assets and the planned calibration/held-out validation workflow.

The shared verdict is **do not submit the current equations as novel collision theory; continue a narrower adaptive heterogeneous reduction study only if it gains distinctness and held-out accuracy/cost evidence**.

Read `research-assessment.pdf` for the typeset mathematical formulation and evidence. `joint-verdict.md`, `critical-review.md`, and `publication-case.md` record the two opposing reviewers' agreed conclusions and checked references.

## Executed moving-polygon research

Read [the new 17-scene study](rigid-study-report.md) for measured solver choices,
reference qualification, failed adaptive speedups and preserved raw histories.
The [headless engine](../rigid_backend/README.md) supports polygon/compound rigid
bodies through pinned block and temporal backends. This extends the implemented
scope beyond the earlier local contact and rod experiments below.

## Implemented local contact scope

- `contact_solver.py`: planar contact multigraph assembly, full force/couple mobility and global energy, a zero-restitution frictionless normal projection, and a proposed single-pair energy-constrained closest-target comparator. The comparator is not exact Coulomb friction; static/dynamic capacity selection is a labeled heuristic.
- `compliant_contact.py`: a local, fixed-geometry finite-contact reference with unilateral normal compliance and tangential/rolling elastic elements in series with dissipative static/dynamic sliders, retaining sliding modes until explicit plastic-slip arrest events. Stored energy at opening is reported, not silently erased. It is not a polygon collision detector or a full moving-frame simulator.
- `benchmarks.py`: layered, force-driven elastic **1D rod** coarse/fine experiment and diagnostic counterexamples. It does not validate 2D frictional collision accuracy or superiority over optimized continuum/reduced models.

## Reproduce

Python with NumPy, SciPy and Matplotlib is sufficient:

```sh
python -m pip install numpy scipy matplotlib
python -m unittest test_contact_solver test_compliant_contact -v
python run_compliant_demo.py
python benchmarks.py
```

The benchmark regenerates `coarse-rod-results.csv`, `diagnostics.json` and the rod plots. The demo regenerates `compliant-results.json` and the compliant energy plot. Synthetic numerical coefficients are not measured material properties. Runtime ratios in the CSV come from one run and must not be reported as an established speed advantage.

To rebuild the PDF with a LaTeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error research-assessment.tex
pdflatex -interaction=nonstopmode -halt-on-error research-assessment.tex
```

## Not completed

A production-quality arbitrary-body engine, adaptive coarse-cell/interface
selection, calibrated material tables, an exact global frictional solver, 3D
implementation and experimental validation remain research work. Executable
planar polygon comparisons and held-out numerical adaptation tests now exist;
those tests do not establish an adaptive speed advantage. No general new model
or publication-worthy superiority has been established.

## Moving containers with many balls

The [executed moving-container study](moving-container/report.md) retains53
histories, exact packed-row checks, 100-ball frictional translation/shaking/rotation,
frame and ordering controls, and independent archive auditing. The current high
preset fails larger exact packed rows; none of the three dense trajectory
references qualifies under the predeclared refinement budgets. See the
[typeset mechanics](moving-container/contact-model.pdf) and
[report PDF](moving-container/report.pdf). All coefficients are synthetic.

## Sparse coupled-island performance

The [measured sparse-kernel report](sparse-islands/report.md) establishes an
81.4x cold-pipeline gain over the dense verification optimizer at 256 balls and
a 27.6x solve gain over the same algorithm with dense factorization at 1,024.
All 139 snapshots independently audit; 96/100 irregular Coulomb stress cases
pass and four are retained as rejected. This is frozen-contact performance,
not full-engine throughput. See the [PDF](sparse-islands/report.pdf) and
[typeset algorithm](sparse-islands/algorithm.pdf).

## Real 3D validation

[3D native backend](../spatial_backend/README.md), [frozen protocol](spatial-validation/plan.json),
[report](spatial-validation/report.md), [typeset PDF](spatial-validation/report.pdf)
and [rendered random shapes](spatial-validation/shapes3d.png). Full Float64 3D
mechanics and fast moving-wall tests pass; 102 archived histories independently
audit. One of six trajectory references qualifies; all five dense frictional
references fail the frozen gates. No general dense accuracy claim follows.

Run `python -m unittest tests.test_spatial_engine -v` after building the native
backend, and `python -m research.audit_spatial_study` to audit retained evidence.

[Exact 3D normal contacts and performance](spatial-normal/report.md),
[typeset report](spatial-normal/report.pdf), [mechanics](spatial-normal/mechanics.pdf),
[frozen protocol](spatial-normal/plan.json): six of six analytic cases pass,
36 histories audit, and exact preassembly elimination gives 1.46–7.22x native
speedup with identical states and nine times less scalar mobility storage.
Restriction: zero friction and restitution; dense frictional accuracy is still
unqualified. Run `python -m research.audit_spatial_normal` to inspect the evidence.
