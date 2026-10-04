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
