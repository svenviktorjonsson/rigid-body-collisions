# Rigid-body research package

The shared verdict is **do not submit the current equations as novel collision theory; continue a narrower adaptive heterogeneous reduction study only if it gains distinctness and held-out accuracy/cost evidence**.

Read `research-assessment.pdf` for the typeset mathematical formulation and evidence. `joint-verdict.md`, `critical-review.md`, and `publication-case.md` record the two opposing reviewers' agreed conclusions and checked references.

## Implemented scope

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

An arbitrary-body production simulator, adaptive coarse-cell/interface selection, calibrated material tables, a full global frictional solver, 3D implementation, convergence/objectivity verification under changing contact geometry, and comparative held-out validation remain research work. No general new model or publication-worthy superiority has been established.
