# Indexed local pressure-patch candidate

Rigid bodies with a small compliant contact patch, no deformable body mesh.
`model.py` evaluates unilateral local normal springs/dashpots and the existing
dynamic sliding coefficient at each pressure site. Integrating traction supplies
both resultant force and an independent contact couple. It includes combined
slip/twist, nonuniform pressure and normal-deformation rolling resistance.

This is a **frozen-frame instantaneous research candidate**, not an adopted
material law, a complete collision integrator or an experimental validation.
Static-friction/shear memory, evolving footprint/frame, brittle yield, material
calibration and coupled contact dynamics remain open. Zero-slip sites do not
resolve static friction. It supplements rather than replaces earlier failures.

The user's full relative velocity direction t and full angular velocity direction
s remain unchanged. `directional_residual` measures any wrench outside n/t and
s/n; no conventional patch resultant is silently declared equivalent to that law.

Foundation stiffness/damping and pressure geometry are independent inputs, never
new undocumented rubber/rock material constants. Existing restitution and static,
dynamic and rolling coefficients remain separate; this branch does not enforce
endpoint restitution on top of compliant energy exchange.

Primary precedent: Elandt et al., IROS 2019,
https://arxiv.org/abs/1904.11433. This simpler flat foundation is **not** an
implementation of their pressure-field intersection geometry or Drake.

The indexing contract is based on Vektor Flow's authoritative Section 0 semantics:
explicit contraction/reduction, one body owner index, flat contact/site/incidence
ranges, deterministic body accumulation. No language-repository changes or
compiler/GPU support claims are made by this external research.

## Verified results

* `audit-v2`: 240 planar/spatial energy and indexed momentum controls (50 fully
  loaded, 190 opening/clipped); scaled instantaneous power error 1.92e-15.
  Pure-spin analytic Hertz moment, normal pressure rolling moment, cached Gram
  equivalence and stored-energy derivative pass. Normal transient energy error
  1.34e-9 J from 0.5 J initial energy.
* `reference-point-v1`: 100 contact-origin and 300 coordinate-length controls.
  Center-of-normal-pressure choice removes the transverse contact couple but
  leaves a mixed force-direction discrepancy. Total body torque/work/energy is
  invariant. This qualifies the fixed-origin directional residual interpretation.
* `benchmark-v1`: 44 synthetic indexed CPU batches, 100 through 1,000,000
  responses. Every measured compact kernel improves, 1.41–27.93x. All same-site
  wrenches match; 48 native/Python controls pass. Baseline-first timing-order bias
  and inaccurate memory estimate are retained, not silently replaced.
* `benchmark-v2`: 22 alternating-order confirmation batches at 10,000/100,000;
  every case improves, 1.42–28.94x. Same 48 controls pass; numeric array-size
  calculation is corrected. Identity frames, allocations/detection/history/time
  integration excluded; ordered gather/scatter/output resets included.
* `refinement-v2`: 18 mixed synthetic states against 147,456 sites. 12 bounded
  estimates admit, six reach the budget without enough consecutive convergence
  evidence. All returned values happen to be within 1e-4 scaled error, but declines
  stay declines and there is no general quadrature certificate. Some branches
  need 65,536 sites, too costly for a cheap general contact.

`report.pdf` contains the six-page checkpoint with material/state policy,
performance tables, error dot plots and unchanged experimental status.
No new material fit, experimental accuracy gain, full engine qualification or
language compiler integration is claimed. The user's removed 2x gate stays removed.

## Reproduce

Use Python with NumPy/SciPy/Matplotlib, C++17 g++, and pdflatex. Every evidence
directory is created exclusively; choose new names instead of overwriting results.

```bash
python research/indexed-pressure-patch/audit.py --output NEW-AUDIT
python research/indexed-pressure-patch/reference_point_audit.py --output NEW-ORIGIN
python research/indexed-pressure-patch/refinement.py --output NEW-REFINEMENT
python research/indexed-pressure-patch/benchmark.py --output NEW-BENCHMARK
python research/indexed-pressure-patch/benchmark.py --output NEW-CONFIRMATION --confirmation
```

`render.py` reads the authoritative committed directories. From this study folder,
run it, then `pdflatex -halt-on-error -output-directory=build report.tex` twice.
Create the build directory first. The optional booktabs dependency was removed;
initial missing-package and row-escape report build failures are retained under
render-build-v1/v2. Numerical v1 site-count label mistakes are corrected in v2;
`metadata-corrections.json` verifies unchanged numerical evidence and pins original
source snapshots. Neither report-build correction changes the mechanics.
