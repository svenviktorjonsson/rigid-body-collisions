# Efficient contact deformation search — 7 October 2026

The bodies remain rigid. This checkpoint tests small contact reductions, preserves
all rejected results, and adds a seven-page downloadable update and signed-error
scatter. It does not replace the native solver or qualify all collision examples.

## Results

* **Rolling:** one nonnegative effective relaxation time per tennis specimen versus
  one constant per specimen. Evaluation RMSE decreases by 51.53%, 67.46% and 33.71%.
  All eight evaluation-point absolute errors and every specimen's maximum error
  decrease. The 17 points and existing 9/8 split are reused after inspecting the
  data and a preview: exploratory evidence, not fresh blind validation. Apparent
  speed/slight skid, shell construction and missing matched material measurements
  prevent identifying a universal bulk relaxation time.
* **Rock normal response:** the one-parameter viscoelastic homogeneous-sphere
  reduction is worse on both the 25-point height split (RMSE 1.00437 to 1.13969 m/s)
  and the 75 pooled angle-fold records (1.14028 to 1.19421 m/s). Rejected. Actual
  facet attitude, contact geometry and inertia are missing. Fixed secondary inputs
  are historical project estimates, not newly documented material coefficients.
* **Independent spin moment:** circular Hertz pressure plus existing dynamic
  sliding friction yields torque magnitude `3*pi/16 * mu_d * N * a`. Net tangential
  force cancels for pure axial spin; the independent moment does not. No new
  spinning-friction parameter is fitted. This is a restricted standard traction
  integral, not an empirical rubber-bounce improvement or novelty claim.
* **Rock geometry:** the primary experiment documents concrete indentation/rim
  damage. Four illustrative local-normal controls show that an apparent restitution
  above one against the mean slab normal can remain energy-passive. Published
  crater photographs cannot be matched to individual collision rows.

## Authoritative evidence and limits

`evidence-v1` contains every data row, split/fold, fit, sensitivity and cost result,
source fingerprints, source snapshot and the normal-response table. `audit-v2`
passes 100 planar/100 rotated spatial rolling controls at three representation
lengths, 20 continuous-ODE comparisons, 25 tighter normal references and data audits.
`audit-v1` retains an audit-script variable-shadowing failure; it is not a physics
failure. The table has 513 nodes on beta 0..8 and rejects extrapolation. Sixty-four
tighter random reference solves find maximum restitution error 1.34e-7. Residual
elastic energy at repulsive force-zero separation is retained explicitly; recovery
into later impacts is not implemented.

`benchmark-v2` compiles warning-free with `-O3 -Wall -Wextra -Werror`, no fast-math,
and passes 16 native/Python response controls. Local scalar kernels for batches of
100, 10,000 and 1,000,000 responses are measured, including resultant impulses and
rigid energy changes. They are **not interacting body simulations**. Uncached
rolling is about 13 ns/response and size/speed-plus-normal-table evaluation about
31 ns/response on this host. Cached factors require unchanged geometry, load and
timestep. Detection, frame transport, contact networks and branch checks are not
timed. `benchmark-v1` retains the earlier harmless compiler-indentation warning
and its timings. No per-case 2x gate is reinstated.

`patch-spin-controls.json` passes 24 spatial distributed-traction controls for
net force/moment/work and energy, plus zero spin/friction and arrest. This is pure
axial full-sliding contact with a circular Hertz pressure distribution and constant
load/radius. It is not mixed sliding/rolling, torsional elastic microslip or
transient-pressure validation.

At zero relative contact velocity the user's full-velocity direction t is
undefined. The rolling prototype obtains the static reaction from the no-slip
constraint; it does not redefine t or claim complete directional-model closure.
Planar rolling controls restrict a spatial sphere law; they do not derive a
plane-strain constitutive law for arbitrary 2D objects. Existing documented
restitution/static/dynamic-friction inputs and production source remain unchanged.

## Next model comparison

Compare a small shear/mode history and pressure-patch reduction for rubber; compare
local plastic indentation plus rim/pressure geometry for rocks. Pressure-field
contact (Elandt et al.; Drake) and dimensionality-reduced plastic contact
(Zunker–Kamrin) are primary literature comparators, not adopted or validated models
in this repository. Default pressure moduli, unlike-material coefficients and
nominal concrete strength are not substitutes for independently measured contact
properties. Rapid groups require a coupled contact solve; isolated endpoint maps
must not be applied as independent whole collisions in overlapping contact groups.

## Reproduction

Use the repository Python environment with NumPy/SciPy/Matplotlib and a C++17
compiler. New output directories must not already exist.

```bash
python research/viscoelastic-relaxation/experiment.py --output NEW-EVIDENCE
python research/viscoelastic-relaxation/audit.py --evidence NEW-EVIDENCE --output NEW-AUDIT
python research/viscoelastic-relaxation/benchmark.py --evidence NEW-EVIDENCE --output NEW-BENCHMARK
python research/viscoelastic-relaxation/patch_spin.py
python research/viscoelastic-relaxation/indentation_demo.py
```

`render.py` renders the authoritative committed evidence and `build.sh` compiles
the update PDF. Third-party PDFs are cached outside Git; source URLs/fingerprints
and experimental provenance are recorded in JSON and the report.
