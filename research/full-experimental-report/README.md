# Experimental report and contact-memory prototype — 7 October 2026

Open [report.pdf](report.pdf) for the complete 19-page report, or [report.html](report.html) for a standalone interactive deviation map. It covers every currently recovered comparison record: **24 glass + 8 ball/surface + 75 rock collision records, and 17 rolling measurements**. These are not 124 independently characterized full-state events. The model does **not** yet reproduce all real-life datasets.

The authoritative recalculation is **evidence-v3/**. v1/v2 remain preliminary checkpoints, not alternate validation results. All original source files remain in the external cache; source-refresh-20261007.json records fresh successful downloads, matching worksheet/PDF/XLSX fingerprints, and the public URLs. Third-party PDFs are not redistributed.

Documented coefficients remain fixed. Missing coefficients may now be estimated with small, explicitly labeled fits, per the latest user instruction. Static friction is not invented from a sliding coefficient. Negative rolling friction is rejected; signed pressure moments are a separate hypothesis. Same-point exact moment inference is calibration, not predictive validation.

Results:

- Glass zero-free-couple comparator: normal RMSE **0.036674 m/s**, COM tangent RMSE **0.017805 m/s**. Source spin is reconstructed, not independently measured. Published coefficients may share characterization trials with the worksheet.
- Ball shared signed-moment hypothesis: conditional leave-one-surface-out spin RMSE **1.047965 → 1.242542 rad/m**, worst error **1.857373 → 2.232542 rad/m**. Rejected. Target-supplied tangential restitution itself contains measured spin; this is an endpoint-consistency test, not blind independent material/outcome prediction.
- Rock historical sphere-proxy comparator: 25 held-out impacts have RMSE **1.004366 m/s normal**, **1.304693 m/s tangent**, **12.876456 rad/s spin**. Actual collision shape, inertia and signed angular vectors remain unavailable; it is not validation of the requested full-angular rock model.
- Tennis quasirolling: one constant coefficient per specimen estimated on 9 points, tested on 8 other points. Positive estimates **0.016438, 0.017675, 0.231073**; all have larger held-out-point residual than the authors' fixed speed curves. Those author curves were fitted to the same published measurements and are reproduction comparators, not independent held-out models. Raw circles, axis mapping, duplicate paths and digitization sensitivity are recorded.

The deviation diagram uses outgoing observable magnitude signatures. Its angle is not heading. Signature channels differ across glass/ball cases, so no pooled accuracy score is computed. Rocks are excluded from this fresh diagram. Material parameters remain separate from symbolic equations.

## Rigid bodies with local deformation memory

`rolling.py` implements only an isolated sustained no-slip rolling branch: independent torque, coupled static reaction, angular arrest and insufficient-capacity rejection. It passes 100 planar + 100 arbitrarily rotated spatial controls, at three coordinate lengths each. This is not a native impact implementation.

`contact_memory.py` implements a small implicit-midpoint elastic-contact matrix with stiffness, damping and accumulated local deformation. Rigid-body motion and inertia remain unchanged; impulse components include the independent angular impulse. The factor is cached for a frozen contact, step and coefficients. It passes 100 planar + 100 spatial direct body-energy controls, including 20 dependent-direction cases, representation-length invariance, and convergence to an independent exact oscillator. `contact-memory-v2/` is authoritative; v1 records a scale-tolerance rejection.

This new core is **a bilateral/elastic sticking branch only**. Unilateral opening, plastic flow, Coulomb transitions, moving direction/history transport and independently characterized material calibration are not implemented. No empirical accuracy improvement, adoption, full native trajectory or all-case performance claim is made. Median local Python cached step: about 45 microseconds, including validation/overhead; not an engine timing claim.

The report recommends contact-local compression, shear and asymmetric-pressure/rocking history to encapsulate deformation without body meshes. A strain/pressure-based bounded correction is preferable to unnormalized powers of accumulated impulse. Hertz/Mindlin and elastic/plastic normal contact are established starting points. Keep documented friction fixed and test additional deformation resistance on held-out conditions. Avoid imposing endpoint restitution a second time on an already resolved compliant contact, and avoid double-counting emergent rolling loss.

Cross 2014 provides force/spin and attached-ball vibration data. Granite bounce, G10 force-plate, and attached-boundary tests have different conditions; do not mix their friction or modal stiffness as if directly interchangeable. The new literature recommendation is not an implemented empirical-fit result.

## Reproduction

In the repository with the verified original source cache and NumPy/SciPy/Matplotlib/PyMuPDF installed:

```bash
python research/full-experimental-report/calculate.py --output research/full-experimental-report/evidence-new
python research/full-experimental-report/audit.py research/full-experimental-report/evidence-new
python research/full-experimental-report/render.py research/full-experimental-report/evidence-new
bash research/full-experimental-report/build.sh
python research/full-experimental-report/audit_contact_memory.py --output research/full-experimental-report/contact-memory-new
```

Use fresh output directories; existing evidence is never overwritten. Calculation snapshots and SHA-256s are preserved. `audit.py` checks direct momentum/energy, full-velocity t reconstruction, direct fit/test separation and all row counts. The independent material/outcome qualifications remain false. Production adapters and all 13 rapid/irregular baseline qualifications are unchanged; the 2x gate remains removed.
