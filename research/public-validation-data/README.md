# Public experimental data extension — October 7, 2026

More real measurements are now available to challenge the contact model. Published
material values remain unchanged. These are branch checks, a rocking comparator
and state imports; the full directional impulse law is not newly validated.

| Dataset / comparison | Evaluation amount | Result |
|---|---:|---|
| GAUGE rubber bounce, conditional on observed impact marker | 42 events, 21 trials | Outgoing normal-speed RMSE **0.380733 m/s** |
| GAUGE wood sliding, fixed published friction | 19 trials, 167 forecast samples | Position RMSE **6.962 mm** |
| GAUGE plastic sliding, fixed published friction | 19 trials, 153 forecast samples | Position RMSE **10.824 mm** |
| GAUGE metal sliding, fixed published friction | 18 trials, 158 forecast samples | Position RMSE **2.880 mm** |
| Limestone rocking, geometry-only Housner comparator | 134 unique trial first events | Angular speed-ratio RMSE **0.039013** |
| MIT non-spherical planar impacts | 1,718 signed pre/post states | Imported; material-independent endpoint prediction not run |
| GAUGE tetrahedron / wedge / pyramid impacts | 160 trials, 4,686 body-pose samples | Imported; full spatial endpoint prediction not run |

Amounts are not interchangeable independent events. Sliding samples within a trial
and rocking cycles are correlated. GAUGE pilot ID1 is excluded per family/material.
The released sliding task has 59 trials, not an assumed 60. Rocking has 135 source
trials and one exact duplicate, retained and excluded from the primary metric.

## Fixed-input and failed-candidate results

GAUGE's source restitution is 0.576733. The corrected marker comparison predicts
`e_source * incoming_speed` without using outgoing flight coefficients to
reconstruct incoming speed at the conditioning timestamp. RMSE is 0.380733 m/s,
maximum error 0.671462 m/s, mean error -0.350346 m/s. A conservative one-frame
timing sensitivity is ±0.515592 m/s; 35/42 errors contain zero within that envelope.
This is not a statistical confidence interval or a forecast of impact time.
Finite contact, deformable ball/plank, coordinate origins and same-pair calibration
still need clarification. Do not refit restitution to the observed rebound ratio.

Read [MEASUREMENT-ERRATUM.md](MEASUREMENT-ERRATUM.md): original joint-arc results
remain in `evidence-v1` and `evidence-v2` as consistency diagnostics. The original
protocol's input-independence claim does not hold for that reconstructed time.

Sliding uses a four-sample state-conditioning prefix and forecasts only subsequent
samples before 0.30 m initial downhill displacement. Source friction values are
wood 0.273673, plastic 0.275018 and metal 0.262758. The source field does not separate
static/dynamic friction; using it in the moving branch is a declared interpretation.
Calibration/evaluation separation and exact material-pair provenance are incomplete.

Using the measured board normal instead of nominal 30° gives **worse results for
all 56 evaluation trials**: RMSE 9.010/12.709/4.573 mm for wood/plastic/metal.
Normals imply 29.56–29.67°. This input correction is retained, not presented as an
accuracy improvement. No friction coefficient changed.

## Sources and readiness

* [GAUGE dataset](https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset),
  [paper](https://arxiv.org/html/2608.05948v1), dataset MIT license. Revision
  `9e0acb70fecc0d4161660264d9a4b08d8f56d45a`. Released trajectories are **30 Hz**,
  although the paper describes 180 Hz capture. Published metadata supports branch
  checks; it does not supply the complete separate coefficients in our model.
* [MIT planar impacts](https://github.com/mcubelab/planar-impact-dataset),
  [Fazeli et al. paper](https://proceedings.mlr.press/v78/fazeli17a.html), revision
  `f24a7e3b31ad0b53652d6b2a6b26a702cc4362da`. Mass 36.4 g, semiaxes 35/25 mm,
  radius of gyration 19.2 mm, planar inertia 1.3418496e-5 kg m². Use the reported
  gyration radius; a uniform ellipse is a different assumption. Source model
  coefficients were fitted to these outcomes. No repository LICENSE found;
  original MAT and code remain outside Git, derived numerical observations are
  attributed here. Source integrity filtering reduced 2,000 drops to 1,718.
* [Colombo et al. limestone dataset](https://experiments.builtenvdata.eu/datasets/92/),
  DOI [10.60756/uminho-jh25](https://doi.org/10.60756/uminho-jh25), CC BY 4.0,
  June 17, 2026 release; [source paper](https://link.springer.com/article/10.1007/s10518-025-02224-8).
  Archive contains 135 trials; paper describes 120. Only 270 processed TXT files
  were extracted here. Raw 3D time histories exist in the archive but have not been
  used for full motion predictions. Nominal inertia is geometric. Processed
  angular ratios and effective geometry are outcomes. `fc=.7` is an assumed
  energy correction, not an independently measured friction coefficient.
* [Rémond et al. compliant silicone / ABS-shell experiment](https://data.hal.science/document/hal-05532284v1),
  DOI [10.1103/mmdr-2mm3](https://doi.org/10.1103/mmdr-2mm3). It reports spin transfer
  against bare glass and four silicone layer thicknesses. Figure 8 is useful for
  a future local-memory comparison. Its friction ~0.92 and local stiffness/mass
  are fitted from rebound/spin data; derived moduli are not independent inputs.
  Figures have not been digitized here and no comparison is claimed.

The rocking comparator predicts `1 - 1.5*B²/(B²+H²)` without a material fit. It
assumes ideal planar no-slip pivot transfer. **22/134** first source ratios exceed
one; all are retained, not converted into material coefficients. All processed
unique pairs contain 7,519 finite event estimates, including 2,820 above one.
Those correlated late-cycle estimates are not 7,519 independent validations.

MIT momentum reconstruction uses observed pre/post states and source contact
Jacobians. The apparent free-couple residual RMS is 0.00118023 N m s. Finite contact,
changing contact point, external/guide reactions, gravity over the measurement
interval and noise can contribute. It is **not independently measured torque**.
Nominal ellipse/contact-Jacobian geometry differs by up to 1.72 mm.

## Indices and model scope

`catalog-v1/spatial-bodies.csv` supplies one flat case index, one body index per
case, and explicit pose offsets/counts into `spatial-poses.npz`. The moving body
is index 0; a recorded support is index 1. MIT is one body (index 0) in each case.
Original v1 mistakenly named the case index `body_index`; v2 fixes that naming
without changing any measurements, metrics or physics. The equivalence receipt
and original source snapshot are preserved.

These observation arrays are suitable inputs to later Vektor Flow kernels. They
do not implement a language port or a group contact solve. Retain full relative
velocity direction t and full relative spin direction s, independent angular
impulse, the user's wedge/transpose convention and reference length ell. Zero-slip
closure and production rolling/twisting integration remain open.

GAUGE nonsmooth source folder labels `task-1/2/3` differ from metadata labels
`task-3/4/5`. The importer preserves the restitution dictionaries and does not
guess their correspondence. All 3D quaternions are finite with maximum norm error
8.98e-7; instantaneous angular velocities are not certified at 30 Hz.

## Reproduce

Use Python with NumPy/SciPy/Matplotlib and `unrar` capable of RAR6. Source originals
stay in an external cache; receipt hashes fix their identity. Every output
directory must be new, so retained evidence cannot be silently overwritten.

```bash
python research/public-validation-data/fetch.py --cache /tmp/physics-public-cache --output /tmp/public-downloads
python research/public-validation-data/fetch_rocking.py --cache /tmp/physics-public-cache --output /tmp/rocking-downloads
python research/public-validation-data/fetch_spatial.py --cache /tmp/physics-public-cache --inventory /tmp/public-downloads --output /tmp/spatial-downloads
python research/public-validation-data/study.py --cache /tmp/physics-public-cache --output /tmp/public-evidence
python research/public-validation-data/orientation.py --cache /tmp/physics-public-cache --baseline /tmp/public-evidence --output /tmp/orientation-evidence
python research/public-validation-data/marker.py --cache /tmp/physics-public-cache --baseline /tmp/public-evidence --output /tmp/marker-evidence
python research/public-validation-data/audit.py --evidence /tmp/public-evidence --output /tmp/public-audit
python research/public-validation-data/catalog.py --cache /tmp/physics-public-cache --output /tmp/public-catalog
python research/public-validation-data/report.py
```

`report.py` renders the retained repository evidence; requires `pdflatex` to build
the PDF. The optional silicone PDF is not required for numerical reproduction.
24 ideal bounce extraction controls recover the analytic response within
1.47e-14 m/s; nine gravity-projection controls pass. These validate extraction
under their ideal assumptions, not empirical model accuracy.

## What this changes

The project now has many more observations with signed states, non-spherical 3D
poses and published material metadata. No empirical accuracy improvement is
claimed from this checkpoint. The immediate useful next step is to resolve the
GAUGE scene/material mapping and reconstruct uncertainty-aware impact states;
then compare a small, passive local shear-history model with the current model
on fixed trial splits. Keep documented coefficients fixed and distinguish any
missing-property estimation from evaluation. The silicone data gives a concrete
deformation/spin challenge, rather than adding an arbitrary fitted correction.
