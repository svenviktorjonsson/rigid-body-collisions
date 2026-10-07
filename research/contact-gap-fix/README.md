# Measured inertia and sustained-contact repair — 7 October 2026

Two concrete gaps are repaired: the production 3D adapter now accepts explicit
measured mass/COM/inertia; a separate public supported-contact primitive adds
static/dynamic friction and independent rolling/spin angular impulses. This is
partial progress toward the requested model, not full experimental qualification.

The [report](report/report.pdf) includes mechanics, synthetic motion traces,
the unchanged 24-case glass experimental comparison, and a scoped native cost
measurement. No restitution/friction coefficient was fitted or changed.

Code/evidence checkpoint: `1997226fda1d9e27e7000ab23691cc2930ccd0bf` (pushed).
Local downloadable PDF and verified 97-member ZIP are recorded in
[downloads.json](downloads.json), under `Physics Reports/2026-10-07/` in the
Vektor Flow project folder. The ZIP contains this repair and source-derived
comparisons; running it requires the full repository and its dependencies.

## Measured mass properties

`spatial_engine.prepare` accepts the body field below. All three keys are
mandatory; the inertia is about the stated COM, expressed in authored body axes.
It must be finite, symmetric, positive definite and satisfy inertia triangle and
geometry-extent bounds. The existing geometry defines collision shape. The input
body `position` retains its existing meaning as world COM.

```python
mass_properties = {
    'mass_kg': 1.0,
    'center_of_mass_m': [0.0, 0.0, 0.0],
    'inertia_body_kg_m2': [[.003, 0., 0.], [0., .004, 0.], [0., 0., .005]],
}
```

Use units appropriate to the actual geometry; the necessary extent bound does
not certify every tensor is realizable by its exact shape. Documented material
profiles check the authoritative mass against their source density and preserve
their declared homogeneous-sphere assumptions. Generic measured-body input does
not override specimen restrictions in a source-specific comparison.

## Supported-contact API

```python
from supported_contact import Resistance, advance_planar

# Synthetic control, not published material data.
result = advance_planar(
    mass_kg=1., inertia_kg_m2=.004, radius_m=.1,
    normal_load_N=9.81, drive_force_N=0.,
    velocity_m_s=1., omega_rad_s=10., duration_s=10.,
    material=Resistance(mu_s=.5, mu_d=.3, mu_r=.02,
                        rolling_length_m=.1),
)
```

This exact piecewise-constant branch handles one disk/sphere with scalar central
inertia on a plane, constant normal load/drive, collinear translation and rolling,
and optional independent axial spin. It advances to slip/rolling/spin arrest,
without reversal from an overlong step. It returns separate force impulse,
independent angular impulses, motion, work, losses and branch records.

`advance_spatial` rotates this branch into 3D and accounts for a translating
support. It preserves **t** as full contact-relative velocity and **s** as full
relative angular velocity. Directions are undefined at zero motion; returned
flags explicitly identify static constraint reactions. A partial angular arrest
that needs a transverse static torque outside the supplied **s/n** span raises
an explicit error. General noncollinear 3D contact, arbitrary inertia, normal
impacts and interacting contact groups are outside this primitive's scope.

The two independent angular capacities are an explicit phenomenological law,
not an established coupled finite-patch traction budget. Their physical moment
lengths are not the coordinate scaling length ell. No material values are inferred
from the synthetic controls. Normal/tangential restitution remain impact inputs;
this sustained-contact branch neither drops nor reapplies impact restitution.

The native engine's existing contact law does not automatically call this new
primitive. Production angular-friction integration remains open.

## Verification and reproduction

Run from the repository root with NumPy/SciPy/xlrd available, plus the existing
native backend build prerequisites. Evidence outputs must be new directories.

```bash
python -m unittest tests.test_supported_contact tests.test_measured_mass_properties tests.test_spatial_engine tests.test_two_channel_restitution tests.test_predictive_contact_review -v
python research/contact-gap-fix/audit.py --output /tmp/contact-audit-new
python research/contact-gap-fix/benchmark.py --controls /tmp/contact-audit-new/controls.txt --output /tmp/contact-benchmark-new
python research/contact-gap-fix/repeat_glass.py --output /tmp/contact-glass-new
python research/contact-gap-fix/build_report.py
```

The glass replay requires the previously fetched original worksheet in
`/home/viktor/.cache/physics-documented-materials-20261006/3mmglass-binary-source`.
Its source hash and URL are in the summary. The replay changes the destination
only. The report build uses archived audit/benchmark/glass JSON, Matplotlib and
pdflatex; it does not rerun expensive native experiments.

15 new tests and 24 existing regression tests pass. 400 randomized branch
controls check energy and step composition; 100 rotated/moving-plane controls
check frame and independent-couple accounting. Three default scene preparations
match the frozen pre-fix revision exactly. A retained native timestep refinement
passes the original free-rotation momentum tolerance at 256 steps; the 64-step
attempt did not pass. Mechanical controls do not establish material accuracy.

The C++17 SoA kernel matches 400 Python fixtures. Five timed repetitions after
warmup include 15 input loads, 11 output stores, branch and energy checks.
Allocation, collision detection, changing load, poses, impacts and group solving
are excluded. One million independent updates takes 81.4 ms median on this host;
this is not a many-body scene benchmark or a speedup against the old model.

The glass comparison still has normal RMSE 0.036674 m/s and translational tangent
RMSE 0.017805 m/s. Its spin target is reconstructed, coefficient/evaluation
independence is incomplete, and it cannot validate the new rolling/spin law.
Other public-data comparisons remain as documented in
[public-validation-data](../public-validation-data/README.md).

Remaining work: recover general zero/static-direction rules, couple angular
resistance to a physical contact patch and efficient local shear/mode memory,
obtain independently characterized parameters and sufficiently resolved impact
states, then perform held-out comparisons and general contact-group integration.
Maw--Barber--Fawcett's [elastic oblique-impact study](https://websites.umich.edu/~jbarber/Wear1976.pdf)
is a basis for the compliance branch, not validation of this constant-load repair.

The disk-full recovery moved only our previously downloaded rock archive to a
checksum-verified temporary cache; see `cache-relocation.json`. Its source URL
allows re-download if the temporary copy disappears. No measurement was deleted.
