# Supported-contact integration package — 7 October 2026

This checkpoint supplies a versioned C-compatible native batch boundary,
an indexed algorithm, import oracles and a separate experimental local-history
component. It is ready for a scoped compiler import, not certification that the
entire collision engine or compiler is complete. No compiler repositories were
modified. Read [the integration contract](../../supported_backend/INTEGRATION.md).
The [three-page report](report/report.pdf) gives the benchmark and import boundary.
The native fields and units also have a [machine-readable schema](../../supported_backend/schema.json).

The integration snapshot `df23b35088268485dfc67229927160d0ccd59a8f` is pushed.
[downloads.json](downloads.json) records the local PDF and checksum-verified
48-member ZIP in the project's `Physics Reports/2026-10-07/` folder. Extracting
that bundle and running all 25 included-component tests passes from its own
directory, including a fresh native build. The bundle contains a standalone
supported core; whole-engine regression runs still require the full repository.

## Performance result and algorithmic changes

The final paired benchmark (`run-v3`) processes one million independent responses
in **12.830 ms median with eight workers**, range **12.417–14.606 ms** across seven
repetitions. The 20 ms target is met in all seven samples for that configuration.
Four workers miss the median target (22.646 ms); the initial four-worker prototype
was faster and is not promoted over the controlled result.

The single-worker candidate takes 50.975 ms versus a matched-layout frozen
reference's 46.463 ms. There is **no demonstrated single-worker algorithmic speedup**.
The earlier 81.4 ms cost used a different benchmark/storage harness; the paired
comparison here is the fair current speed comparison. Timings include field loads,
all eleven response stores, finite/energy/branch gates and worker entry. Preparation,
validation, allocation, detection, impacts, changing frames/loads, group solves and
contact-history updates are excluded. Repeating 400 independent fixtures is not
an interacting million-body simulation. No 2x all-scene gate is reinstated.

Algorithmic changes are portable to later BKF lowering: use rim speed `q=R*omega`
and force-equivalent couple `b=M/R`, put both mobility equations in linear units,
precompute reciprocals/capacities, and skip active-set enumeration when both
nonzero directions are known. This removes a real dimensional-tolerance bug:
frictionless drive was rejected for small radii. The corrected Python/native paths
work over radii 1e-12–1e6 m. Exact axial arrest prevents a roundoff reversal;
finite-response rejection includes impulses that overflow while doing zero work.

Native finite validation checks the energy residual plus linear/rolling integrals:
under validated positive mass/inertia and non-fast-math IEEE arithmetic, a nonfinite
velocity, distance/work or loss propagates to the residual; a static impulse can
overflow without work and requires its own check. Returned finite-channel and
overflow controls remain part of acceptance. Do not remove these gates in lowering.

## Correctness and local state

54 focused tests pass (including a C11 header probe). Native responses are exactly
equal across worker counts. 400 updated Python controls differ by at most 1.04e-15
on physical output scales; another 2,000 cases spanning mass 1e-6–1e6 kg and radius
1e-9–1e6 m pass at 1.02e-15 maximum scaled difference. Energy and step-composition
audits and 100 rotated supported controls remain passing. Prior benchmark runs
and the first failing C-probe test are retained. That test passed a noncontiguous
NumPy view to a field-major native interface; correcting the probe's layout did
not change the native algorithm or acceptance criterion.

`contact_history.ContactHistory` adds one to three passive spring/slider modes
without body deformation meshes. Implicit midpoint elastic storage is coupled
through rigid-body mobility; a static-capacity test precedes a dynamic-capacity
convex return map. Small factorizations are cached; at most 27 active sets are
examined. Opening applies no impulse and reports released stored energy into a
separate internal-mode ledger instead of silently deleting it. The caller must
retain/relax that energy physically; it is not measured heat or automatically
returned kinetic energy. 300 randomized coupled-mode controls check passivity
and permutation invariance; elastic histories conserve body-plus-stored energy.

The memory component is experimental and does not run inside the native batch.
Declared mode directions, diagonal stiffness, fixed timestep and independent
mode capacities are assumptions. It does not complete evolving full t/s history,
an authentic coupled patch traction budget, joint normal impact or group solving.
The user's full t/s definitions and independent angular impulse remain binding.
Partial spatial arrest outside the allowed s/n span is still explicitly unsupported.
Published normal/tangential restitution and friction values remain unchanged;
no fit or new experimental accuracy gain is claimed.

Tangential compliance/partial slip are established physics, as in
[Maw, Barber and Fawcett (1976)](https://websites.umich.edu/~jbarber/Wear1976.pdf).
[LAMMPS' documented granular models](https://doc.lammps.org/pair_granular.html)
illustrate history and unloading treatment. These motivate local state; they do
not validate this reduced model or independently supply its stiffness/capacities.

## Run and integrate

```python
from supported_batch import PreparedBatch, INPUT_FIELDS, OUTPUT_FIELDS
batch = PreparedBatch(inputs)  # Float64 shape (15, count); owned validated copy
outputs = batch.run(threads=8, out=preallocated_outputs)  # shape (11, count)
```

Use an explicit worker count suitable for the host; eight is the measured target
configuration, not a universal fastest choice. Each body index is independent.
Do not use this loop to scatter simultaneous contacts to shared body storage.

```bash
python -m unittest tests.test_supported_contact tests.test_supported_batch tests.test_contact_history tests.test_measured_mass_properties tests.test_spatial_engine tests.test_two_channel_restitution tests.test_predictive_contact_review -v
python research/contact-gap-fix/audit.py --output /tmp/contact-audit-fresh
python research/supported-batch-optimization/benchmark.py --controls /tmp/contact-audit-fresh/controls.txt --output /tmp/contact-benchmark-fresh
python research/supported-batch-optimization/verify_scale.py --controls research/supported-batch-optimization/scale-audit-v1/controls.txt --output /tmp/contact-scale-fresh
```

Evidence directories are immutable and require new paths. `run-v1/v2/v3` preserve
different source revisions/receipts. `scale-audit-v1` provides importer inputs and
Python expected outputs. Verification logs/source hashes and the downloadable
integration package/report are recorded alongside them. Compiler acceptance,
native/WASM/GPU equivalence and actual full-model empirical validation need their
own gates; this result covers only the stated Linux native supported branch.
