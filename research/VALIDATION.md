# Benchmark validation and authenticity

Use this repository as the home for the engine, its physical models, calibration
evidence and comparisons. The repository now contains a headless rotating-polygon engine, an earlier
disk demo and a local compliant contact integrator. The new
[rigid study](rigid-study-report.md) executes 17 planar scenes with source-locked
Box2D backends. The older IPC/GetFEM continuum catalog remains a catalog; those
cases have not been executed as independent continuum references here.

## What the evidence currently supports

The executed rigid study checks full position/velocity/spin histories at matched
physical sample times, independent refinement axes, analytic rebound/friction
and measured controller overhead. Its rigid outputs are numerically verified,
not experimentally authenticated. The two audit commands verify
35 public source files (6 new plus 29 old),
153 archived histories and recomputes all 119 rigid comparisons.

The local regression tests check impulse coupling, momentum/energy accounting,
passive sliders, calibrated isolated normal rebound and some numerical refinement.
The rod data concern a synthetic one-dimensional force-driven elastic problem.
They do not establish arbitrary-shape collision accuracy or material authenticity.

The catalog's `authenticity` record separates reported numerical inputs from
experimental validation. None of its ten cases is marked physically validated.
The foam impact example has reported properties and a video comparison on the
IPC project page, but quantitative observations and uncertainty have not been
established here. The remaining inputs are useful public numerical cases.

Source manifests pin upstream commits and SHA-256 hashes. Original files remain
unchanged and include upstream copyright/license notices. `sources/` contains IPC
assets; `getfem-sources/` contains GetFEM assets. These folders are not complete
installations of either reference solver. Build the corresponding pinned upstream
project to reproduce a case, preserving its script-defined motion and units.

IPC implements 3D dynamics. The planar gear case is static and frictionless. The
plate example has a 2D mid-surface and transverse bending. Neither can be silently
substituted for planar frictional impact. A 2D continuum adapter must explicitly
state plane stress/plane strain and thickness.

## Run the existing checks

From the repository root:

```sh
python -m pip install numpy scipy matplotlib
python -m unittest discover -s tests -v
python -m unittest discover -s research -p 'test_*.py' -v
python -m research.benchmark_tools audit
```

To reproduce the synthetic contact and rod data:

```sh
cd research
python run_compliant_demo.py
python benchmarks.py
```

## Engine adapter result contract

Export one JSON result for each case and fidelity setting. The tools compare
scalar quantities extracted by an adapter: velocity/spin components, integrated
impulses, contact duration, peak force, maximal deformation or energy residuals.
This scalar-output tool does not compare full histories or run a continuum
solver automatically. Separately, `research.run_rigid_study` now compares full
rigid histories and calibrates numerical effort thresholds, preserving fixed
physical coefficients. For long chaotic scenes export declared ensemble statistics
with appropriate uncertainty, rather than matching individual late trajectories.

Each result declares `schema_version: 1`, `case_id`, `physical_setup_id`,
`evidence_kind`, `wall_time_s`, `fidelity` and an `observables` map. An observable
contains a finite scalar `value` and its `unit`. Use SI and keep spin in rad/s.
`evidence_kind` is `numerical_simulation`, `experimental_measurement` or
`synthetic_fixture`. The report preserves this label.

Derive `physical_setup_id` from a canonical record of rest geometry, material and
surface-pair properties, initial state, boundary conditions and physical duration.
Keep numerical timestep/mesh/tolerance outside that identity. Include the declared
2D reduction and thickness. Preserve mass, COM and inertia across resolutions.
The comparison tool checks identity equality; an adapter must construct it honestly.

An error budget names the quantities being checked, their units, absolute
tolerances and relative tolerances. Allowed error is the absolute tolerance plus
the relative tolerance times the reference magnitude. Absolute tolerances prevent
division problems near zero. Velocity and spin have separate budgets. Passing
only establishes agreement for those declared outputs against that reference.

The included examples are hand-written **synthetic tool fixtures**, not engine
outputs, measurements or runtime benchmarks. Exercise the comparison with:

```sh
python -m research.benchmark_tools compare \
  --reference research/adaptive-benchmarks/examples/fine.json \
  --candidate research/adaptive-benchmarks/examples/fast.json \
  --budget research/adaptive-benchmarks/examples/budget.json
```

## Establish the detailed reference before calibration

Check at least three levels for each refinement axis independently: timestep,
mesh spacing, then nonlinear/contact solver tolerance. Record the other controls
and hold them fixed within each sequence. The tool checks the last two successive
differences against a declared reference error budget:

```sh
python -m research.benchmark_tools convergence \
  --axis dt_s \
  --runs research/adaptive-benchmarks/examples/coarse.json \
         research/adaptive-benchmarks/examples/medium.json \
         research/adaptive-benchmarks/examples/fine.json \
  --budget research/adaptive-benchmarks/examples/budget.json
```

This is a consistency check, not a proof of convergence. The mesh, timestep,
contact discretization and solver checks all matter. Keep reference uncertainty
below the fast model's allowed error, and check whether the reference model itself
matches experiments. Running for longer does not improve numerical resolution.

## Calibration and adaptive engine milestones

1. The planar arbitrary-shape rigid adapter is implemented. Add an independent
   continuum reference adapter with shared geometry, material and initial-state
   records for tests where deformation is physically relevant.
2. Characterize material/surface pairs and collect experimental observables with
   uncertainty. Preserve fixed physical properties across representations.
3. Fit shared reduced-model corrections on several calibration cases. Keep the
   existing passive energy/contact-history constraints. Avoid fitting a new free
   parameter set for every geometry or absorbing integrator error into material
   damping.
4. Freeze fitted parameters and evaluate unseen shapes, angles, speeds, spin,
   material contrast and simultaneous contact islands. Use separate validation
   data to choose switching thresholds, then keep a final untouched test set.
5. The first rigid-only comparison is complete and found no held-out adaptive
   speed advantage. Continue comparing fast-only, detailed-only and adaptive runs
   at matched physical-output
   accuracy, including adaptation/detection/state-transfer costs. Report repeated
   timings and false-safe switching decisions.

The full [protocol](adaptive-benchmarks/benchmark-plan.txt) also covers elastic
history transfer, conservation, long-run statistics and out-of-domain fallback.
