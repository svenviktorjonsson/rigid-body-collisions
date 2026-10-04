# Rigid body collisions

A physics-engine prototype and an experimental adaptive collision research package.
The executable engine currently simulates smooth circular disks in a unit square.
Its implementation is `test_v3.py`; `test_v1.py`, `test_v2.py`, and `test.py` are
historical experiments, not automated tests.

The [research package](research/README.md) contains the coupled planar contact
model, a compliant contact reference, the typeset assessment and opposing reviews,
coarse/fine rod experiments, and pinned public benchmark assets. The
[benchmark catalog](research/adaptive-benchmarks/benchmark-catalog.json) records
geometry, dimensionality, parameters, source revisions and authenticity status.
The [calibration protocol](research/adaptive-benchmarks/benchmark-plan.txt) defines
reference convergence, fitting, held-out tests and adaptive fidelity selection.

The intended approach is to calibrate reduced elastic/contact models against
converged detailed simulations and measurements, then choose fidelity by error
in physical outputs. Arbitrary-shape production simulation and adaptive switching
are not implemented yet. Downloaded numerical example parameters are not presented
as measured material properties, and the benchmark cases are not yet physically
validated.

## Run

Use Python 3.11 or newer:

```sh
python -m pip install numpy scipy matplotlib
python test_v3.py
```

With Poetry, use `poetry install --no-root` and `poetry run python test_v3.py`.
The demo uses elastic collisions and zero gravity. Set `e` between 0 and 1 for
inelastic collisions, and set `g` to a positive value for downward gravity.
Run the regression suite with `python -m unittest discover -s tests -v`.

Run the contact research tests and verify benchmark provenance:

```sh
python -m unittest discover -s research -p 'test_*.py' -v
python -m research.benchmark_tools audit
```

See [the validation tools guide](research/VALIDATION.md) to export engine results,
compare fast/reference outputs and check separate reference-refinement axes.

## Physics model

Disk mass is density times area: `m = density * pi * radius**2`. Between impacts,
disks move freely. `physics.advance_disks` finds the earliest disk or wall impact,
advances all disks to that time, applies an impulse, and recomputes collisions
for the remaining time. Multiple impacts can happen within a frame, including
impacts exactly at its endpoints. Walls are fixed straight boundaries.

For a contact normal `n` from disk 2 to disk 1, the impulse magnitude is
`J = -(1 + e) * dot(v1 - v2, n) / (1/m1 + 1/m2)`.
The velocities change by `J*n/m1` and `-J*n/m2`. This conserves momentum in each
disk collision and kinetic energy when `e = 1`. Fixed walls exchange momentum
with the disks; total disk momentum is therefore not conserved across wall hits.

Gravity uses symmetric half-step velocity kicks around the collision step.
Free flight under gravity is exact, but impact times under gravity and energy
across those impacts are approximate; decrease `dt` to improve accuracy.

This is a frictionless disk model: it has no angular velocity, friction, arbitrary
shapes, or solver for resting stacks. Simultaneous contacts are processed
sequentially, so their outcome can depend on contact order. Initial overlaps and
objects outside the box are rejected by `Simulation.add_object`. Extremely dense
or inelastic scenes can exceed the event limit; that raises an error instead of
silently advancing through unresolved collisions. Collision detection checks all
pairs, so it is intended for small scenes rather than thousands of disks.
