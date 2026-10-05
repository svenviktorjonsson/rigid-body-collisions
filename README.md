# Rigid body collisions

A physics-engine research prototype with real 2D and 3D native collision backends
and reproducible speed/accuracy benchmarks.

The [Float64 3D backend](spatial_backend/README.md) supports arbitrary convex
hulls, boxes, spheres and compounds, full inertia tensors, quaternion rotation
and prescribed moving walls. Twelve mechanics tests include 100 m/s walls driving
64 bodies and 20 m/s containers with 64 spheres or 27 random rotating hulls.
The [3D evidence report](research/spatial-validation/report.pdf) retains
102 histories. Only the frictionless row qualifies its frozen trajectory gate;
all five dense frictional references remain unqualified. Passing containment
and no-tunneling tests does not establish their trajectory accuracy.

The [polygon engine](rigid_backend/README.md) supports rotating convex polygons,
compound concave bodies, persistent multiple contacts, dry friction, many-body
contact chains and continuous collision detection through two pinned Box2D
backends. Fast, standard, accurate and high numerical presets are available.
An experimental dynamic controller changes solver effort while preserving the
world and its contact caches.

The [executed study](research/rigid-study-report.md) tests 17 scenes and retains
all 153 state histories. Its conservative measured default is the coupled
normal block solver with eight collision updates and 32 velocity iterations per
1/120 s output frame. It met the declared RMS budgets on all ten initially
qualified scenes and the three additional scenes qualified in exploratory
refinement. The cheaper 4×16 preset misses the triangle-drop budget. The adaptive prototype was slower than the cheapest
passing fixed setting on all four qualified held-out scenes and missed the
rebound trajectory budget. It remains an experiment rather than the default.

These are numerical and analytic checks of idealized rigid mechanics. Public
sample provenance is preserved; friction/restitution values are not presented
as experimentally measured material properties. Seven initial reference cases
were unresolved; higher-work follow-up checks and uncertainty are reported
separately. No universal engine ranking or novel friction law is established.

## Run polygon scenes

Use Python 3.11+, CMake 3.22+ and a C/C++ compiler:

```sh
python -m pip install numpy scipy matplotlib cmake ninja
cmake -S rigid_backend -B build/rigid_block -DCMAKE_BUILD_TYPE=Release -DRIGID_BLOCK_BACKEND=ON
cmake --build build/rigid_block -j 4
cmake -S rigid_backend -B build/rigid_backend -DCMAKE_BUILD_TYPE=Release
cmake --build build/rigid_backend -j 4
```

Create a scene from the declared benchmark inputs and run it:

```sh
python -c 'import json; from research.rigid_scenes import scenes; print(json.dumps(next(s for s in scenes() if s["id"] == "concave_L_drop")))' > /tmp/rigid-scene.json
python rigid_engine.py /tmp/rigid-scene.json --output /tmp/rigid-result.json --preset accurate
python rigid_engine.py /tmp/rigid-scene.json --output /tmp/rigid-fast.json --preset fast
python rigid_engine.py /tmp/rigid-scene.json --output /tmp/rigid-adaptive.json --adaptive --policy research/rigid-benchmarks/results/frozen-policy.json
```

`high` is the default; presets describe numerical effort, not certified
physical accuracy. `--primary-steps` and `--substeps` override the preset.
For the block backend, the latter means velocity iterations; for the temporal
backend it means temporal substeps. See the backend guide for geometry, units,
collision skin, friction mixing and rolling restrictions.

The [seeded random-shape study](research/random-shapes/report.md) adds convex
hulls, concave triangle compounds and shaking boxes with 36 mixed shapes.
All 24 frozen physical-contact solves pass independently audited mechanics;
the initial study qualified four of eight full-trajectory references and retained
two native geometry rejections.
The [illustrated PDF](research/random-shapes/report.pdf) and archived inputs,
trajectories, timings and rejection reasons preserve these limits. Disk-row
speedups are not established for arbitrary random shapes.

The [follow-up fix validation](research/random-shape-resolution/report.md)
resolves both geometry rejections and qualifies both original concave drops.
Full Float64, exact convex partition merges, analytic wall motion and separate
position iterations are available as tested diagnostic controls. Of two new
concave seeds, one qualifies and one remains unresolved; both 36-body packed
boxes still fail. Sixty new histories and the unchanged accuracy gates are
independently audited. The [illustrated report](research/random-shape-resolution/report.pdf)
records these limits. `fidelity.select()` provides offline cost selection only
after reference qualification and returns no verified choice for failed cases.
The controller saves time against its continuous fine setting on three qualified
scenes, while cheaper passing fixed settings still exist. All validation is 2D.

## Reproduce and inspect evidence

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s research -p 'test_*.py' -v
python -m research.benchmark_tools audit
python -m research.audit_rigid_study
python -m research.run_rigid_study --repeats 5 --output /tmp/new-rigid-study
python -m research.audit_rigid_study --pack --directory /tmp/new-rigid-study
```

The [research package](research/README.md) also contains the coupled contact
mathematics, typeset assessment, opposing reviews, compliant contact-history
model, rod reduction experiments and public continuum benchmark catalog.
The [validation guide](research/VALIDATION.md) distinguishes numerical
verification, fitting and experimental material validation.

## Earlier disk demonstration

`python test_v3.py` runs smooth disks in a unit square, using `physics.py` for
event-driven frictionless collisions. Disk mass is density times area. Equal
and opposite isolated impulses preserve momentum and, with restitution one,
kinetic energy. Fixed walls exchange momentum with the disks. Gravity uses
symmetric half-step kicks; impact timing under gravity is approximate.

This older demo has no rotation, resting-contact friction or arbitrary shapes.
Simultaneous contacts are processed sequentially and can depend on contact order.
Initial overlaps are rejected, and exceeding the event limit raises an error.
`test_v1.py`, `test_v2.py` and `test.py` are historical experiments, rather than
automated tests. The polygon engine is a separate implementation.
