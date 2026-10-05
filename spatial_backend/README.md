# Native 3D validation backend

Build the unmodified, hash-pinned Bullet 3.25 CPU backend in Float64:

```sh
cmake -S spatial_backend -B build/spatial -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/spatial --target spatial_runner -j 2
python -m unittest tests.test_spatial_engine -v
```

`spatial_engine.run(scene)` supports spheres, boxes, arbitrary convex 3D hulls
and compounds (use authored convex decomposition for concavity). Density is
volumetric kg/m³. Mass, COM and full body inertia are integrated from geometry;
compound parallel-axis terms are included. A principal-frame transform supplies
Bullet's diagonal inertia without discarding off-diagonal terms. Poses use XYZW
unit quaternions. Output states are COM x/y/z, quaternion x/y/z/w, world vx/vy/vz,
world omega x/y/z. Gravity defaults to -z.

Body position denotes the COM. Authored fixture coordinates are shifted to their
aggregate COM. Output orientation uses the original body axes. Static and
prescribed kinematic bodies have infinite mobility mass (reported mass zero).
Velocity schedules include translation and angular velocity and split steps at
command times. Bodies are integrated independently of the container.

The travel guard accounts for both bodies' translation, angular tip speed and
gravity, and limits updates to 15% of the smallest fixture half-width/radius by
default. It prevents the tested fast-wall tunneling example; it is a conservative
timestep heuristic, **not an exact swept CCD proof for every concave feature**.
When the scene specifies `container_interior_half_extents_m`, a native monitor
checks every fixture support against every container plane at **each internal
update**, retaining maximum surface excess. The offline audit independently
reconstructs sampled surface supports.

Setting `travel_fraction=0` is an explicit negative-control diagnostic.

`solver='sequential'` selects projected sequential impulses; `'coupled'` selects
Bullet's Dantzig MLCP solver. The adapter configures Dantzig's impulse sanity bound to 1e30 N·s rather than
its default 1000, which is too small for some 100 m/s rows. Bullet may fall back to sequential iterations when
an MLCP fails; `coupled_fallbacks` exposes those events. No run with fallbacks may
be described as a pure direct coupled solve. Worlds/contact caches persist across
updates. `solver='adaptive'` uses 8 sequential iterations until at least 12
positive-impulse contacts or a closing residual above .01 m/s triggers 64
iterations of MLCP (or the requested iteration count), with 24-update dwell.
This is an observable heuristic, not a certified online error estimator.
No invented compliance or material changes distinguish the two modes.

Friction uses two independent bounded tangent directions, a pyramid approximation
to the isotropic Coulomb cone. Body friction and restitution coefficients multiply
at a contact. For a desired pair coefficient mu between identical bodies set each
body coefficient to sqrt(mu); a wall coefficient one retains the object's value.
One friction coefficient supports sticking and sliding; separate static/dynamic,
rolling/twisting and elastic tangential history are **not implemented here**.
Nonzero rolling/twisting fields, explicit mass/inertia overrides and per-fixture
material fields are rejected so authored physics cannot be silently ignored.
Restitution is a normal velocity rule with zero velocity threshold. Prescribed-wall
work is summed from normal and tangential impulses, including positive-gap predictive contacts,
at wall point velocities;
split position corrections do not count as physical impulses.

The full engine is 3D; analytic regression tests and held-out scenes must establish
each claim. Synthetic values in `research/spatial_scenes.py` are not calibrated
physical materials. This backend is public Python/C++ research, not a Vektor port.

## Exact normal-contact profile (3D)

`solver='normal_coupled'` is restricted to **zero friction and zero restitution**.
It retains native discovery, full 3D lever arms/inertia, integration and all
multiple-contact coupling. It solves the normal nonnegative quadratic program
with Cholesky on positive definite active faces and a bound active set. Every
accepted result checks unilateral feasibility and complementarity; singular or
failed faces fall back to disclosed upstream Dantzig/sequential handling. No
regularization or compliance is added. Pressure may be nonunique while velocity
is unique. Four native analytic QP checks cover inactive, redundant, coupled and
separating constraints; full 3D packed-box regressions check actual trajectories.

The normal profile eliminates exactly fixed-zero tangent variables **before**
assembling mobility. `preassembly_elimination=False` retains them through assembly
and removes them afterwards, using the identical normal algorithm. This provides
a fair numerical/timing ablation. Both modes preserve every physical coupling
that acts on a nonzero impulse. With two tangent rows per point, mobility has c
rather than 3c rows, giving nine times less scalar matrix storage, not nine times
less total process memory. This restriction must not be applied to frictional
contacts. Existing frictional modes retain both tangent directions.

`kinematic_contact_phase='start'` computes contacts at the current wall pose with
its explicitly commanded velocity and advances the wall after dynamic integration.
The original `'end'` option (default) advances walls before contact discovery and
retains the frozen frictional study's numerical convention. Neither moves contents
by assignment. The start option avoids injecting wall-travel penetration into an
already touching row. It overrides Bullet's automatic inference of kinematic
velocities; rotation and translation still follow the same prescribed schedule.

`position_stabilization='velocity_only'` disables split projection and ERP. It is
useful for the exactly touching, frictionless analytic fixtures: no invented
position-correction impulse enters their velocity solution. It does not repair
preexisting macroscopic overlap. Default `'split'` retains Bullet position
projection. Geometric violations must be monitored independently in either mode.
The analytic packed fixtures disclose initial overlaps around 1e-10m to make
floating-point touching-contact discovery reliable, and require them to stay
below 1e-8m. They are not measured material experiments.

## Circular 3D friction and residual-driven work

`solver='coulomb'`, `kinematic_contact_phase='start'`, zero restitution and
`iterations=4096` enable the project circular Coulomb lane. Both tangent
coordinates share the disk radius mu times the normal impulse. Normal
complementarity is enforced separately; this is **not** an associated cone QP
that adds artificial normal dilation. Off-centre and inter-contact couplings
remain in the full mobility matrix. Sparse column updates propagate impulses;
block Gauss-Seidel uses exact two-variable disk solves. Semismooth Newton with a
merit line search accelerates smaller stalled islands (at most 512 rows), without
adding diagonal compliance. Matrix assembly remains dense/quadratic.

The solver checks a velocity-scaled projection residual every eight sweeps and
stops as soon as `contact_tolerance_m_s` (default 1e-8 m/s) is satisfied. Eight
sweeps are therefore the cheap path; the iteration argument is a maximum budget,
not fixed work on every island. A contact-energy upper bound checks passivity.
Separate normal-only position projection first tries the active-set QP, then
residual-gated iterations of the same normal equations. It applies no tangential
position impulse. Failed gates reject the run with a reason; **this lane never
falls back to Bullet's friction pyramid**. Convergence is not universal.

`contact_slop_m=1e-9` treats gaps within one nanometre as touching in this lane,
removing inconsistent gap/time targets at almost redundant face points. This is
an explicit geometry tolerance, not physical compliance. It is limited to 1e-5
of the minimum feature; setting zero disables it. Position correction ignores
penetrations within the same tolerance. Earlier modes and archived evidence are
unchanged. Prescribed velocity reversals now reach contact discovery immediately.

Output includes Coulomb solve counts, islands accepted within eight sweeps,
maximum sweeps, Newton steps, projection residual and passive-energy bound.
The velocity residual certifies a frozen contact solve; containment, boundary
work and whole-trajectory refinement are independent gates. `spatial_fidelity`
provides an **offline** refinement ladder and refuses to recommend a candidate
when the three finest consecutive levels fail quarter-budget comparisons. The
travel guard changes timestep with current motion; this is not a certified online
local-error controller or a cache-preserving timestep rollback.

This lane uses one coefficient for sticking and sliding, no contact elasticity,
normal restitution, rolling or twisting couple. Independent elastic torque and
stored tangential energy are studied separately in `research/elastic-patch`.
Body coefficients are multiplied and upstream Bullet clamps the pair coefficient
at 10. Parameters remain synthetic until measured material data support them.
