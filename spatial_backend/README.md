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

For a rejected circular-contact solve, `spatial_engine.run(...,
rejected_contact_path=Path(...))` can save the exact assembled matrix, free-velocity
RHS, row dependencies, bounds, final rejected iterate, velocity residual,
internal timestep and iteration budget as JSON. This diagnostic is opt-in,
requires a new destination in an existing directory, and keeps the original
exception. It never applies the rejected iterate or substitutes another friction
law. A snapshot describes one failed contact solve, not a completed trajectory.

Circular-contact recovery keeps the same isotropic law and residual/passivity
gates. With recovery enabled and an iteration budget of at least 64, the first
block-iteration phase uses at most 256 sweeps. Normal-only recovery first tries
bounded descent along numerical pressure-null directions, then a pressure-face search
and a warm active-contact search run next, followed by full continuation and the
earlier minimum-norm polisher. If none passes, the solver spends the remainder
of its original block-iteration budget. `contact_recovery=False` disables these
searches. The physical mobility and material are never regularized or replaced.
All 20 retained captured systems pass when the native LAPACK recovery option is
enabled. Passing a capture does not establish trajectory accuracy.

The pressure-null search also supports translation-only position repair. It
permits at most 384 normal rows and 128 active-face states/SVD calls per attempt.
It moves pressure to a nonnegative boundary along the numerical null projection
of the current gradient, or takes a minimum-norm range step. The original
mobility, bounds, absolute residual and finite passivity gates remain unchanged.
Counters record actual SVD calls, face moves and maximum trial velocity change.
Recovery requires a budget of at least 64 and honors `contact_recovery=False`.
Nine native controls include singular redundancy, infeasibility, unchanged output
on failure, and explicit enable/disable behavior. Frozen replay evidence is in
`research/normal-null-integration`; it retains all 20 successes and failures.

The additional difficult-friction fallback runs only after all earlier stages
and the remaining original sweep budget reject, and initially supports at most
64 rows. Warm/cold Fischer–Burmeister searches are followed, when needed, by a
numerical continuation guide, mobility-null pressure relocation and Moré trust
search using LAPACK DGESDD. No archived answer is used as an initial guess. Only
the original circular law, normal bounds and finite passivity gate can accept.
Each call has fresh counters and at most 1024 nonlinear/projector SVD calls and
2048 outer search steps. Pressure initialization has a separate 128-SVD limit
and at most one upstream pivot-guide call, whose internal pivot count is not
exposed. Engine counters disclose actual work. These limits are additional to
the earlier solver stages and do not certify a wall-clock deadline.

`SPATIAL_LAPACK_RECOVERY=ON` is the default native CMake setting and requires
32-bit-integer LAPACK/BLAS; Linux versioned runtime libraries are supported when
development symlinks are absent. Configure `-DSPATIAL_LAPACK_RECOVERY=OFF` for the
previous dependency-free solver. Its three later friction captures remain
unresolved; no WebAssembly portability claim follows from the native fallback.
`research/native-recovery-integration` records exact source, executable and linked
library hashes and all 20 independently checked outputs. Independent review adds
16 analytic and rejection controls. No new trajectory accuracy or speed claim
follows from these captured systems.

Active search supports up to 4096 original rows and 384 reduced rows. Each
candidate must pass the original all-row contact and finite energy gates;
violated inactive contacts expand the search. Up to eight subset passes share
512 search steps, 256 SVD calls, 512 damped factorizations and 96 continuation
attempts. Pressure guides separately permit 128 face attempts/SVD calls and up
to eight upstream normal-only pivot calls. Full continuation and polishing have
separate limits; there is no single global 256-SVD ceiling across all stages.
An upstream pivot call exposes no internal iteration cap, so bounded call counts
are not a hard wall-clock guarantee. Counters disclose actual search work.

Before the gauge and cold restarts, a bounded fallback tries up to eight positive-pressure contacts per
starting face with tangential traction opposing the current slip. This is a
feasible numerical initialization, which can change trial velocity; it is not a
mechanical-null pressure move or a physical impulse. The unchanged full contact
and finite energy gates still accept only the final result. All restarts share
the same global SVD work budget, and counters disclose their use.

For long runs, `progress_checkpoint_path=Path(...)` optionally preserves accepted
output frames through atomic replacement. Its schema is
`native-spatial-progress-v1`: world COM/velocity/spin, **principal-inertia-axis**
quaternions, wire geometry, times, boundary work, output-frame counts and native
residual/containment monitors. Ordinary Python results instead use authored body
axes. A partial checkpoint is explicitly incomplete and cannot resume persistent
contact caches. The destination must be new and have an existing parent directory.

The circular solver defaults to `contact_point_policy="shared"`. Both finite
bodies use the midpoint of their surface contact endpoints; against a fixed or
kinematic body, the finite body's endpoint is used. Normal and tangent lever
arms, world-inertia angular mobility, free-velocity and split RHS, and already
applied warm angular impulses are transported before matrix assembly. Boundary
work uses the same common point. `"separate"` is an explicit legacy comparator;
other solver lanes retain their existing separate-point convention. Unsupported
CFM, contact stiffness/damping and friction anchors reject in this shared lane.
This corrects the internal moment produced by separated equal/opposite forces;
full-tensor orientation integration has a separate finite-step momentum error.
Old trajectory/performance archives qualify their frozen geometry only.

One bounded retry after warm Newton stalls drops Jacobian singular directions
below 1e-10 of its maximum singular value, versus the ordinary 1e-12 search cutoff.
This changes only the numerical Newton increment; it does not regularize physical
mobility, add compliance or waive any contact/energy condition. Weak numerical
modes can require huge increments for tiny residuals and stall line search. The
48-row shared-hull fixture now passes the exact gate; all retries still share
256 SVD calls. Counters and numerical metadata disclose the alternate rank retry.

## Two-channel endpoint restitution

The circular shared-point solver now accepts explicit `normal_restitution` and
`tangential_restitution` together through `spatial_engine.run`. This opt-in
impact law supersedes the zero-restitution restriction for configured impacts;
the unconfigured lane retains its original checks. Explicit values override
body normal-restitution mixing. Positive tangent restitution requests slip
reversal subject to circular impulse capacity. Full simultaneous coupling and
an additional actual kinetic-energy/boundary-work gate remain mandatory.
Optional `record_contact_impacts=True` records contact geometry, basis, before/
after relative motion and actual impulses. See the
[two-channel evidence](../research/two-channel-restitution/README.md).
