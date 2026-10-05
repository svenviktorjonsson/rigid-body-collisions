# Headless planar rigid engine

Two independent pinned solver backends use the same scene and state protocol:

- `block`: Box2D 2.4.1 with its two-point normal block solver. Solver steps mean
  velocity iterations; three position iterations are held fixed.
- `temporal`: Box2D 3.1.1. Solver steps mean temporal substeps. Contact recovery
  frequency and damping are held fixed, and frequency clipping is rejected.

Build both from the repository root with CMake 3.22+ and a C/C++ compiler:

```sh
cmake -S rigid_backend -B build/rigid_block -DCMAKE_BUILD_TYPE=Release -DRIGID_BLOCK_BACKEND=ON
cmake --build build/rigid_block -j 4
cmake -S rigid_backend -B build/rigid_backend -DCMAKE_BUILD_TYPE=Release
cmake --build build/rigid_backend -j 4
```

CMake fetches source archives and verifies their SHA-256 digests. Upstream
licenses and relevant sample/solver sources are stored in
`research/rigid-benchmarks/public-sources/`. The compatibility layer does not copy
the old solver implementation; it adapts public APIs to the common runner.

`rigid_engine.run(scene, backend="block")` runs a headless scene. Its default
is eight primary updates and thirty-two velocity iterations per output frame, the
measured conservative setting in [the executed study](../research/rigid-study-report.md).
The CLI supports `--preset fast|standard|accurate|high`, numerical overrides, and
`--adaptive --policy path/to/frozen-policy.json` for reproducible controller
experiments. The controller did not beat the cheapest passing fixed settings in
the first held-out test, and is therefore not the default. Bodies are
static, dynamic or kinematic and have circle fixtures or strictly convex polygon
fixtures with three to eight ordered vertices. Concave bodies require a supplied convex
decomposition, whose non-overlap the caller must ensure. Geometry is meter-scale;
density is areal kg/m². Output positions are centers of mass. State columns are
COM x,y, angle, vx,vy,omega, using m,rad,m/s,rad/s.

Mass, COM and inertia are calculated from unrounded polygon cores and exact disks and
explicitly assigned in both engines. A common 0.01 m collision skin supports
the older backend's continuous collision algorithm; it is a contact geometry
tolerance, not extra mass. The block backend requires skin at least 0.005 m.
The temporal backend supports an explicit zero skin if desired. Do not silently
compare different collision skins. Compound fixture seams can affect contacts.

Already ordered temporal-backend polygons are authored at a temporary scale
when needed to preserve shallow corners through native hull construction, then
restored to their original core dimensions. World units and solver slop do not
change. `run(..., position_iterations=12)` independently refines block pose
correction; the default remains three. Higher diagnostic collision/velocity work
is allowed, but increased work is not an accuracy certificate.

Two explicit scene controls support diagnosis: `analytic_kinematics: true`
corrects prescribed poses from a double accumulator at each primary update,
while retaining prescribed velocities; `suppress_internal_edges: true` filters
points whose outward reaction direction enters another fixture's core. Both are
off by default. The latter is experimental and does not coalesce duplicate
exterior manifolds. Numerical metadata records both controls.

`research/convex_partition.py` merges equal-material adjacent pieces only when
their convex hull has the same area as their union; it preserves the boundary,
mass and inertia. Stateful or heterogeneous patch boundaries must be retained.

An experimental full-Float64 diagnostic can be built with
`python -m research.build_precision_backend`, then selected in Python with
`run(scene, binary="build/rigid_double/rigid_runner")`. The script checks the
pinned upstream archive and retains original/transformed source hashes in
`build/rigid_double/precision-source.json`. It changes project scalar types,
literals, math calls and precision constants throughout narrowphase, transforms
and solving. This is a locally transformed comparator, not an upstream Float64
release or verification of every Box2D feature. Output labels the precision and
requires its source manifest. Geometry, primitive mechanics and precision
controls are tested separately; packed trajectory convergence remains a gate.

Both backends use their established single-coefficient dry friction model and
default mixing: geometric mean for friction, maximum for restitution. Separate
elastic static/dynamic material laws are not implemented here. The block backend
rejects nonzero rolling resistance; the temporal backend supports its built-in
moment-impulse bound. Numerical contact frequency in the temporal backend is
not a measured Young's modulus or contact stiffness.

Sleeping is disabled during these experiments, continuous collision enabled,
and restitution threshold set to zero. Initial velocities that would be clipped
by this adapter are rejected. The engine's own limits can still affect later
states in extreme scenes, so benchmark regimes must remain within supported
scales and rates.

With `policy={}`, adaptation preserves one world and its contact caches. It
chooses among levels (primary steps, solver steps): (1,1), (1,4), (2,8), and
(4,16). High-level work and thresholds are configurable. Features include
motion relative to fixture size, penetration in reported contact pairs, dynamic
contact island size and mass contrast. Promotion is immediate; demotion has a
dwell period. The whole world uses the chosen setting, so unrelated islands can
be over-refined. This is a heuristic controller, not a certified error estimator.

Timing separates solver and controller work. State extraction and output are
reported separately through end-to-end wall time. Reported penetration covers
known contact pairs, rather than proving no collision was missed. Use independent
CCD, momentum, energy, analytic and reference-refinement checks.

The restitution failure and comparator evidence are recorded in
`research/rigid-benchmarks/rebound-counterexample.json`. Different numerical
stabilization/contact models must be acknowledged when comparing the backends;
only refinement within one backend holds that formulation fixed.

Moving containers use one kinematic body with four wall fixtures. Prescribed
`velocity_schedule` entries contain `time_s`, `velocity` and optional `omega`,
with changes aligned to output frames. The world and contact caches persist.
Dynamic histories remain in `states`; prescribed-body histories are in
`kinematic_states`, with the same six columns. Contents exchange momentum and
energy with the actuator: their momentum is not an invariant.

The runner uses the `rigid-v2` wire protocol; rebuild both backends after updating.
`research/container_scenes.py` supplies moving boxes, ball grids and an exact
closed packed-row constraint case. The global normal-impulse projection is an
independent frozen-contact verification oracle, not the implemented frictional
engine or a demonstrated performance improvement.
