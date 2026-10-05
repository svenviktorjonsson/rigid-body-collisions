# Qualified rapid-motion friction benchmarks

Later pushes through `a799038` were merged before this continuation. The
production 3D projection recovery tail now passes all 23 retained contact
systems, with every earlier 22 response byte and original counter preserved.
The 2D adapter now records signed moving-boundary work and actual friction
impulses. Its observer preserves prior state histories exactly.

Three complete synthetic scenes qualify with friction **0.4**, zero
restitution, gravity **9.81 m/s²**, container reversals at **±20 m/s**, and a
**0.12 s** horizon. Both adjacent final refinement edges pass the original
quarter budgets; all timed repetitions pass full budgets and repeat states
bitwise. Timings are medians of three alternating, sequential repetitions
after one warmup per variant.

| Scene | Fine reference | Selected setting | Native gain |
|---|---:|---:|---:|
| 2D, 9 disks | 5.003 s | 0.217 s | 23.07× |
| 2D, 25 disks | 13.070 s | 0.835 s | 15.65× |
| 3D, 27 spheres | 13.281 s | 1.753 s | 7.58× |

The 2D adapter/process medians are 5.010/0.221 s for nine disks and
13.082/0.842 s for 25 disks: 22.71× and 15.54× respectively. Native 2D
`step_s` times the physics loop and observer, excluding per-frame output and
diagnostic work. Native 3D also includes state recording. Earlier prospective
plan wording grouped these timer scopes together; this paragraph gives their
actual implementation scopes. Ratios compare settings within one scene and
model. External host load was uncontrolled; these are not universal speed
ratios or real-time performance claims.

The qualified 2D build uses an explicitly declared **1 µm numerical penetration
slop**, while preserving authored 0.01 m polygon collision skin, disk radii,
mass, inertia, friction and prescribed motion. It is the experimental full
Float64 Box2D comparator. The supported builder reproduces both prototype
histories byte for byte:

```sh
python -m research.build_precision_backend \
  --linear-slop-m 0.000001 --output build/rigid_double_tight
```

Use a fresh build destination. Numerical metadata records the slop and source
manifest. Nine disks select 5 µs updates and 32 velocity iterations; 25 disks
select 10 µs updates and 128 iterations. Both use 12 position iterations. The
finest reference is 0.625 µs with 128 velocity iterations. The 3D candidate uses
10 µs updates against a 1.25 µs reference, shared contact points, the original
circular Coulomb law and the combined translation repair. Exact scene hashes,
paths and API arguments are in `verified-settings.json`; they do not certify
unseen scenes.

The original 5 mm planar slop did not qualify the tested 10/5/2.5 µs ladder.
The tighter study also used additional refinement levels, so those two changes
do not isolate slop as the sole cause. Increasing velocity iterations alone to
512/2048 did not qualify the rotating disk or mixed polygon cases. Clearing
3D manifolds or tightening their cache threshold did not qualify any of the
prototype sphere/box/hull references, and those policies were not adopted.

**Remaining failures:** rapidly rotating groups, mixed convex/concave polygons,
boxes and random 3D hulls still fail original trajectory accuracy checks.
The six original adaptive hull runs at integrated source `8c7065e` retain
three histories and three actual later contact rejections, with zero qualified
references. Neither complete histories nor accepted instantaneous equations
close that accuracy gap. No failed gate or archived trajectory was relaxed.

`audit.py` independently recomputes geometry, mass/full inertia, endpoint
energy/work, aligned trajectory errors, both refinement edges, repeated-state
bytes and timing medians from the saved outputs. The 2D work observer has
independent signed normal/tangent impulse controls. The independent state audit
does not reconstruct every contact impulse or certify continuous containment.
Run the portable archive audit from the repository root:

```sh
python -m research.rapid-friction.audit
```

Results are under `results-spatial`, `results-planar-tight` and
`results-planar-optimized`. Earlier planar and cache failures, the first harness
interruption and actual execution adapters remain preserved. Current checks:
163 engine tests plus 35 subtests, ten contact-model tests and all eleven native
check executables pass. These are external Python/C++ experiments, separate
from Vektor compiler/Section 0, WASM/GPU or calibrated-material acceptance.
