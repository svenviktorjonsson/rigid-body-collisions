# Method selection for the planar rigid-body engine

The initial candidate was Box2D 3.1.1. Independent analytic checks found a
restitution/symmetry failure, so the study now compares it with Box2D 2.4.1's
coupled two-point normal solver. Adaptive work is tested within each backend,
retaining its world and contacts. Neither backend is established as the most
physically accurate engine or a universal winner. These comparisons will not establish
superiority over MuJoCo, Drake, another hard-contact solver or deformable FEM.

## Evidence behind the choice

- [Box2D's simulation documentation](https://box2d.org/documentation/md_simulation.html)
  states that substeps increase contact/joint accuracy, recommends four substeps,
  and notes that more substeps reduce stretching in long constrained chains.
  This supports a tunable accuracy/cost mechanism; it is an author recommendation,
  not our benchmark result.
- [Catto's Solver2D article](https://box2d.org/posts/2024/02/solver2d/) discusses
  projected Gauss-Seidel, temporal substeps, soft constraints and relaxation.
  It explains why spending work on smaller integration increments can help
  beyond merely repeating iterations at the same frozen geometry.
- [The latest release endpoint](https://api.github.com/repos/erincatto/box2d/releases/latest)
  returned v3.1.1 when checked for this study. Its pinned source includes polygon
  manifolds, persistent impulses, continuous collision handling and rolling
  resistance. The exact commit and archive digest are in
  [source-lock.json](rigid-benchmarks/source-lock.json).
- [MuJoCo's computation documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html)
  explains that hard complementarity and convex soft friction are different
  approximations. With friction, dropping complementarity changes the model.
  Solver convergence cannot by itself identify which constitutive approximation
  matches a real material or experiment.
- The prior [review](joint-verdict.md) identifies established coupled contact
  formulations and energy/restitution issues. In particular, independently
  assigned restitution targets can be incompatible or energetic in simultaneous
  impacts. The new study must preserve that limitation.

The backend's contact constraints contain numerical softness and penetration
recovery. Bodies themselves are rigid. Neither its contact frequency nor rolling
resistance is a measured material property just because it appears in a public
example. Box2D uses a single dry-friction coefficient, rather than the distinct
static/dynamic elastic-history contact law in our local compliant model.

## First falsifying experiment

Two equal, square bodies collide centrally at velocities +2 and -2 m/s with
restitution 0.6 and zero friction. Symmetry and the isolated normal impulse law
give outgoing velocities -1.2 and +1.2 m/s, with zero spin. The temporal backend
at four primary steps and sixteen substeps produces approximately -1.056 and
+1.056 m/s and opposite spins of magnitude 0.288 rad/s. The block comparator
produces -1.2000003 and +1.2000003 m/s with zero spin. Both preserve linear
momentum in this case. The [full sweep](rigid-benchmarks/rebound-counterexample.json)
varies both primary resolution and solver work, including adverse refinements.

This is numerical evidence against using the temporal setting as an accurate
default for this symmetric rebound case. It is not experimental validation and
does not prove that the older comparator wins on stacks, friction or throughput.
The block source explicitly solves a two-point normal complementarity problem
inside its global sequential-impulse iterations. The comparator keeps that
coupling instead of treating its two points as unrelated impacts.

## Executable study

1. Add a headless pinned backend and a Python scene adapter. Support convex
   polygons and compounds of convex polygons for concave bodies. Preserve mass,
   COM, inertia, friction, restitution and rolling settings across fidelity modes.
2. Check independent analytic cases: free flight; isolated normal rebound;
   frictional stopping; incline stick/slip; internal momentum transfer. Use
   synthetic declared coefficients to verify the governing idealized model.
3. Add coupled stacks, chains, large mass contrasts, spinning/off-center impacts,
   thin obstacles and compound shapes. Public sample topologies and exact source
   parameters remain traceable; newly generated variants are labeled synthetic.
4. Refine primary collision-update time and solver substeps separately. Treat a
   highest-work run as a candidate reference until output convergence is checked.
   Use analytic results where available; exclude unconverged reference cases
   from favorable accuracy/cost claims.
5. Compare fast fixed settings, an established four-substep setting, a higher-work
   fixed setting and an adaptive controller. Hold physical parameters constant.
   Calibrate controller thresholds on a fixed training split, then freeze them
   before evaluating held-out scenes.
6. Record separate position/velocity/spin errors, penetration and mechanical
   energy behavior. For long unstable/chaotic stacks report aggregate observables
   and trajectory-divergence limits. Repeat timings and include controller work.

Initial adaptation will apply to the whole world, retaining its bodies and
contact caches. It will vary solver substeps and the number of collision-update
steps within a fixed output frame. This avoids transferring between different
physical models. It may over-refine independent contact islands; an island-local
scheduler is a later optimization that requires a supported backend interface.

## What would justify the recommendation

The adaptive configuration must meet declared tolerances on held-out qualified
cases while reducing measured runtime against a fixed configuration that meets
the same tolerances. A low-cost setting that misses the tolerances does not win.
Failed cases, reference uncertainty, controller overhead and fixed-setting
alternatives belong in the report. A heuristic error predictor is not a certified
error bound. If adaptation offers no advantage, retain a fixed setting as the
default and report that result.

Experimental material identification and quantitative validation remain separate
work. The new rigid study can establish numerical behavior and agreement with
classical idealized mechanics; it cannot turn the existing numerical sample
coefficients into experimentally authenticated properties.
