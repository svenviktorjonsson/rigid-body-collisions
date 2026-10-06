# Rigid body collisions

A physics-engine research prototype with real 2D and 3D native collision backends
and reproducible speed/accuracy benchmarks.

The [Float64 3D backend](spatial_backend/README.md) supports arbitrary convex
hulls, boxes, spheres and compounds, full inertia tensors, quaternion rotation
and prescribed moving walls. Mechanics tests include 100 m/s walls driving
64 bodies and 20 m/s containers with 64 spheres or 27 random rotating hulls.
The [3D evidence report](research/spatial-validation/report.pdf) retains
102 histories. Only the frictionless row qualifies its frozen trajectory gate;
all five dense frictional references remain unqualified. Passing containment
and no-tunneling tests does not establish their trajectory accuracy.

The [3D normal-contact improvement](research/spatial-normal/report.pdf) verifies
100 m/s walls moving full 27/64-body boxes and 64/128-body rows. All six analytic
cases pass with no fallback. Removing fixed-zero tangent variables before assembly
gives 1.46–7.22x measured native gain with bitwise-identical trajectories and nine
times less mobility matrix storage. This optimized profile requires zero friction
and restitution; it does not qualify the failed frictional scenes.

The [circular 3D friction study](research/spatial-friction/report.pdf) adds
residual-driven sticking/sliding solves, immediate prescribed-wall reversals and
correct first-collision world inertia. Three of six new references qualify:
slow shaking with 27 spheres, 20 m/s translation with 27 spheres and 20 m/s
shaking with eight boxes. The archive retains 76 attempts, 52 complete histories
and 24 solver rejections. Its original fast 27-sphere shaking refinement fails.
A separately verified [gyroscopic RHS correction](research/spatial-friction-gyro/report.pdf)
restores free angular velocity to tangent equations; all six paired hull attempts
remain rejected. No fallback to a different friction law is accepted in this lane.

A [same-law fast-shaking follow-up](research/fast-shake-diagnostic/report.pdf)
qualifies two finer references while retaining the earlier nonmonotonic failure.
The verified fast setting passes both references in all three repetitions.
A [predeclared repeated cost comparison](research/shake-performance/report.pdf)
measures **6.11x faster native stepping** (2.97 versus 18.18 seconds median)
for this 27-sphere scene with unchanged material and accuracy budgets.
This is a scoped numerical result for the frozen separate-endpoint geometry, not calibrated material or a universal ranking. The corrected shared-point default requires fresh qualification.

[Bounded contact recovery](research/coulomb-diagnostics/README.md) adds
minimum-norm Newton steps, certified velocity-neutral pressure relocation and
cold restarts on the same circular law. Five captured hull systems pass native
and independent contact/passivity gates, including a 267-row system. The
[six-attempt full follow-up](research/spatial-friction-recovery/report.pdf) and
[larger-cap follow-up](research/spatial-friction-recovery384/report.pdf) still
reject later systems; general hull trajectory accuracy remains unqualified.

A [contact-point conservation review](research/predictive-contact-review/review.pdf)
finds an angular-momentum defect in separate endpoint force impulses at nonzero
contact gap or overlap. The Coulomb default now transports the complete contact
rows and warm starts to one shared world point before mobility assembly; boundary
work uses the same point. Analytic pair and full-tensor impulse tests verify
momentum conservation. Explicit `contact_point_policy="separate"` retains the
legacy comparison. A fresh [corrected-model shaking study](research/shared-shake-study/report.pdf)
qualifies both reference edges and all repetitions, measuring **7.17x native gain**
(3.17 versus22.77 seconds median) for the same synthetic27-sphere scene.
The separate-endpoint6.11x result and all historical failed archives are preserved.
The fresh [shared hull follow-up](research/shared-hull-followup/report.pdf)
retains six later solver rejections and zero qualified references; this sphere
result does not establish general hull accuracy.
A [numerical-Jacobian follow-up](research/shared-hull-rank-followup/report.pdf)
recovers the weak 48-row capture and advances one trajectory, which then rejects
later. Its six complete attempts remain zero qualified references. Only the
numerical search increment changes; physical mobility and gates remain unchanged.

The [elastic wrench study](research/elastic-patch/report.pdf) separately models
stored tangential energy and an independent twisting couple at a computational
contact point, bounded by normal load and an effective contact length. It tests
spin reversal and floor/ceiling rebounds with energy accounting. The initial
28-history archive qualifies four of ten cases and retains two integration-budget
rejections. The [refinement follow-up](research/elastic-patch-refined/report.pdf)
keeps those failures and the same physics/gates, adds 15 histories and three
rejections, and verifies five of six remaining cases. Together nine of ten
distinct scenarios qualify, including one material at 0.01 and 100 m/s. These are
synthetic sphere/plane material hypotheses, not calibrated rubber parameters or
a completed elastic many-body engine. [Research review](research/elastic-patch/review.md)
includes experimental support and measured no-reversal counterexamples.
A [conditional exact elastic path](research/elastic-patch-refined/fast-path.md)
returns both force and independent couple impulses and their full contact energy
history; its dispatcher uses resolved integration of the same material when the
exact assumptions fail. The [computational strategy](research/CONTACT_STRATEGY.md)
records supported branches and the remaining many-body elastic integration work.

The [elastic completion study](research/elastic-completion/summary.json) corrects
friction-limit event chatter and zero-time grazing loops, and uses exact ballistic
free flight between impacts. It preserves the original material and accuracy
gates: all **10 original cases** and eight additional signed/chained-bounce cases
qualify, with **54 complete histories and zero rejections**. Independent auditing
checks energy stores, plastic work, force and independent couple impulses, yield
capacity and both refinement edges. With enough configured elastic capacity,
normal-axis spin reverses; three oblique floor bounces under gravity alternate
horizontal motion and spin, while five vertical floor/ceiling bounces alternate
surfaces. The low-friction control retains its spin sign, as the material law
requires. The [interactive playback](research/elastic-completion-visuals/demo.html)
and [figures](research/elastic-completion-visuals/spin-and-bounces.pdf) show these separate
sphere/plane prototype results. Measured rubber calibration and native arbitrary-body
elastic many-contact integration remain open.

A supplementary [original-law continuation search](research/coulomb-trust/README.md)
recovers all six latest captured hull failures. The combined primary, polishing
and continuation solver initially passed eleven captured systems. The
[active-contact follow-up](research/coulomb-normal/README.md) passes all **16**
retained captures with independent circular-friction and passivity checks,
including the 231- and 321-row systems. It checks every original contact after
searching a smaller active system and expands that system when omitted contacts
remain violated. Numerical trial friction and damping do not change the accepted
material law. Full trajectory qualification is evaluated separately in
`research/hull-active-completion`: its fresh seed-42 runs have reached later
failures, which remain retained. Captured solutions do not qualify a scene.
Optional atomic output-frame checkpoints preserve accepted partial trajectories
if a later solve or the execution environment fails.
The opt-in `position_stabilization="split_translation"` removes angular pose
correction, which otherwise can add kinetic energy to a spinning anisotropic body.
It preserves the physical velocity solve; positional translation still requires
separate gravitational-energy and orbital-momentum accounting.

The [earlier translation-only shared-hull study](research/hull-translation-completion)
retains six uninterrupted attempts: three eight-body histories and three
27-body solver rejections, with **zero qualified references**. Both completed
refinement edges exceed every unchanged trajectory budget. The two new velocity
captures have strict original-law roots in the
[exact-component recovery review](research/translation-native-review); its
frozen combined strategy passes all 22 retained contact systems. Production
integration and its final-seed replay are recorded separately in
[the integration receipt](research/component-recovery-integration).
The remaining 74-row translation-only position-repair system is
[provably inconsistent within its recorded finite bounds](research/translation-position-certificate).
This is a numerical pose-repair defect, not a proof that the physical geometry or
Coulomb velocity law is infeasible. A revised repair/refinement protocol requires
fresh complete trajectories and energy/momentum accounting.

The opt-in `position_stabilization="split_translation_gap"` permits position
repair to consume the available clearance at separated cached contacts. Physical
normal/tangent impulse equations and penetrating repair targets stay unchanged.
The existing native solver passes the revised saved 74-row problem at
**1.79e-12 m/s**, with no angular correction and no worsening contact pair in an
independent geometry re-query. This is a declared change to the numerical repair,
not a solution of the old inconsistent equations. Native output and atomic
progress disclose numerical gravitational-energy and orbital-momentum changes.
The [completed six full trajectories](research/hull-gap-completion/RESULTS.md),
frozen at `108a9bb4c7899f75d760b27b179cc56557904a08`, have **six complete histories,
zero solver rejections, and zero qualified references**. Individual physical,
contact and ledger checks pass, but both scenes fail both original refinement
edges. Passing contact equations does not establish trajectory accuracy.

The new opt-in `position_stabilization="split_translation_combined"` subtracts
accepted physical linear and angular contact motion from every desired numerical
position rate. This prevents physical motion and pose repair spending the same
gap twice. A real two-wall sphere control changes a 50-micrometre overlap into
50 micrometres of clearance without changing velocity or energy. Hull construction
also sets its declared margin before recomputing cached bounds; eight independent
cache controls pass. The opt-in `early_component_recovery=True` uses the actual
rejected first256 iterate before the old recovery pipeline. Its isolated66 outputs
pass original-law gates; default22 outputs remain exact, and all5 early declines
retain the original endpoint with their added work disclosed. No universal speed
ranking is claimed. The integration passes162 engine and10 contact-model tests.
The [fresh six-trajectory results](research/hull-combined-completion/RESULTS.md)
retain **five complete histories and one actual42-row velocity rejection**,
with zero qualified references. Every completed lane passes its individual gates;
the eight-body middle rejection blocks both accuracy edges, and the27-body edges
fail unchanged trajectory budgets. Independent exact-system searches find roots
for the new42-row capture which the unchanged native final gate accepts; a bounded
search correction now passes23 captured inputs, preserves all22 prior impulse/response Float64bytes and counters, and passes122 strict audit checks. The bounded tail is now integrated and live23 preservation passes; read [HANDOVER.md](HANDOVER.md) for current continuation. All three numerical changes were
declared together, so these timings do not establish causal speed gains.

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

The [rapid-motion friction benchmarks](research/rapid-friction/README.md) qualify
2D nine/25 disks and 3D27 spheres under +/-20m/s container reversals with friction0.4.
Median native gains versus qualified fine references are23.07x,15.65x and7.58x.
Experimental Float64 planar settings explicitly use1um penetration slop; authored
geometry and materials remain unchanged. Rotating groups, polygons, boxes and
hulls remain unqualified. Exact settings, all failures and an independent audit
of189 retained baseline records accompany the report. Current tests:163 engine tests,
35 subtests, ten contact-model tests and eleven native checks pass.

The [larger position recovery fix](research/large-position-recovery/README.md)
clears the retained390-row125-hull position system with the original absolute
gate, exposing512-row position-only numerical search metadata. Velocity search
defaults remain384; all earlier22 contact response bytes/counters remain exact.
Full rotating and irregular-shape trajectory qualification remains open.

The [larger velocity fallback](research/large-contact-recovery/README.md) uses
bounded native mobility-null seeds after all existing lanes decline. Actual
162/297-row failures now pass the original production gate;423 rows still
decline. The repaired125-hull finest history completes0.12s and passes physical
gates, while its reference refinement remains pending. The separate13-case
working-baseline and per-case2x performance gate is still open.

The larger-contact fallback now supports exact components up to128 rows while
retaining six pressure seeds and the original physical gates. An actual393-row
capture passes independently at5.60e-9m/s; other larger contact failures remain.
See [current continuation status](HANDOVER.md) and
[the component128 proof](research/component-cap128/production393-independent.json).
The all-case working baseline and subsequent2x performance gate remain pending.
