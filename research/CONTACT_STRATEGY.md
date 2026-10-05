# Computational contact strategy and verification boundary

Use a declared material law first, then choose the cheapest verified numerical
method for that law. A speed setting must not silently change elastic contact
into dissipative rigid contact. Body state uses COM position, world velocity,
world angular velocity and a unit quaternion; geometry supplies mass and full
inertia in the correct frame before the first contact.

| Configured contact model | Cheap method | Resolved method and gate |
|---|---|---|
| Inelastic, zero tangential force and no contact couple | Eliminate exactly zero tangent variables before normal matrix assembly | Normal complementarity; tested packed 3D rows/boxes; disclose unsupported fallback |
| Inelastic circular Coulomb, one coefficient | Warm-started coupled block iterations; accept within eight sweeps when residual passes | Same equations with more sweeps, semismooth Newton, bounded minimum-norm recovery, velocity-neutral pressure redistribution and cold restarts; strict residual/passivity gate; reject rather than substitute friction pyramid |
| Matched linear elastic sphere/plane, no damping/history/yield, bounded declared deformation | Exact normal/tangent/twisting impulse and half-sine contact trajectory | Same configured material integrated with explicit energy stores; instantaneous shared force/couple capacity proves the fast branch |
| Other supported sphere/plane elastic material | Adaptive resolved material integration | Exact ballistic free flight and explicit entry/lift-off/yield/release events, stored normal/shear/twist energy, plastic/damping/separation loss, full energy/yield gates and bounded rejection |
| Arbitrary many-body elastic wrenches, separate static/dynamic coefficients, measured rubber material | Not verified yet | Integrate persistent histories with the full contact graph; verify before promoting a production preset |

The rigid graph includes all simultaneous contacts and retains normal/tangent,
lever-arm and inter-body coupling. Relative translation and angular tip speed
control the native travel-guard timestep; exact prescribed command times split
updates. Scheduled wall velocities are installed before contact discovery, so
shaking uses the new velocity immediately. World/contact caches persist across
internal updates. Position correction is separate from physical impulse and
boundary work.

The elastic contact carries both a resultant force and an **independent couple**
at a computational point. The body angular impulse is the moment of the force
impulse plus this couple impulse. Its normal-load capacity includes an effective
material contact length to preserve torque units. Tangential and twisting
elastic history share one phenomenological yield budget. Conservative normal
force includes the derivative of stored shear/twist energy when stiffness varies
with compression. Contact release either smoothly releases stored energy or
records residual energy as a declared loss. No hidden energy reset or imposed
spin reversal is accepted.

The exact elastic branch is conditional on matching normal, tangential and
spin oscillator frequencies. It proves the force/couple capacity throughout the
contact, returns both impulses separately and checks total kinetic energy.
Its trajectory includes position, velocities, normal/shear/twist stores, force
and couple; spherical orientation is omitted. The dispatcher falls back to
integration of the **same** material, not another restitution/friction law.
This is a sphere/plane method, not an exact general many-body collision map.

For offline trajectory qualification, compare consecutive refinement levels
including angular velocity and quaternion orientation. Separate contact residual,
containment, energy/work and trajectory accuracy: a run can pass the first
three while failing the last. Preserve failures and return no verified choice
against an unqualified reference. `spatial_fidelity` chooses effort against a
qualified reference; it does not claim a certified online truncation-error bound.

Current evidence includes independently qualified slow shaking, rapid translation
and fast eight-box shaking. The corrected shared-point 27-sphere fast-shaking
study qualifies both adjacent reference edges and verifies 7.17x native gain
for its declared scene and budgets. Complete random-hull trajectories require
separate qualification. Eleven captured hull systems now pass the combined
primary/polish/continuation solver, including all six latest rejected systems.

The elastic completion study qualifies all ten original scenarios without
changing their material, simulation budgets or gates, plus eight additional
signed/chained cases: 54 histories, zero rejections. Both normal-axis spin signs
reverse with sufficient configured capacity. Three oblique same-floor gravity
bounces alternate horizontal motion and spin; five vertical floor/ceiling bounces
alternate surfaces. Insufficient capacity does not imply reversal. Plastic flow
in the short gravity release tail is retained in the loss ledger, and strict
force/couple capacity is checked separately from energy. Historical budget
rejections remain archived. These sphere/plane tests use an undeformed-radius
contact lever; large-compression physical fidelity requires additional evidence.

Synthetic parameters establish mechanics and numerical checks. They do not
identify authentic rubber coefficients, convergence order or a novel contact
law. The literature review includes experiments supporting tangential compliance
and experiments where balls do not reverse spin despite substantial friction.
Material calibration needs independent measured normal/tangential/rotational
trajectories and must be separated from numerical speed/accuracy selection.
The Vektor target is `vektor-flow/bootstrap`, paired with spec. This research is
public Python/C++ evidence, not a completed native/WASM/GPU language port.

The native recovery operates only after iterative exhaustion and is optional.
It searches neighboring friction faces through a certified mechanical null
direction, checks the resulting velocity change, and accepts only the unchanged
full contact residual and finite passivity bound. Its work is capped at 384
rows and 256 numerical SVD calls. Bounded opposing-slip guesses run before gauges/cold restarts and never reach bodies unless the original gates pass. Five captured random-hull systems recover;
the separate six-attempt whole-trajectory follow-up still rejects. Preserve
that distinction when choosing or porting this method.


The Coulomb default uses a shared world contact point, with both complete signed
rows transported before assembly and boundary work evaluated consistently.
Finite pairs use their endpoint midpoint; finite-infinite contacts use the finite
surface endpoint. Separate-endpoint friction at a nonzero gap or overlap can
produce an internal couple and violate total angular momentum. The native and
Python conservation checks distinguish exact impulse momentum from subsequent
full-tensor orientation integration drift. Historical separate-point references
and the 6.11x timing result remain frozen evidence for that earlier convention;
fresh shared-point protocols must qualify independently.


A bounded numerical-Jacobian rank retry drops weak search directions below1e-10
of the maximum singular value, versus the ordinary1e-12 cutoff. It changes only
a Newton increment, never the physical mobility or exact final contact law.
The48-row shared capture passes; its fresh full trajectory then fails later.
Both six-attempt shared hull follow-ups retain every rejection and qualify no
reference. The corrected-geometry27-sphere study independently qualifies a7.17x
native gain with fixed budgets; this does not establish general hull accuracy or
authentic material coefficients.


Supplementary continuation uses a Fischer–Burmeister normal search merit and
internal trial friction continuation, accepting only the final original circular
projection/complementarity/passivity gate. Per call: 384 rows, 512 outer search
steps, 256 SVD calls, 512 damped factorizations and 96 stage attempts. These limits
are separate from the preceding polishing budget. A frictionless normal QP and
at most one normal-only Dantzig call provide initial guesses; Bullet exposes no
internal pivot cap, so this is not a hard wall-clock guarantee. Trial friction
and Jacobian damping are numerical search devices; physical mobility and material
remain unchanged. Cost counters disclose actual recovery use.

Opt-in translation-only split repair uses a linear normal Gram matrix and leaves
physical angular velocity and orientation repair unchanged. Clearing numerical
turn velocities prevents fake pose correction from rotating world inertia and
injecting kinetic energy. Translation can still change orbital momentum and
gravity potential; it is a geometric repair, not a physical force impulse.
