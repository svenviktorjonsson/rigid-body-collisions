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
separate qualification. Twenty-two retained captured contact systems now pass the combined
primary/active-contact/continuation/polishing solver. Fresh full trajectories
have reached later failures; they remain unqualified until all physical and
refinement gates pass.

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

Native recovery is optional and begins after at most 256 primary sweeps when
the iteration budget permits it. Normal-only systems can try bounded pressure
face release. General circular friction tries a warm active-contact subsystem,
then full continuation and the earlier polisher, before spending any remaining
primary iteration budget. Active search supports at most 4096 original rows and
384 reduced rows. Every original equation and finite passivity condition must
pass, including inactive contacts; violated inactive normals expand the search.
At most eight subset passes share 512 search steps, 256 SVD calls, 512 damped
factorizations and 96 continuation attempts. Pressure-guide search has a separate
128-attempt/SVD budget. The subsequent continuation and polishing stages have
their own disclosed limits; 256 SVD calls is not a combined solver-wide bound.
Upstream pivot calls have no exposed internal pivot limit, so these caps are not
a hard wall-clock guarantee. Failed trial impulses never reach bodies.


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

`progress_checkpoint_path` atomically records every accepted output frame,
boundary work and native containment/residual monitors. Quaternion coordinates
in this diagnostic are the backend principal-inertia axes, unlike the ordinary
authored-axis result. An incomplete checkpoint is a retained prefix, not a
completed reference or a resumable contact-cache snapshot. The six prospective
full runs at source `95d224f` keep the original material, scene and gates.


The final exact-component tail recovery passes22 retained captured contact
systems at source9e97be0, with the prior20 accepted impulses exactly preserved.
It uses the actual final rejected PGS seed, exact nonzero mobility components,
complete normal/tangent triples and every original contact in final acceptance.
All numerical-guide budgets and finite-metric validation are disclosed in
research/component-recovery-integration. It runs only after existing stages
reject; early recovery performance remains a separate prospective experiment.

The source52f7 six-run translation-only hull study retains3 histories/3 actual
rejections and0 qualified references; completed8-body refinement edges fail
all original trajectory budgets. An exact rational finite-bound certificate
proves the captured74-row translation-only pose-repair equations inconsistent.
That rejection cannot be fixed by more iterations under the same bounded repair
model. A revised geometric or rollback/refinement protocol must explicitly
account for pose-induced energy/momentum effects and qualify fresh trajectories.

`split_translation_gap` explicitly replaces the zero-closing position constraint
on separated cached contacts with their signed available clearance divided by the
internal timestep. Within-slop and penetrating targets remain unchanged; physical
velocity rows and material coefficients retain their original law. The actual
74-row saved system passes the existing native repair at 1.79e-12 m/s, with
independently reconstructed translations and a geometry re-query. Its old bounded
infeasibility certificate remains valid for the old targets.

The numerical pose ledger uses the final physical velocity, including applied
velocity and external-force increments, to record displacement cross momentum.
It separately reports signed and absolute gravity-potential and orbital-momentum
changes. These corrections are disclosed, not subtracted from the original energy
gate. The source108 six-run study completed all six histories without rejections;
all individual physical/contact/ledger gates pass, but both scenes fail both
original quarter-budget refinement edges. No dense-hull trajectory is qualified.

The next combined rule subtracts accepted physical point motion, including spin,
from every desired numerical position rate; a native writeback oracle and actual
opposing-wall sphere test verify that clearance is not consumed twice. Hull bounds
are recalculated after setting the authored margin. An optional early component
search uses the actual first256 rejected iterate, restores its exact seed on
decline and preserves every original final law gate. Default ordering stays
unchanged; early and later helper caps are separate, and declines can add work.
The fresh full protocol declares all three numerical choices together, retains
all original gates and forbids causal attribution or a speed ranking from this
joint experiment. Its results must be independently audited before qualification.
