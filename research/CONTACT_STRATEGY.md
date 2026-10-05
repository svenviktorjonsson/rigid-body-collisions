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
| Other supported sphere/plane elastic material | Adaptive resolved material integration | Entry/lift-off events, stored normal/shear/twist energy, plastic/damping/separation loss, full energy/yield gates and bounded rejection |
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

Current evidence: the circular native study qualifies three of six cases,
including slow 27-sphere shaking, rapid 27-sphere translation and rapid eight-box
shaking. Fast 27-sphere shaking remains unqualified; random-hull attempts still
reject after a verified gyroscopic RHS correction. The elastic original and
refined studies together verify nine of ten synthetic scenarios, including spin
reversal, vertical floor/ceiling alternation and a fixed material at 0.01 and
100 m/s. One high-spin/low-friction weighted case remains budget-rejected.
A separate same-floor gravity regression shows approximate horizontal back/forth
motion with spin reversal and accounts for its small separation loss.

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
rows and 256 numerical SVD calls. Bounded opposing-slip guesses run before gauges/cold restarts and never reach bodies unless the original gates pass. Four captured random-hull systems recover;
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
