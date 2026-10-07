# Normal and tangential restitution restored

The user's model uses BOTH normal and tangential restitution. The earlier
zero-restitution Coulomb comparison omitted this central part of the model;
its rubber mismatch does not demonstrate failure of the two-channel model.

For incoming relative velocity at the shared contact point, the configured
endpoint targets are u_n_after = -e_n u_n_before and u_t_after = -e_t u_t_before.
Normal coefficient e_n is in[0,1]; scalar isotropic e_t is in[-1,1]. e_t=-1
requests unchanged tangent velocity,0 requests sticking,positive e_t requests
slip reversal. Tangential impulse is projected onto the circular μ*p_n capacity;
when that capacity binds, achieved tangent rebound differs from the target.
This is an explicit impact endpoint model, not the zero-tangential-restitution
continuous Coulomb law or a resolved deformation/force-history calculation.

`spatial_engine.run` accepts both coefficients with solver='coulomb', shared
contact point and start-phase contacts. Explicit coefficients override body
normal-restitution product mixing. Targets use the full contact Jacobian,
world inertia and free gyroscopic velocity. The complete simultaneous graph
is retained. A separate gate checks actual kinetic change minus boundary work,
not a quantity computed by treating restitution targets as incoming velocities.
Incompatible coupled coefficients that inject energy are rejected. Rebound
is applied to geometrically touching approaching contacts; all authored original
zero-restitution scene settings remain unchanged.

```python
result = spatial_engine.run(
    scene, solver='coulomb', kinematic_contact_phase='start',
    position_stabilization='split_translation_combined',
    normal_restitution=0.78, tangential_restitution=0.49,
    record_contact_impacts=True,
)
```

Optional contact diagnostics retain body identifiers, world contact point,
normal/tangent basis, incoming/outgoing relative velocities and actual world
impulse. This supports shape/location/direction comparisons. The coefficients
currently apply uniformly; no location-dependent or anisotropic material map
has been fitted or asserted. Local normal/tangent directions and full inertia
already account for geometric differences. Diagnostics have an explicit100000
record cap and reject on overflow rather than silently truncate.

`rigid_engine.run` accepts the same coefficients with backend='block', selecting
`build/rigid_double_restitution_v1/rigid_runner` when no binary is supplied.
The isolated2D build uses the simultaneous whole-island solver, existing shape
handling and independent actual-body impulse/passivity gates. Build with
`python -m research.two-channel-restitution.build_planar` after the existing
`planar-global-rounded-union` source build is available. Historical binaries
remain frozen. Backend acknowledgement is mandatory: an old unsupported binary
cannot silently ignore coefficients. Defaults retain legacy selection/output.
This2D build is not yet the qualified all-world production baseline.

## Evidence

-144 native2D/3D size/spin/friction-cap controlsPASS, including e_n/e_t=(0,0),
 (.78,.49),(1,1),(.6,-1),three radii,three initial spins andtwo friction capacities.
 Final source run and previous pre-diagnostic run are both retained.
-24 earlier default control trajectories remain Float64 byte-exact.
-Three rotated,off-centre box impactsPASS contact restitution,full world-inertia
 angular impulse,linear impulse,circular capacity and physical energy checks.
-Regression tests check positive tangent rebound,capacity-limited rebound,
 fully elastic pair momentum/energy and translating-wall actuator work.
-Cross2010 Superball/granite replay now uses BOTH.78normal/.49tangential values:
 spin factor15.5099 at25deg;14.9270–16.0880 over24–26deg,overlapping measured
 14.9±.1. Normal/tangent targets match; this comparison is consistent with the
 reported angular uncertainty, not an independently calibrated material proof.
 μ=.9 is a disclosed hypothesis; homogeneous-sphere inertia/fixed plane/approximate
 incident speed remain assumptions. Both restitution inputs come from the same
 paper's target impact. Do not claim they were predicted from elastic constants.
-Actual rock before/after scalar rotation data are imported separately. A speed
 ratio is not e_t. Full contact-frame motion and independent parameters are still
 needed before a quantitative full-state rock validation.

No all13-case accuracy baseline or≥2x gate is claimed. Shape-dependent coefficients
require measured evidence separate from shape/inertia effects and material fitting.
