The signed-gap position rule does not by itself guarantee nonpenetration after
physical motion and numerical pose repair are combined. This is a first-order
mechanics limitation demonstrated by the algebraic control in this directory.
It is **not an observed containment failure in the frozen six-run study** and
does not retroactively change that study's qualification rules.

The reviewed integration is frozen source
`108a9bb4c7899f75d760b27b179cc56557904a08`. Its separated position target uses
the start-of-step available clearance, without subtracting motion already
allocated by the accepted physical contact solve. The source and units below
identify the two separate operations.

| Quantity | Meaning | Units |
| --- | --- | --- |
| d | Signed manifold distance, positive for separation | m |
| s | Declared contact slop | m |
| h | Accepted internal time step | s |
| G = d − s | Start available clearance for a separated row d > s | m |
| u* | Final accepted physical normal separation velocity at the shared point | m/s |
| a | Signed normal separation velocity from translation-only pose repair | m/s |
| q | Numerical normal push impulse | kg m/s |
| Apos | Translation-only normal mobility matrix | 1/kg |

Positive normal velocity opens the contact. With fixed current normal and
contact-point Jacobians, the first-order combined available clearance is

```text
G_after = G + h (u* + a) + higher-order geometry terms.
```

The existing separated pose rule requires only `a ≥ −G/h`. A speculative
physical normal row also allows some closing motion over the step. Consequently
those two individually admissible motions can consume the same clearance.
Nonnegative numerical pressures do not imply positive separation velocity at
every contact: an impulse opening one face can close an opposing face through
the off-diagonal entries of the unchanged position mobility.

The algebraic control uses one translating mass between opposing flat faces.
There is no angular motion, frictional slip or geometric truncation error. Its
two normal axes are +x and −x; the unit-mass position mobility is
`[[1, −1], [−1, 1]]`. This matrix is positive semidefinite with a pressure gauge.
The control uses the study's 1 nm declared slop, a 1 ms illustrative step,
1 mm right-face available clearance and 0.5 mm left-face penetration.

The prescribed final physical velocity is +0.95 m/s, already admissible at both
normal rows. Physical normal impulses are exactly zero. A left-face ERP of 0.2
requires a numerical +0.1 m/s push, also admissible under the independent
signed-gap position rule. The numerical pressure vector is `[0.1, 0]` kg m/s;
both numerical complementarity residuals are zero. Combined motion consumes
1.05 mm against 1 mm of available clearance. The final right-face distance is
approximately −50 micrometres. No native engine or stored accepted answer is
used: this is an explicit admissible first-order row example, rather than an
engine execution claim.

For a future declared protocol, the remaining-clearance expression at separated
rows is

```text
G_physical = G + h u*
b_pose_remaining = −G_physical / h = −G/h − u*.
a ≥ b_pose_remaining.
```

`G_physical` must remain signed. Clamping a negative predicted clearance to zero
would hide a physical-motion overshoot instead of requiring its repair. The
current target is assembled before the physical solve, so replacing it would
require constructing this target *after* the accepted physical impulses are
known. It is not justified to substitute the pre-contact velocity or the last
output-frame velocity for `u*`.

Keeping all penetrating targets unchanged while tightening separated targets
can itself make a translation-only repair infeasible. In the control, the
right remaining-clearance bound allows only +0.05 m/s numerical x push, while
the unchanged left target still demands +0.1 m/s. Yet physical motion alone
has already resolved the left overlap. A separate prospective alternative
would apply a desired **combined** geometric rate to all rows, subtracting the
accepted physical rate from each numerical target. For example, a penetrating
desired rate `k = ERP × (−d)/h` would give `b_pose = k − u*`. This also changes
the penetrating discretization and must be declared separately; it is not part
of the current signed-gap study.

The shared normal Jacobian uses both physical translation and rotation:

```text
u* = nA · vA* + (rA × nA) · omegaA*
   + nB · vB* + (rB × nB) · omegaB*.
```

The signed B row already incorporates the opposing impulse direction.
Translation-only pose repair uses only its linear Jacobian and suppresses fake
angular pose motion. Physical angular velocity must nevertheless remain in
`u*`; dropping it would miss normal approach from rotating bodies or walls.
The shared contact point changes tangential lever arms. For exact manifold
endpoints differing only along the contact normal, moving either endpoint to
that shared point leaves the corresponding *normal* angular Jacobian unchanged:
the extra lever-arm component is parallel to the normal and its cross product
with the normal is zero. Numerical endpoint defects and changing normal/shape
geometry remain separate issues.

Even the remaining-clearance expression supplies only a first-order condition.
Finite orientation changes, curved surfaces, changing normals and a changing
contact set require geometry requery and a bound on neglected motion; rapid
rotation is particularly relevant here. A travel guard reduces motion but does
not certify continuous nonpenetration or an error order. Reusing a stale
manifold after a failed update is not independent geometric validation.

The implementation sequence is concrete:

1. `shared_contact.h:50` transports contact rows to the declared shared point
   before matrix assembly; `contactRowFreeVelocity` includes external force
   velocity and the normal-row external angular contribution.
2. `coulomb.h:325` prepares the physical and split targets. The separated gap
   target is assigned before `coulombSolve` accepts physical `m_x`.
3. `coulomb.h:345` assembles the linear-only position matrix and solves it with
   the separate `m_bSplit` target.
4. Upstream `btMLCPSolver.cpp:570` calls this virtual solve, then applies accepted
   physical and push impulses to solver bodies. Thus final accepted `m_x` exists
   in the override before ordinary solver-body physical delta writeback.
5. `coulomb.h:294` records the pose ledger after that application and clears
   position turns. The recorded linear physical velocity includes delta contact
   velocity and external force velocity.
6. Upstream `btSolverBody.h:263` applies push displacement to the pose;
   `btSequentialImpulseConstraintSolver.cpp:1808` writes the physical velocities
   and corrected pose to the rigid body. `btDiscreteDynamicsWorld.cpp:480` solves
   constraints before `:486` integrates physical transforms. The runner's
   start-phase kinematic walls advance by their prescribed motion after
   `world.stepSimulation` (`runner.cpp:171` at this frozen source).

Upstream Bullet normal-row construction explicitly subtracts positive distance
over h from its physical velocity target (`btSequentialImpulseConstraintSolver.cpp:949`).
`runner.cpp:73` sets the split penetration threshold to zero; default split ERP
is 0.2 and Bullet's own linear slop is zero in `btContactSolverInfo.h:87,98`.
The repository's `spatial_backend/CMakeLists.txt:26` pins upstream Bullet to
`2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5` with a declared tarball SHA256.
`source-pointers.json` records the inspected files' byte hashes and precise
function/line pointers, including local third-party bytes. This note does not
claim a new rebuild or runtime attribution beyond the study's saved provenance.

The old captured 74-row position geometry is insufficient to measure the actual
remaining-clearance error. Its observer saves base solver/body velocities and
external increments, but does not save accepted physical `m_x`, all applied
warm-start impulses or the solver delta velocities. The corresponding matrix
snapshot contains **position** pressure, not the accepted physical pressure.
No calculation on that old capture is presented as actual final `u*`. A physical
velocity rejection candidate likewise must not be treated as accepted motion.
Full output-frame states cannot identify an unrecorded internal contact solve.

If a future remaining-clearance linear program has no feasible translation-only
correction, use a bounded transactional rollback/refinement protocol or reject
the update. Rollback must restore body poses and velocities, force/torque and
interpolation state, all contact manifold/cache and warm-start state, and
kinematic schedule progress before retrying a smaller step. Geometry and all
accepted physical impulses must be recomputed at that smaller step; complete
failure evidence and bounded retry counts must remain visible. This is a future
design suggestion, not implemented or retrospectively substituted here.
