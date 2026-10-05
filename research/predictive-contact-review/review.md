# Angular momentum at separated friction-contact endpoints

The native rows conserve linear momentum but do **not** conserve the total angular momentum of two isolated finite bodies when a noncentral impulse is applied at two separated contact endpoints. A residual-converged Coulomb solve and decreasing kinetic energy do not detect this defect. This diagnostic establishes the row-level mechanism; it also reproduces the defect in the published adapter for penetrating sphere and box contacts. It does **not** establish that the adapter currently creates CCD predictive contacts.

## Mechanical identity and the user's wrench model

Let $x_A,x_B$ be physical centres of mass and $c_A,c_B$ the world-space points used by a contact row. The impulse on A is $p$ and that on B is $-p$. With no additional couple, their angular increments about their own centres are $(c_A-x_A)\wedge p$ and $-(c_B-x_B)\wedge p$. Therefore, about any fixed inertial origin,

$$
\Delta P_{\rm total}=0,\qquad
\Delta L_{\rm total}=(c_A-c_B)\wedge p.
$$

For ordinary surface contact endpoints, $c_A-c_B=g\hat n$, where signed $g$ is the gap (negative in penetration). The normal impulse is central, while tangential impulse produces

$$
\Delta L_{\rm total}=g\hat n\wedge p_t,
\qquad |\Delta L_{\rm total}|=|g|\,|p_t|.
$$

This result is independent of the origin because total linear momentum is conserved. No external force, driven wall, torque, damping, anisotropic gyroscopic approximation, or split-position correction is present in the diagnostic.

The user's common-point generalized impulse, including an independent angular impulse $\ell$, conserves both momenta:

$$
\begin{aligned}
\Delta p_A&=p,&\Delta L_A&=(c-x_A)\wedge p+\ell,\\
\Delta p_B&=-p,&\Delta L_B&=-(c-x_B)\wedge p-\ell.
\end{aligned}
$$

An equal-and-opposite independent couple alone does **not** repair separated endpoints: its total contribution is zero. Transporting the wrench to the shared point changes each angular Jacobian. If retaining the separate endpoint representation, equivalent transport terms are

$$
\ell_A=(c-c_A)\wedge p+\ell,
\qquad
\ell_B=-(c-c_B)\wedge p-\ell.
$$

These terms include lever-arm transport; they are not an added material rolling or twisting friction law. Updating only angular writeback would leave the contact mobility and impulse solve inconsistent with the resulting kinetic energy. A candidate correction must use common-point lever arms throughout Jacobians, mobility, contact velocities, force application, and wall-work accounting.

## Numerical evidence

The frozen baseline is project commit `0f17e758c731da1cf7d32f4e5f7ba6233d539951`, with Bullet 3.25 commit `2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5`. The native diagnostic supplies one manifold directly to the real project Coulomb solver; collision discovery is bypassed and the manifold's processing threshold is explicitly permissive. The pair has two spheres of radius 0.1 m, mass 1 kg each and isotropic inertia 0.004 kg m². Initial velocities are $(1,0,-1)$ and $(-1,0,1)$ m/s, with zero initial spin, gravity and other applied forces. The step is 0.01 s; restitution is zero; pair friction is varied over 0, 0.4 and 1. The circular-contact residual tolerance is $10^{-10}$ m/s, iteration budget 4096, position stabilization disabled, contact slop $10^{-9}$ m. There is no material compliance or independent contact couple.

For separate sphere surface endpoints, mobility is $\operatorname{diag}(7,7,2)$ kg$^{-1}$ in tangential/tangential/normal coordinates. With $V=1$ m/s, $m=1$ kg, $R=0.1$ m and $I=0.004$ kg m², outside the geometric slop,

$$
p_n=\max\left(0,\frac{2V-g/h}{2/m}\right),\qquad
p_x=-\min\left(\frac{2V}{2/m+2R^2/I},\mu p_n\right).
$$

Within positive slop the declared adapter treats the normal contact as touching. The following observed values are for pair friction 1:

| Representation | Signed gap (m) | Impulse on A (N s) | Total angular increment along y (kg m²/s) | Final kinetic energy (J) |
|---|---:|---|---:|---:|
| Separate endpoints | -0.001 | (-0.285714, 0, 1) | +0.000285714 | 0.714286 |
| Separate endpoints | 0 | (-0.285714, 0, 1) | 0 | 0.714286 |
| Separate endpoints | +0.001 | (-0.285714, 0, 0.95) | -0.000285714 | 0.716786 |
| Shared midpoint | +0.001 | (-0.283683, 0, 0.95) | $2.78\times10^{-17}$ | 0.718817 |
| Separate endpoints | +0.010 | (-0.285714, 0, 0.5) | -0.002857143 | 0.964286 |

Initial kinetic energy is 2 J. Every solve is passive. The shared-point comparator changes tangential mobility, explaining its different impulse; it is not an implemented or validated engine correction.

All 42 supplied-manifold cases are retained, including negative, zero and positive gaps, zero-friction and inactive-contact controls, and the shared-point comparator. The endpoint identity matches measured full orbital-plus-spin angular momentum within $4.61\times10^{-17}$ kg m²/s. Linear momentum error is exactly zero. Shared-point angular momentum error is at most $2.78\times10^{-17}$ kg m²/s.

Thirty additional full-engine cases use the same isolated velocities and zero-force setup, two spheres or two 0.2 m cubes, signed gaps -0.001, 0, $10^{-10}$, 0.001 and 0.01 m, and the same three friction coefficients. Cubes have volume density 125 kg/m³, mass 1 kg and inertia $1/150$ kg m² per axis. The production velocity-only profile runs one internal step. A 1 mm overlap produces angular error +0.000285714 kg m²/s for the spheres and +0.000400000 kg m²/s for the cubes at pair friction 1. Full world-inertia tensors and orbital momentum are used, including the final orientations. Coincident contact and frictionless controls conserve angular momentum.

In all positive-gap full-engine cases the initial contact is **not generated**, impulse is zero, and both momenta are conserved. These zero results are retained; supplied positive-gap solver results must not be represented as collision-discovery tests.

## Source interpretation and scope

Bullet's `convertContact` uses `cp.getPositionWorldOnA()` and `cp.getPositionWorldOnB()` separately, then subtracts each body's centre to construct both normal and friction lever arms. `setupContactConstraint` subtracts $g/h$ from the normal closing-velocity target for positive distances. The adapter's small-positive-gap slop correction changes this target; it does not make the two points coincide. These source facts agree with the measured increments.

The sphere–sphere narrowphase explicitly generates no new positive-gap contact when its closest-point distance threshold is zero. The adapter does not call `setCcdMotionThreshold` or `setCcdSweptSphereRadius`; the default rigid-body CCD threshold is zero, disabling Bullet's predictive-sweep creation. Thus the synthetic positive-gap manifold isolates a potential row-level defect and must remain separate from demonstrated production overlap defects.

Bullet's separate CCD code constructs predictive point A at the current body centre and point B at its predicted centre-at-hit. Their displacement need not align with the contact normal. The algebra then permits a net couple even for a normal impulse. This is a **source-level prediction**, not a measured CCD runtime result in this archive. It is not a claim that the adapter currently exposes that path.

The common-point representation is a candidate conservation improvement, not proof of a calibrated contact model. Speculative gap impulses already approximate events before the actual physical contact time. Additional assumptions are needed for where to place the shared point and how to account for motion and position stabilization. A coupled elastic or finite-patch model can carry additional physical angular momentum, but the current two-rigid-body state has no hidden deformation degrees of freedom to absorb the observed error.

For future accuracy gates, bound cumulative angular error by $\sum_k |g_k|\,|p_{t,k}|$ over ordinary separated endpoints, and measure the actual total momentum. A smaller timestep alone cannot remove a deliberately fixed initial overlap. This study establishes neither a long-time convergence order nor that gap leakage explains the separate 27-sphere shaking branch anomaly.

## Verified references and reproduction

- Bullet pinned source: [contact conversion and friction lever arms](https://github.com/bulletphysics/bullet3/blob/2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5/src/BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp), functions `convertContact`, `setupFrictionConstraint`, `setupContactConstraint`.
- Bullet pinned source: [sphere–sphere discovery](https://github.com/bulletphysics/bullet3/blob/2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5/src/BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.cpp), `processCollision`.
- Bullet pinned source: [CCD predictive construction](https://github.com/bulletphysics/bullet3/blob/2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5/src/BulletDynamics/Dynamics/btDiscreteDynamicsWorld.cpp), `createPredictiveContactsInternal`.
- Stewart, D. E. (2000), *Rigid-Body Dynamics with Friction and Impact*, SIAM Review 42:3–39. [DOI 10.1137/S0036144599360110](https://doi.org/10.1137/S0036144599360110). Crossref metadata and abstract verified. This is context for unilateral Coulomb contact and its mathematical limitations; it does not establish the specific native defect measured here.
- Baraff, D. (1994), *Fast contact force computation for nonpenetrating rigid bodies*, SIGGRAPH. [DOI 10.1145/192161.192168](https://doi.org/10.1145/192161.192168). Crossref title verified. Conventional contact-force literature is background; this diagnostic does not assert a new material friction law or novelty from the momentum identity.

Run `python -m research.predictive_contact_review` from the repository root after the documented native build exists and the frozen `/tmp/fast-shake-diagnostic-runner` has been restored if necessary. The script extracts the frozen adapter and headers from the existing source archive, checks them against the recorded published commit, compiles the independent harness against the pinned Bullet libraries, reruns all cases, and checks the momentum identity and controls. `execution-source.zip` stores the adapter, contact headers, harness, diagnostic script and inspected Bullet source files. `provenance.json` records library/executable/source hashes, compiler flags and numerical versions. All raw cases, full scenes, summaries and artifact hashes are retained. No native solver or material-law file was edited for this review.
