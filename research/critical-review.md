# Critical research review: coupled frictional impacts and coarse heterogeneous bodies

Prepared 2026-10-04. This is a research assessment, not a claim that the proposed method has been validated. I inspected `rigid-body-friction-model/friction-model.tex` and `rigid-body-combined-notation/velocity-V-impulse-P.tex`. References below were checked through Crossref/OpenAlex metadata and, where stated, a publisher abstract or public technical documentation. A metadata match proves that a work exists, not that every equation in it matches this proposal.

## Shared verdict after exchange with the publication advocate

**The present formulation should not be submitted as a novel general collision theory.** The compact velocity/impulse notation, contact mobility matrix, energy quadratic, normal restitution, static/dynamic friction distinction, rolling resistance, and simultaneous-contact Jacobian are established mechanics. Existing research also directly combines tangential restitution and friction and studies energy anomalies of frictional restitution.

**A narrower research project remains worth pursuing:** adaptive heterogeneous contact patches or rigid blocks with a small number of compliant interface/internal states, a representation-invariant energy-consistent solver, and demonstrated improvement in bulk impulse/outgoing translation/spin per computational cost. This is a hypothesis about a possible contribution, not established novelty or measured superiority. The advocate and critic agree: continue investigation; do not publish novelty or accuracy claims until a concrete method and held-out benchmarks exist. The critic does not need to reject useful pedagogical exposition; teaching the algebra well is a different contribution from a new mechanics method.

## What is correct in the existing mathematics

The proposed wedge convention consistently maps a contact linear impulse to center-of-mass angular impulse, with its transpose mapping angular velocity to linear velocity at the contact. The full contact response includes translational, rotational, and cross-coupling blocks. With positive masses and positive-definite inertias, a single unrestricted pair contact carrying a full force/couple impulse has a symmetric positive-definite mobility matrix. Scaling angular velocity by a fixed length and angular impulse by its reciprocal preserves work and energy. These are useful choices for exposition, but are changes of representation rather than new physics.

For a network of contacts, let U contain scaled center-of-mass velocities and spins, H contain body mass and scaled inertia blocks, and G map U to relative contact velocities. The standard contact mobility is K = G H⁻¹ Gᵀ; impulses update U by H⁻¹GᵀJ. The global kinetic-energy change is V⁻·J + ½JᵀKJ, where V⁻ = GU⁻. This formulation automatically includes contact coupling through shared bodies. It preserves internal total linear/angular momentum when G uses equal/opposite actions at the same world contact location, including equal/opposite free couples.

Important limits: K is generally positive semidefinite, **not necessarily invertible**, for many contacts. Distinct contacts between the same body pair must remain distinct edges. A single antisymmetric pair linear-impulse sum loses lever arms, moments, and material-dependent directions. Pairwise energy tests cannot replace the global energy calculation because contact impulses have cross terms.

## Physical objections that the final model must resolve

### 1. Independent restitution targets can inject energy

The draft correctly identifies this. For a coupled two-channel example, K⁻¹ = [[1, 0.9], [0.9, 1]], incoming velocity (−1, 1), and restitution diag(1, 0) give outgoing (1, 0), impulse (1.1, 0.8), and kinetic-energy increase 0.4. A sufficiently large static coefficient accepts the impulse; capacity does not fix the energy defect. Every restitution value being between zero and one is insufficient. Stronge's work [1, 2] establishes this broader problem decades earlier.

### 2. Local restitution targets can be mutually impossible

Consider a planar rigid body with three upward-normal bottom contacts at x = −1, 0, 1. Their normal velocities always satisfy u_middle = (u_left + u_right)/2. Incoming normal speeds all equal −1. Local choices e_left = e_right = 0 and e_middle = 1 request outgoing speeds (0, 1, 0), contradicting that kinematic identity. No impulse solver can satisfy these targets. More contacts do not automatically provide more independent rigid-body velocities.

Thus heterogeneous material labels cannot be translated into arbitrary exact simultaneous Newton restitution constraints. Finite compliance, sequential contact timing, or an explicitly relaxed/feasible global impact law must decide the outcome. Liu, Zhao and Brogliato [10] already study multiple-impact redistribution through local compliance/energetics.

### 3. Static friction is a capacity; elastic reversal is a constitutive process

A static inequality bounds the contact action. It does not prescribe restitution or prove that an entire finite collision can remain stuck. Even when the *total* tangential impulse fits a static impulse cone, intermediate tangential force can exceed the instantaneous normal-force-dependent limit. Tangential rebound can reflect stored and released tangential elasticity, partial slip and changing contact load. Maw, Barber and Fawcett's oblique-impact work [4] predates this proposal and supplies experimental evidence that tangential compliance changes rebound angles near the friction angle.

Positive tangential/rolling restitution may be useful as an effective endpoint approximation, but call it rebound, not final sticking. A final nonzero reversed tangential speed is not a sticking final state. A purely passive friction force opposes instantaneous slip; it cannot by itself supply elastic restitution. Contact-memory variables or an explicit compression/restitution process are needed for sustained or repeated elastic contacts.

### 4. Direct angular impulse needs a finite patch or an explicit contact law

An ideal geometric point with a force cannot transmit a free couple. A free rolling/twisting couple stands for a finite pressure/friction patch, deformation, adhesion/bonding, or a phenomenological rolling-resistance law. For arbitrary 2D shapes the out-of-plane couple is a patch moment, not automatically the rolling law of a circular particle.

Rolling-resistance length κ has units of length; writing κ = μ_roll a needs a physical contact/radius/resistance scale a. The arbitrary representation length ℓ is not that material scale. In 3D sliding, twisting and rolling occupy different subspaces; one initial tangential direction cannot capture coupling-generated orthogonal slip. Simultaneous independent force/moment capacity bounds may also overestimate the wrench realizable by one finite patch. Patch size, pressure support, and local friction impose shared wrench constraints. Review [6] shows that rolling models are already a developed, nonunique subject.

### 5. A final-slip Coulomb formula does not recover the whole impact path

The draft explicitly labels its final-slip convention as an idealization, which is appropriate. An integrated impulse opposed to the final slip can disagree with instantaneous Coulomb friction when slip halts or reverses. Distinguish a defined algebraic endpoint law from a force law integrated through compression and restitution. Do not claim that static/dynamic switching plus an endpoint sign rule uniquely reproduces authentic impacts.

### 6. Kinetic passivity is conditional on initial internal energy

For two initially undeformed passive bodies with no external impulse, nonincrease of kinetic energy is a sound requirement for a reduced instantaneous map. If a coarse block model retains prestrain or vibration, stored elastic energy can become kinetic energy during impact. Then it is **total mechanical energy** that must be tracked. Rejecting every kinetic-energy increase would incorrectly suppress legitimate release of stored energy. External forces/work and contact creation/deletion also need consistent accounting.

## Assessment of an energy-constrained impulse projection

Root proposed retaining the normal target and projecting a full restitution target onto convex friction bounds plus the energy inequality. For a single approaching normal channel with normal e in [0, 1], a normal-only impulse supplies a feasible passive endpoint, so a constrained optimization is plausible. This can define a reproducible phenomenological contact law.

But passivity is necessary, not sufficient to establish material authenticity. The projection's objective determines *which* passive response occurs; a generic Euclidean norm changes when ℓ changes. Use an explicitly covariant metric, such as a mobility-derived velocity metric on the attainable subspace, or prove length invariance. Singular multi-contact K needs a quotient/pseudoinverse or a body-velocity objective. Strict convexity cannot be inferred from semidefinite K. Uniqueness of outgoing body velocities does not imply uniqueness of every contact impulse.

A convex bound using static μ does not implement kinetic saturation at a distinct smaller dynamic μ. Branch selection introduces extra constitutive choices and may be nonconvex. The projection should be described as a **proposed bounded dissipative impulse law**, compared against maximum-dissipation, energetic restitution and compliant references. It should not inherit the label Coulomb automatically, or be advertised as novel before closest-prior-art review. For simultaneous impacts, even retaining all normal Newton targets can destroy feasibility, as the three-contact example shows.

## Coarse heterogeneous bodies: what is plausible, what is not

Assigning local material/contact parameters to an otherwise rigid exterior changes its boundary response without introducing internal deformation. Dividing it into pieces rigidly welded together still leaves one rigid kinematic body. Allowing pieces to move independently produces a different physical object unless interfaces provide stiffness, damping, moment transfer, and possibly failure. Such a model is related to discrete/bonded elements and discontinuous deformation analysis [7, 8, 9], not a general alternative newly discovered here.

The strongest directly competing coarse-contact example is Elandt et al.'s pressure-field/hydroelastic approximation [11]. Its public abstract explicitly advertises continuous contact wrenches on coarse meshes and speed compared with elasticity-theory models. Drake documentation exposes contact/material parameters and distributed contact modeling. Consequently “coarse cells rather than fine FEM” is too broad a novelty claim.

Large cells might be accurate for **selected bulk outputs** when missing deformation and wave modes have little influence, or when a few interface/modal states summarize them. This does not imply accurate stress, peak force, contact duration, damage, or vibration. Restitution is an effective property of a colliding system and contact process; there is no reason a fixed location-dependent value transfers to every speed, orientation, thickness, backing constraint, preload, or history.

## Strongest falsifier and minimum validation

**Structural falsifier of the memoryless model:** construct two impacts with identical represented exterior contact state (location, normal, masses, inertias, velocities, spin, and local coefficients), but different unresolved prestrain, internal vibration, or tangential contact history. A memoryless local map predicts identical outgoing motion; an adequately resolved compliant reference can predict different outgoing translation/spin. This defeats a universal memoryless coarse-model claim. It does not defeat a revised model that retains the missing interface/modal state—the advocate correctly made that distinction.

**Practical falsifying experiment:** oblique impacts on a thin heterogeneous beam/block, at held-out locations and spin, with contact duration comparable to wave transit time; follow with two-contact simultaneous impacts and a three-body collision chain. Freeze calibrated coefficients before evaluation. Compare outgoing translation, spin, total impulse, contact sequence and energy. If the proposed reduction does not match these bulk outputs within a declared tolerance at lower cost than existing reductions, its proposed advantage fails in that regime.

Minimum paper-worthy evidence:

1. Define a distinct algorithm/constitutive reduction and which observable it targets; do not substitute notation for contribution.
2. Prove momentum consistency, objectivity, representation-length invariance, well-defined feasible update, and relevant total-energy consistency, with singular/dependent multi-contact cases handled.
3. Compare to standard impulse and energetic/compliant models, hydroelastic/coarse-patch contact, and bonded/reduced flexible-body alternatives appropriate to the application.
4. Use a converged fine compliant reference and ideally measured impacts; calibrate on a separate training set and quantify held-out errors.
5. Report runtime/error tradeoffs over refinement and increasing body/contact counts, including failures and parameter-identification uncertainty. More parameters can fit training data without adding predictive accuracy.

An illustrative 1D calculation is now available, described below. It does not supply the 2D frictional-impact comparison data requested here. Neither reviewer has established a publishable superiority result. An adaptive heterogeneous interface/patch selection rule with an error estimator and demonstrated held-out accuracy/cost advantage is a defensible **candidate** contribution; whether it is actually novel still requires a closer review of reduced contact/interface mechanics.

## Newly generated 1D evidence and its limits

Root generated a reproducible layered unit-rod experiment in `benchmarks.py` and `coarse-rod-results.csv`; I inspected the reported CSV and algorithm descriptions. It uses unit length, density and area, modulus 1 in the left half and 4 in the right, a Gaussian prescribed left-end force of approximately unit integrated impulse, lumped masses/linear springs, and velocity-Verlet integration. This is an externally loaded elastic chain, **not a collision/friction experiment**. The fine comparator has 2048 elements, with a 1024-element reference-resolution check.

At the same coarse resolution of 16 elements:

| Gaussian pulse width | Right-end velocity relative L2 error | Final energy relative error | 1024/2048 velocity discrepancy |
|---|---:|---:|---:|
| Broad, σ = 0.25 | 1.3345% | 0.4043% | 0.0002393% |
| Sharp, σ = 0.015 | 107.5259% | 23.9313% | 1.3513% |

The sharp case's reference check is sufficient to distinguish order-one coarse error from fine-resolution discrepancy; it is not enough to claim a subpercent converged reference for that case. The broad result supports the restricted proposition that coarse cells can preserve selected smooth, low-bandwidth responses. The sharp failure is counterevidence to a universal coarse-cell accuracy claim. Final energy error is a reference-relative accuracy metric, not evidence that elastic Verlet dynamics dissipate energy correctly as a collision law.

The reported node-update reductions compare with a very fine explicit reference; the single-run runtime ratios do not establish superiority over an appropriately resolved finite-element, hydroelastic, bonded-element or reduced-mode solver. This experiment is valuable regime evidence and a reproducibility starting point. It establishes neither new physics nor novelty, nor arbitrary 2D/3D contact accuracy.

## Checked references and evidence boundaries

1. **W. J. Stronge (1990).** “Rigid body collisions with friction.” *Proceedings of the Royal Society A* **431**, 169–181. https://doi.org/10.1098/rspa.1990.0125. **Publisher abstract verified via Crossref and OpenAlex.** It explicitly discusses energetically consistent collision theory, slip halting/reversal, and erroneous energy increase under a Newton-based treatment. This verifies prior art on the central energy concern; it is not a universal proof that every implementation of a Newton target fails.

2. **W. J. Stronge (1991).** “Unraveling Paradoxical Theories for Rigid Body Collisions.” *Journal of Applied Mechanics* **58**, 1049–1055. https://doi.org/10.1115/1.2897681. **Crossref metadata and publisher abstract verified.** The abstract separates normal-work compression/restitution from frictional dissipation and warns of Newton-law energy anomalies when slip reverses.

3. **Y. Wang and M. T. Mason (1992).** “Two-Dimensional Rigid-Body Collisions With Friction.” *Journal of Applied Mechanics* **59**, 635–642. https://doi.org/10.1115/1.2893771. **Metadata and publisher abstract verified.** Routh processes, contact-mode classification, and Newton/Poisson comparisons are prior art. The abstract advocates Poisson; Stronge's papers state additional limitations. Do not cite this disagreement as a settled universal guarantee for every Poisson model.

4. **N. Maw, J. R. Barber and J. N. Fawcett (1981).** “The Role of Elastic Tangential Compliance in Oblique Impact.” *Journal of Lubrication Technology* **103**, 74–80. https://doi.org/10.1115/1.3251617. **Metadata and publisher abstract verified.** Abstract reports significant rebound-angle effects near friction angle and agreement with steel/rubber experiments. Their earlier “The oblique impact of elastic spheres,” *Wear* **38** (1976), 101–114, is metadata-verified: https://doi.org/10.1016/0043-1648(76)90201-5. The spherical setting is evidence about contact compliance, not validation of arbitrary-body shapes.

5. **A. Doménech-Carbó (2014).** “On the tangential restitution problem: independent friction–restitution modeling.” *Granular Matter*. https://doi.org/10.1007/s10035-014-0507-3. **Springer abstract directly read.** It describes independent tangential restitution/friction, stick/gross-slip regimes, the same normal/tangential/friction coefficients and literature-data comparisons, for a homogeneous sphere against a massive wall. An erratum exists: https://doi.org/10.1007/s10035-014-0538-9 (**metadata verified**). The 2023 extension is “On the friction/tangential restitution problem: Independent friction-restitution modeling of sphere rebound with arbitrary spin,” *Powder Technology*, https://doi.org/10.1016/j.powtec.2022.118141 (**metadata verified**). This anticipates the general modeling ingredients; full equation-level identity is not claimed without reading the corrected full articles.

6. **J. Ai, J.-F. Chen, J. M. Rotter and J. Y. Ooi (2011).** “Assessment of rolling resistance models in discrete element simulations.” *Powder Technology* **206**, 269–282. https://doi.org/10.1016/j.powtec.2010.09.030. **Crossref title/authors/pages/references verified; full article not read.** This establishes substantial prior work on rolling resistance; it does not prove this draft's exact rolling law is valid. Crossref lists older rolling-resistance, rotation and bonded-particle models.

7. **P. A. Cundall and O. D. L. Strack (1979).** “A discrete numerical model for granular assemblies.” *Géotechnique* **29**, 47–65. https://doi.org/10.1680/geot.1979.29.1.47. **Crossref/OpenAlex metadata and Crossref publisher abstract verified.** The abstract describes contact-by-contact, particle-by-particle explicit mechanics and validation against photoelastic force-vector plots. This is foundational discrete-element prior art; it is not a quantitative coarse-continuum error bound.

8. **D. O. Potyondy and P. A. Cundall (2004).** “A bonded-particle model for rock.” *International Journal of Rock Mechanics and Mining Sciences* **41**, 1329–1364. https://doi.org/10.1016/j.ijrmms.2004.09.011. **Crossref/OpenAlex metadata verified.** Relevant prior art for internally connected blocks/particles and material behavior; full paper not read here.

9. **G. H. Shi (1992).** “Discontinuous deformation analysis: a new numerical model for the statics and dynamics of deformable block structures.” *Engineering Computations* **9**, 157–168. https://doi.org/10.1108/eb023855. **Crossref/OpenAlex metadata and Crossref publisher abstract verified.** Abstract describes deformable block systems, contact/block equilibrium, no-tension/no-penetration constraints and static/dynamic Coulomb law. This is relevant block-method prior art, not a complete comparison of its constitutive assumptions.

10. **C. Liu, Z. Zhao and B. Brogliato (2008/2009).** “Frictionless multiple impacts in multibody systems. I. Theoretical framework.” *Proceedings of the Royal Society A* **464**, 3193–3211. https://doi.org/10.1098/rspa.2008.0078. Part II, “Numerical algorithm and simulation results,” **465**, 1–23, https://doi.org/10.1098/rspa.2008.0079. **Crossref metadata and publisher abstracts verified.** Part I explicitly relates kinetic-energy evolution to relative contact stiffness and stored contact potential energy, with local energetic restitution coefficients. Part II describes impulse-scale integration and Newton's-cradle/Bernoulli examples. These concern frictionless multiple impacts; they do not validate the proposed full frictional/rolling law. **W. Yao, B. Chen and C. Liu (2005)**, “Energetic coefficient of restitution for planar impact in multi-rigid-body systems with friction,” *International Journal of Impact Engineering* **31**, 255–265, https://doi.org/10.1016/j.ijimpeng.2003.12.007, is metadata-verified frictional energetic prior art.

11. **R. Elandt, E. Drumwright, M. Sherman and A. Ruina (2019).** “A pressure field model for fast, robust approximation of net contact force and moment between nominally rigid objects.” *IROS 2019*. https://doi.org/10.1109/IROS40897.2019.8968548; public preprint https://arxiv.org/abs/1904.11433. **Advocate checked Crossref and public abstract; critic's contribution uses that exchanged evidence.** Coarse-mesh distributed wrench/contact approximation directly narrows the claimed advantage. Implementation/background: https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html. These do not prove the proposed adaptive reduction cannot improve a specific task.

12. **M. Anitescu and F. A. Potra (1997).** “Formulating Dynamic Multi-Rigid-Body Contact Problems with Friction as Solvable Linear Complementarity Problems.” *Nonlinear Dynamics* **14**, 231–247. https://doi.org/10.1023/A:1008292328909. **Crossref/OpenAlex metadata verified.** **D. E. Stewart (2000)**, “Rigid-Body Dynamics with Friction and Impact,” *SIAM Review* **42**, 3–39, https://doi.org/10.1137/S0036144599360110, metadata-verified survey. These establish that global multi-body contact solving is not by itself a new contribution.

13. **R. Featherstone (2008).** *Rigid Body Dynamics Algorithms*. Springer. https://doi.org/10.1007/978-1-4899-7560-7. **Crossref metadata verified.** General rigid/spatial-vector mechanics background for the stacked velocity/wrench representation.

## Publication decision criteria

At present the critic's argument is stronger against **submission now**, because a specific new method and validation data are absent and close prior art exists. The advocate's argument is stronger for **continuing a focused research project**, because adaptive heterogeneous reduction may have a useful accuracy/cost niche that the current review has not disproved. This is an agreed evidence-based verdict, not a declared victory for either side. A later publication decision should follow a concrete method, closest-prior-art comparisons and held-out evidence rather than the preference to publish.

## Final technical review of the finite compliant reference

I additionally inspected `compliant_contact.py`, `compliant-results.json` and Section 4.2 of `research-assessment.tex`. The local frozen-frame scalar energy algebra is sound under the stated positive-stiffness/damping assumptions and a symmetric positive-definite mobility matrix. For the sliding branch, with trial force F₀ = −kz − cu and selected force F = μ_dynamic F_n sign(F₀), the implied plastic velocity is w = (F − F₀)/c. Since the branch requires |F₀| > μ_static F_n ≥ μ_dynamic F_n, Fw ≤ 0. Consequently D = c ż² − Fw ≥ 0. Sticking gives w = 0 and D = cu². The same proof applies to the scalar angular channel. The clipped normal law's dissipation expression is also correct on both its active and zero-force unloading branches.

The reported example is a **local effective reference**, not a validated arbitrary-body impact model. Its energy residual approximately 1.27×10⁻⁹ is numerical accounting evidence for that run. Isolated normal restitution calibrated to 0.6 becoming 0.493860 under coupled mobility correctly illustrates that a calibration target need not be the coupled outcome. Energy retained at opening is reported rather than silently deleted.

One material limitation requires explicit disclosure: the implementation tests the static trial threshold afresh at every right-hand-side evaluation; it retains no sliding mode. With unequal static/dynamic coefficients it can chatter near the threshold. For example, take F_n = 1, k = 50, c = 3, μ_static = 0.6, μ_dynamic = 0.4, u = 0.01, z = 0.0114. Then F₀ = −0.6. The static branch has ż = 0.01 and increases |F₀|, whereas just outside the threshold the dynamic branch has ż approximately −0.05667 and decreases |F₀|. The vector fields push toward the switching surface from both sides. Passing eight regression tests does not establish a globally unique classical trajectory or general adaptive-ODE convergence for this discontinuous rule.

A well-defined comparator can use equal friction coefficients, or a deliberate mode/history law with sliding maintained until a specified plastic-slip arrest event, or a documented regularization/differential-inclusion interpretation. The changing mode is part of the constitutive model. Scalar passive branch algebra alone does not settle this issue. I recommended that root explicitly label the implementation's branch selection and avoid claiming general well-posedness or exact static/dynamic Coulomb dynamics. I also recommended input validation for finite symmetric positive-definite 3×3 mobility: an ordinary matrix inverse by itself does not enforce those physical conditions. No root implementation files were edited by this reviewer.

### Follow-up: local transition fix inspected

Root subsequently revised `integrate_contact` to retain elastic or signed plastic-slip modes and integrate piecewise between yield, slip-arrest and opening events. The static capacity is tested only in the elastic mode; a dynamic mode persists until its plastic slip arrests. Equal-coefficient channels retain the continuous baseline. The revised entry point checks finite 3×3 near-symmetric mobility and positive definiteness by Cholesky, and rejects invalid initial velocities. I inspected the revised code and the added invalid-mobility and transition/refinement tests. These changes address the previously identified **memoryless threshold chatter mechanism locally**; that criticism should not be presented as an unresolved defect of the revised test implementation.

The new reported example has four phases and nearly unchanged coupled normal restitution 0.49386034. Its maximum reported energy-accounting residual is now approximately 5.04×10⁻¹⁰. Ten total regression tests include outgoing-velocity agreement between tolerances 10⁻⁹ and 10⁻¹⁰ for the chosen example. I found no further blocking algebraic or event-mode issue in this scoped local reference. These checks remain local numerical evidence, not a general existence/uniqueness theorem for hybrid transitions, arbitrary contacts or rotating geometry. The original physical scope and publication verdict remain unchanged.
