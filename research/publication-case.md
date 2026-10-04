# Evidence-based case for continued research

## Recommendation agreed with the critical reviewer

Do not submit the existing formulation as a new collision theory. The wedge notation, generalized velocity/impulse stacking, contact mobility matrix, coupled friction and restitution, and simultaneous-contact Jacobian are established mechanics. There is no empirical evidence yet that this implementation is more accurate or less expensive than the appropriate alternatives.

Continue investigating a narrower contribution: **adaptive, heterogeneous contact patches or compliant interfaces that preserve mechanical energy accounting and predict held-out bulk impact outcomes at lower cost than competing reduced contact models.** Publication becomes defensible only after a distinct algorithm, theorem, or reproducible accuracy–cost advantage is demonstrated. This is a conditional research recommendation, not a verdict that novelty has already been established.

The advocate and critic explicitly agreed on this wording through direct exchange. The strongest objection won against the current novelty claim; the advocate won support for a revised, testable research question. Neither side agreed to predetermine the result of that test.

## What the literature supports

1. It is physically useful to predict integrated contact force and moment without resolving every continuum deformation degree of freedom. Elandt et al. [1] already propose this, including friction and dissipation, continuous contact wrenches, and coarse meshes. This strongly supports the usefulness of the problem and strongly weakens any claim that coarse contact cells themselves are new.
2. Static friction capacity, dynamic slip, and restitution can be treated together. Aghili [2] explicitly derives a critical static friction coefficient and energy-admissible friction/restitution regions. Consequently the proposed static-capacity and energy tests are sensible but not novel as concepts.
3. Tangential rebound needs contact elasticity/history or an explicitly phenomenological impact law. Oblique elastic impact has been studied since Maw, Barber and Fawcett [3]. It must not be described as an automatic consequence of ordinary static Coulomb friction.
4. Rolling resistance is useful, but choosing a torque bound is only the start of a model. Ai et al. [4] compare existing rolling-resistance models. Their paper supports comparing torque laws and calibration, not assuming that one coefficient generalizes to every contact shape and speed.
5. Reduced deformation models are credible alternatives to fine continuum meshes, but substructuring has a long history: Craig–Bampton [5]. Generalized coarse-grained DEM with variable scale ratios is also an active area [6]. These are comparators, not proof that the current model has their accuracy.

These sources provide plausibility and prior-art boundaries. They provide no measured performance data for the user's method.

## Distinguish two different meanings of subdivision

**Material patches on one rigid body.** Patches share the same pose, translational velocity and angular velocity. They can have different local contact properties. Several simultaneous contacts can be integrated into a net wrench, and patches can distinguish, for example, rubber-coated and bare surface regions. No internal deformation or stress wave appears. Refining such a partition cannot converge to elastic continuum dynamics by itself.

**Compliantly connected coarse bodies.** Cells have independent translations and rotations, connected by interfaces with deformation states. In a corotated/objective interface frame, retain relative extension, shear and rotation as required, a nonnegative stored potential energy, and positive-semidefinite damping. Joint forces and moments must be equal and opposite with consistent transport to body centers. The relevant passive energy is kinetic energy plus interface/contact stored energy. Rebound can release stored elastic energy; it need not dissipate kinetic energy at every instantaneous update.

The second interpretation could approximate low-frequency deformation using fewer unknowns than fine finite elements. It needs interface constitutive laws and appropriate bending/shear modes; attaching normal springs between arbitrarily lumped cells does not establish convergence to a general 2D solid. Preserve total mass, center of mass and inertia, then fit static compliance and selected low-frequency modes before impact calibration. Craig–Bampton or another established reduced flexible-body model should be a baseline.

## Strongest defensible candidate contribution

A tentative claim for testing is:

> A mechanically consistent adaptive patch/interface reduction for heterogeneous impacts can meet prescribed errors in outgoing translation, rotation and integrated contact wrench with fewer retained contact/interface states than uniform coarse discretizations and established reduced contact models, over a stated class of low-deformation impacts.

The distinctive element would have to be the **adaptive reduction and its error control**, not the generalized impulse algebra. For example, split a patch when material contrast, contact geometry or an estimator of omitted interface-mode work predicts an unacceptable error in the target observables; merge only while preserving mass/inertia and energy accounting. This remains a proposed method. No novelty search performed here establishes that this exact combination is new.

Three possible publishable results, each needing evidence, are:

- A bound on the error in integrated wrench or outgoing generalized velocity under explicit bandwidth, contact-patch and deformation assumptions.
- A reproducible adaptive rule that dominates matched-cost homogeneous and uniform patch models on held-out heterogeneous impacts.
- A specific validated application where a small set of physically identifiable contact/interface parameters transfers across location, speed and spin better than an ordinary fitted rigid impact law.

## Complete the physics before comparing it

The existing seven-parameter planar proposal describes requested response and capacities. It is not yet a universally closed contact law. Independent normal, tangential and rolling restitution targets can be incompatible with both friction limits and passive energy. A clipped target is a modeling decision and must state which targets were relaxed.

Two defensible routes are:

**An algebraic phenomenological impact map.** Explicitly specify admissible impulse sets, branch selection, and the objective for compromising incompatible rebound targets. Use the mobility-induced metric, rather than an ordinary norm in arbitrarily scaled generalized coordinates. Verify reference-length invariance. If the resulting impulse is projected onto a convex energy/capacity set, call it a constrained response map; do not label it exact Coulomb friction unless its slip/stick complementarity conditions are actually enforced. A passive endpoint map is useful but need not reproduce intermediate slip reversal, contact duration or force history. Compare with Aghili's established energy-consistent model [2].

**A compliant contact/interface model.** Retain normal compression, tangential elastic slip and rolling-angle history, with stored potentials and dissipative forces/couples. Impose unilateral contact and Coulomb-type yield caps, allowing elastic loading, plastic slip and unloading. Integrate through compression and restitution. Report effective rebound numbers measured from that process rather than prescribing arbitrary independent numbers at every impact. Rolling stiffness, normal/tangential stiffness ratios and damping then supplement the friction coefficients. A small parameter count is helpful only if it can identify the behavior of interest.

Do not invent universal material values. Restitution is usually an effective property of the contacting pair, geometry, incident speed and unresolved dynamics. Ramírez et al. [7] explicitly derive velocity-dependent restitution for viscoelastic spheres; that is an example demonstrating speed dependence, not a constitutive law for arbitrary polygons. Parameters associated with local coatings may need patch/material-pair tables. Roughness, pressure, contact size or elastic memory may require additional state. Friction and restitution should be calibrated with uncertainty and held-out validation.

Finite-patch torque capacity also depends on the patch geometry and pressure/traction distribution. In 3D, sliding and torsional limits may share a wrench budget and cannot automatically be treated as independent scalar boxes. A contact center and an independent couple must be defined consistently to avoid counting a shifted center of pressure twice.

For many contacts the standard assembly is a contact Jacobian mapping body generalized velocities to local contact velocities, with mobility formed by sandwiching the inverse body mass matrix. Shared bodies generate cross-contact terms. A pairwise loop omitting these terms does not model the simultaneous event. Redundant contacts can make the mobility singular, so a general many-contact algorithm cannot simply require its ordinary inverse.

## Tests that could defeat the proposed contribution

Use the same geometry, material inputs and computational budgets for all comparisons. Calibrate on one subset, freeze the parameters, and test on impacts with held-out locations, speeds, angles and incoming spins.

| Question | Benchmark | Failure that would matter |
|---|---|---|
| Is surface heterogeneity useful? | A nominally rigid object with two distinct contact regions, tested near and away from their boundary | One homogeneous calibrated model predicts equally well at less cost |
| Do interfaces retain necessary dynamics? | A compliant heterogeneous beam/block with impact duration varied relative to wave-transit time | Coarse predictions fail when unresolved modes participate; refinement does not repair them |
| Is multiple-contact treatment correct? | A plate striking at two corners; repeat after permuting contact order | Predicted outcome depends on ordering or violates momentum/energy accounting |
| Does the method transfer? | Freeze parameters, then change impact location, speed, incidence and spin | Accuracy requires per-case refitting or extra uncounted parameters |
| Does history matter? | Same exterior contact state but different prestrain/vibration state | Memoryless patch law gives the same output while the reference does not |
| Is the speed gain substantive? | Error-versus-wall-clock curves, including preprocessing/calibration and online integration | Hydroelastic, bonded DEM or standard reduced flexible-body model dominates the proposed model |

The hidden-state test refutes a *memoryless* patch model, not every possible coarse compliant model. A few extra modes/interface states may repair it; their cost and identification burden must then be included.

Use fine-mesh/time-step convergence to verify the numerical reference, and measurements if claiming material accuracy. Report integrated normal/tangential impulses, outgoing center-of-mass velocity and spin, dissipated energy and contact mode classification. Peak contact force, stress and local strain are separate harder targets. An apparent fit to outgoing speed alone is not evidence of accurate force history.

Prespecify application-specific tolerances rather than presenting a chosen number as a universal standard. Report median and worst-case/quantile error, parameter confidence intervals, and failure rates. Include parameter-count and training-cost ablations. Measure conditioning, conservation residuals, contact-order invariance and independence from the artificial reference length.

A 1D heterogeneous mass–spring-chain experiment can demonstrate low-frequency reduction mechanics and furnish real reproducible numerical data. It cannot establish superiority for arbitrary 2D frictional impacts, general PDE convergence, or contact-wrench fidelity. For an axial-wave analogy, a cell size small relative to the wavelengths that materially contribute to the target output is a useful scale criterion; decreasing the cell count can miss short impact transients even when static stiffness matches.

## References and what was actually verified

Metadata and public abstracts/documentation were checked on 2026-10-04. No inaccessible full-text result is treated as a proof here.

1. Elandt, R.; Drumwright, E.; Sherman, M.; Ruina, A. (2019). *A pressure field model for fast, robust approximation of net contact force and moment between nominally rigid objects*. IROS. [Publisher DOI](https://doi.org/10.1109/IROS40897.2019.8968548); [arXiv abstract/full-text link](https://arxiv.org/abs/1904.11433). Crossref title/authors verified; public abstract read. It explicitly claims much faster evaluation than elasticity-theory models and continuous wrenches even for coarse meshes. Those are its authors' claims, not measured results for this project. [Drake hydroelastic documentation](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html) read for implemented pressure fields, meshes, friction and dissipation.
2. Aghili, F. (online 2019). *Energetically consistent model of slipping and sticking frictional impacts in multibody systems*. Multibody System Dynamics. [DOI](https://doi.org/10.1007/s11044-019-09703-2). Crossref metadata and publisher abstract read. It states the critical static friction and energy-admissibility results summarized above.
3. Maw, N.; Barber, J. R.; Fawcett, J. N. (1976). *The oblique impact of elastic spheres*. Wear. [DOI](https://doi.org/10.1016/0043-1648(76)90201-5). Crossref metadata checked. The companion 1977 paper *The rebound of elastic bodies in oblique impact* is [DOI](https://doi.org/10.1016/0093-6413(77)90045-3). These are directly relevant prior-art leads; their full derivations were not independently audited in this review.
4. Ai, J.; Chen, J.-F.; Rotter, J. M.; Ooi, J. Y. (2011). *Assessment of rolling resistance models in discrete element simulations*. Powder Technology. [DOI](https://doi.org/10.1016/j.powtec.2010.09.030). Crossref title/authors checked; no quantitative result from it is asserted here.
5. Craig, R. R., Jr.; Bampton, M. C. C. (1968). *Coupling of substructures for dynamic analyses*. AIAA Journal. [DOI](https://doi.org/10.2514/3.4741). Crossref title/authors/date checked. Establishes long-standing substructuring prior art; full derivation not audited here.
6. Fang, Y.; Liu, G.; Zhang, Y.; Zhu, Z.; Li, S. (2025, DOI registered 2024). *A generalized coarse-graining discrete-element method with variable scaling ratios based on non-dimensional contact equation*. Powder Technology. [DOI](https://doi.org/10.1016/j.powtec.2024.120569). Crossref metadata checked. It demonstrates the relevance of modern variable-scale coarse-graining prior art; its quantitative performance was not audited.
7. Ramírez, R.; Pöschel, T.; Brilliantov, N. V.; Schwager, T. (1999). *Coefficient of restitution of colliding viscoelastic spheres*. Physical Review E. [DOI](https://doi.org/10.1103/PhysRevE.60.4465). Crossref metadata checked; cite as a specific example of speed-dependent restitution, not a general polygon contact model.
8. Stewart, D. E.; Trinkle, J. C. (1996). *An implicit time-stepping scheme for rigid body dynamics with inelastic collisions and Coulomb friction*. International Journal for Numerical Methods in Engineering. [DOI](https://doi.org/10.1002/(SICI)1097-0207(19960815)39:15%3C2673::AID-NME972%3E3.0.CO;2-I). Crossref metadata checked. Standard multibody frictional-contact baseline; exact correspondence to this proposed solver must still be demonstrated.

The search is targeted, not exhaustive. A paper-level novelty claim needs a broader forward/backward citation review, including the closest flexible-body, contact-reduction and energy-consistent impact models, after the candidate algorithm is fixed.

## Additional reviewed local compliant reference

The subsequently implemented [compliant_contact.py](compliant_contact.py) is a useful restricted comparator, not a new general collision theory or a material-validated global model. It retains three planar contact channels, normal compression, tangential elastic displacement and an elastic relative angle; geometry and a positive-definite mobility matrix are held fixed during one finite contact episode. The tangential and angular channels use a Kelvin–Voigt element in series with a Coulomb-type slider, with distinct static/dynamic capacities. The angular capacity uses a physical rolling-resistance length.

Its constitutive passivity can be checked directly. With elastic state z, relative rate u, stiffness k and damping c > 0, the trial force is T = −kz − cu. In the static branch the elastic rate is u and plastic rate is zero, so the dissipation is cu². On yield the distinct-coefficient implementation retains a plastic-slip direction σ and selects F = −μ_dynamic Nσ. The plastic rate is (F−T)/c. This branch persists while σ times plastic rate is nonnegative, ending at a located arrest event; consequently F times plastic rate is nonpositive. Thus c times elastic-rate squared minus F times plastic rate is nonnegative. The same calculation applies to the angular channel with moment capacity κN. When static and dynamic coefficients are equal, a continuous clipping baseline gives the same dissipation sign without the distinct-coefficient mode state.

The clipped normal force is F_n = max(0, k_n δ + c_n δ_dot) while compression δ is nonnegative. Its dissipation is (F_n−k_nδ)δ_dot: it equals c_nδ_dot² on the unclipped branch, and −k_nδδ_dot ≥ 0 when clipping occurs during unloading. Hence kinetic plus retained spring energy plus integrated dissipation is conserved in the unforced local reference, subject to numerical integration error.

Root reports ten passing tests, including retained slip modes and integration-tolerance refinement, and a sampled total-energy accounting residual approximately 5.04×10⁻¹⁰ for the demonstrated case. An isolated normal restitution calibration of 0.6 becomes approximately 0.493860337 in a coupled oblique case, illustrating that effective restitution changes with contact coupling. The implementation validates finite, symmetric, positive-definite 3×3 mobility inputs. These checks support the stated local comparator; they do not establish material authenticity, global well-posedness, general convergence, or superiority over established contact laws.

The critic identified a concrete chatter mechanism in the original memoryless distinct-coefficient switch: at N = 1, k = 50, c = 3, μ_static = 0.6, μ_dynamic = 0.4, u = 0.01 and z = 0.0114, the static and immediate dynamic vector fields drive the trial force toward opposite sides of the same switching boundary. The current implementation addresses that mechanism by retaining slip mode and direction and segmenting integration at yield/arrest events; static capacity is tested on the elastic branch. This is an explicit constitutive transition rule, not a theorem of well-posedness for all contact histories. The equal-coefficient comparator remains a simpler continuous baseline.

Elastic energy can remain at geometric opening. Its value is tracked, but deleting a contact must retain, release or discharge that energy with a stated rule; silently dropping the contact states would break complete energy accounting. Contact detection, rotating geometry, multiple simultaneously evolving contacts, deformable interfaces and external work still require separate implementation and validation.
