# Joint publication assessment

Prepared 2026-10-04 by the critical reviewer and the evidence-based publication advocate. Both inspected the existing formulation, exchanged verified prior-art sources, challenged the original scope, and explicitly agreed on the conclusions below.

## Agreed decision

**No-go: submit the present formulation as novel general collision theory.** Its compact notation, contact mobility matrix, kinetic-energy quadratic, coupled friction/restitution ingredients, and simultaneous-contact Jacobian are established mechanics. Coarse contact patches and connected discrete blocks also have substantial prior art. There is no demonstrated improvement in arbitrary-body impact accuracy or computational cost.

**Conditional go: research a narrower contribution.** Develop an adaptive heterogeneous contact/interface reduction, retaining needed elastic/history states, with mechanically consistent energy and momentum accounting. Test whether it predicts held-out integrated impulses, outgoing translation and spin at lower cost than appropriate established contact and reduced deformation methods. Publish a new method only if a distinct algorithm, theorem, or reproducible advantage survives those comparisons.

The critic's argument prevails against submission now. The advocate's argument supports further research in a restricted regime. This is a shared evidence-based decision; neither reviewer was asked to manufacture a favorable winner.

## Strongest argument from each side

**Critical reviewer:** Stronge's frictional-impact work, established multibody contact solvers, tangential-compliance/restitution research, rolling-resistance models, DEM/bonded-particle methods and pressure-field contact already cover the main ingredients. Independent restitution targets can inject energy or become kinematically impossible with multiple contacts. A memoryless local coefficient cannot distinguish hidden prestrain, internal vibration or contact history. Algebraic static capacity does not establish an authentic force path throughout impact.

**Publication advocate:** Integrated contact wrench and bulk outgoing motion often require fewer unknowns than full stress/strain resolution. Heterogeneous adaptive patches plus a few well-chosen interface/internal states may produce a useful accuracy–cost niche. The hidden-state objection defeats a universal memoryless model, but a revised small-state model can represent the missing information. The promising contribution is its adaptive reduction, error control and transfer across held-out impacts—not its notation or its generic use of coarse cells.

**Agreed restriction:** seek bulk outcome accuracy in explicitly specified low-deformation/bandwidth regimes. Do not claim universal continuum accuracy, accurate peak force/stress/damage, or arbitrary finite-mesh convergence from a surface material partition.

## Reviewed algebraic comparator

The proposed energy-constrained closest restitution target is a legitimate **phenomenological endpoint comparator** under stated assumptions. It is not Coulomb maximum-dissipation friction, an established material law, or a novelty result.

For one unrestricted approaching pair contact with positive-definite mobility K, relative contact velocity u, and impulse j, impose:

- u⁺ = u⁻ + Kj;
- the exact normal target u_n⁺ = −e_n u_n⁻, with 0 ≤ e_n ≤ 1;
- j_n ≥ 0 and static sliding/rolling capacity inequalities;
- ΔT = (u⁻)ᵀj + ½jᵀKj ≤ 0;
- minimize ½(u⁺ − u_target)ᵀK⁻¹(u⁺ − u_target).

The feasible set is nonempty: the normal-only impulse j_n = −(1+e_n)u_n⁻/K_nn, j_t = j_s = 0 satisfies the capacities and gives ΔT = −(1−e_n²)(u_n⁻)²/(2K_nn) ≤ 0. The objective is strictly convex in j for positive-definite K, so the feasible endpoint minimizer is unique. This uniqueness applies to that comparator and those assumptions, not to all Coulomb or multibody impacts.

The metric is invariant under consistent representation changes: if u′ = Su, j′ = S⁻ᵀj and K′ = SKSᵀ, then both the kinetic-energy expression and the metric objective are unchanged. Rolling bounds must transform consistently; the artificial reference length must not become a physical resistance length. An unweighted Euclidean velocity norm would not have this property.

Static capacity inequalities alone do not enforce dynamic friction at a distinct coefficient, its saturation, or slip/stick complementarity. Projection may yield a passive endpoint without an authentic intermediate slip path. Positive tangential/rolling target restitution describes rebound and does not mean final no-slip sticking.

For many contacts K can be singular. Independently specified simultaneous restitution targets may be incompatible, even when every number lies in [0,1]. Consequently the single-pair feasibility proof cannot be promoted to a universal global-contact solver. Use feasible body-velocity/subspace formulations or explicit relaxation if proposing a many-contact endpoint comparator, and state the resulting constitutive choices.

## Preferred full model

Use an established compliant contact model with normal compression, tangential elastic/contact history, and a physically justified patch angular response. Static friction is a force/couple capacity; dynamic slip dissipates work; stored contact elasticity accounts for rebound. Integrate compression and unloading rather than assuming every channel's independent restitution target must be exactly satisfied.

For deformable coarse bodies, give cells independent poses and interfaces objective extension/shear/rotation states, stored elastic potentials and nonnegative dissipation. Account for kinetic plus stored energy and external work. Rigidly welded subdivisions only label material regions and cannot represent internal deformation. Preserve aggregate mass/center/inertia, fit static compliance and selected low-frequency modes, and compare against standard substructuring or bonded-element reductions.

Normal/tangential/rolling stiffness and damping, contact-patch scales and history may therefore be needed in addition to friction coefficients. Effective restitution is an outcome or calibrated response of a contacting pair/process, not a universal constant belonging independently to each material. Freeze coefficients and calibration rules before evaluating held-out impacts.

## Evidence required to change the publication decision

1. Fix a concrete adaptive reduction/constitutive algorithm and its target observable; complete closest-prior-art comparison.
2. Verify objectivity, reference-length invariance, internal momentum consistency and relevant total-energy accounting, including contact creation/deletion and dependent contacts.
3. Compare against standard impulse/energetic models, compliant contact, hydroelastic/distributed patches, and an appropriate bonded or reduced flexible-body model.
4. Use separate calibration and evaluation cases across location, incidence, incoming spin, speed, material contrast and simultaneous contacts. Include hidden-state/history tests.
5. Report converged reference accuracy, output errors, computational cost including calibration/preprocessing, failure cases and parameter uncertainty. A narrow improvement can be useful; universal superiority is unnecessary.

## Directly relevant checked sources

- Stronge (1990), *Rigid body collisions with friction*, [DOI](https://doi.org/10.1098/rspa.1990.0125): critic verified metadata/public abstract describing energy consistency and slip transitions.
- Aghili (online 2019), *Energetically consistent model of slipping and sticking frictional impacts in multibody systems*, [DOI](https://doi.org/10.1007/s11044-019-09703-2): advocate read publisher abstract specifying critical static friction and energy-admissible parameter regions.
- Maw, Barber and Fawcett (1981), *The Role of Elastic Tangential Compliance in Oblique Impact*, [DOI](https://doi.org/10.1115/1.3251617): critic checked public abstract reporting rebound-angle effects and experimental comparisons.
- Elandt et al. (2019), *A pressure field model for fast, robust approximation of net contact force and moment between nominally rigid objects*, [preprint](https://arxiv.org/abs/1904.11433), [DOI](https://doi.org/10.1109/IROS40897.2019.8968548): advocate read public abstract explicitly advertising coarse meshes and faster evaluation than elasticity models. These are its authors' results, not this project's results.
- Craig and Bampton (1968), *Coupling of substructures for dynamic analyses*, [DOI](https://doi.org/10.2514/3.4741): advocate checked metadata; established deformation-reduction comparator.
- Anitescu and Potra (1997), *Formulating Dynamic Multi-Rigid-Body Contact Problems with Friction as Solvable Linear Complementarity Problems*, [DOI](https://doi.org/10.1023/A:1008292328909): critic checked metadata; simultaneous contact solving is established prior art.

The individual reports contain broader references and precise evidence boundaries: [critical-review.md](critical-review.md) and [publication-case.md](publication-case.md). This targeted review is not an exhaustive novelty search.

## Synthetic reduction experiment

Both reviewers inspected the completed synthetic results in [coarse-rod-results.csv](coarse-rod-results.csv) and the setup in [benchmarks.py](benchmarks.py). The system is a force-driven, free-boundary, linear elastic 1D rod represented by lumped masses and springs. Length, density and area are one; elastic modulus is one on the left half and four on the right. A Gaussian boundary force has unit integrated impulse. The measured observable is the **right-end velocity time history**, not center-of-mass outgoing velocity from a collision. Final energy includes kinetic and spring energy.

| Prescribed pulse width σ | Coarse elements | Right-end velocity relative L2 error | Final mechanical-energy relative error | 1024-versus-2048 reference velocity discrepancy |
|---|---:|---:|---:|---:|
| 0.25, broad pulse | 16 | 1.3345% | 0.4043% | 0.0002393% |
| 0.015, sharp pulse | 16 | 107.5259% | 23.9313% | 1.3513% |

For the broad pulse, refinement to 32, 64 and 128 elements reduces the velocity error to approximately 0.3284%, 0.0817% and 0.0203%. For the sharp pulse, even 128 elements leave approximately 57.32% error. The sharp-pulse reference check is sufficient to distinguish order-one coarse error; it does not establish a subpercent converged truth for that case.

This is real, reproducible numerical evidence that a coarse heterogeneous elastic chain can reproduce a broad response while badly missing a narrow-band-duration/high-frequency excitation. It supports a **bandwidth-limited reduction hypothesis** and refutes a uniform coarse-accuracy claim. It is neither a frictional collision experiment nor evidence of adaptive-patch novelty.

The calculation is elastic and driven by a prescribed force. Differences in final energy relative to the fine reference are not automatically unphysical dissipation: the driven work depends on the discretization's boundary response. The separate work–energy residuals in the CSV measure integration accounting.

Node-update reductions are arithmetic workload comparisons against a fixed 2048-element explicit reference. Runtime ratios come from one implementation and one run. Neither establishes superiority over optimized FEM, hydroelastic contact, bonded DEM or a reduced flexible-body method, and neither includes an identified adaptive algorithm's calibration costs.

This scoped evidence does not change the shared publication decision. It supplies a useful demonstrator and a falsifying case; the requested arbitrary-2D frictional/multicontact and comparative validation remain to be done.

## Follow-up implementation reviewed by both agents

After the agreed verdict, root implemented the limited comparators and ten tests. The local finite-contact reference initially lacked persistent static/dynamic mode history; the critical reviewer gave a concrete chatter counterexample. Root replaced repeated threshold switching with retained slip directions and segmented yield/arrest events. Both reviewers inspected the repair and accepted it for the scoped frozen-frame example, while retaining all general convergence and material-validation limits. Mobility is checked as finite symmetric positive definite. Current synthetic energy-accounting residual is approximately 5.04e-10. This improves the reviewable implementation, not its publication novelty verdict.

The implemented global normal solver is frictionless and zero restitution; the graph assembly and global energy identities extend the representation to multiple bodies, but a full heterogeneous global frictional impact simulator has not been completed. The two reviewers' follow-up notes are in their individual reports.
