# Adoption order for the full force-and-angular-impulse model

This plan distinguishes implemented numerical components from unimplemented
world integration. Published material values stay fixed. A faster solve is
accepted only if it solves the same current contact law and meets the original
mechanics and trajectory gates. No new experimental accuracy claim follows from
a solver speed measurement.

## 1. Preserve complete contact blocks — implemented foundation

The scheduler groups all scalar components of a contact together, including the
independent angular impulse. The body angular update must retain the lever moment
of the force impulse **plus** the independent angular impulse. It is tested for
prescribed 2D/3D impulses, not yet selected by a nonlinear world solver.

Keep the full contact-relative velocity direction t, the full relative angular
velocity direction s and normal n. Never reinterpret s as a second tangent.
Topology grouping does not justify changing the physical directional basis.

## 2. Replace dense assembly with body-local matrix application — next priority

Use the article's complete shared-point wrench map and its transpose. For each
trial contact-component impulse: form the physical force/couple, accumulate its
signed body momentum changes, apply each body's inverse mass and full world
inertia, then gather the resulting contact-relative linear/angular motions and
project back into the supplied component map. All stages use flat body/contact
indices. Prescribed boundaries contribute known motion and zero inverse mass.

This computes the same contact mobility action without storing a quadratic-size
contact matrix. It retains cross-contact coupling through common bodies; it does
not treat a connected group as independent collisions. Coloring can handle body
accumulation, while gathering reads body responses independently. Direct small
block solves remain useful for modest islands; large islands need sparse or
matrix-free iterative methods with the original full acceptance gate.

First qualify against explicit dense assembly on small random 2D/3D systems:
full tensors, off-center contacts, pure couples, prescribed walls, body/contact
permutations, length scaling, and dependent n/t or s/n directions. Do not assume
the four-column component map is invertible. Then measure complete irregular
and rapid-group scenes, including discovery, assembly, solve and integration.

## 3. Reuse work within a frozen solve — partially implemented

The local contact-history component now prepares Cholesky factors and active-set
metadata, tests a caller-owned prior face and the current unconstrained face,
and falls back to exhaustive exact face search. Hints are checked against the
current capacities and energy ledger. Its present scope is one to three declared
fixed modes; it is not a joint general normal-impact solver.

In the future graph solver, reuse inverse tensors, local block factors and
preconditioners within a frozen geometry/direction evaluation. Changing pose,
contact point, stiffness, timestep or component basis can invalidate them.
Previous physical impulses require a persistent contact identity and transport
of their physical directions; copying scalar amplitudes into a new basis is
insufficient. Static-direction closure and history transport remain explicit
model tasks, not numerical guesses.

## 4. Persist topology and reuse collision infrastructure — partial foundation

The new cache checks all ordered endpoints, body mutability and color budget
before reusing a plan. It can safely reuse connectivity across geometry changes
but cannot reuse physical patch history on that basis. Controlled edit logs and
dirty-island rebuilds could later reduce this checking cost; never replace exact
invalidation with an unchecked timestamp, pointer or collision-prone hash.

Use the existing backend's collision discovery/manifold infrastructure where it
fits. Preserve shared world contact points, rapid prescribed-wall updates and
the authored mass/COM/full-inertia inputs. Profile before replacing discovery.
Sleeping must consider all connected constraints, moving boundaries and stored
contact-mode energy; low body velocity alone is not sufficient.

## 5. Qualify parallel iteration before world adoption — not implemented

Coloring allows concurrent complete contacts with disjoint writable bodies.
High-degree conflicts retain a serial tail. Include other joints and shared
mutable constitutive states in the graph. Use private scratch and deterministic
reductions rather than racing a shared energy or work scalar.

The current visitor commits prescribed/gated blocks. It does not prove that a
colored nonlinear iteration converges as quickly as the original ordering.
Residual/passivity checks alone also do not prove trajectory accuracy. Start
with isolated impact and supported branches, then stacks, rocking/rolling,
irregular 3D hulls and rapid moving groups; retain every rejection/refinement
failure. Keep performance and experimental prediction errors separate.

## 6. Adopt hardware tuning after the algorithm is qualified

Use field-major arrays, contiguous index batches, cached per-worker workspaces
and persistent worker pools. SIMD can speed body-response and contact projection
without redefining the law. Measure preparation, allocations, scheduling and
barriers as well as the contact kernel. GPU/BKF lowering is a later compiler
acceptance task, not demonstrated by these host C++/Python results.

## Techniques that require a physical decision

Soft constraints, positional friction anchors, Baumgarte bias, XPBD compliance,
clamped velocities and dropping weak couplings can alter the model or its energy
accounting. Do not adopt them as invisible numerical optimizations. Contact
elasticity/history should represent justified material behavior, with stored,
released and dissipated energy accounted separately. A numerical pose correction
also needs its existing momentum/energy ledger and fresh trajectory qualification.

Primary references and the predeclared experiment are in [README.md](README.md).
The full directional closure and experimental limitations remain in the
[model fidelity audit](../scaled-contact-article/MODEL-FIDELITY-AUDIT.md).
