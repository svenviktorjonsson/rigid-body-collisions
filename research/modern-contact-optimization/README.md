# Reusing modern solver techniques without replacing the contact law

Started 7 October 2026, from `1098616`. This is a staged numerical research
continuation, not a competing-engine speed ranking or main-compiler integration.

## Literature and compatibility

| Technique | Primary reference | Compatible use and boundary |
|---|---|---|
| Warm starts and cached small solves | [Catto, Solver2D](https://box2d.org/posts/2024/02/solver2d/) | Reuse a previous active set as a guess, then verify every constraint. Do not reuse physical impulse without transporting its directions and checking the current law. |
| Islands and persistent connectivity | [Catto, Simulation Islands](https://box2d.org/posts/2023/10/simulation-islands/) | Group contacts by bodies whose state they can change. A prescribed/static shared support must not couple otherwise independent bodies. Other constraints must be included. |
| Graph coloring and data layout | [Box2D 3.0](https://box2d.org/posts/2024/08/releasing-box2d-3.0/) | Keep each complete force/couple contact block together; parallel contacts must not write a shared dynamic body or shared material state. Colors change iteration order, so convergence needs fresh qualification. |
| Multicore island solving | [Jolt architecture](https://jrouwe.github.io/JoltPhysicsDocs/5.5.0/index.html) | Solve independent components concurrently; never mistake one million independent local responses for one million coupled world contacts. |
| Coupled angular friction | [MuJoCo computation](https://mujoco.readthedocs.io/en/latest/computation.html) | Useful comparator for torsional/rolling constraints. Its constitutive law is not a substitute for our motion-directed force/couple law. |
| Substeps, softness and relaxation | [Catto, Solver2D](https://box2d.org/posts/2024/02/solver2d/) | Event/substep reuse is potentially useful. Softness, bias and friction anchors can change physical compliance or material history; they require explicit physical and energy accounting. |

The older Bullet-backed lane already retains contact caches, warm impulses and
sparse iterative propagation. That lane is a historical comparator, not a
complete implementation of the requested independent angular-impulse model.

## Predeclared first experiment

Freeze the current `contact_history.py` before editing. Retain its midpoint
spring/slider law, separate static/dynamic capacities, exact singleton intervals,
opening-energy ledger and final finite/passivity checks. Improve only numerical
work: prepare the at-most-27 active sets and small factors once; optionally try a
caller-owned previous active set before exhaustive search. Current geometry,
mobility and capacities remain authoritative; hints cannot bypass KKT checks.

Compare frozen and candidate results on seeded one/two/three-mode cases, changing
loads/capacities, reversals, zero capacities, opening/recontact, coupled positive
semidefinite mobilities and mode permutations. Preserve input/output fields,
allow roundoff-level differences from a different triangular solve, and compare
all impulses, motions, stores and loss/release channels. Use a separate convex
optimizer as an independent reference on a subset of yielded cases.

Time alternating frozen/candidate runs after warmup, seven repetitions, same
inputs and output consumption. Include validation and energy gates. Report
preparation separately; disclose that these are Python local updates, not full
world steps. Benchmark both no-hint and coherent caller-hint execution, mixed
sliding directions and sizes 100/1,000/10,000. No universal 2x gate is reinstated.
Retain regressions as well as gains.

First numerical audit: 1,500 seeded cases agree with the frozen implementation
to 6.67e-16 scaled error. The initial SLSQP oracle stopped with a 2.72e-6 impulse
error on one scaled quadratic; its failure is retained in `history-v1.stderr`.
The independent check now transforms the SPD quadratic into bounded least
squares and uses SciPy BVLS, explicitly eliminating zero-capacity coordinates.
It does not start from the candidate result or relax the mechanics gates.

The first candidate and `run-v2` are preserved. Optional previous-face hints
sometimes add overhead, especially when all signs reverse or cold enumeration
already starts at the right face. Before the final run, add a second numerical
guess from the current unconstrained solution. Validate it with the same KKT
conditions; if both hints fail, keep the complete original exhaustive search.
Extend the final benchmark with coherent upper-bound and mixed-sign cases, while
retaining every original scenario and count. No material inputs change.

Declare an additional `coupled_face` case before its timing: a two-mode coupled
matrix whose unconstrained solution predicts lower/upper bounds, but whose
constrained solution is lower/free. The three-mode variant adds an independent
free mode. This specifically tests when a valid previous-face hint is useful
even after current-solution prediction is added. Run it separately after the
five-scenario `run-v3`; no one-mode coupling case is fabricated. The explicit
fallback regression already checks the exact impulse [-0.1, -0.06].

## Next structural experiment

Prepare a dimension-independent contact schedule from flat body indices and an
explicit mutable-body mask. Split independent islands and form conflict-free
contact batches. Preserve force and independent angular impulses as one contact
block. Test static-support separation, contact multiplicity, irregular graph
degree, stale topology and dynamic-body ownership. This is scheduling metadata,
not a new impact/friction law or a qualified colored world solver.

Full moving t/s history, normal impact plus persistent memory, a physical coupled
patch budget, and general irregular/group trajectory accuracy remain open.
Existing public material coefficients and experimental errors remain unchanged.
