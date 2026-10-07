# Reusing modern solver techniques without replacing the contact law

Started 7 October 2026, from `1098616`. This is a staged numerical research
continuation, not a competing-engine speed ranking or main-compiler integration.

## Accepted results

`run-v5` uses the corrected code at `e8f6752`. All 63 focused repository tests
pass. The 1,500 random controls preserve physical outputs to 2.17e-19 scaled
difference; 141 independent BVLS checks have at most 5.55e-17 modal impulse
difference. All 300 exact/adjacent static-limit cases retain the original branch.
The earlier fast unbounded solve is rejected and retained as evidence below.

Across 51 repeated-response combinations (100/1,000/10,000; one to three modes),
default median gains are 1.03–12.97x. Simple sticking is essentially unchanged.
At 10,000 three-mode responses, reversing/mixed/upper cases gain 6.91/10.27/12.90x;
a correct prior-face hint improves the coupled-face case by 2.86x, versus 1.31x
without a hint. Wrong hints add overhead. Preparation costs 0.106–0.842 ms versus
0.068–0.199 ms for the frozen object; changing mobility frequently can erase
the amortized benefit. Validation and energy gates are timed. These are Python
local response replays, not evolving many-body scenes or native batch timings.

The C++ planner passes 300 independent BFS/conflict controls and 2D/3D prescribed
force-plus-couple momentum tests, with bit-identical serial/eight-worker colored
outputs. Portable and OpenMP builds both pass. One million contacts require
14.97–27.84 ms for fresh planning and 0.85–1.00 ms for exact checked reuse.
The million-edge hub has 999,968 serial-tail contacts: no shared-body independence
is invented. `schedule-v1` includes sizes 100 through 1,000,000, but excludes
discovery/physical solving and is not an irregular-shape scene benchmark.

The [three-page report](report/report.pdf) includes all 10,000-response rows,
preparation costs, failed candidates and current limits. [ROADMAP.md](ROADMAP.md)
prioritizes full wrench matrix-free evaluation, current-geometry factor reuse,
qualified parallel iteration and fair matched-law competitor benchmarks. Public
material values and experimental errors remain unchanged.

## Literature and compatibility

| Technique | Primary reference | Compatible use and boundary |
|---|---|---|
| Warm starts and cached small solves | [Catto, Solver2D](https://box2d.org/posts/2024/02/solver2d/) | Reuse a previous active set as a guess, then verify every constraint. Do not reuse physical impulse without transporting its directions and checking the current law. |
| Islands and persistent connectivity | [Catto, Simulation Islands](https://box2d.org/posts/2023/10/simulation-islands/) | Group contacts by bodies whose state they can change. A prescribed/static shared support must not couple otherwise independent bodies. Other constraints must be included. |
| Graph coloring and data layout | [Box2D 3.0](https://box2d.org/posts/2024/08/releasing-box2d-3.0/) | Keep each complete force/couple contact block together; parallel contacts must not write a shared dynamic body or shared material state. Colors change iteration order, so convergence needs fresh qualification. |
| Multicore island solving | [Jolt architecture](https://jrouwe.github.io/JoltPhysicsDocs/5.5.0/index.html) | Solve independent components concurrently; never mistake one million independent local responses for one million coupled world contacts. |
| Sparse body-space operators | [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/) | Avoid unnecessary dense contact-matrix formation; reuse the full wrench map and its transpose. A different optimization objective is acceptable only if it represents the same constitutive law. |
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

An additional exact-static-limit audit invalidated the first fast candidate:
9 of 100 cases changed static/dynamic branches solely from triangular-solve
roundoff, with up to 6.52 difference in a modal motion. Its source and failed
audit are retained. The correction uses the original full unconstrained solve
for every static-capacity decision. Specialized triangular substitution remains
only inside the continuous constrained-face problem after yielding. No static
limit is expanded by a tolerance and no material parameter is changed. Static
steps return no dynamic-face hint. The final audit includes exact limits and
both adjacent representable limits; final timings must use this corrected code.
Runs v2/v3/v4 are preliminary evidence, not qualified current performance.

## Reproduction

From the repository root, with Python 3.11+, NumPy, SciPy, g++ and OpenMP:

```bash
python -m unittest tests.test_contact_history tests.test_contact_history_warm_start tests.test_contact_schedule -v
python research/modern-contact-optimization/boundary_audit.py --output /tmp/contact-boundary-audit.json
python research/modern-contact-optimization/history_experiment.py --output /tmp/contact-history-repeat
g++ -std=c++17 -O3 -Wall -Wextra -Werror research/modern-contact-optimization/schedule_benchmark.cpp -o /tmp/contact-schedule-benchmark
/tmp/contact-schedule-benchmark > /tmp/contact-schedule-repeat.json
```

The history experiment takes several minutes; output folders must not already
exist. Its final default has all six scenarios and three sizes, with the
one-mode coupled case omitted: 51 combinations. Earlier v2 can be reproduced
with `--candidate research/modern-contact-optimization/candidate_history_v2.py
--scenarios stick coherent_slide reversing_slide`. v3 uses
`candidate_history_v3.py` with those scenarios plus coherent_upper/coherent_mixed.
Those preliminary versions intentionally retain the boundary defect for audit;
use the current root module for simulations. The 15 component tests above are
separate from the 63 focused full-repository tests recorded in verification.

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
