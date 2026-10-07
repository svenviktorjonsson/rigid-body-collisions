# Moving boxes with many balls

This study tests exact many-contact impulse propagation and three 100-ball
frictional scenes: constant translation, scheduled reversals and rotation.
One kinematic compound body carries the four walls; the contents are dynamic.
Their momentum changes through the actuator. It is not an internal invariant.

[Executed report](report.pdf) retains the results and failed reference gates.
[Typeset mechanics](contact-model.pdf) gives the wedge convention, point
mobility, global contact map, impulse/energy equations and friction limits.
The [plan](plan.json) was committed before executing the study. Parameters are
synthetic declarations, not experimentally authenticated materials. These tests
execute the pinned native Box2D comparators and an independent Python normal
projection; they do not execute a language compiler, WASM or physical GPU.

Build both backends using [these instructions](../../rigid_backend/README.md), then:

```sh
python -m unittest discover -s tests -q
python -m unittest discover -s research -p 'test_*.py' -q
python -m research.run_container_study --repeats 3
python -m research.audit_container_study
python -m research.make_container_report
```

Every trace is published in `results/traces.zip`; `summary.json` records its
SHA-256, source provenance, all timings, failed cases and reference gates.
The runner saves completed cases atomically to ignored local
`results/checkpoints/` on subsequent executions. These preserve partial results
after interruption; they are not automatically trusted as a resumed study.
Push coherent source/evidence checkpoints before long runs. Reproduce in a
separate checkout/output tree when preserving a published evidence set.

The analytical packed row requires every unit-mass ball at the box speed of
1 m/s. The budget is a maximum individual error of 0.01 m/s, not an average.
First-frame native errors include contact discovery and floating geometry as
well as solver convergence. The exact rigid model has instantaneous constraints;
it does not model finite-speed elastic waves.

Grid reference qualification separately refines collision updates and velocity
iterations. All four refinement edges must meet quarter error budgets. A
highest-work trajectory that fails this gate remains a candidate reference.
The geometry audit observes wall containment and disk overlap at output frames;
it is not a continuous-time CCD proof. Work for shaking and rotation is not
claimed from a final momentum difference; an exact reaction ledger is future work.

The frozen zero-restitution, frictionless global normal projection is a
correctness oracle. It does not implement the complete frictional engine or
establish a scalable performance advantage. The current adaptive controller is
heuristic. Physical parameters must stay fixed when varying numerical fidelity.
See the existing [joint research verdict](../joint-verdict.md) for prior art and
the evidence required for a publication claim.
