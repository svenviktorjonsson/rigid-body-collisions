# Position geometry diagnostic and bounded pose guide

The exact finite-bound certificate proves that the retained 74-row translation-only numerical repair cannot pass its original 1e-8 m/s gate. It does not prove that the physical bodies cannot be separated. This protocol records the missing geometry rather than adding solver effort to an impossible bounded problem.

`position_geometry.h` records the exact pre-repair normal rows, endpoint geometry, signed linear/angular Jacobians, original/body IDs, split targets and body transforms/mobility. The native failure capture keeps its original schema in a separate file. `runner.py` reproduces the original seed7301 reference_2 authored scene at travel fraction .015, requires a committed source SHA, archives execution sources and guards the native binary and runtime libraries throughout. A changed rejection is retained and explicitly compared with the original capture hash.

After the observer replay, `analyze.py` independently reconstructs every entry of the actual translation mobility matrix. It then evaluates the positive weighting of rows 11/54/55 against the linear and angular geometric Jacobian. Two bounded prospective full-pose guides use the same split target: a feasibility LP and minimum mass/inertia-weighted pose QP. The protocol limits each translation component to 1e-4 m and each angular component to 1e-3 rad; Euclidean norms are reported explicitly. A numerical optimizer success flag is insufficient: the actual original velocity tolerance is independently checked on the full linear guide.

`requery.cpp` constructs a separate Bullet collision world from the authored backend shapes and exact failed-step transforms. It applies the proposed translations/world-frame rotation increments only to this clone, clears retained contacts, and discovers the actual convex contact gaps again. It does not modify the engine simulation. Its synthetic overlapping-box control records .01 m penetration before a .02 m lift and no overlap afterward. That control establishes the geometry-clone mechanism only, not a result for the actual random hulls.

A linear guide or reduced overlap in the clone does not qualify a complete trajectory. Full rotation changes the position discretization. Holding angular velocity fixed while changing orientation can change rotational energy; preserving angular momentum can also change energy. Any applied angular correction therefore needs a declared numerical energy/momentum ledger and a new trajectory/refinement study. A transactional rollback with smaller physical advances and fresh contact discovery may preserve the intended collision mechanics more directly; such a strategy must restore the complete state, prescribed boundary motion and work accounting and use a bounded retry budget. None of these changes are silently applied to the original captured problem.

Freeze the prospective driver and observer before execution:

```sh
python3 research/translation-position-geometry-review/runner.py --check-plan
python3 research/translation-position-geometry-review/runner.py --source-commit <frozen-sha>
```

The current materials, tolerance and historical failure are retained. The diagnostic is not experimental rubber calibration, a publication novelty claim, or qualification of a full random-shape shaking trajectory.
