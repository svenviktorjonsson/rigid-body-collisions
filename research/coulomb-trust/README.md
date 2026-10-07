# Captured circular-contact recovery

All six new frozen shared-contact rejection systems are recovered by the isolated native helper, with independent normal complementarity, circular-capacity, maximum-dissipation and passive-energy checks. Final original-law residuals are at most 6.71e-11 m/s, below the unchanged 1e-8 gate. The integrated solver also passes the five older capture regressions; standalone continuation does not replace the existing primary/polishing lanes.

The search uses a normal-only QP/pivot guide, Fischer–Burmeister normal merit and internal friction continuation. Only the original A, b and friction coefficients can be accepted/applied. Search damping acts on the numerical Jacobian, never the physical mobility. The final gate explicitly recomputes the original projection map.

`native-final-six.jsonl` records the final isolated six-case impulses, independent physical checks and source hashes. `final-native-provenance.json` and `final-native-source.zip` preserve compilation/input evidence. `combined-solver-replays/` contains the parent’s integrated compatibility checks. `artifact-hashes.json` covers retained sources, receipts and documentation.

Earlier Python and native diagnostic failures remain visible. Files named `native-first-*`, `native-guide-diagnostic`, `native-inexact-guide`, `native-FB-six` and `native-FB-old5` are intermediate prototypes. Some early aborted trials report a zero residual placeholder alongside `accepted:false`; these are not zero physical residual measurements. Native files before the final snapshot have no complete archived build provenance. `native-FB-old5` has four standalone failures expected to remain handled by existing solver lanes. None is a full trajectory accuracy result.

The helper limits rows to384, iterations to512, SVD calls to256, damped factorizations to512, continuation attempts to96 and stage iterations to48. The optional normal-only Dantzig guide has one call but no exposed internal pivot cap; there is no hard real-time wall-clock guarantee.

[Detailed mechanics and retained negative evidence](review.md), with [typeset equations](review.pdf), explain the method and limitations. Full hull trajectories require a new frozen integration run; these captures alone establish no experimental authenticity, convergence order, speed ranking or universal robustness.
