# First executed rigid collision study

Read [the assessment](../../rigid-study-report.md) before using these numbers.
All 17 scenes are numerical simulations. No measured material properties or
experimental observations were collected for this study.

- `scenes.json`: exact input geometry, initial states, coefficients, split and horizon.
- `reference-checks.json`: independent primary-step and velocity-iteration/substep refinement, analytic checks and reference diagnostics.
- `frozen-policy.json`: all eight calibration candidates, selected policy and excluded training cases. Calibration preceded held-out evaluation.
- `comparisons.csv`: 119 mode/case comparisons, RMS errors, median/min/max of five solver-plus-controller timings, switches, work and penetration diagnostic.
- `summary.json`: decisions, including failed and unqualified cases.
- `trace-manifest.json`, `trace-archive.json`, `traces.zip`: SHA-256-verified complete histories for all 153 evaluated/reference runs. Each NPZ contains states, physical sample times, mass, inertia and selected fidelity levels. Extract with `python -m zipfile -e traces.zip traces` in this directory.
- `accuracy-cost.png`: qualified held-out comparisons.

Re-run from the repository root after building both native backends:

```sh
python -m research.run_rigid_study --repeats 5 --output /tmp/new-rigid-study
python -m research.audit_rigid_study --pack --directory /tmp/new-rigid-study
python -m research.audit_rigid_study
```

The first study stored median/min/max timings, rather than every repetition.
The runner now additionally writes `run-records.json` with all repetition times,
analytic checks and per-mode diagnostics. The existing first-study summaries are
preserved; those missing repetition samples cannot be reconstructed. Calibration
uses measured runtime, so noise can change which tied candidate a rerun selects.

`follow-up-refinement.json` is explicitly exploratory and does not change
the first study's frozen policy or qualify its previously excluded comparisons.
`follow-up-traces.zip` and its manifest preserve the forty corresponding
histories. `adaptive-rebound-check.json` distinguishes outgoing impulse accuracy
from sampled collision-time error. `interleaved-timings.json` preserves all twenty
follow-up repetitions, mode order and end-to-end times for qualified held-out
cases. Neither follow-up tunes a new controller on those cases.
`follow-up-comparisons.json` compares the preserved trajectories with qualified
higher-work references. This exposes the 4×16 triangle failure and supports the
8×32 conservative default, without revising the original held-out score.
