# Bounded original-law contact recovery: successful higher-effort tail, failed comparisons retained

The new 42-row captured system has a strict passive original-law solution. A prospectively frozen **2,048-SVD/iteration numerical tail accepts in 1,283 calls**, with independently recomputed circular projection residual **7.60597e-9 m/s** below the unchanged 1e-8 gate, and original impulse energy change **−3.60146098 J**. Nine API/physics controls pass. This is a captured-system result, not full-hull trajectory qualification or measured material validation.

The helper is `projection_more_v2.h`, namespace `projection_recovery_v2`, frozen at `170d798b113863dc4bee3df515bb8ab63175ea21`. It searches the original projection-map equations with their analytic Jacobian, monotonic Jacobian-column scaling and an existing LAPACK SVD trust subproblem. Scaling and damping affect numerical coordinates/search only; A, b, friction, normal bounds and tolerance remain unchanged. It accepts only through the original all-row normal/circular and finite-passivity gate. The candidate is limited to 64 rows and a fresh maximum of 2,048 SVD calls and iterations; failed searches leave caller impulses unchanged. Intended integration is strictly after all previously accepted lanes decline, to preserve earlier accepted outputs. A separate generic face-guide/original-PGS candidate is being reviewed; no fastest method is claimed here.

## Every earlier comparison failed honestly

Frozen source `877023f2341c682a325481c4c7b9e58f9ef09ed3` predeclared four merit/scaling comparisons at 1,024 calls. All nine API/physics controls passed, while all four numerical searches exhausted the budget and rejected:

| Numerical normal merit | Column scaling | Calls | Last original residual (m/s) | Accepted |
|---|---|---:|---:|---|
| Projection | Jacobian norms | 1,024 | 1.4976114e-6 | No |
| Projection | None | 1,024 | 1.5183357e-6 | No |
| Fischer–Burmeister | Jacobian norms | 1,024 | 1.4976276e-6 | No |
| Fischer–Burmeister | None | 1,024 | 1.5183527e-6 | No |

For a normal row, write u=Akk·pn and w=(Ap−b)n. The exact projection merit is min(u,w); the numerical Fischer–Burmeister merit is u+w−hypot(u,w). Both encode the same complementarity zeros, but their off-root objectives/Jacobians differ. At this captured start the maximum residual difference is only 6.45e-11 m/s, and at the stalled scaled candidate it is 1.21e-13. The corresponding Jacobian differences are approximately2.36e-5 and9.39e-7. Projection gradients remain nonzero. These computations do not support blaming a different normal merit alone. Stronger independent SciPy searches previously needed more than1,200 Jacobian evaluations, motivating the separately declared higher numerical budget.

## V2 comparison

| Trial | Global call cap | Actual calls | Original residual (m/s) | Accepted |
|---|---:|---:|---:|---|
| Full 42-row projection/scaled tail | 2,048 | 1,283 | 7.60597e-9 | Yes |
| Exact mobility components/dependency closure | 1,024 | 1,024 | 1.4976114e-6 | No |

Exact-component reduction groups only nonzero mobility couplings and complete normal/tangent dependencies, shares one global1,024-call budget, and requires the full original42-row gate after recombination. It does not rescue this budget-limited case. Its partial numerical candidate and unchanged caller output are retained separately. The successful full solve took1,220 accepted steps; rejected trust steps still count toward the1,283-call total.

Both plans were published before isolated compilation/execution. Read-only production binaries, native/Bullet headers, static and runtime libraries stayed unchanged; research binaries are separate. Source archives, compilation warnings/output, runtime guards, every successful/failed impulse and independent audits are retained under `results/` and `results-v2/`. The unexecuted original relative-include mistake was corrected before freezing; its previous bytes and erratum are retained. No compilation or native trial failure is omitted. Local research executables are excluded from publication manifests; provenance retains their hashes.

The prospective controls cover an analytic sticking contact, separating contact, zero/negative budgets, unsupported66-row input, negative friction, nonfinite mobility, invalid dependency and a contradictory semidefinite normal target. They demonstrate decline preservation and bounded work, not exhaustive constitutive correctness. Full-world geometry/contact discovery and trajectory refinement remain separate mandatory checks.

Recompute archived outcomes without building or rerunning native solves:

```sh
OPENBLAS_NUM_THREADS=1 python3 research/new-combined-projection-review/audit_failed_trials.py
OPENBLAS_NUM_THREADS=1 python3 research/new-combined-projection-review/audit_v2.py
```

These results establish bounded numerical recovery under greater declared effort. They do not establish friction novelty, unique rigid-impact histories, trajectory accuracy, a universal speed advantage, or authentic rubber parameters.
