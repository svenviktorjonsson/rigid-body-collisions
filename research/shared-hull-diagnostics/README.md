# Near-threshold shared-contact rejection: a numerical Newton increment

The immutable shared-contact seed42, travel fraction 0.03 run rejects a 48-row
system at 1.2099446e-8 m/s, against the unchanged 1e-8 gate. This folder preserves
that diagnosis without modifying the original failed trajectory archive.

Native replay spends 39 of its 256 SVD calls, makes 13 opposing-slip restarts,
and has no SVD convergence or work-budget rejection. The frozen helper reports
rank 48, retained-column correlation 1.18226e-14, and a 4.20146 N s Newton step.
The independent LAPACK result agrees. Thus this failure is not explained by
exhausting the effort ceiling or by unconverged Jacobi rotations.

The exact projected-equation Jacobian has largest singular value 24.65647 and
smallest 1.7732483e-9, ratio 7.19182e-11. The small-singular-value direction is
**not a mechanical gauge**: its mobility response has infinity norm 0.03618.
Dividing a tiny residual component by that singular value produces a large
pressure step. Its full step violates the final law badly; merit line search
reduces it to tiny fractions and stagnates.

| Numerical Jacobian increment | Rank | Step norm (N s) | Final original-law residual (m/s) | Gate |
| --- | ---: | ---: | ---: | --- |
| Relative SVD cutoff 1e-12 | 48 | 4.20147 | 13.3095 | Reject |
| Relative SVD cutoff 1e-10 | 47 | 3.50581e-9 | 5.57701e-9 | Accept |

Only the **numerical Newton increment** is truncated. The physical mobility
matrix A, b, friction coefficient, normal complementarity, circular capacity,
velocity tolerance, and final passivity gate remain unchanged. The accepted
step is independently checked against those original equations and passive
work. A truncated increment's acceptance does not imply that its linearized
Jacobian residual vanishes: the declared tolerance is checked on the full
nonlinear contact law.

Twelve declared full-step variants are retained in `bounded-experiments.json`:
five SVD rank cutoffs and seven smooth numerical-J damping values. Cutoffs
1e-10, 1e-9, and 1e-8 accept; full-rank 1e-12 and 1e-13 reject. Relative damping
lambda/max_sigma of 1e-12 through 1e-7 rejects; 1e-6, 1e-5, and 1e-4 accepts.
The damping experiment affects J only, never the physical A. The simpler
rank-47 increment needs one numerical solve. Published trust-region
least-squares independently accepts at approximately 5.57701e-9 m/s; the
published opposing-only restart fails.

The successful captured contact solve demonstrates a numerical-search issue
for this system. It is not an infeasibility certificate and does not repair or
qualify the already rejected six full trajectories. A production integration
and separately frozen full-trajectory follow-up must pass unchanged gates
before making such a claim.

Replay from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python -m research.shared_hull_diagnostics
g++ -std=c++17 -O2 -I build/spatial/_deps/json-src/include research/shared-hull-diagnostics/native_linear.cpp -o /tmp/shared48-linear
/tmp/shared48-linear research/shared-hull-diagnostics/seed42-48row-linear-input.json
```

The native numerical-helper header is archived from source
`38f407d12208654075c07315e96b9fc213612b91`; subsequent production edits cannot
change this replay. `manifest.json` pins inputs, sources, and each diagnostic
output. All failed variant results remain available.
