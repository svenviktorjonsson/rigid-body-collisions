# Same-law circular-contact recovery diagnostics

The six new frozen rejection systems have valid, passive solutions. The final isolated native continuation helper recovers all six and passes independent contact-law and energy checks. This establishes recovery of these **captured contact systems**, not completion, convergence or speed of the six full hull trajectories. The parent study must rerun those trajectories with frozen integrated source and retain any later rejections.

The input systems remain exactly as captured in `research/shared-hull-rank-followup/results/rejections/`. They contain the original unregularized physical mobility $A$, free/target velocity vector $b$, circular friction coefficients and normal limits. No mobility entries, normal targets, capacities or final $10^{-8}$ m/s acceptance tolerances were changed.

## Numerical search and physical acceptance

For impulse vector $p$, let $w=Ap-b$. Physical acceptance uses the original natural projection map, normal nonnegativity/complementarity, circular tangential capacity, maximum-dissipation support identity and finite passivity:

$$
p_n\geq0,\quad w_n\geq0,\quad p_n w_n=0,
\qquad\|p_t\|\leq\mu p_n,
\qquad p_t\mathbin{\cdot}w_t+\mu p_n\|w_t\|=0.
$$

The finite energy bound is

$$
\Delta E_{\rm bound}=\frac12 p^{\mathsf T}(Ap-2b).
$$

A frictionless normal QP supplies an initial complementary pressure face. If it declines, one normal-only Bullet Dantzig solve is tried and independently checked. There are no tangent pyramid rows in this initializer. These states are numerical guides, never applied to bodies.

During search, a scalar $\alpha$ raises trial friction from zero to the original coefficient: $\mu_{\rm trial}=\alpha\mu$. Only the final $\alpha=1$ solution can be accepted. Intermediate trial coefficients are continuation parameters, not changed material properties or fallback outputs.

The normal equation uses a Fischer–Burmeister **merit representation** during search, with velocity-dimensional $u=p_n/\rho_n$:

$$
\phi(u,w_n)=u+w_n-\sqrt{u^2+w_n^2}.
$$

Its zeros give the same unilateral normal law. The final gate explicitly recomputes the **original normal projection map**, rather than trusting the merit value. Tangential search and acceptance use the original circular projection law.

Newton/SVD steps and optional Levenberg damping operate on the numerical Jacobian $J$ of this merit map. A damping term in $J^{\mathsf T}J$ changes only the trial search increment. It adds no diagonal term to physical $A$, no compliance, no elasticity and no alternate friction law. The output impulse vector changes only on final acceptance.

Per-call search bounds are: at most 384 rows, 512 outer search iterations, 256 SVD calls, 512 damped factorizations, 96 continuation attempts and 48 iterations per trial stage. Trial $\alpha$ increments start at 0.1, grow to at most 0.2, and halve on failure; increments smaller than $1/16384$ stop the attempt. Each SVD retains the existing 64 Jacobi-sweep limit. The normal QP has its existing active-set loop bounds. The optional upstream Dantzig initializer has one **call**; Bullet exposes no internal pivot-count limit. Consequently these are explicit numerical call/iteration limits, not a hard real-time wall-clock guarantee.

## Retained evidence

| Scene and capture | Rows | Final original-law residual (m/s) | SVD calls | Independent contact and energy gates |
|---|---:|---:|---:|---|
| seed42 reference0 | 48 | $1.07\times10^{-12}$ | 17 | Pass |
| seed42 reference1 | 51 | $3.81\times10^{-16}$ | 21 | Pass |
| seed42 reference2 | 30 | $6.70\times10^{-11}$ | 18 | Pass |
| seed7301 reference0 | 216 | $5.93\times10^{-15}$ | 12 | Pass |
| seed7301 reference1 | 192 | $9.08\times10^{-14}$ | 12 | Pass |
| seed7301 reference2 | 123 | $4.87\times10^{-15}$ | 19 | Pass |

All six use six accepted continuation stages. No damped factorization was required in these final six solutions. Maximum independent friction support gap is $1.62\times10^{-11}$ J; every recorded passivity bound is negative. `native-final-six.jsonl` records the impulses and all independent acceptance results. It is a captured-system experiment; no shape/contact discovery or trajectory was simulated by this executable.

Failures remain in the directory. Warm and cold direct SciPy trust-region/LM searches stalled at nonzero natural-map merit minima. A first native natural-map continuation implementation recovered four systems but stalled at the initial zero-friction face of the other two. Allowing an inexact initial guide alone failed. FB search plus an independently checked normal pivot guide resolved these cases. Early failed native receipts used a zero residual **placeholder** when the initial stage aborted; this is not a measured zero physical residual. Their `accepted:false` flags remain authoritative, and subsequent guide diagnostic receipts record the actual stalled residuals. Do not use those placeholder fields for accuracy claims.

For the 51-row system, the stalled pressure face included obsolete normal row3. The exact zero-friction face [2,5,16] has residual approximately $6\times10^{-16}$ m/s. This is evidence for a numerical mode/pressure-face issue, not proof of physical infeasibility. A separate reviewer found all six captured physical matrices symmetric and PSD up to Float64 roundoff and normal inequalities feasible.

`native-FB-old5.jsonl` retains a **standalone continuation** experiment on five older captures. Four of those are not recovered by this new helper alone; they are already handled by the existing primary/polishing solver. The integrated implementation must therefore retain those existing lanes and add this helper after their failure. Combined-solver old-capture regression is required. No universal existence or robustness theorem is claimed.

## Reproduction and scope

Build `research/coulomb_trust_replay.cpp` against `spatial_backend/coulomb_trust.h`, pinned Float64 Bullet and nlohmann JSON with C++17, `-O3 -DNDEBUG -DBT_USE_DOUBLE_PRECISION -ffp-contract=off`. Pass capture paths as arguments. The executable reports helper acceptance separately from independent support-function and energy acceptance. `final-native-provenance.json` records the final helper/source/library/compiler and input hashes; `final-native-source.zip` preserves the final isolated source. Intermediate Python scripts and their receipts are exploratory numerical diagnostics, not prospectively frozen performance studies.

There is no material calibration, novelty claim, experimental validation or full-trajectory accuracy qualification in this directory. The numerical methods used here are established. The practical result is recovery of previously rejected captured circular-contact systems while keeping the original physical equations and final gates.
