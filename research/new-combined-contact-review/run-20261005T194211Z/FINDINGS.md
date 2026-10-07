# Terminal new combined-contact velocity failure

The archived 42-row velocity system has valid original-law roots. The native failure is a numerical search failure, not a certificate of bound infeasibility. This statement concerns the frozen instantaneous contact system, not the continued hull trajectory.

## Immutable input and planned search

`plan.json` and its hash preceded execution; the terminal capture path and SHA256, source commit, native replay executable and production-file guards are recorded there. Neither mobility, right-hand side, friction coefficient, bounds, dependencies, nor the archived 1e-8 m/s acceptance tolerance was changed. The captured impulse is the final runtime rejected iterate; it is not the runtime's original warm-start impulse. The existing native executable was not rebuilt.

## Reproduced decline and independent root

Replaying the unchanged captured system from its archived rejected impulse at budget4096 genuinely declines. The replay reaches256 SVD calls in each active, continuation and polish lane, and1024 SVD calls in each supplemental and support lane. The declined baseline receipts/exit2 remain recorded. Exhausted numerical budgets are not infeasibility evidence.

The exact normal and circular-tangent projection equations, with their analytic Jacobian, were solved in SciPy least squares. Five of eight bounded LM/TRF searches found roots; three stalled at residual about1.53933e-4 and are retained. Successful captured-seed LM used1221 function evaluations and TRF1293. For each successful result, only the impulse field in a separate research input changed; the unchanged native executable at budget0 independently accepted the original full normal/friction/bounds/passivity gates.

For captured-seed LM, the native accepted root has maximum active normal velocity1.7764e-15 m/s, maximum interior sticking velocity1.6653e-16 m/s, maximum friction support gap1.3280e-16 J, and passivity bound−3.6014610008861405 J. The largest normal impulse is about1.595, far below the1e10 upper bound. No restitution/friction relaxation or tolerance inflation was used.

## Why a small velocity residual is difficult

The42x42 mobility has numerical rank29 at relative1e-12 and exact nonzero connected components of sizes3,12,3,21,3. The active4-contact principal mobility has9 physical rank directions among12 impulse coordinates. Redundant pressure coordinates and tiny sliding velocities make a small velocity residual require a substantial friction-boundary rotation and pressure redistribution.

Normal row4 and tangent row30 (contact8's first tangent) agree to floating-point roundoff: maximum matrix coefficient discrepancy1.9984e-15, RHS discrepancy8.1567e-15. They are **not bitwise identical**. The earlier informal equality wording was corrected. In the accepted root, imposing active normal4 nearly annihilates that tangent velocity, so sliding contact8's first tangent impulse is nearly zero. The archived iterate instead has p30≈.28014. The accepted root changes normal4 by+.28015 and normal8 by−.12150 while sticking contact7's pressure changes only−6.57e-7. The root has contact4 slip speed3.1183e-4 m/s, contact8 slip speed4.9553e-5 m/s, and contact7 sticks.

This evidence supports changing the numerical residual/starting face, while preserving the constitutive equations and full acceptance gates. It does not establish uniqueness or general optimizer convergence.

## Additional numerical guide and retained failures

A prospective exact-equality row detector found no guide and declined before any numerical solve. Its failed control is retained. An initial launcher also failed before execution because its exec namespace lacked `__file__`; the original source/stderr and corrected version remain.

A separately planned roundoff-relation guide zeroes contact8's first tangent impulse and saturates its orthogonal impulse against the archived slip. This changes only the numerical starting impulse. The unchanged native replay accepts this guide with budget4096 after3432 total PGS sweeps, including the existing recovery lanes. The same guide at256 genuinely declines. SciPy LM from this guide stalls; TRF finds a native-accepted root after1326 evaluations. It is therefore a useful numerical guide, not a universal or proven cheap repair.

All source/runtime/capture guards passed after these experiments. No production file, executable, library, archive, or live world state was written. Costs are not ranked.

## Prospective implementation validation

The next frozen23-input plan contains the original22 accepted captured systems plus this new42-row terminal capture. An isolated candidate must run the existing solver first and attempt any new projection recovery only after all old lanes fail, using the actual rejected seed. Thus the22 prior accepted endpoints and original work receipts should remain exactly unchanged. The new candidate must retain finite explicit per-call caps, preserve caller impulses on decline, and satisfy every original global normal/friction/bounds/passivity gate. Full-world continuation and refinement remain separate root-owned work.
