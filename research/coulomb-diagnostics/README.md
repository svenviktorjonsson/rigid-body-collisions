# Recovering rejected circular Coulomb contact systems

Two frozen matrices captured from rejected random-hull simulations now admit independently verified impulses without changing the mobility matrix, targets, friction coefficients, restitution or acceptance tolerance. Eight independent diagnostic tests pass. These results concern contact systems; full trajectories must still be tested with a native implementation.

| Snapshot | Rows | Original residual (m/s) | Recovered residual (m/s) | Recovery |
|---|---:|---:|---:|---|
| Hull seed 42 | 18 | 1.77895e−8 | 4.80955e−9 | Cold semismooth Newton restart |
| Hull seed 7301 | 108 | 2.30410e−5 | 1.53184e−10 | Mechanical-nullspace relocation to a cone boundary, then Newton |

Both use the original **1e−8 m/s** residual gate and also pass the frozen-system mechanical passivity bound. The independent solver preserves normal complementarity, the circular tangential impulse bound, and the full normal/tangential and inter-contact mobility coupling. It does not minimize energy over an associated friction cone, which would change the normal contact law. One scalar tangent step size preserves changes of the tangent basis.

For seed 42, the warm iterate has an ill-conditioned Jacobian: its smallest singular value is approximately 5.03e−10 and its condition number is about 4e10. Its Newton direction requires large positive and negative pressure changes and the line search stalls. Reinitializing the **same frozen problem** at zero impulse finds a verified adjacent contact state in seven Newton steps. Warm-starting is therefore an optimization whose failure must not prevent a verified restart.

For seed 7301, the current active normal and sticking conditions impose 22 equations with rank 21. Their target vector has an approximately 8.19e−6 m/s inconsistency in that fixed face. Optimizer convergence in the face is not a contact solution. A right-null Jacobian direction also lies in the mechanical mobility nullspace: its maximum mobility response is 2.73e−15. This direction redistributes tangent impulses between neighboring contacts 5 and 6 without changing the rigid-body contact velocities. Following it to the first feasible circular-friction boundary at contact 6 changes velocity by at most 1.21e−15 m/s. Two subsequent Newton steps find a verified contact solution.

A strict-decrease merit line search rejects this initially neutral relocation and can remain trapped in an inconsistent sticking face. The recovery algorithm instead enumerates both signs of each eligible null direction, stops at the first unilateral-normal or friction-cone boundary, verifies mechanical neutrality and impulse feasibility, and restarts the ordinary contact solve. Only the final unchanged physical gates accept an impulse. Numerical rank truncation applies to Newton effort; no softness is added to the mechanical matrix.

`research/coulomb_diagnostics.py` contains the independent analytic Jacobian, SVD Newton steps, a disclosed least-squares fallback, normal-target inconsistency diagnostics, and deterministic recovery. The captured inputs retain the original failed iterates and tolerances. Recovery JSON files retain all attempted restart outcomes and the successful gauge certificate. `manifest.json` hashes every input, output, solver and test source. The captured matrices came from public project-generated numerical scenes, not experimental rubber or impact measurements.

Replay:

```sh
python -m unittest tests.test_coulomb_diagnostics -v
python -m research.coulomb_diagnostics research/coulomb-diagnostics/hull42-rejected.json
python -m research.coulomb_diagnostics research/coulomb-diagnostics/hull7301-rejected.json
```

This demonstrates an actionable numerical recovery for two previously rejected systems. It does not establish that every random-hull contact system is feasible, that frictional trajectories converge, or that the mechanism is novel in contact mechanics.
