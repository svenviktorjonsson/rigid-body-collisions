# Larger velocity-contact recovery

Production now has a bounded mobility-null traction-seed fallback after every
existing recovery lane, including the original projection tail, declines.
Exact nonzero matrix/dependency connectivity preserves every contact triple and
coupling. Full systems are capped at4096 rows; each component at64. One warm
search plus at most six contact-drop seeds per component receives fresh2048
iteration/SVD caps. Numerical Jacobi factorizations use64 sweeps and1e-12 rank
cutoffs. These affect search coordinates only; original mobility, target,
friction, bounds, residual and passivity remain unchanged.

The search uses the right derivative of the original max(0,normal pressure)
cone radius at zero pressure. The older projection helper's default derivative
is unchanged. Numerical null relocation followed by cone projection can change
trial response; its maximum is exposed in native/Python policy metadata.
No rejected trial is applied. Final acceptance uses the original production
zero-budget gate, including eager cone projection and full finite passivity.

Actual162-row and297-row saved rejections now pass the new native helper and
unchanged production gate. Direct fallback residuals are7.187e-9 and9.956e-9m/s,
against the original1e-8m/s limit. The297-row input's difficult component has21
rows; another already accepted component has27. The423-row125-hull capture
still declines. These are instantaneous-system regressions, not completed
larger trajectories or reference accuracy/performance qualification.

All23 saved original replays pass with the new helper bypassed throughout.
Prior22 impulse/response Float64 bytes, including signed zeros, and all original
counters are exact.163 engine tests, ten contact-model tests, eleven existing
native checks, both actual-failure regressions, the390-row position regression
and a no-LAPACK inelastic-wall control pass. CI builds/runs both actual failures.
See validation/receipt.json and live23/receipt.json.

Earlier Python and native trials remain archived: projection and FB searches,
null seeds, inactive/support faces, traction directions, selected sliding faces,
and exact linear elimination. Most decline. Native root audits distinguish
independent-law acceptance from production acceptance: the older297-row result
passes its independent gate but crosses1e-8 after eager cone projection, so it
is explicitly retained as a production decline. Both right-derivative results
pass that final gate. The initial failed audit is preserved alongside its
successful recheck; no threshold was relaxed.

Reproduce generic direct recovery with `build/spatial/spatial_null_traction_replay`
and an original JSON capture. Production pipeline uses
`build/spatial/spatial_coulomb_replay`. Preserve outputs in fresh directories.
Next: fresh original64-hull trajectories and unchanged refinement gates;
continue423-row recovery and planar convergence before freezing the13-case
working baseline and running the separate per-case2x performance gate.
