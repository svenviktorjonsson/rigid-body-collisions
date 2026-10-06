# Shear and rocking contact hypothesis

This experiment tests a passive compliance extension without modifying the production solver. It fails the measured-prediction improvement requirement and is not adopted.

For a fixed lever arm R and inertia I, angular impulse from tangential force satisfies I Δω = -R Jt. Together with a supplied contact tangential restitution, this fixes outgoing spin regardless of shear-force history. Merely changing stiffness/damping cannot repair the measured pad discrepancy at the same restitution. The existing elastic prototype's normal-axis twist also cannot supply the required horizontal-axis torque.

The new isolated candidate adds a horizontal-axis rocking spring to a tangential contact spring. Its positive potential is `0.5*k*(x-theta)^2 + 0.5*b*theta^2`, with dimensionless mass/radius 1 and inertia 0.4. All retained strain energy at separation is explicitly recorded as dissipation. This is a declared finite-duration hypothesis, not a derived deforming patch or a calibrated normal-force history. The given normal restitution remains separate, rather than adding normal damping on top of it.

The first shear-stiffness root is selected from supplied tangential restitution alone. A shared rocking stiffness is selected on three surfaces; the fourth spin outcome is withheld. Four such folds are run. The prospective plan defines bounds and branch selection before outcomes. The 161-point shared-parameter grid is limited; its result is not a global optimality proof.

| Withheld surface | Measured spin factor (rad/m) | Current model | Candidate | Shared rocking stiffness |
|---|---:|---:|---:|---:|
| Granite | 14.9 | 15.510 | 15.510 | 0 |
| Rubber sheet | 14.5 | 14.677 | 14.677 | 0 |
| Superball pad | 18.2 | 16.343 | 15.841 | 0.375 |
| Tennis strings | 9.0 | 9.368 | 9.368 | 0 |

Pooled held-out spin RMSE is **0.999 rad/m for the current law**, **1.235 rad/m for this candidate**, about **23.7% worse**. No improvement is claimed. All 36 analytic controls and 40 independent integration/history-energy checks pass. These establish numerical/passivity consistency, not experimental realism. Integrated resultant friction/moment budgets pass for the four folds under the disclosed mu=0.9 and moment-length=0.3R hypotheses. Instantaneous force/patch admissibility and physically determined duration are not established.

Next development should derive a force distribution and its horizontal-axis moment from deformation and support response, supported by measured geometry/load history. Do not insert an independently fitted torque or offset for each surface. Additional material and contact-duration data are necessary to distinguish competing explanations; the existing four table rows alone cannot identify a full contact law.

Run `python research/shear-rocking-contact/experiment.py`, then `python research/shear-rocking-contact/audit.py` in the repository environment. Preserve results before rerunning. Source: [Cross (2010), Table I](https://physics.usyd.edu.au/~cross/PUBLICATIONS/48.%20EnhanceBounce.pdf). The supplied restitution comes from these same experiments, so this remains a conditional spin test.
