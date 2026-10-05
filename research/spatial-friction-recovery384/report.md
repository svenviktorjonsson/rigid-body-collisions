# Larger contact recovery: retained full-trajectory results

The 267-row later hull system passes native and independent contact/passivity checks
with residual **1.97e-12 m/s** after raising the numerical recovery row limit from
256 to 384. Its global recovery work budget remains 256 SVD calls; the successful
captured-system replay needs four. This establishes a numerical capacity fix.

The separately frozen rotating/shaking 27-hull full-trajectory follow-up retains
**three rejected attempts and no completed histories**. The later contact failures
remain unqualified under unchanged accuracy and physical gates.

| Attempt | Failed residual | Outcome |
|---|---:|---|
| fast_rotate_shake27_hulls7301/reference_0 | 0.00254761 m/s | Rejected |
| fast_rotate_shake27_hulls7301/reference_1 | 0.00453103 m/s | Rejected |
| fast_rotate_shake27_hulls7301/reference_2 | 1.79796e-05 m/s | Rejected |

Execution source: `4910fbca7cacf8dfd218be97a082d32b7ea13ea2`.
Authored geometry, density/mass/inertia, friction, restitution, gravity, prescribed
20 m/s translation/reversals and 10 rad/s rotation, timesteps and travel fractions
are unchanged. Both earlier failed archives remain intact. The new capacity
permits resolving one later system; subsequent unresolved systems still reject.

Sources, exact plan, authored scenes and every error are retained in `results/`.
Audit with `python -m research.audit_spatial_recovery384`. The appended 267-row
fixture and capture provenance are in `research/coulomb-diagnostics/`; its native
replay is mandatory in CI. The 45-row second seed42 system remains unresolved.
No claim of general random-hull feasibility, material authenticity, convergence
order or arbitrary-body trajectory accuracy follows from these captured solves.
