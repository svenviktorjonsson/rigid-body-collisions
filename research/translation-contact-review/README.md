# Independent review of the next 261-row velocity rejection

The frozen six-attempt study source is `52f7e6d244e92a8405134e0222ce14fc3eda0ef6`. Its seed7301 reference_0 velocity capture is pinned by SHA-256 `7b7a7649645426d21030a51310cdc235a417cbc830f0a13cfa951d4581c018d6`. Production, binaries, authored scenes, material and the original 1e-8 m/s acceptance tolerance are unchanged by this review. Captured-system feasibility is distinct from full trajectory/refinement qualification.

The 261 scalar contact rows form one exact mobility-connected component, including inactive constraints. There are 87 contacts, of which the native rejected iterate loads 12 contacts (36 rows). Numerical searches may restrict impulse coordinates to this support, but acceptance always recomputes all 261 original rows, including contacts provisionally assigned zero pressure. Omitting their mechanical gate would be incorrect.

Full warm FB trust least-squares and reduced warm/cold FB or warm natural searches stall near 9.73e-6 m/s. Their complete receipts and optimizer-success flags are retained. Worst remaining residuals are normal velocities at contacts 31, 29 and 30; tangent equations already pass. The active normal rows are full rank 12, with smallest singular value 1.724e-5 and largest 17.971. This is a nearly dependent pressure/mode configuration; there is no normal-target infeasibility certificate. Search failures do not prove global infeasibility.

A whole-contact release guess succeeds. Dropping contact 31 fails (all-row residual 1.53e-5); dropping contact 29 and solving the remaining 33 active rows satisfies the complete original equations. Both attempts remain in `normal-release.json`.

## A contact-index-independent rule

`native_seed_ranked_release.py` derives the loaded contacts directly from the captured native iterate. It ranks contacts with positive pressure and positive normal velocity by outward normal velocity, then tests at most four complete-contact releases. There is no encoded contact 29 or accepted impulse target. The first derived guess passes in eight FB/TRF function and Jacobian evaluations:

- Original all-row residual: **3.9715e-15 m/s**.
- Normal pressures: nonnegative and within the original upper bounds.
- Frozen-system passive energy bound: **−100.9525566 J**, finite.
- Independent physical normal complementarity, circular friction capacity, maximum-dissipation support and nonpositive friction work: pass.

The original iterate ranks contact 29 first because its outward normal velocity is 2.8277e-5 m/s; next are 31, 27 and 70. Ranking by pressure times outward velocity would select contact 27 instead. A second generic proof bootstraps the 36-row support, then applies the same velocity ranking; contact 29 again ranks first and the complete original gate passes in eight additional evaluations. Its failed bootstrap remains archived.

The rule is a bounded numerical initializer, not a guarantee that this support or any release works for arbitrary systems. New inward violations at omitted contacts must remain rejection reasons or cause a separately bounded support expansion. A native implementation still needs its usual dimension/input/work validation and final all-row acceptance. No altered matrix compliance, friction radius, restitution or acceptance gate is introduced.

`audit.py` imports neither the solver nor its contact-system helper. It recomputes the full original circular projection gate, pressure bounds, normal complementarity, disk capacity, maximum-dissipation support, friction work and finite energy bound from the archived matrix and impulses. Its verdict and every accepted/rejected released-support candidate are recorded in `independent-audit.json`. Sources, frozen input and diagnostic import hashes are pinned by `provenance.json` and `diagnostic-source.zip`. Native integration and a new full-hull study are separate work.
