# Rejected equivalent cone-map search

`research/coulomb_cone_probe.py` tested a numerical De Saxcé corrected-velocity cone map on the three new source-95d224f velocity failures. For Coulomb coefficient mu, the corrected velocity is (w_n + mu*norm(w_t), w_t); complementarity in the friction cone is equivalent to the original no-dilation normal complementarity and maximum-dissipation traction law. This is an alternative residual for numerical search, not a new constitutive law.

All six warm/cold trials declined under their declared budget, with original physical residuals approximately 6.13e-5/3.68e-5 (60 rows), 8.39e-7/9.72e-7 (39 rows), and 3.43e-7/6.44e-6 (51 rows). Intermediate merit or optimizer termination was never treated as physical acceptance. Trial receipts record the script hash.

The archived script's zero-friction projection Jacobian has an untested degenerate-cone edge case: at zero tangent vector the inside-cone derivative should project onto the normal ray. Every trial used mu=0.4; this issue does not affect these six receipts, but the script is not qualified for zero-friction cases. Preserve the original script when fixing that separate numerical edge case.

Bibliographic metadata verified via Crossref: De Saxce and Feng (1991), *New Inequality and Functional for Contact with Friction: The Implicit Standard Material Approach*, https://doi.org/10.1080/08905459108905146; and De Saxce and Feng (1998), *The bipotential method: A constructive approach to design the complete contact law with friction and improved numerical algorithms*, https://doi.org/10.1016/S0895-7177(98)00119-8. This verification covered title/author/DOI metadata, not full-text claims. No novelty is claimed for the cone formulation.
