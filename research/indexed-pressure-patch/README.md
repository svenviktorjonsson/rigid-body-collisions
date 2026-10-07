# Indexed local pressure-patch candidate

Rigid bodies with a small compliant contact patch, no deformable body mesh.
`model.py` evaluates unilateral local normal springs/dashpots and the existing
dynamic sliding coefficient at each pressure site. Integrating traction supplies
both resultant force and an independent contact couple. It includes combined
slip/twist, nonuniform pressure and normal-deformation rolling resistance.

This is a **frozen-frame instantaneous research candidate**, not an adopted
material law, a complete collision integrator or an experimental validation.
Static-friction/shear memory, evolving footprint/frame, brittle yield, material
calibration and coupled contact dynamics remain open. Zero-slip sites do not
resolve static friction. It supplements rather than replaces earlier failures.

The user's full relative velocity direction t and full angular velocity direction
s remain unchanged. `directional_residual` measures any wrench outside n/t and
s/n; no conventional patch resultant is silently declared equivalent to that law.

Foundation stiffness/damping and pressure geometry are independent inputs, never
new undocumented rubber/rock material constants. Existing restitution and static,
dynamic and rolling coefficients remain separate; this branch does not enforce
endpoint restitution on top of compliant energy exchange.

Primary precedent: Elandt et al., IROS 2019,
https://arxiv.org/abs/1904.11433. This simpler flat foundation is **not** an
implementation of their pressure-field intersection geometry or Drake.

The indexing contract is based on Vektor Flow's authoritative Section 0 semantics:
explicit contraction/reduction, one body owner index, flat contact/site/incidence
ranges, deterministic body accumulation. No language-repository changes or
compiler/GPU support claims are made by this external research.
