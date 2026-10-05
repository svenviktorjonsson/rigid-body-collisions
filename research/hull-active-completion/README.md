# Full active-contact completion attempt

The six prospective runs use execution source
`95d224f1cfb36ec5e34bf4e5bd1a1e8484f9e8eb`, with exact source and native binary
hashes. Authored scenes, materials, discretization controls and accuracy gates
match the earlier shared-point protocol. No attempt was interrupted or omitted.

All three seed-42 lanes reject later velocity systems (60, 39 and 51 rows).
Seed-7301 reference 0 rejects a 291-row normal position projection. Its reference
1 and 2 complete all 12 output frames and independently pass their quaternion,
containment, contact-residual and endpoint energy-minus-wall-work gates. Costs
are descriptive under concurrent workload: 503.49 and 220.64 seconds for 0.12
seconds of simulated motion. No speed-ranking conclusion follows.

**Zero references qualify.** Besides the missing first reference edge, the
completed seed-7301 pair fails the original quarter-budget RMS comparison:

| Quantity | Observed difference | Required limit |
|---|---:|---:|
| Position | 0.108616 m | 0.00125 m |
| Velocity | 6.12246 m/s | 0.0125 m/s |
| Spin | 74.2622 rad/s | 0.025 rad/s |
| Orientation | 1.33364 rad | 0.0025 rad |

Complete histories, rejected systems and accepted output-frame prefixes all
remain archived. The progress auditor independently converts principal-inertia
quaternions back to authored axes and verifies full tensors and wire geometry.
Partial prefixes never qualify a trajectory. Endpoint energy balance also does
not prove local rotational-energy preservation by angular split pose correction;
the separate retained split counterexample and translation-only option remain
relevant.

The four rejected systems have strict, passive independent diagnostic solutions
in `research/new-hull-contact-review`. The standalone normal-null prototype in
`research/active-next-review` solves the 291-row capture in three SVD calls.
These results support further numerical recovery work, not retrospective
acceptance of this failed study or authentic rubber calibration.

```sh
python -m research.audit_hull_active_completion --source-commit 95d224f1cfb36ec5e34bf4e5bd1a1e8484f9e8eb
python -m research.audit_new_hull_contacts
```
