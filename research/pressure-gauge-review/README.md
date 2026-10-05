# Neutral pressure guidance for the later 39-row failure

These are captured-system numerical diagnostics of production source `95d224f1cfb36ec5e34bf4e5bd1a1e8484f9e8eb`. They do not qualify a full hull trajectory. Physical A, b, friction and the original 1e-8 m/s acceptance tolerance remain unchanged.

## Why the previous numerical start matters

The independently successful continuation base and the native failed-restart output differ by as much as 0.0162451 N s in their contact impulses. Their full mobility responses differ by only 1.479e-11 m/s. Different distributions of nearly redundant contact impulses therefore produce almost identical body motion, but initialize different friction disks and Newton faces. A restart that opposes slip at one contact is sensitive to that pressure distribution.

Canonical constrained minimum-norm and pressure-extremum trials are retained, including their optimizer failures. The all-row gate rejects almost all outputs. One cone-infeasible pressure-maximum numerical trial followed by an opposing-slip guess reaches a strict root; this is evidence that neutral relocation can help, not a claim that the canonical constrained optimization succeeded. `canonical-positive-only-probes.json` records the interrupted first pass; `canonical-probes.json` records the complete larger near-zero-normal pass. Every completed trial is retained.

## A guide derived only from the current system

Collect rows belonging to positive-pressure contacts or contacts whose normal velocity is within 10 tolerances of zero. An SVD of the full mobility columns on those rows yields a numerical right-null basis N at relative rank cutoff 1e-13. Its projector `Q = N Nᵀ` is independent of the SVD basis orientation.

For a candidate contact, construct the existing opposing-slip target by changing only its two tangent impulse coordinates to `−μ max(p_n,0) w_t / ||w_t||`. Let `d = Q (target − p)`. A bounded numerical relocation `base = p + α d` retains essentially the same complete contact velocity. Apply the same contact's opposing-slip guess at the relocated pressure, then solve the original coupled circular equations. These trial impulses are search initializations; they are never applied to a body, and may violate individual friction capacities. Only the final original all-row residual, finite bounds and finite passivity can accept the output.

| Numerical base | Contact | Fraction α | Full neutral velocity change (m/s) | Final original residual (m/s) | Final passivity bound (J) |
|---|---:|---:|---:|---:|---:|
| Native output after failed restart attempts | 2 | 0.25 | 2.318e-16 | 1.277e-15 | −0.489904899 |
| Actual native 40-stage continuation candidate before any restart | 2 | 0.5 | recorded in native-base receipt | 3.731e-16 | negative, independently gated |

Both receipts contain derived directions, initial guesses, raw roots and roots after tiny negative pressures are clipped to zero and the original gate is recomputed. No accepted archived impulse is used as a target. The first successful Python guide uses 251 Jacobian/SVD evaluations and 270 function evaluations, recorded separately in `quarter-contact2-work.json`. The independent native Moré implementation recovers its derived initial guess in 136 SVD calls, original residual 9.962e-9; its receipt is `research/active-direct/39-native-neutral-more-v2.jsonl`.

A useful basis-independent ordering is the tangent impulse's fraction of its friction capacity among contacts still failing their tangent residual. In the actual native candidate, these ratios are 0.9811 at contact 2, 0.7253 at contact 3 and 0.4194 at contact 5. The closest disk boundary prioritizes the successful contact; a largest-residual ordering spends work on contact 3 first. A prospectively bounded fraction schedule can test 0.5 and 0.25 before larger relocations. This is an initialization heuristic, with mandatory unchanged acceptance, rather than a guarantee of convergence or a new contact model.

Failures of other contacts/fractions and of all canonical searches remain visible. No general infeasibility, novelty, experimental authenticity or whole-engine performance claim follows from these captured roots.
