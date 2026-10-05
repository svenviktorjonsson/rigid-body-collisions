# Next rejected-contact review

This directory preserves independent diagnostics of failures reached after production source `95d224f1cfb36ec5e34bf4e5bd1a1e8484f9e8eb` recovered all earlier captures. Captured-system acceptance is not proof of a completed or refinement-qualified random-hull trajectory. Production source, study protocols and every earlier failed archive remain unchanged.

## Original circular Coulomb systems

The `probe.py` receipts retain every warm/cold natural-map or Fischer–Burmeister-normal trust-region trial. Only the original circular projection residual, finite impulse bounds and finite passivity bound accept an output; optimizer convergence does not. Both successful outputs have their tiny negative numerical pressures clipped to zero and the complete original gate recomputed in `nonnegative-certificates.json`.

| New seed42 capture | Independently successful search | Original residual (m/s) after clipping | Passive energy bound (J) |
|---|---|---:|---:|
| reference_0, 60 rows | Cold full-friction FB trust search, 81 function evaluations | 2.4013e-15 | −16.9465974 |
| reference_2, 51 rows | Warm full-friction FB trust search, 240 evaluations | 6.7872e-15 | −0.0795359 |

Warm natural and FB searches on the 60-row capture stall near 9.63e-5 m/s. For the 39-row reference_1, this review's four direct searches, 64 sticking/sliding mode combinations with three angular starts and seven normal-release subsets all fail the unchanged gate. Those failures are retained and do not establish infeasibility. Another independent reviewer found a strict solution by continuation followed by an opposing-slip contact-2 starting guess; its separate receipt is `research/new-hull-contact-review/fast_shake8_hulls42-reference_1.json`.

## Semidefinite normal-pressure recovery

The new seed7301 position capture has 291 rows with 97 normal impulses and zero configured tangent capacity. Its normal problem minimizes `0.5 pᵀ M p − bᵀ p` subject to `p ≥ 0`; the physical matrix M is unchanged. Unscaled SLSQP and FB/natural trust searches stall between 2.78e-8 and 3.87e-8 m/s; all attempts are retained in `normal291-probes.json`.

`normal_active.py` and the independently compiled `normal_null.h` apply a bounded active-set search. On an active face, write `g = M p − b`. An SVD gives the face's numerical null basis N. If its projected gradient is significant, use `d = −Nᵀ N g`, normalize d, move only until the first nonnegative pressure boundary, and release that pressure. Otherwise use the minimum-norm range Newton increment. New inward normal violations enter the active set. This is a numerical search over pressure faces; intermediate trial impulses are never applied to bodies.

A null direction of a positive-semidefinite face can support descent in the linear term when redundant normal targets differ. Selecting the correct boundary avoids combinatorial guesses about which pressures to remove. These matrices are floating-point approximations: the recorded full mobility response is small rather than identically zero. The largest complete trial velocity change from a null move is 1.82e-10 m/s in the new capture. The final unmodified mechanical gate remains mandatory.

| Native replay | States checked | SVD calls | Null/range moves | Original full residual (m/s) | Passive energy bound (J) |
|---|---:|---:|---:|---:|---:|
| New 291-row position capture | 4 | 3 | 2 / 1 | 1.4764e-16 | −0.00640216 |
| Earlier 54-row position capture | 5 | 4 | 2 / 2 | 1.0735e-17 | −0.000232510 |

The helper returns success only for finite pressure bounds, the original normal projection residual and finite passivity. It leaves caller output unchanged on every failure. Seven native controls verify unequal redundant targets, compatible redundancy, contradictory opposing normals, independent compression/separation, upper-bound rejection, bounded-work rejection and nonfinite-input rejection. The contradictory control rejects; no general infeasibility inference is made from search failure.

Limits are explicit: at most 384 normal rows, 128 active states by default (caller may set 1–512), relative SVD rank cutoff 1e-13, existing 64 Jacobi sweeps and retained-column correlation gate 1e-10, null-gradient trigger 1e-12 in the numerical search. No diagonal term is added to physical M. There is no material, contact-discovery or acceptance-tolerance change. `provenance.json` pins the source archive, compiler flags, binaries and limits. Costs are diagnostic and are not a full-engine performance ranking.
