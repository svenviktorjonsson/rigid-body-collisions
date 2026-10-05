# Independent completion review: large contact systems

The retained 231-row and 321-row velocity rejection captures both have solutions
to their original circular Coulomb law at the unchanged 1e-8 m/s gate. This rules
out physical infeasibility for these captures. It does not qualify subsequent
trajectory steps, whole-step energy, or an experimentally calibrated material.

| Candidate | Full original rows | Residual, m/s | Passive kinetic-energy change, J |
|---|---:|---:|---:|
| Independent 231-row trial | 231 | 2.05e-16 | -0.773343197 |
| Independent 321-row restart | 321 | 6.13e-16 | -1.260432083 |
| Final native seed42 reference 1 | 45 | 1.47e-9 | -1.204860016 |
| Final native seed42 reference 2 | 48 | 9.39e-15 | -0.282046018 |
| Final native seed7301 reference 2 | 231 | 8.87e-14 | -0.773343197 |
| Final native seed7301 reference 0 | 321 | 1.01e-11 | -1.260432083 |

## Search and acceptance

The independent search initially selects all three rows for each contact with
captured warm normal impulse above 1e-9 N s. The 231-row capture has nine selected
contacts (27 rows); the 321-row capture has 31 selected contacts (93 rows).
Unselected contacts have zero trial impulse. Acceptance always evaluates every
row of the full original capture; active-face omission by itself is not evidence
that a contact is inactive.

The trial normal pressure guide minimizes the original normal-only quadratic
energy with nonnegative pressure. Friction continuation changes coefficients
only in numerical search problems, from zero to the original coefficient. Normal
Fischer–Burmeister residuals are an alternative zero-equivalent complementarity
representation. SciPy TRF uses their analytic Jacobian. Physical mobility,
original coefficient, and final tolerance remain unchanged.

The 231-row search converges through ten continuation stages. The 321-row search
stalls at 3.25e-7 m/s, exclusively at opposite tangential residuals on contacts 69
and 70. Both tangential impulses lie inside their disks, making the current face
nominally sticking. Setting only contact 69's trial tangential impulse to its
original friction capacity opposing the current residual slip permits recovery
in 14 TRF evaluations. The final original-system gate, rather than the guessed
mode, establishes physical validity. `reference_0-sticking-mode-trials.json`
retains the initial, raw returned, and verified final impulse and velocity.

Tiny negative normal roundoff is explicitly clamped before independent final
verification. The original 231-row raw search has a normal impulse about
-1.1e-24 N s. The auditor records this correction and recomputes the full law and
energy. Normal impulses must be nonnegative in the final independent audit.

The independent auditor imports no engine acceptance implementation and
recomputes normal projections, circular tangential projections, finite values,
normal bounds and passive energy from captured A and b. Both independent
candidates and the two intermediate native v3 candidates pass
`independent-final-audit.json`. The final native four-capture receipt passes
`independent-final-native-audit.json`. The final native 321-row recovery uses the
full aggregate 256 SVD-call allowance; the 45-row case uses 218. Subset recovery
is a numerical performance opportunity, not a proven speed superiority claim.

## Retained failures and limitations

The ten earlier pressure restarts at contacts 30/31/32 targeted the initial warm
residual cluster, which was already solved after continuation. They are retained
as failures rather than evidence of infeasibility. Natural-map continuation and
direct LM also fail the 321-row gate in the recorded trials.

For the 45-row capture, plain boundary restarts, neutral gauge relocation,
minimum-norm Newton, residual preconditioning and normal-face release are
retained. Plain TRF after 500 evaluations passes marginally around 8.9e-9 m/s.
No independent trial establishes a strong fast root there. Two exact gauge
directions and weak Jacobian singular values around 1e-6 and 1e-8 explain why
direct merit descent progresses slowly. Some optimizers report success while
the full physical gate rejects them. The final native method combines a slip
boundary guess with ranked pressure-face release; its independently verified
1.47e-9 residual has margin. Neither plain restart nor plain release reproduces
that result in this review.

No solver guess is physically applied. No regularization is added to A, and no
friction coefficient or acceptance tolerance is relaxed. Recorded wall times are
descriptive diagnostics from a shared machine, not controlled benchmarks.

The native arbitrary-body Coulomb lane is an inelastic rigid impulse model.
Persistent elastic tangential/twist storage and spin-reversing rubber-like
fixed-sphere examples belong to the separate elastic prototype and archive.
These capture results do not establish that those prototype mechanics have been
integrated into the native arbitrary-body world.

## Reproduction

Run from the repository root with NumPy and SciPy installed. Restrict BLAS thread
counts for small dense diagnostic solves.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.large_contact_completion_review research/hull-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_2.json --stages 10 --representation fb --method trf --start normal-qp --subset
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.large_contact_completion_review research/hull-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_0.json --stages 10 --representation fb --method trf --start normal-qp --subset
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.large_contact_sticking_review
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.audit_large_contact_completion --final-native
```

The continuation module accepts 40-stage and natural/LM variants to reproduce
the retained failures. `research.large_contact_mode_review` reproduces the
unsuccessful initial pressure guesses. `research.completion_45_sticking_review`
and `research.completion_45_followup_review --experiment …` reproduce the bounded
45-row trials. Reproduction rewrites the selected output; preserve this archive
when experimenting with new solver settings.
