# Contact-direction recovery continuation — 2026-10-05

The frozen research-only row-relation recovery at public source
`5666bbc` passes the prospective 23-capture comparison on this Ubuntu host.
The existing solver accepts the original 22 systems and declines the new
42-row velocity system. The candidate runs the existing solver first and
invokes the extra helper only after its entire recovery pipeline declines.
Every candidate passes the unchanged native and independently recomputed
normal/circular-friction/finite-bound/passivity gates.

All 22 existing impulse arrays, resulting velocities and original solver
counters compare exactly; the helper is bypassed in every case. The new
system uses the actual final rejected seed, one 21-row exact component,
one numerical direction guide and 1,800 component sweeps. Its maximum
projection residual is 8.907309345473049e-9 m/s against the original 1e-8
gate. The separate direct-helper test, initialized from the archived impulse,
also succeeds, with 3,432 sweeps. These are different initial numerical seeds.
Near-equal row detection changes a search seed only; it does not delete
couplings or change mobility, material, tolerance or pressure bounds.

Nine C++ controls pass: absent relation, invalid tolerance, nonfinite RHS,
asymmetric matrix, invalid friction bound, invalid dependency, an oversized
connected component, reused statistics and the full-row limit. Every decline
preserves caller impulses. The original eight-control version and its
execution receipts are retained alongside the nine-control version.

The Python suite reports 109 passed, 53 skipped and 31 subtests passed.
Compiled-engine tests skip because the usual production build paths do not
exist in this checkout; this is not a complete engine regression receipt.
Float64 Bullet dependencies and three replay executables were built in a
separate cache tree with LAPACK recovery and contraction disabled. All 35
guarded production source files retain their hashes. Timings reflect concurrent
host load and establish no speed improvement.

`results.zip` retains all 46 baseline/candidate outputs and receipts, direct
helper results, compile commands and diagnostics, final guards, both control
receipts, regression output and execution log. `receipt.json` pins the archive.
The captured inputs remain in their existing immutable repository paths.
Recheck with:

```sh
python research/relation-continuation/audit.py
```

`run.py OUTPUT_DIR DEPENDENCY_BUILD_DIR` reproduces the frozen direct and
23-capture comparison using the declared Bullet dependency build. It requires
a new output directory. The C++ controls have their exact compilation commands
in the archive. No production engine/header or executable was changed, and no
world trajectory was executed in this continuation.

Next work is to strengthen the helper's production acceptance and cost-counter
contract, integrate it after the existing tails with transactional rejection,
then rerun complete hull histories and both refinement edges. A solved contact
matrix does not qualify those trajectories. This remains external Python/C++
research, not a Vektor compiler port or Section 0 acceptance.
