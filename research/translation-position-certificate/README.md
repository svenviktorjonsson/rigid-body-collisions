# Exact finite-bound certificate for the 74-row position rejection

The retained seed7301 reference_2 translation-only position-repair matrix has no
impulse satisfying its original nonnegative finite bounds and the exact frozen
normal projection equations to 1e-8 m/s. This is a certificate about one numerical
repair subproblem, not about feasibility of the physical Coulomb velocity law,
the authored geometry, or every random-hull trajectory.

Capture: `research/translation-position-diagnostic/results/rejected-normal-system.json`,
SHA-256 `c26bc4e45377fdd7fa069b41f355065c6deed9d112de0fdb58fd26c173a3d617`.
Execution source: `23751e6a29b0c5f8241e16581dd54ac172a80bdf`.
The capture has 74 normal rows, upper bounds 1e10, original tolerance 1e-8 m/s,
and observed rejection residual 1.0607115809e-4 m/s.

An approximate normalized left-null LP identifies rows 11, 54 and 55 as a
discovery guide. That LP includes small negative weights and nonzero row error;
its success status is not a certificate. Exact rational elimination on the
selected principal block constructs nonnegative weights summing to one:
approximately 0.11489686070142884, 0.43524438325305803 and
0.4498587560455131. `witness.json` stores their exact integer ratios.

The standard-library-only independent verifier uses exact rational values of
every round-tripped binary64 input. It retains every original matrix column,
including tiny positive coefficients; no null residual is rounded to zero.
For this witness y, c = y A and 0 <= p <= upper imply

```
y A p <= sum_j max(c_j, 0) upper_j = 1.7471591374e-5 m/s
y b                               = 4.9386342780e-5 m/s
```

For positive A_ii, the original normal projection residual
`abs(p_i - max(0, p_i - w_i/A_ii))*A_ii`, with `w = A p - b`,
is at least `max(0, -w_i)`. Passing all original rows therefore requires
`y w >= -1e-8 m/s`. The two exact bounds above instead imply
`y w <= -3.1914751406e-5 m/s`. The original maximum residual is consequently
at least **3.1914751406e-5 m/s**, approximately **3191 times the gate**.
The strict rational margin, not those printed decimal approximations, proves
the contradiction. No complementarity assumption beyond the necessary inward
velocity bound is needed.

The finite upper bounds are essential to this particular certificate. It does
not prove infeasibility with unbounded pressure or under perturbed coefficients.
It certifies exact evaluation of the frozen binary64 equations; it does not prove
that floating-point accumulation at enormous pressures cannot spuriously pass a
native residual calculation. Four controls cover inconsistent and feasible
opposing rows, finite-bound dependence, and a corrupted real-capture witness.

More solver effort cannot produce an exact passing answer to this captured
bounded repair problem. A future repair policy needs explicit geometry/event
rejection or a separately declared pose-repair/refinement protocol; its complete
trajectories must be requalified. This evidence changes no solver, physical
matrix, material, timestep, tolerance, historical outcome or compiler input.
The separate 243-row velocity capture has independently demonstrated roots;
this position certificate does not override that evidence.

Reproduce the exact verification without NumPy, SciPy or native builds:

```sh
python3 research/translation-position-certificate/verify.py
python3 -m unittest discover -s research/translation-position-certificate -p 'test_*.py' -v
```

`generate.py` requires NumPy/SciPy and refuses to overwrite its witness. Its LP
is only a discovery mechanism; the stored witness can be checked independently
without reproducing that numerical search. Hosted validation now checks the
certificate and controls on Python 3.11 and 3.12.
