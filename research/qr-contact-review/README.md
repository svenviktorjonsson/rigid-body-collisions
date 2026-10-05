# QR v1: preserved negative result

The research-only QR trial fails functional qualification and is not suitable
for production adoption. Baseline accepts 17 of the 20 frozen captured systems;
QR v1 accepts 16. The original 45-row seed42 reference 1 is the regression:
baseline residual is 1.75e-9 m/s while the QR variant fails its original 1e-8
gate. Every accepted QR result independently satisfies the original circular
law, nonnegative normal bounds, finite values and passive energy.

The QR trial changes only a numerical Newton direction. It uses column-pivoted
Householder QR to return a basic rank-deficient solution, leaving free increment
coordinates zero. Its physical mobility, coefficients and final gate remain
unchanged. Existing SVD runs when QR factorization, its model check or line search
fails. Pressure/nullspace extraction still uses the original SVD.

On the regressed case, QR accepts 999 search steps that provide insufficient
progress, exhausting the search work. A merit-decreasing basic direction can
change the search gauge and basin; falling back only after rejected steps does
not preserve baseline recovery. Reducing SVD calls alone is insufficient evidence
of an improvement.

The prospective 20-capture plan was committed at d0da9dc before execution.
`build-provenance.json` freezes exact 95d224f baseline header copies, modified QR
copies, compiler settings and both binaries. `results/validation-receipts.json`
retains all 40 untimed attempts, including failures and independent gates.
`executed_validation_driver.py` matches the exact driver hash in the validation
provenance. A later correction to the unused timing aggregation in
`run_experiment.py` computes the predeclared median of paired ratios; no timing
execution used either aggregation.
Source and production binary were not edited. Numerical fixtures test full rank,
rank redundancy, retained weak directions, inconsistent linearizations and the
1024-call QR allowance.

The timing experiment was **not executed because qualification failed**, as
recorded in `results/timing-not-executed.json`. Solver times embedded in untimed
validation receipts are diagnostic observations, not a speed-ranking result.
No speed claim or trajectory-accuracy claim follows from this experiment.

An additional prospective variant can examine a complete minimum-norm QR
correction and reject tiny-progress QR steps into the unchanged SVD path. This
directory remains the immutable v1 record.
