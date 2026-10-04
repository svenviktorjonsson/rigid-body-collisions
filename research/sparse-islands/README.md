# Sparse coupled contact performance

The implemented frozen-contact normal and Coulomb kernels keep one declared
physical law while choosing numerical algorithms from contact structure. They
do not perform collision discovery or integrate a full body trajectory.
No Vektor compiler, WASM or GPU performance is inferred from this experiment.

The [plan](plan.json) was committed before execution. Seven interleaved timing
repeats follow one warm-up per setting, with BLAS/OMP threads set to one.
Normal packed-row velocity error must be at most 1e-8 m/s. Cold totals include
input/contact construction, matrix preparation and solve. Solve timings exclude
matrix preparation; every repetition still factors and solves the problem.
Explicit operator array storage is reported, not peak process memory.

The archive in `results/states.zip` retains all final states/impulses and all
100 held-out irregular algebraic stress inputs, including four rejected cases.
`results/summary.json` includes every timing sample, source hashes and gates.
Synthetic coefficients are not measured material properties. Dense and sparse
versions of the same active-set algorithm provide a control distinct from the
older dense optimizer. Normal and friction fast paths retain the same residual
and energy gates. Redundant rigid contact pressures may remain nonunique.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.run_sparse_study
```

Use a separate checkout/output tree to preserve published evidence. Completed
cases and the partial timing journal are saved during execution. Compiler
porting remains a separate task in the private bootstrap handover.
