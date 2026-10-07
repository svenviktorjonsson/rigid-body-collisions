# Native pressure-gauge recovery: full hull follow-up

Two independently verified frozen contact systems recover with residuals
4.81e-9 and 1.53e-10 m/s, using a cold restart and a mechanically neutral
pressure redistribution respectively. The native implementation reproduces
both results and separately checks the unilateral normal law, circular capacity,
maximum dissipation and finite passivity bound.

The full-trajectory study retains **six attempts and zero completed histories**.
All six therefore remain unqualified. This is a numerical improvement for two
contact systems, not a claim of general random-shape accuracy.

| Attempt | Failed velocity residual | Outcome |
|---|---:|---|
| fast_rotate_shake27_hulls7301/reference_0 | 9.902e-08 m/s | Rejected |
| fast_rotate_shake27_hulls7301/reference_1 | 0.00453103 m/s | Rejected |
| fast_rotate_shake27_hulls7301/reference_2 | 1.79796e-05 m/s | Rejected |
| fast_shake8_hulls42/reference_0 | 1.06032e-05 m/s | Rejected |
| fast_shake8_hulls42/reference_1 | 1.16451e-08 m/s | Rejected |
| fast_shake8_hulls42/reference_2 | 3.47786e-05 m/s | Rejected |

Execution source: `f6d03878e7d187b18be8d590c6e66f976232e020`.
The source and plan were committed before execution. The archives preserve the
exact source bytes, authored geometry, checksums and every error. The paired
baseline is the gyroscopic-RHS follow-up; geometry, materials, gravity, prescribed
translation/rotation, timesteps, travel fractions and physical/trajectory gates
are unchanged. Only the numerical contact search gains bounded recovery.

The ordinary iterative solve runs first. Recovery uses the same circular-contact
projection equations, a numerical SVD Newton step, mechanically neutral gauge
moves to neighboring impulse-boundary faces and a cold Newton restart. It adds
no compliance to the physical mobility and substitutes no friction pyramid.
Recovery is restricted to 256 rows, at most 64 Newton steps per attempt and a
global ceiling of 256 SVD calls, each with at most 64 Jacobi sweeps. Unconverged
numerical steps reject. Complete contact residual and finite energy/work gates
remain mandatory. Work counters are part of the native output.

The successful frozen systems can still be followed by different, unresolved
systems later in a simulation. Those later failures must be diagnosed before
promoting an arbitrary-body accuracy preset. The effective contact model remains
synthetic and rigid; independent elastic couple/history integration into the
many-body engine and experimental material calibration remain open.

Reproduce with `python -m research.run_spatial_recovery` from the frozen source;
audit with `python -m research.audit_spatial_recovery`. The captured-system solver
and certificates are in `research/coulomb-diagnostics/`.
