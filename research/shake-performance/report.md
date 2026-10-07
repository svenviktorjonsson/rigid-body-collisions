# Verified fast shaking: repeated whole-trajectory cost

The verified fast setting is **6.11 times faster** in median native stepping time than the resolved reference in this predeclared six-run comparison. Every repetition passes unchanged trajectory and physical budgets against both previously qualified references. Histories repeat bitwise within each setting.

| Setting | All three native times (s) | Median (s) | Qualified |
|---|---|---:|---|
| reference | 18.184, 20.211, 17.719 | 18.184 | True |
| candidate | 3.337, 2.809, 2.975 | 2.975 | True |

The scene is 27 spherical rigid bodies inside a six-wall container, shaking at
plus/minus 20 m/s with reversals at 0.04 and 0.08 s, simulated for 0.12 s.
The synthetic pair friction is 0.4, restitution is zero and gravity is 9.81 m/s².
Geometry, masses, full inertia, contact law and prescribed motion are identical.
The candidate uses travel fraction 0.015, four primary steps and 10 ms output
frames; the reference uses fixed 1.25 microsecond internal steps and 5 ms output
frames. Comparisons use common 10 ms times. No material parameter was adjusted.

Full RMS budgets: 5 mm position, 0.05 m/s velocity, 0.1 rad/s spin and 0.01 rad
orientation. Every run also passes quaternion norm, finite energy change minus
measured wall work and actual internal-update surface containment. Both reference
ladders independently pass quarter budgets. Original failed refinement evidence
remains unchanged in the earlier archive.

Execution source: `63330336b8d0839d971dcd500e5a7ed977defb5e`.
Native stepping includes state recording and excludes process/JSON overhead.
Whole-process timings and every repetition are retained in the summary and raw
histories. Runs launch sequentially in a predeclared interleaved order. Heavy
team experiments were paused during execution; external host load cannot be fully
controlled. This is a scoped cost comparison on one environment, not a universal
engine ranking, experimentally calibrated material result or Vektor-native gain.

The faster setting is selected by full-trajectory evidence, not by assuming a
contact residual proves trajectory accuracy. General random-hull trajectories
still reject, and arbitrary many-body elastic angular history is not integrated.

Reproduce from the frozen source with `python -m research.run_shake_performance`;
independently audit with `python -m research.audit_shake_performance`. Plans,
source snapshots, source and binary hashes, six histories, physical checks,
repeated-state checks and all timings are retained in `results/`.
