# Native hull-cache constructor control

All **8/8 controls pass**, independently audited. Frozen execution source is `de7eead49ac732a24a3d6ffe570fe877e793d64a`, published before compilation/execution. A separate research executable links existing read-only double-precision Bullet libraries; the production executable, headers, static libraries and runtime dependencies were unchanged through the run. No world/engine simulation was performed. Concurrent workload makes compilation/execution time descriptive only.

The old constructor recalculates a hull's cached bounds at Bullet's default 0.04 m margin, then sets the declared margin. Native support uses the declared margin, but the cached bounds keep the old padding. Moving `setMargin` before `recalcLocalAabb` and compound insertion removes this inconsistency. Both variants have identical declared vertex support in every control.

Controls use an asymmetric tetrahedron and the actual original seed42 body4 backend hull, declared margins 0 and 0.003 m, and identity/rotated-offset child transforms. They verify cached-bound formulas, actual native support, compound bounds and relative contact-breaking thresholds. Independent scalar/corner enumeration reproduces the eight controls with maximum discrepancy **2.78e-17 m**. At nonzero margin Bullet cached bounds pad twice the declared margin, while support pads once; these are conservative bounds, not exact support equality. Exact vertex-bound equality is tested for zero-margin identity transforms.

| Geometry | Declared margin (m) | Child transform | Old threshold (mm) | Corrected threshold (mm) |
|---|---:|---|---:|---:|
| asymmetric-tetrahedron | 0.0 | identity | 4.184585 | 2.807987 |
| asymmetric-tetrahedron | 0.0 | rotated | 5.740481 | 3.883470 |
| asymmetric-tetrahedron | 0.003 | identity | 4.288082 | 3.013890 |
| asymmetric-tetrahedron | 0.003 | rotated | 5.879860 | 4.161780 |
| original-seed42-body4-hull-child | 0.0 | identity | 4.278207 | 2.900077 |
| original-seed42-body4-hull-child | 0.0 | rotated | 5.836724 | 3.980219 |
| original-seed42-body4-hull-child | 0.003 | identity | 4.381764 | 3.106350 |
| original-seed42-body4-hull-child | 0.003 | rotated | 5.976073 | 4.258444 |

The thresholds here are native compound-shape estimates, not a live manifold trace. These controls establish a cache-construction bug and its correction. **They do not establish that it caused the previous refinement failure or that corrected trajectories qualify.** A fresh full-horizon study must keep the material and original physical/refinement gates.

`results/compile.json` retains compilation output; `results/controls.json` retains every native control; `results/independent-audit.json` recomputes geometry/provenance. The source zip contains exactly the four frozen execution sources. The independent auditor was prepared before execution but is separate from that four-file source archive. The local executable is intentionally excluded from the publication manifest; its hash is retained in provenance. No attempted native control failed.

Reproduce the audit without rebuilding or running the engine:

```sh
OPENBLAS_NUM_THREADS=1 python3 research/hull-aabb-cache-review/audit.py
```
