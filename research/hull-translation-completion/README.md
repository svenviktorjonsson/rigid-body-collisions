# Completed translation-only full-history study

All six predeclared attempts ran from production source
52f7e6d244e92a8405134e0222ce14fc3eda0ef6, with unchanged authored shapes,
material, boundary commands and physical/refinement gates. The only declared
numerical protocol change from the earlier study is split to split_translation.
Source, executable and runtime library hashes remained unchanged throughout.

The three eight-body histories complete and pass individual physics/contact
checks. Neither adjacent refinement edge passes the original quarter budgets:
position RMS errors are 0.07251 and 0.08915 m, velocity 5.545 and 5.381 m/s,
spin 80.604 and 78.212 rad/s, orientation 1.300 and 1.063 rad. This is a clear
accuracy failure. Zero references qualify.

All three 27-body spinning/shaking attempts reject, with no infrastructure
interruption. Two velocity systems are retained exactly: 261 rows at
2.82774e-5 m/s residual and 243 rows at 5.33651e-4 m/s, against 1e-8 m/s.
The third rejects translation-only position repair before producing a matrix
snapshot; the missing diagnostic is retained explicitly, not reconstructed or
interpreted as proof of physical infeasibility. Native accepted prefixes survive
at 5, 6 and 5 output frames. Prefixes are not complete or resumable histories.

The independent audit recomputes raw authored-frame physical measures, full
principal tensors, containment, quaternion norms and both refinement edges.
The initial five-snapshot receipt remains alongside final six-snapshot receipts.
Runtime records have two separate checks: portable archive/metadata integrity
for other hosts, and explicit local executable/library-byte attestation. The
saved original local attestation and exact auditor source remain immutable;
archive CI does not require another machine's compiler/library bytes to match.

Passing 20 earlier captured contact systems does not qualify later trajectories.
New 261-row research recovers a strict passive solution using a generic ranked
pressure-face release; native recovery and the other later failures remain work
for separate future source checkpoints. No new speed or authenticity claim.
