# Larger translation-position recovery

The125-hull finest reference rejects a390-row translation-only position system
at3.35429e-7m/s against the unchanged1e-8m/s absolute gate. The normal active-face
helper previously declined above384 full rows, leaving4096 projected sweeps.

Production now permits512 rows **only for translation-position recovery**,
with the same128 active states and64 Jacobi sweeps per numerical factorization.
Velocity normal-pressure recovery retains its384-row default. Mobility, target,
friction, geometry, bounds and acceptance tolerance are unchanged. Native final
and progress JSON plus Python numerical metadata expose the new position policy.

The isolated full pipeline reproduces the old rejection and solves the new
variant in two factorizations at3.55e-15m/s. The live production replay reproduces
that success. An independent geometry/Gram audit reconstructs a translational
witness satisfying every original constraint. The LP solver reported success
but its point had6.96e-6m/s negative slack: its status is explicitly rejected as
a feasibility proof. The valid witness comes from the audited native solution.

163 engine tests/35 subtests, ten contact-model tests, eleven native check
executables and the actual390-row replay pass. Live23 velocity captures pass
and prior22 impulse/response Float64 bytes and original counters remain exact.
CI builds and runs the actual390-row regression.

Original larger-study source/runtime attestation was sealed before integration.
Original binaries are preserved on this host and mapped in
../rapid-friction/reference-binary-snapshots.json. The state-only portable archive
audit remains independent; the runtime provenance audit is host-specific.

This establishes one instantaneous position-system fix, not full trajectory
qualification or the requested all-case2x performance gate. Nine retained larger
velocity rejections and rotating/irregular trajectory accuracy still need work.
