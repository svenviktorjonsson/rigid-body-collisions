# Exact contact-component recovery integration

Production attempts exact-component support recovery only after the existing
contact solver and its remaining original iteration budget reject. The seed is
the final rejected PGS vector, never an applied physical state. Every returned
answer passes every original circular projection row, normal bound and finite
passivity gate at the unchanged 1e-8 m/s tolerance. Existing accepted contacts
return before this new stage.

The prospective research V3 algorithm is retained separately. This production
copy additionally rejects asymmetric external matrices, nonpositive tangent
blocks and nonfinite intermediate residual metrics. Adversarial checks cover
overflow and invalid matrices that could otherwise hide a NaN in std::max.
The physical matrix is assumed to come from the engine's Gram assembly; these
checks do not prove arbitrary external matrices positive semidefinite.

Limits per call: 4096 full rows, 64 rows per exact active component, 8 passes,
8 numerical helper calls, 1024 nonlinear/projector SVD calls, separately at most
1024 pressure SVD calls and 8 pivot-guide calls. Internal guide pivots have no
exposed bound. LAPACK remains optional; the OFF build excludes this stage.
Counters are fresh per call and aggregated as scalar engine statistics.

prebuild-interrupted.log preserves a short replay attempt deliberately stopped
because its binary rebuild had not finished. It is not a physics rejection or
an accepted study. The final replay uses binary/source drift guards and a new
receipt destination. Captured contact roots do not qualify dense trajectories.
The separately certified 74-row translation-only pose-repair failure requires
a revised explicit repair protocol; increasing solver work cannot fix its
original bounded equations.

Final production replay at source9e97be07b833503c15d9a96f1f92afc59c11a292:
**22/22 accepted** by native independent-law bookkeeping and the independent
Python original-row projection/bounds/passivity audit. All20 earlier impulse
arrays remain exactly identical. New261/243 residuals are1.4374e-10 and7.6613e-11
m/s, using5 and103 supplemental nonlinear/projector SVD calls respectively.
The full previous failed pipeline still runs first, so this is a correctness
improvement rather than a controlled performance gain.

The receipt records every actual compiled native source hash, binary and linked
library hash. Its broad header fingerprint also includes the then-unused
position_geometry.h serializer before integration; that file was untracked at
execution and is not part of the9e commit or compiled model. The source archive
retains it explicitly rather than claiming a false committed-source identity.
The plan's changed production sources all match the9e commit.
