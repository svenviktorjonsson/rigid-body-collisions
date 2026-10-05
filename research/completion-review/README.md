# Independent mechanics review

This review separates mechanics defects from numerical solver failures. It
does not claim that every shape, speed, initial overlap or material works.

`mechanics_review.py` finds all six latest captured mobility matrices exactly
symmetric and positive semidefinite to floating-point roundoff. The normal-only
inequality problems are feasible, and each warm positive-normal set has
independent normal target equations. These checks rule out the corresponding
simple assembly or target contradictions; they do not prove full Coulomb
existence. The 51-row fixed-active-set mode search in `active_modes.py` retains
48 failed sticking/sliding initialization attempts. Their nonzero minima are
not an infeasibility certificate. Independent friction-continuation recovery
provides the appropriate next test.

The frozen before/current elastic event checks expose a real integration
defect: a sphere initially tangent to a plane with zero normal velocity could
alternate zero-time entry/lift-off events and exhaust its unchanged work
budget. This also broke a side impact while grazing the floor. The corrected
approach classification admits those trajectories without altering stiffness,
friction, stored energy or the force/couple law. Simultaneous inward corner
entry already worked before the change; that control is retained.

`split-energy-counterexample.json` exposes a second distinct limitation. A
spinning anisotropic box initially overlapping a fixed floor by 5 mm gains
0.001429948 J in a single original split update, although the physical velocity
contact solve reports zero work and zero energy change. Numerical angular pose
repair rotates the world inertia while retaining angular velocity. The
velocity-only control changes kinetic energy by just -5.68e-9 J. This is a
counterexample involving disclosed initial overlap, not a spontaneous failure
of an initially separated physical collision. The frozen scene, full results,
binary hash, source hashes and independent full-tensor energy calculation are
retained. Subsequent corrections must not overwrite this evidence.

A translation-only position projection leaves the inertia orientation and
physical velocity solve unchanged, so its repair phase cannot change kinetic
energy. Its linear Gram matrix may have harmless pressure redundancy or
genuinely incompatible position targets; the exact normal residual gate must
distinguish these. Translation repair still changes gravitational potential
and orbital angular momentum. Its numerical displacement and energy effects
require their own ledger and trajectory gates. It is not an additional
physical impulse or a proof of exact whole-step conservation.

The elastic model has a conservative spring potential and nonnegative plastic
and compression damping losses. Its independent twisting couple can reverse
normal-axis spin while respecting a normal-load capacity. Exact reversal
requires suitable stiffness, contact duration and available capacity. Static
friction capacity by itself does not prescribe reversal. This model is a
rubber-like material hypothesis; no measured rubber parameter fit or
experimental validation is supplied here. Chained vertical floor/ceiling
bounces with twisting spin and oblique same-floor bounces are distinct from
arbitrary oblique floor/ceiling retroreflection, which additionally requires
appropriate rolling-couple channels and material evidence.

`elastic_tail_review.py` independently recomputes kinetic, gravitational and
spring energy and both impulse balances for a stiff matched material under
gravity. The finite-gravity release tail lasts about 33 ns. Its loss converges
to 8.980095e-7 J of plastic work, while the recorded residual separation-store
loss falls to 2.18e-20 J. The coarse absolute integration tolerance 1e-13 has
a retained force-capacity failure (maximum excess 0.000793 N). Tolerances
1e-14, 1e-15 and 1e-16 pass the unchanged peak-load-scaled capacity budget;
1e-15 and 1e-16 also pass a stricter 1e-5 N absolute excess check. Whole-energy
error alone therefore cannot substitute for the force/couple capacity check.

The material uses an undeformed-radius tangential lever even during normal
compression. Its energy-consistent small-deformation approximation and declared
compression bound do not establish real surface-torque accuracy at large
deformation. Likewise a constant effective torsion length is a material
parameter, not a measured evolving rubber contact radius.
