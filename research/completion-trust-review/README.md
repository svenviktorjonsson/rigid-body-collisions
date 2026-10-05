# Critical review of numerical continuation

The original inelastic contact law is retained. Normal impulses obey unilateral
complementarity independently of the tangential disk. Tangential impulses
oppose slip and maximize instantaneous dissipation within the configured
circular capacity. The shared physical contact point transports lever arms,
free velocities, mobility and warm angular impulses before assembly.

The new continuation modifies numerical search problems, not accepted body
updates. Intermediate friction multipliers provide a pressure-face guide;
only multiplier one can return. The Fischer–Burmeister normal merit is an
equivalent zero representation of unilateral complementarity. Numerical
Levenberg damping modifies the search Hessian only. Final acceptance recomputes
the original projection equations and the original full mobility energy bound,
checks finite values and upper normal limits, clamps only permitted tiny
negative normal roundoff, and then recomputes the original residual.

The implementation limits 384 rows, 512 stage iterations, 256 SVD calls, 512
damped subproblems and 96 continuation attempts. Each stage has at most 48
iterations and each trial at most 24 backtracking evaluations. Its optional
frictionless Dantzig guide has a single-call limit; Bullet does not honor the
provided iteration argument as a hard internal pivot or wall-time bound.
Consequently the adapter must not advertise a strict total runtime bound.
Minimum-norm and damped search steps can be expensive on a dense large island;
this review supplies no new performance-ranking claim.

`audit_coulomb_trust_recovery.py` independently recomputes the original law and
energy from all six retained native final impulse vectors, without importing
the implementation or its acceptance helpers. All six captured systems pass.
The 51-row recovered solution changes its positive normal set from the failed
warm iterate; unsuccessful searches within the old set were not physical
infeasibility evidence. Later whole-trajectory failures remain separate work.

These velocity contact gates do not cover angular pose correction or finite
orientation integration. The retained anisotropic split-rotation counterexample
requires its own numerical repair policy and whole-step ledger. Translation
repair preserves kinetic energy but may change gravitational potential and
orbital angular momentum. Likewise the native arbitrary-body solver is
inelastic and currently contains no persistent elastic twisting store. The
rubber-like spin and bounce verification applies to a separate sphere/fixed-
plane elastic prototype, not an integrated arbitrary-body elastic engine or
an experimentally authenticated rubber law.
