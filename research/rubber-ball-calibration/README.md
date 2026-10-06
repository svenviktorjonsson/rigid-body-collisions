# Rubber-ball measurements and calibration

Public experimental papers provide useful spin/contact tests. A public raw dataset
with matched rubber compound, surface, several diameters, measured inertia and
per-trial incoming/outgoing velocity and spin has **not yet been verified**.
An exact experimental match cannot be claimed from rounded summary tables.

Cross (2002), [Grip-slip behavior of a bouncing ball](https://physics.usyd.edu.au/~cross/Gripslip.pdf),
Tables I/II, reports a 46 mm, 46.4 g Superball and a homogeneous-sphere inertia
factor 0.40. For one smooth-surface impact: speed 2.69 m/s, angle 36 degrees to
horizontal, zero initial spin, outgoing spin 98.4 rad/s, horizontal-speed ratio
0.59, normal restitution 0.91. Typical measurement errors are 2–3%. The support
is a moving instrumented block; include its velocity and kinetic energy when
replaying the experiment. Force reversal during contact demonstrates grip and
elastic deformation. The author could establish only a lower friction bound
for the Superball in the relevant comparison, not a reliable unique coefficient.

Cross (2010), [Enhancing the Bounce of a Ball](https://physics.usyd.edu.au/~cross/PUBLICATIONS/48.%20EnhanceBounce.pdf),
reports a 58 mm, 103 g Superball, impact speed approximately 4 m/s, and angle
25 ± 1 degrees to vertical. Table I compares granite, rubber, Superball material,
and tennis strings. It supplies restitution and outgoing-spin factor, with
estimated errors; incident speed is approximate, so it cannot supply exact
per-trial state vectors. The two published Superballs are different specimens;
the diameter comparison also confounds compound, surface and measurement setup.

`derive_energy.py` reconstructs the 2002 table's velocity components and ball-only
energy ledger. `derived-energy.json` records inferred quantities and limitations.
The approximately 0.0816 J reduction in translational energy includes 0.0475 J
of outgoing rotation; ball-only total kinetic energy falls by about 0.0341 J.
These calculations use rounded measurements and the author's inertia model.
Do not treat the remaining loss as complete system dissipation without the block.
Downloaded manuscripts remain outside the repository; the author permits personal
use and restricts redistribution of articles.

## Prospective validation protocol

1. Distinguish experimental observations, graph-digitized estimates and exact
   synthetic controls in every case. Record mass, diameter, solid/hollow structure,
   inertia provenance, surface, temperature, impact speed, angle, incoming spin,
   support motion and measurement uncertainty. Never assume hollow-ball inertia
   equals the homogeneous solid sphere's 2 m R² / 5.
2. Add radius and spin sweeps as **synthetic** single-impact controls in 2D and 3D,
   using analytically declared mass/inertia. These test implementation scaling,
   impulse/torque consistency and energy accounting, not experimentally fitted
   rubber properties. A 2D disk and a 3D sphere have different inertia factors.
3. First fit normal restitution and sliding friction only on verified sustained
   sliding impacts. Compare fitted friction with independently measured friction
   on the same material/surface/speed; literature values from other pairings are
   context, not same-specimen validation. In gripping impacts, net tangential
   impulse divided by normal impulse is not a measured sliding coefficient.
4. Test held-out whole specimens and impacts, including backspin, zero spin,
   topspin, incidence angle, speed and diameter. Fit linear and angular velocities
   jointly; report uncertainty and residuals, not just energy or fitted parameters.
5. Where a rigid Coulomb model cannot reproduce slip reversal, retain its mismatch.
   Evaluate tangential stiffness/damping and compliant contact as a separate
   physical model, with duration/force histories where available. Do not hide
   missing deformation by assigning an arbitrary effective friction coefficient.

This calibration study does not change authored materials or acceptance gates in
the existing 13-case benchmark. Its all-case working baseline and ≥2× performance
gate remain open.
