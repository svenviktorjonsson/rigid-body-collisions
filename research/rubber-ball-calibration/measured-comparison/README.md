# Measured rubber-impact model adequacy

User requires independently sourced contact/material properties to predict
measured post-impact motion. This requirement is now separate from mathematical
solver checks; prescribed restitution from the target impact is not independent
validation. No friction is fitted to hide model error.

Cross2010 TableI Superball/granite:58mm,103g,approximately4m/s,25±1deg to vertical,
normal restitution0.78±0.01,tangential restitution0.49±0.01,spin factor14.9±0.1
rad/m. Source:https://physics.usyd.edu.au/~cross/PUBLICATIONS/48.%20EnhanceBounce.pdf
The paper does not supply an independent rubber/granite friction value or measured
inertia for this specimen. Fixed support and homogeneous sphere are assumptions.

Seven complete current native3D coupled-solver impacts with μ0/.05/.1/.2/.4/.9/1.5
retain all inputs/outputs. Normal restitution is imposed from this same experiment.
It is reproduced at0.78 by construction, not predicted from elastic material data.
Tangential restitution reaches0 and spin factor10.4093, whereas observations are
0.49 and14.9. The independent rigid-law maximum is10.4093; even26deg gives10.7973,
below measured lower bound14.8. Under this rigid contact/inertia/support model,
no increase in sliding friction can reproduce the measured spin. This diagnostic
is not a complete experimental replay: support approximation and missing measured
inertia/friction remain explicit. Independent impulse/energy checks pass all7.
Initial result-reporting KeyError is retained; the correction changes a dictionary
key, not native physics. No production source change or all-world acceptance.

## Required physical inputs and validation

A rigid prediction needs geometry, mass/inertia, initial poses/linear/angular
velocities, contact-pair friction and normal restitution, with documented
conditions. A deformable rubber model additionally needs elasticity and a
measured dissipation/contact-duration model, including tangential deformation.
Density/shape do not determine rubber hysteresis. A generic material label is
insufficient for quantitative properties; use specified compound, opposing
surface, temperature, speed and uncertainty. Do not silently combine unrelated
friction measurements with an assumed surface coefficient through product mixing.

Properties used to predict a validation impact must come from independent tests
or a training set distinct from the held-out impacts. Independently measured
restitution may be a legitimate input, but matching that input is not evidence
that restitution was predicted from material constants. When sufficient elastic
and dissipative properties are available, predict rebound from the compliant
law and validate restitution as an output. Record unknown properties explicitly;
do not substitute default0.4 friction or zero restitution and call it rubber.

Required outputs:normal/tangential rebound,linear/angular velocity,contact
duration where available,translational+rotational energy and support work/loss.
Numerical convergence must precede physical acceptance. Report model mismatch
against measurement uncertainty, not only that the solver ran. Main next model
work is nonzero rebound with isotropic friction and tangential compliance; the
current Coulomb production lane remains zero-restitution only.
