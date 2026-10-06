# Actual limestone impacts: coefficient conventions and geometry check

[Wang et al.2018](https://doi.org/10.5194/nhess-18-3045-2018) supplies a public
workbook of75 impacts for10cm/20cm faceted natural-limestone specimens on concrete.
The paper gives average masses1.2kg/10kg and elastic material properties, but no
per-specimen exact meshes/inertia/contact orientation. Published normal/tangent
ratios apply to COM velocity. They are NOT the contact-slip e_n/e_t convention
used in our spinning rigid-body law. Reading the rock Rt straight into e_t would
be a definition error. The paper bounds incoming angular speed at3rad/s and
uses a sphere-inertia approximation to calculate rotational energy.

`compare.py` imports every record and reconstructs that published energy formula.
It also checks the conditional sphere/plane identity linking measured loss of
COM tangential speed to angular impulse. This is a deliberately disclosed
geometry diagnostic: it is not a replay of the actual faceted rock shape and
cannot prove or disprove the complete arbitrary-shape model. It does not fit μ,
e_n or e_t and does not assume a scalar spin ratio is restitution. The actual
angular velocities remain comparison targets, with source row provenance.

The next valid full-state comparison needs actual contact point, inertia/pose
and signed angular motion, or a controlled specimen for which these are provided.
Any remaining dependence on contact location/direction must be distinguished
from geometric lever-arm/inertia effects before assigning material variation.

Source measurements in the derived comparison retain attribution to Wang,Y.,
Jiang,W.,Cheng,S.,Song,P.,and Mao,C.(2018),NHESS18,3045–3061 and the linked supplement.
The article is published under CC BY4.0; cached originals are outside this repo.
