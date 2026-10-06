# Public rock-impact calibration discovery

The [Chant Sura dataset](https://doi.org/10.16904/envidat.174) supplies ideal
reinforced-concrete EOTA shapes and reconstructed trajectories, not exact natural
rock specimens. Public archive HTTP receipts and SHA256 values are retained in
`discovery/`; raw archives remain outside the repository. The output resource
metadata still says "under embargo" although its ordinary public URL returned
HTTP200. Metadata uses WSL Data Policy; redistribution permission is not assumed.

`inventory_chant.py` inspects all 82 published reconstruction CSVs. Forty-one have
finite gyro/rotational-energy records. In these files, resultant angular speed
is consistently 180/pi times the norm of its component vector: mixed unit
conventions require checking against author methods before importing motion.
Energy-implied masses are 210,250,780,840,2670 kg, rather than assuming nominal
filename categories. Diagonal inertia values can be reconstructed from the
rotational-energy formula with negligible residual, but this only recovers
parameters used by the authors, not an independently measured inertia tensor.
Sensor/body/world alignment, full attitude and contact normal are unverified.
Do not fit a supposedly measured friction coefficient before resolving those
inputs and soil/contact deformation.

The [Tschamut2014 dataset](https://www.envidat.ch/metadata/tschamut2014) provides
natural-block scans, specimen mass tables, trajectories and jump/impact tables.
The inspected impact tables report scalar before/after rotation in rps; a complete
three-component angular velocity and attitude record has not been verified.
Point-cloud-derived inertia would additionally assume density and reconstructed
watertight geometry. Record these as inferred properties, not measured tensors.
Its metadata declares ODbL+DbCL. Neither dataset currently qualifies as an exact
full-state single-impact replay in this engine.

Calibration must preserve measured uncertainty, infer contact frames explicitly,
fit restitution/friction jointly against linear and angular motion, and reserve
whole trajectories/specimens as holdouts. Compare fitted friction with independent
measurements on the same contact pair when available. Literature compilations
supply comparison ranges, not same-specimen independent validation. No fitted
friction values or exact experimental matches are claimed yet. Existing authored
synthetic benchmark materials and all acceptance gates remain unchanged.
