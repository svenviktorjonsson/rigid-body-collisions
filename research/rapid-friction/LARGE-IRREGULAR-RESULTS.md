# Larger irregular-shape rapid-friction cases

All cases retain friction0.4, gravity9.81m/s², +/-20m/s reversals at0.04/0.08s and0.12s simulated. Counts exclude the container. Shapes rotate freely; prescribed containers translate. Planar shapes use seed7301; spatial hulls seed42.

2D mixed populations contain one-third concave triangular compounds and two-thirds irregular convex polygons. 3D hulls use12 asymmetric random vertices within0.1m support radius. These hulls are convex; this suite does not test concave3D compounds. Planar cell spacing is0.24m; spatial spacing0.205m. Disk scaling controls therefore do not isolate body count from the earlier qualified disk packing.

Planar execution uses supported experimental Float64 Box2D with1um numerical slop,128 velocity and12 position iterations, authored0.01m collision skin. Spatial execution uses production circular Coulomb law,4096 iterations and original bounded recovery. Defaults and accuracy gates are unchanged.

Five predeclared reference levels are10,5,2.5,1.25,0.625us. Qualification requires both adjacent quarter-budget edges; only qualified scenes receive candidate selection and three alternating timed repetitions. Failed histories/rejected systems remain archived.

| Case | Bodies | Complete reference histories | Native costs by level (s) | Process elapsed by level (s) | Reference qualified | Timed gain |
|---|---:|---:|---|---|---|---|
| 2d_mixed_polygons_36 | 36 | 5/5 | 3.309, 6.106, 12.786, 24.311, 51.990 | 3.371, 6.175, 12.856, 24.410, 52.134 | False | unqualified |
| 2d_mixed_polygons_100 | 100 | 5/5 | 9.826, 20.049, 40.065, 78.502, 155.016 | 9.969, 20.192, 40.176, 78.714, 155.331 | False | unqualified |
| 2d_disks_100 | 100 | 5/5 | 1.639, 3.181, 6.037, 11.938, 23.167 | 1.672, 3.220, 6.076, 11.996, 23.260 | False | unqualified |
| 3d_hull_64 | 64 | 0/5 | rejected, rejected, rejected, rejected, rejected | 7.653, 44.390, 85.466, 96.349, 30.273 | False | unqualified |
| 3d_hull_125 | 125 | 0/5 | rejected, rejected, rejected, rejected, rejected | 35.607, 176.079, 243.384, 258.714, 657.058 | False | unqualified |

Single reference costs are descriptive, not repeated benchmark medians. Rejected runs cover only their accepted prefix: their process elapsed is time to failure, not a full-horizon timing. No speed ratio is assigned to an unqualified reference. Native2D times physics/observer excluding frame output/diagnostics; native3D includes state recording. Compare within a case. Process elapsed and exact numerical settings are saved in each record. External host load is uncontrolled.

Reproduce in a fresh output directory with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m research.rapid-friction.large_irregular`; source/binary/runtime hashes are in provenance.json. Portable archive audit: `python -m research.rapid-friction.audit`.

This expands the external synthetic benchmark coverage; it does not establish material calibration, a VKF port or general irregular-shape qualification.
