# Native 3D validation backend

Build the unmodified, hash-pinned Bullet 3.25 CPU backend in Float64:

```sh
cmake -S spatial_backend -B build/spatial -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/spatial --target spatial_runner -j 2
python -m unittest tests.test_spatial_engine -v
```

`spatial_engine.run(scene)` supports spheres, boxes, arbitrary convex 3D hulls
and compounds (use authored convex decomposition for concavity). Density is
volumetric kg/m³. Mass, COM and full body inertia are integrated from geometry;
compound parallel-axis terms are included. A principal-frame transform supplies
Bullet's diagonal inertia without discarding off-diagonal terms. Poses use XYZW
unit quaternions. Output states are COM x/y/z, quaternion x/y/z/w, world vx/vy/vz,
world omega x/y/z. Gravity defaults to -z.

Body position denotes the COM. Authored fixture coordinates are shifted to their
aggregate COM. Output orientation uses the original body axes. Static and
prescribed kinematic bodies have infinite mobility mass (reported mass zero).
Velocity schedules include translation and angular velocity and split steps at
command times. Bodies are integrated independently of the container.

The travel guard accounts for both bodies' translation, angular tip speed and
gravity, and limits updates to 15% of the smallest fixture half-width/radius by
default. It prevents the tested fast-wall tunneling example; it is a conservative
timestep heuristic, **not an exact swept CCD proof for every concave feature**.
Setting `travel_fraction=0` is an explicit negative-control diagnostic.

`solver='sequential'` selects projected sequential impulses; `'coupled'` selects
Bullet's Dantzig MLCP solver. Bullet may fall back to sequential iterations when
an MLCP fails; `coupled_fallbacks` exposes those events. No run with fallbacks may
be described as a pure direct coupled solve. Worlds/contact caches persist across
updates. No invented compliance or material changes distinguish the two modes.

Friction uses two independent bounded tangent directions, a pyramid approximation
to the isotropic Coulomb cone. Body friction and restitution coefficients multiply
at a contact. For a desired pair coefficient mu between identical bodies set each
body coefficient to sqrt(mu); a wall coefficient one retains the object's value.
One friction coefficient supports sticking and sliding; separate static/dynamic,
rolling/twisting and elastic tangential history are **not implemented here**.
Restitution is a normal velocity rule with zero velocity threshold. Prescribed-wall
work is summed from normal and tangential impulses at wall point velocities;
split position corrections do not count as physical impulses.

The full engine is 3D; analytic regression tests and held-out scenes must establish
each claim. Synthetic values in `research/spatial_scenes.py` are not calibrated
physical materials. This backend is public Python/C++ research, not a Vektor port.
