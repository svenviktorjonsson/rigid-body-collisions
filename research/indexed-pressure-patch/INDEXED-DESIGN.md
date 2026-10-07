# Single-index contact fields for a later Vektor Flow port

This is a design contract and executable Python/C++ research, **not a compiled
VKF port**. Read authority: `spec/language/00-foundations.md`, sections 0.1,
0.3, 0.10–0.12 and 0.20, inspected 7 October 2026. Repeated dimensions do not
imply a sum; explicit reductions and their numerical order are required.

Use one index per entity, rather than body-pair tensor dimensions:

| Entity | Single index | Flat fields / ownership |
|---|---|---|
| Body | k | v, omega, mass, inertia, accumulated force and moment |
| Contact | i | incidence_start, site_start, frame, compression, profile_id |
| Signed incidence | h | owner (one body k), sign, arm; contiguous range of contact i |
| Pressure site | j | x, y, weight; contiguous range/shared template of contact i |
| Material profile | m | fixed documented coefficients and source/conditions |

Body components are arrays rather than a dense body-by-contact tensor. Native
`Fields` uses separate x/y/z arrays; `Contacts` uses separate scalar arrays.
Footprint points/weights and the small Gram matrix are shared immutable data.
Native benchmarks use two incidences per contact and a shared fixed wall; the
Python work audit permits arbitrary world frames. Native frame transformations
are not timed: benchmark frames are the identity.

First gather the body's local contact motion over incidence h. Each local patch
calculation then receives an ordinary vector, with no body-pair indices in the
physics equations. Finally scatter equal/opposite force and the sum of its lever
moment and the independent patch couple back through owner k. Multiple contacts
writing the same body must use ordered segmented reductions or a declared
floating-point reassociation contract. Independent whole-impact maps are not a
substitute for a coupled contact solve during rapid group motion.

For one contact's local site view, valid Section 0 reduction notation is
`f_x: sum_j(f_x_j)` and `moment_z: sum_j(x_j f_y_j - y_j f_x_j)`.
The underlying C++/Python uses CSR-like ranges. Ragged gather/scatter and history
transport have not been compiled/verified against current VKF; no invented
indexing syntax is presented as accepted language support.

## Local mechanics and cache

In a contact frame, normal is z; offset is **r** = [x,y,0]. Let **d** contain
compression and two small virtual tilts, **q** = [v_z, omega_x, omega_y], and
**b** = [1,y,-x]. Site compression is b·d, normal speed is b·q and the compression
rate is minus that speed. This is frozen affine normal deformation, not an
arbitrary finite rotation of a footprint. For weight w and total K,C:

    elastic = K max(compression, 0)
    normal force = w max(elastic - C normal_speed, 0), if compression > 0
    local sliding velocity = [v_x - omega_z y, v_y + omega_z x]
    local dynamic force = -mu_d normal_force sliding_velocity / sliding_speed

The last line applies only at positive local sliding speed. Static equilibrium,
shear stiffness and mu_s transitions require a separate local history solver.
The zero-speed kinetic contribution is zero; it does not predict static friction.

Sum local forces and **r** wedge force to obtain the resultant and independent
couple about the contact origin. Body angular impulse adds its own center-to-
contact lever moment as well. Integrating these forces in time would produce
delta p_n, delta p_t, delta L_s, delta L_n only when the resultant belongs to the
user's direction spans. The audit measures residuals instead of relabeling t or s.

Cache G = sum_j(w_j b_j b_j transpose). If every site is compressed and its trial
pressure is positive, the normal force/moment triple is exactly
G (K d - C q). For zero axial twist, sliding direction is uniform across the flat
patch, so shear force and its moment also reduce exactly using G. Otherwise only
the dynamic-friction loop remains. A conservative footprint-box bound admits
caching; open/clipped sites fall back to local evaluation. There is no relaxed
physics tolerance, erased residual or new fitted coefficient in this reduction.

For symmetric circular Hertz weights and zero center sliding/normal speed,
normal damping produces a rolling couple
`moment_x = -C a^2 omega_x / 5` and similarly for y. The independently derived
axial full-sliding torque is `-3*pi*mu_d*N*a/16` times spin sign. General mixed
motion shares the local friction budget, rather than independently applying
maximum sliding force and maximum spin torque. These formulas assume the supplied
footprint/pressure shape; they are not a universal material identification.

Reference length ell changes only coordinates: **V** = [**v**; ell **omega**],
**P** = [**p**; **L**/ell], independent impulse has angular entries delta **L**/ell.
The physical radius a and physical inertia are not replaced by ell. Code uses
unscaled physical units and passes force/couple separately. The earlier article's
wedge/transposed-wedge convention and single body k notation remain unchanged.

## Separate material and state inputs

| Symbol | Meaning | Policy in this candidate |
|---|---|---|
| e_n, e_t | normal/tangential endpoint restitution | Kept in the material catalog; not imposed on top of foundation dynamics |
| mu_s | static coefficient | Kept fixed where documented; static branch still unresolved |
| mu_d | dynamic coefficient | Existing coefficient used at every sliding site |
| mu_r | documented rolling resistance, with physical length | Comparison input; not added again to rolling pressure dissipation |
| K, C | effective contact-foundation stiffness/damping | Independently characterized or small bounded missing-parameter estimate; synthetic in current controls |
| a / patch sites | physical footprint and pressure shape | Geometry/measurement input, not an arbitrary friction fit |
| ell | numerical reference length | Coordinate choice, never a material constant |

Mapping measured E/Poisson ratio to K, contact footprint growth, shear memory,
unloading energy carried into subsequent contact, local brittle yield and
documented restitution consistency remain research tasks. A default soft modulus
cannot certify authentic rubber or rock behavior.

Future history buffers need stable contact IDs plus generation tags; footprint
changes need interpolation with an explicit energy ledger, and frame changes need
transport. New/open contacts must not read a recycled contact's shear/mode state.
Those operations are not yet implemented by this frozen kernel.

## Scope of the benchmark

CPU local response includes body gather, point/patch force+couple, body scatter,
and output resets; allocations are outside timing. Counts up to one million are
**responses**, not a million interacting bodies evolved through collisions.
The same prescribed sites are used by reference and optimized kernels. Quadrature
error versus a continuous patch is a separate audit and can dominate near a local
slip-zero. Fixed low site counts cannot be claimed accurate for every mixed state.
Detection, footprint generation, frame transport, constitutive branch integration
and contact-group solve are not timed. The production example gates stay open.

Primary contact-patch precedent: [Elandt et al. (2019)](https://arxiv.org/abs/1904.11433);
implementation context/parameter cautions:
[Drake hydroelastic guide](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html).
Neither source establishes that our synthetic K,C or footprint matches the
experimental specimens in the earlier full report.
