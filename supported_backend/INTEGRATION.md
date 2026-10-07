# Supported-contact compiler boundary — ABI v1

This package prepares a native reference and an indexed algorithm for later BKF /
Vektor Flow lowering. No compiler repository, accepted fixture, main branch,
baseline, seed or hardware-acceptance state is changed by this package. The
compiler's authoritative Section 0 and its paired append-only workflow remain
the integration gate; a C header is not proof that public FFI already exists.

## Build and interface

```bash
g++ -std=c++17 -O3 -Wall -Wextra -Werror -fopenmp -fPIC -shared supported_backend/batch.cpp -o supported.so
```

`batch.h` is compatible with C11 and C++17. ABI version, input/output field counts
and host worker limit are callable. Inputs and outputs are IEEE Float64 with
field-major contiguous storage. Use one body index `k`:

`schema.json` provides the same field offsets, names, units and status/ownership
rules as machine-readable importer metadata. The Python field lists are checked
when generating this artifact; actual channel behavior is tested against the
archived physical oracles.

```
input[field * count + k]
output[field * count + k]
```

The caller owns disjoint storage for `15*count` input and `11*count` output
doubles and retains it through the synchronous call. No pointer is retained;
there is no per-body heap allocation. Every successful response writes all
eleven fields. A count of zero is valid. Native pointer lengths cannot be inferred:
the caller must provide those buffers and prevent concurrent mutation/aliasing.

Call `supported_validate(input,count)` when creating or changing inputs.
`supported_batch(input,output,count,threads)` includes finite-response, branch and
energy gates but assumes validated physical inputs. The Python `PreparedBatch`
copies and validates inputs once, owns the immutable storage, and allows repeated
zero-copy runs into caller-supplied outputs. Preparation and allocation costs must
be reported separately when comparing updates with an end-to-end workload.

`SUPPORTED_SUCCESS` indicates success. `SUPPORTED_INVALID_ARGUMENT` identifies a
bad pointer/count/worker argument. Other returns identify the earliest failed
body index. Discard the whole output after failure; a failed batch can have some
written responses. There is no fallback contact law or partially accepted batch.

## Physical fields

| Input offset | Name | Units |
|---:|---|---|
| 0 | mass | kg |
| 1 | central scalar inertia | kg m² |
| 2 | sphere/disk radius | m |
| 3 | supported normal load | N |
| 4 | constant tangential drive | N |
| 5 | translation relative to support | m/s |
| 6 | rolling angular speed | rad/s |
| 7 | axial spin | rad/s; zero for planar bodies |
| 8 | update duration | s |
| 9 | static friction μs | dimensionless |
| 10 | dynamic friction μd | dimensionless |
| 11 | rolling resistance μr | dimensionless |
| 12 | physical rolling moment length | m |
| 13 | axial resistance μn | dimensionless |
| 14 | physical axial moment length | m |

| Output offset | Name | Units |
|---:|---|---|
| 0 | outgoing translation | m/s |
| 1 | outgoing rolling speed | rad/s |
| 2 | outgoing axial spin | rad/s |
| 3 | traveled displacement | m |
| 4 | contact impulse along declared translation axis | N s |
| 5 | independent rolling angular impulse | N m s |
| 6 | independent axial angular impulse | N m s |
| 7 | sliding loss | J |
| 8 | rolling loss | J |
| 9 | axial spin loss | J |
| 10 | energy/work residual | J |

These scalar planar channels are signed integrals on fixed declared axes, not
the coefficients on the user's changing motion-defined directions. In particular,
offset 4 is not automatically δpt: on a nonzero-slip branch its contribution
along t has the slip-direction sign applied. Static reactions have no defined t.
Apply the branch directions before accumulating physical vectors; do not multiply
an integrated fixed-axis output by a final t or s to recover its physical impulse.
These scalar planar channels are not a redefinition of full motion directions.
In an allowed spatial embedding, resolve **t** from the full contact-relative
velocity and **s** from full relative angular velocity. The body angular transfer
includes the lever moment **plus** the independent impulse: in the scalar branch,
`I*(omega_after-omega_before) = -R*output[4] + output[5]` on the declared rolling axis.
Normal support impulse is `N*duration`; it is balanced by external support load
and is not an inferred normal-impact restitution impulse.

Physical angular capacities use their material/contact moment lengths; they do
not use coordinate normalization length ell. Normal and tangential restitution
remain inputs of impact branches elsewhere in the engine. Do not double-apply
them when importing this sustained-contact primitive.

## Algorithm to lower

Use rim speed `q=R*omega` and force-equivalent independent couple `b=M/R`.
The coupled mobility of slip and rim speed is

```
[(1/m + R²/I), -R²/I]
[-R²/I,         R²/I]
```

Both rows now use compatible linear units. Precompute reciprocal mass/inertia,
mobility and capacities once per update. Continued nonzero sliding/rolling has
known signs and needs no active-set enumeration. At zero motion, solve the small
static/onset candidates in the specified order; never invent t/s unit directions.
Advance exactly to slip or rolling arrest, update constraints, and cap axial
impulse at exact spin arrest. Preserve the energy and finite-range rejection gates.

The body loop is independent and has no hidden cross-body reductions. Worker
scheduling is an optional host implementation detail, not a mathematical change.
Contact incidence gather/scatter and simultaneous contact-group solving belong
to a separate layer and cannot be parallelized by blindly reusing this loop.

## Import acceptance

The Python reference, 400 frozen/updated controls, 2,000 wide-scale controls,
native one/multiworker equality and C11 header probe provide importer oracles.
Recompute all eleven channels and compare using the archived physical scales;
also test sign symmetry, split-duration composition, zero motion, small real
drive, exact spin arrest, invalid inputs and finite-range rejection.

Read the compiler's current handover and Section 0 authority before adding
fixtures. Use new identifiers and preserve every accepted oracle/source byte.
Native/WASM/GPU equivalence, ownership and compiler acceptance require their own
actual runs; this Linux native package does not certify those routes.

The public `contact_history.py` component is experimental: passive spring/slider
memory, distinct static/dynamic capacities and opening-energy transfer in one to
three declared modes. It is not called by this batch ABI. General evolving full
t/s directions, a physical coupled patch budget, joint normal impact and contact
groups remain unresolved. Keep that component behind an explicit experimental
boundary until those gates and independent empirical comparisons are met.
