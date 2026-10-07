# Indexed contact scheduling, C++17

`schedule.h` is dimension independent and does not choose a contact law. Its
flat original contact index `k` refers to an entire block: scalar force and
independent angular impulse components, with lever moments included by the law.
It never separates linear and angular components into independently scheduled
constraints. Shapes, masses, inertia and material values are not planner inputs.

`build(a, b, movable, colors=32)` accepts equal-length endpoint arrays of UInt32
body indices and a UInt8 mutable-body mask (0/1). Distinct valid endpoints and at
least one mutable endpoint are required. A prescribed body is read-only during
this solve. Contacts joined through mutable bodies belong to the same island;
sharing only a prescribed wall does not connect them. All other constraints and
shared mutable material states must be included in connectivity before these
islands can be used as independent solves. Sleeping/waking or a change of body
type changes the mask and invalidates the cached plan.

`Plan.island_order/island_offsets` group complete contacts into independent
components, preserving original order within each component. Isolated bodies
with no constraints are omitted. `color_order/color_offsets` are conflict-free
batches across all components. Every color has at most one contact per mutable
body. A bounded greedy search (1–64 colors) puts excess conflicts into
`serial_order`, which must run sequentially. No contact is dropped. The plan is
deterministic for the exact ordered input graph; it does not promise invariance
to reordering contact discovery.

`Cache.prepare` compares the full ordered topology and mutable mask before reuse,
including the color budget. A cache hit has no allocation/factorization, but
still costs O(bodies + contacts) for equality checking. A miss builds before
publication; an invalid request leaves the previous plan intact. Returned
references are valid until the next successful miss or cache destruction.
Cache mutation is not thread safe; a completed immutable plan can be read by
workers. Build cost is O(bodies + contacts α(bodies)), with bounded color work;
storage is O(bodies + contacts). A high-degree star remains inherently serial.

`visit_colors(plan, threads, callback)` uses an optional OpenMP build and an
explicit worker count, barriers between colors, then a serial tail. The
`noexcept` callback receives the original contact index and must write only
declared mutable endpoint state and contact-exclusive state. Do not accumulate
an ordinary shared energy/statistics scalar in the callback; reduce separately.
For a nonlinear solver, use trial buffers and preserve the full current physical
acceptance gate. This helper is suitable for committing already gated contact
blocks; it is not a new iterative solver or a complete many-contact law.

Geometry, current full relative-motion t/s directions, patch state, loads,
restitution and friction must be recomputed/transported independently. Topology
reuse authorizes none of those to be reused. A changed color order can alter
nonlinear iteration convergence, so full trajectory qualification remains
required before this schedule is selected in a world engine. The existing native
world default and supported-response ABI v1 remain unchanged.

The topology cache does not establish persistent physical contact identity:
two patches can exchange order while retaining the same endpoint arrays. Track
material history by verified contact features, and transport it separately. The
planner is currently a C++ header interface, not a new C ABI or a compiler port.

Native controls exercise 2D/3D force-plus-independent-couple momentum application,
parallel worker identity, static-support separation, high-degree overflow,
topology invalidation and a fresh independent connected-component reference.
Planner timings are preparation/cache costs, not collision-engine step timings.
