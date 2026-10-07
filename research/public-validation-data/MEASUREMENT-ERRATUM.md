# Bounce measurement audit

The frozen `evidence-v1` bounce reconstruction intersects the incoming and outgoing
ballistic arcs. Its reconstructed incoming speed is evaluated at that joint
intersection time. Consequently the measured outgoing trajectory indirectly
affects the reconstructed input time. The protocol's statement that no measured
outgoing speed is used as an input is too strong for this reconstruction.

Retain the original numbers as a **joint-arc consistency diagnostic**, not a
strict prediction from independently observed incoming state. Neither restitution
nor friction was fitted. The other dataset branches are unaffected.

`marker-v1` corrects this dependency by evaluating the incoming-only flight fit
at the recorded valley-frame timestamp. It evaluates the outgoing-only fit at
the same timestamp for comparison. The observed impact marker is a conditioning
input, not a forecast of the time of contact. Its true time remains unresolved
within a 30 Hz frame. This is a conditional normal-branch check, still not a blind
trajectory forecast or independently characterized material validation.

Declare the correction after observing the original outcomes; keep exactly the
same 42 evaluation events, source coefficient, fit windows and gravity. Report
all corrected errors, the conservative one-frame sensitivity envelope, and all
original results. Do not select timestamps to minimize errors.
