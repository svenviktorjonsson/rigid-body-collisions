# Model fidelity audit — 6 October 2026

The user identified a substantive mismatch: existing force-only contact calculations and their comparisons do not validate the requested directional force-and-angular-impulse model. Their results are retained as historical comparator evidence. Correct mechanics identities do not imply the correct constitutive model.

## Confirmed requirements

Use lowercase delta and lowercase direction indices. Combined quantities use uppercase P, V, F, with fixed reference length ell and angular momentum L. Direction t follows the relative velocity at the collision point; direction s follows relative angular velocity. Neither is an arbitrary second tangent axis. No linear impulse component p_s is allowed. Angular impulse components include L_s and L_n. Do not invent independent fitted moment coefficients or silently impose a conventional three-axis Coulomb decomposition.

The user's earlier reference to the other linear direction as p remains to be reconciled with the original direction definition; do not silently replace it with n. It is also necessary to recover the original zero-motion convention, whether contact velocity is projected before normalization, how directions update during collision, and exact restitution/capacity rules. These have not been established by the current audit.

## Located evidence and mismatch

- research/research-assessment.tex, section Notation and the two-body response, already includes a free angular impulse in addition to the force lever-arm moment. Its general mechanics is useful, but its later endpoint/compliance comparators are not proof of the user's specific directional law.
- research/scaled-contact-article/article.tex previously reduced predictions to force-only point contact. The latest full-wrench algebra restores a free couple, but does not implement the requested directional law. Its prescribed pure-spin example is an identity check only.
- research/two-channel-restitution/README.md describes a translational normal/tangent endpoint law and circular friction capacity. Native predictions using that path do not implement an independent angular contact impulse.
- Existing public-data residuals and benchmarks characterize those historical models. They must not be reported as accuracy or performance of the user's requested model.

## Required recovery

Recover and transcribe the original directional equations before implementing a replacement. Trace every allowed impulse component through body momentum, contact-relative motion, restitution and friction rules. Preserve coupled contacts, equal/opposite impulses, lever-arm moments, energy/work checks and coordinate-length invariance. Run new pure-spin, sliding and mixed-state controls, followed by new public-data comparisons. No revised numerical agreement or model equivalence has yet been established.
