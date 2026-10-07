# Model fidelity audit — 6 October 2026

## Latest continuation — 7 October 2026

The user now permits small fits for missing material/contact parameters while documented inputs stay fixed. That supersedes the previous blanket prohibition on fitting. The user also asks to encapsulate deformation in rigid-body contact features rather than deformable meshes for every body.

The active article derives the projection of conventional material constraints into the user's full-velocity n/t map, without changing t. Scalar directional p_n is not the entire physical normal transfer when t contains normal approach. Rolling resistance uses the physical normal transfer and physical moment length, normalized separately by ell. A new small contact-memory elastic branch includes the independent angular impulse and exact midpoint energy accounting; it passes isolated 2D/3D controls but does not implement opening/friction/yield transitions or authentic material calibration.

The full experimental report is `../full-experimental-report/report.pdf`, with 107 collision records and 17 rolling measurements, all limitations and row tables. It does not validate the complete directional production model. Target-supplied restitution, reconstructed spin and fitted spherical rock proxies remain distinct from independent prediction. A shared signed-moment fit worsens the ball comparison; its failure is retained. No full experimental agreement is claimed.

The user identified a substantive mismatch: existing force-only contact calculations and their comparisons do not validate the requested directional force-and-angular-impulse model. Their results are retained as historical comparator evidence. Correct mechanics identities do not imply the correct constitutive model.

## Confirmed requirements

Use lowercase delta and lowercase direction indices. Combined quantities use uppercase P, V, F, with fixed reference length ell and angular momentum L. Direction t follows the relative velocity at the collision point; direction s follows relative angular velocity. Neither is an arbitrary second tangent axis. No linear impulse component p_s is allowed. Angular impulse components include L_s and L_n. Do not invent independent fitted moment coefficients or silently impose a conventional three-axis Coulomb decomposition.

Latest user clarified that capitalization in dictation is incidental and the sliding label delta p_t was already settled. The active rewrite uses delta p_n, delta p_t, delta L_s and delta L_n. Do not reopen notation questions based on speech capitalization. It remains necessary to recover the original zero-motion convention, any projection explicitly present in the source law, how directions update during collision, and exact restitution/capacity rules. The active article preserves the supplied t = relative contact velocity / magnitude and s = relative angular velocity / magnitude definitions.

## Located evidence and mismatch

- research/research-assessment.tex, section Notation and the two-body response, already includes a free angular impulse in addition to the force lever-arm moment. Its general mechanics is useful, but its later endpoint/compliance comparators are not proof of the user's specific directional law.
- research/scaled-contact-article/article.tex previously reduced predictions to force-only point contact. The latest full-wrench algebra restores a free couple, but does not implement the requested directional law. Its prescribed pure-spin example is an identity check only.
- research/two-channel-restitution/README.md describes a translational normal/tangent endpoint law and circular friction capacity. Native predictions using that path do not implement an independent angular contact impulse.
- Existing public-data residuals and benchmarks characterize those historical models. They must not be reported as accuracy or performance of the user's requested model.

## Required recovery

Recover and transcribe the original directional equations before implementing a replacement. Trace every allowed impulse component through body momentum, contact-relative motion, restitution and friction rules. Preserve coupled contacts, equal/opposite impulses, lever-arm moments, energy/work checks and coordinate-length invariance. Run new pure-spin, sliding and mixed-state controls, followed by new public-data comparisons. No revised numerical agreement or model equivalence has yet been established.

The full symbolic rewrite (6 October) removes all numerical outputs and plots at the user’s request. Its baseline wedge, transpose notation, body/contact impulse distinction and constrained component maps have been checked. Independent evaluation passed 100 planar and 100 spatial explicit mobility/component-response/energy identities, including collinear spin/normal directions; these prescribed-impulse checks do not validate a constitutive law.
