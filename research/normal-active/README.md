# Retained failed normal-pressure prototype

The cold normal-only active-set prototype retained here declined the 291-row source-95d224f position capture under 128 SVD calls. Its best internal residual was approximately 2.78e-8 m/s, above the unchanged 1e-8 gate. It left the original caller impulse unchanged; the resulting physical output still has the original approximately 5.79e-8 residual. The receipt field `attempts` was a prototype entry counter rather than a count of SVD calls; use the explicit SVD field for factorization cost.

This negative experiment is superseded by the independently developed semidefinite normal null-pressure descent in `research/active-next-review/normal_null.h`, which recovered the exact original system with two boundary releases and one range step. Do not use the failed cold prototype as a production fallback or count its internal best merit as accepted output.
