# Length-scaled rigid-body contact article

Read **article.pdf**; edit **article.tex**. The notation is `(v, ell*omega)` and `(p, L/ell)`, with uppercase L the angular momentum and lowercase script ell a fixed coordinate length. Equations and matrix algorithms use symbols; numerical material/contact inputs are tabulated separately from measured prediction results.

The article derives dual momentum, contact velocity, impulse/torque, coupled effective mass, normal/tangential restitution, circular friction capacity, and energy/work in 2D/3D. Its practical section describes matrix-free scatter/solve/gather evaluation without discarding shared-body contact coupling. Figures compare eight measured ball/surface cases and held-out limestone impacts, with synthetic geometry verification explicitly separate.

`matrix_tools.py` provides educational matrix-free operations and actual native prediction wrappers. Its 100 random shared-contact chains match dense calculations within2.85e-14. `prediction_examples.py` reproduces retained2D/3D native histories exactly using parameters read from separate input tables. The planar build is isolated and not a qualified all-case baseline. These scripts do not establish a speedup, new material validation or an adopted compliance extension.

Build PDF: `bash research/scaled-contact-article/build.sh` (requires base LaTeX with amsmath, geometry, graphicx and hyperref). From the repository environment, run `python research/scaled-contact-article/matrix_tools.py`, `python research/scaled-contact-article/prediction_examples.py`, and `python research/scaled-contact-article/plot_comparisons.py`. Preserve existing evidence before rerunning. The additional200case scaling audit is in research/contact-moment-identification.

A passive shear/rocking candidate remains rejected: held-out spin error worsens23.7%. No per-impact normal-force offset or moment correction is used to make measured cases pass. Full experimental shape/inertia and independently sourced matched friction remain missing. The removed2x performance requirement is not reinstated by this article.
