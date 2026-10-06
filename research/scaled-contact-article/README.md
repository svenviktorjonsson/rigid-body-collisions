# Symbolic article rewrite — 6 October 2026

Read `article.pdf`; edit `article.tex`. This reading version has been rewritten throughout using a baseline wedge: `r wedge` is the cross-product matrix, `wedge r` its transpose in both 2D and 3D. There is no minus-sign reversal rule. Inertia is written with double-struck I (`\mathbb I`). Linear/angular impulses use lowercase delta and lowercase indices; combined P uses uppercase Delta, as do state and global changes. Body indexing uses only k; relative quantities are unindexed sums of signed body contributions. Vectors are bold again at the user’s latest request; scalar components and planar angular scalars stay plain. Reduced mobility is W_d, not A. Combined V, P, F and mass M retain the dual fixed-length scaling. The directional labels delta p_n, delta p_t, delta L_s and delta L_n are scalars; their unit directions have hats. Full impulses are scalar-weighted sums of directions. Identity and zero blocks use double-struck digits. t follows relative contact velocity and s follows relative angular velocity. No independent linear spin component or arbitrary tangent basis is introduced.

The article includes full force-plus-angular-impulse mechanics, explicit two-body mobility, permitted-component projection, energy/work, symbolic 2D and 3D examples, and coupled matrix-free calculations. The remaining exact constitutive closure is explicitly symbolic; this rewrite does not invent its missing rules or claim corrected native predictions.

At the user's request, all numerical outputs, material values, comparison tables and plots are omitted pending recalculation. Prior artifacts and scripts below remain historical evidence for their originally specified models; their presence is not validation of this directional model.

Build with `bash research/scaled-contact-article/build.sh`.

---

The following documentation describes historical artifacts and preceding article versions. It is superseded for the active PDF by the scope above.

# Length-scaled rigid-body contact article

Read **article.pdf**; edit **article.tex**. The notation is `(v, ell*omega)` and `(p, L/ell)`, with uppercase L the angular momentum and lowercase script ell a fixed coordinate length. Equations and matrix algorithms use symbols; numerical material/contact inputs are tabulated separately from measured prediction results.

The article derives dual momentum, contact velocity, impulse/torque, coupled effective mass, normal/tangential restitution, circular friction capacity, and energy/work in 2D/3D. Its practical section describes matrix-free scatter/solve/gather evaluation without discarding shared-body contact coupling. The main experimental figure compares24Cornell glass impacts using unchanged published inputs. Eight conditional ball/surface cases and held-out fitted limestone comparisons are preserved in an explicitly historical appendix. Synthetic geometry verification remains separate.

The worked3D example is a rotated homogeneous cuboid hitting a stationary plane at its unique lowest corner, with non-diagonal world inertia and three-component incoming velocity/spin. `worked_3d_inputs.json` is the separate illustrative input table; `worked_3d.py` generates the result table, full matrix calculation, geometry/motion figure and independent unscaled cross-product/energy/impulse checks. Evidence is in `worked-3d-v1/calculation.json`; four reference lengths and three tangent-basis rotations retain the physical output within6.22e-15. This frozen impulse example is not a native trajectory replay or a measured-material validation. Preserve it when rerunning: pass a new `--output` directory.

The new prior-art section identifies established spatial motion/force duality and characteristic-length normalization. Translation-first ordering, the dual length scaling and signed lever operator are carried consistently through the contact calculation. The article claims a coherent presentation, not new physics or priority for the notation. Negative tangential restitution is explicitly explained as continued contact slip under the signed convention; fitted coefficients are not promoted to material properties.

`matrix_tools.py` provides educational matrix-free operations and actual native prediction wrappers. Its 100 random shared-contact chains match dense calculations within2.85e-14. `prediction_examples.py` reproduces retained2D/3D native histories exactly using parameters read from separate input tables. The planar build is isolated and not a qualified all-case baseline. These scripts do not establish a speedup, new material validation or an adopted compliance extension.

Build PDF: `bash research/scaled-contact-article/build.sh` (requires base LaTeX with amsmath, geometry, graphicx and hyperref). From the repository environment, run `python research/scaled-contact-article/matrix_tools.py`, `python research/scaled-contact-article/prediction_examples.py`, and `python research/scaled-contact-article/plot_comparisons.py`. Preserve existing evidence before rerunning. The additional200case scaling audit is in research/contact-moment-identification.

A passive shear/rocking candidate remains rejected: held-out spin error worsens23.7%. No per-impact normal-force offset or moment correction is used to make measured cases pass. Full experimental shape/inertia and independently sourced matched friction remain missing. The removed2x performance requirement is not reinstated by this article.

## Single deviation map

The active article now uses figures/deviation-map.png/pdf in place of the bar charts. Open deviation-map.html to inspect dots, or regenerate with plot_deviation_map.py. Each dot maps prediction versus measurement into the angle between observable magnitude signatures and relative signature-norm error. The signature is (abs(v_normal),abs(v_tangent),ell*abs(omega)); ell is the reported radius under a fixed, unfitted policy. This angle is not a spatial heading.25held-out rock impacts and8ball-surface summaries are plotted, with ball input/spin uncertainty lines; synthetic checks are excluded. Ball velocities are reconstructed from target-supplied restitution and spin, so they are conditional, not independent full-state measurements. The full33point vectors, metric definitions and limitations are retained in deviation-map.json. Previous figures remain historical artifacts and are not in the active article.

## Expanded literature review

The targeted review in `../literature-novelty-review/report.pdf` and `review.json` identifies an explicit planar velocity/torque dual-scaling precedent in Vose et al. RSS2011 (appendix footnote1, PDFp7) and spatial deformation/moment scaling in Zhang et al.2014 (section3.3 eq21). Momentum follows by dual impulse integration; exact symbol arrangement was not found, which does not establish priority. Prior art also covers two restitution parameters, coupled Delassus operators, matrix-free contact evaluation and dissipation issues. No completed new research contribution is claimed. The suggested research direction is independently specified material-pair prediction on held-out full-motion data, with model comparisons and uncertainty.

## Independent torque impulse correction

The complete contact update is `delta L = r cross j + k`, where `k` is the independent torque impulse about the declared contact reference point. The article now derives the full scaled wrench, its 2D/3D dual map, coupled effective mass and energy/work identity. A central normal impact with axial spin explicitly demonstrates why a force-only point contact cannot resist that spin. The prescribed torque example is an algebra control, not a measured material prediction.

`wrench_impulse_audit.py` checks 100 random two-body 3D controls and 100 planar controls, including scaling invariance, reference-point shifts, angular momentum, duality and moving-boundary work. Latest evidence is `wrench-impulse-v2/`; v1 preserves the preceding audit before the central-spin example was added. Rerun with a fresh `--output` directory. The production solver remains force-only: no independently characterized predictive rolling/torsional moment law has been implemented, and existing experimental comparisons have not been relabeled as full-wrench validation.
