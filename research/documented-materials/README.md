# Documented pair parameters, without project fitting

Open **catalog.html** or inspect **catalog.json**. Eleven profiles preserve the published normal restitution, tangential restitution and sliding friction, their reported uncertainties, nominal particle diameter, source and available experimental context. Parameter origin is experimental characterization by the source authors. No values are fitted by this project or substituted from another material pair.

The sources are [Cornell's impact chart](https://grainflowresearch.mae.cornell.edu/impact/data/Impact%20Results.html), its [experimental-method page](https://grainflowresearch.mae.cornell.edu/impact/impact.html), and [Joseph & Hunt (2004)](https://authors.library.caltech.edu/records/ks4y3-4ct92), p78 for dry12.7mm steel/glass spheres against Zerodur. The latter documents normal velocities0.05–0.38m/s. Missing context stays unknown. A printed zero uncertainty is not treated as mathematical exactness.

`material_profiles.documented_profile(id)` returns a source-backed nominal profile. `run_documented_pair(scene,id,**options)` passes both restitution coefficients unchanged to the native3D solver and realizes the measured pair friction through Bullet's product mixing (`sqrt(mu)` on each side). It requires one pair, documented geometry and nominal size. It does not define a generic per-body mixing law or heterogeneous restitution graph. Parameters remain in the separate catalog; equations stay symbolic.

The 7 October measured-properties adapter accepts explicit mass/COM/inertia, but
this source-specific wrapper checks authoritative mass against the documented
density and preserves declared homogeneous-sphere COM/inertia assumptions.
Conflicting overrides are rejected. A fresh 24-case replay in
`../contact-gap-fix/glass-repeat-v2/` has exactly the prior residuals. The new
supported rolling/spin primitive is separate and is not validated by this worksheet.

The81native/analytic/energy checks pass for the nine profiles with documented density. They verify parameter loading and mechanics, not real-world accuracy. The initial loader error assignedmu to both Bullet bodies, producingmu²; its failed control and diagnosis are preserved separately. This correction changes the adapter, not production contact physics or source values. The corresponding3D educational wrapper is corrected too; Box2D2D mixing remains different.

## Comparison with recorded impact data

Cornell's original [3mm-glass binary worksheet](https://grainflowresearch.mae.cornell.edu/impact/data/Results-3mmglass-binary) supplies24photo records. With the published glass/glass parameter triple unchanged, native predictions have:

| Quantity | RMSE |
|---|---:|
| Relative outgoing normal velocity |0.03667m/s|
| Relative outgoing COM tangential velocity |0.01780m/s|
| Reconstructed contact tangential velocity |0.06232m/s|

The normalized translational joint RMSE is0.02416, divided by each incoming relative speed. This is a reproduction using source-cached contact reconstruction, not independent validation: the chart and worksheet may share characterization trials. Angular velocity is reconstructed by the author using angular momentum and sphere inertia; it is not an independent measured vector. The reduced instantaneous scene does not replay the photographed3Dtrajectory.

**material-backed-deviation-map.html** allows point inspection; the PNG/PDF version uses hollow points to identify reconstructed spin. The fixed plotting policy and source assumptions accompany every figure. No changes are made to parameters or plotted measurements to move points toward zero. The separate plate worksheet is retained in the external cache; its coefficient summary differs from the catalog profile, so it is not silently combined into this comparison.

## Still missing for the original requested pairs

The limestone/C25concrete source provides COM velocity ratios and elastic properties, not a complete compatible contact en/et/mu triple. Cross2010Superball surfaces report both restitution values but not matched independently specified sliding friction. They remain incomplete profiles; fitted rock coefficients and assumed rubber friction are excluded. Public values are empirical and condition-specific, not universal exact constants for a material name.

The recent fixed-material Pareto plan is superseded before execution. Historical geometry/inertia fits remain preserved as rejected development evidence, with no production adoption. The user removed the2x performance gate.

Rebuild catalog with `build_catalog.py` (primary files in the external cache); use `verify_predictions.py`, `compare_glass_worksheet.py` (requiresxlrd), then `render.py` in the repository Python environment. Native result directories deliberately require fresh paths; preserve prior evidence before rerunning. Source PDFs/XLS files are not redistributed in the repository; hashes and derived comparisons are retained.
