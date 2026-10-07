# Independent next 243-row velocity review

This capture comes from the immutable study source `52f7e6d244e92a8405134e0222ce14fc3eda0ef6`, seed7301 reference_1. SHA-256: `87c8637e8df9bbcbe085c184b27e012808afcc897d26547bdf53ae5312c49d7b`. No production, authored geometry, material, binary, physical matrix or original 1e-8 m/s tolerance is changed by this review.

There are 81 contacts and 30 positive-pressure contacts. Full mobility/contact-dependency connectivity gives components of 204, 18, 6, 12 and 3 rows. On the native positive-pressure support, exact connectivity gives smaller components of 6, 9, 39, 6, 6, 6, 3, 9, 3 and 3 rows. Contact dependency joins its normal and both tangent rows even when their mobility cross entries are zero, because the friction capacity depends on normal pressure.

The outgoing-normal rule from the preceding 261-row example selects contact 5 first. That rule fails here: dropping 5 solves the remaining 87 rows but leaves its omitted normal moving inward at 1.6835 m/s. This is a mandatory original-gate rejection. Dropping either 39 or 19 also fails, with omitted-contact violations 0.8443 and 1.0681 m/s. All three unsuccessful release trials remain archived. An initializer is not a physical acceptance rule.

Warm FB trust least-squares on the entire 243-row matrix finds a strict original root in 178 Jacobian evaluations. Restricting search variables to the native 90-row positive support also finds a root in 177 Jacobian evaluations. Both outputs pass the original all-row gate, finite pressure bounds and passivity after tiny negative pressure roundoff is clipped and the gate is recomputed.

A smaller exact numerical search suffices. `component.py` derives connected components from the current positive support, orders them by their current residual and solves only the first failing component. Its selected nine rows belong to contacts 5, 19 and 39; these identities are derived, not encoded as a fixture solution. All other native impulses remain unchanged. The resulting complete 243-row impulse satisfies:

- Original full projection residual: approximately **4.6464e-12 m/s** under independent recomputation.
- Nonnegative pressures and original finite upper bounds: pass.
- Frozen-system passivity bound: **−0.871982323 J**, finite.
- Physical normal complementarity, circular capacity, maximum-dissipation support and friction work: pass within the original numerical tolerance.

The small component search uses 188 function evaluations and 177 Jacobian evaluations. It does not omit the final mechanical constraints: `audit.py`, importing neither the solver nor its contact helper, recomputes every original row and physical work gate. It additionally proves that the selected component has exactly zero mobility coupling to the other positive impulse coordinates and that all other impulses are unchanged. Those checks are in `independent-audit.json`.

Inactive contacts can connect positive-support components if they later carry pressure. Therefore this numerical decomposition must retain the full-system mechanical gate and must reject or separately expand the support when new inward violations appear. It is an exact restricted search for this accepted impulse, not unconditional physical separation of bodies or contact graphs.

Native integration and fresh full-hull refinement qualification are separate work. The companion native reviewer reports a general support-expansion solver accepting this captured system in 103 SVD calls; its receipts live in `research/translation-native-review`. No accepted archived impulse was used to initialize that native solve. Costs here are diagnostic, with collaborative workload, and establish no whole-engine speedup.
