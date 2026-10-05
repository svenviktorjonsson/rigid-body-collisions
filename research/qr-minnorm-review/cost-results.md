# Complete captured-system paired cost results

All 240 scheduled attempts are retained: 40 warmups and 200 timed solves. Each table entry contains five timed paired repetitions. Costs are native solver seconds, excluding parsing and process startup. The ratio is the predeclared median of paired baseline/QR costs; above one means lower QR cost in these observations. These are descriptive measurements under a shared workload, with no isolated-machine or whole-trajectory performance claim.

| Capture | Rows | Baseline median [range], s | QR median [range], s | Median paired ratio | Accepted baseline / QR |
|---|---:|---:|---:|---:|---:|
| [initial hull42](../../research/coulomb-diagnostics/hull42-rejected.json) | 18 | 0.00276751 [0.0016225, 0.00678968] | 0.00224992 [0.00161386, 0.00453987] | 1.431 | 5/5 / 5/5 |
| [initial hull7301](../../research/coulomb-diagnostics/hull7301-rejected.json) | 108 | 5.2984 [4.42662, 10.1181] | 3.8726 [2.64624, 15.2437] | 1.248 | 5/5 / 5/5 |
| [later hull7301](../../research/coulomb-diagnostics/hull7301-later-rejected.json) | 267 | 51.0389 [37.0651, 68.0342] | 23.0184 [18.7251, 31.7545] | 2.143 | 5/5 / 5/5 |
| [followup hull42](../../research/coulomb-followup/hull42-second-rejected.json) | 45 | 0.417136 [0.281415, 1.1972] | 0.193899 [0.140717, 0.388645] | 2.049 | 5/5 / 5/5 |
| [shared followup hull42 ref1](../../research/shared-hull-followup/results/rejections/fast_shake8_hulls42/reference_1.json) | 48 | 0.00315011 [0.00256552, 0.00773803] | 0.004802 [0.00362147, 0.00791206] | 0.689 | 5/5 / 5/5 |
| [rank hull7301 ref0](../../research/shared-hull-rank-followup/results/rejections/fast_rotate_shake27_hulls7301/reference_0.json) | 216 | 0.0524391 [0.0335941, 0.0997598] | 0.0501416 [0.0311448, 0.107616] | 1.241 | 5/5 / 5/5 |
| [rank hull7301 ref1](../../research/shared-hull-rank-followup/results/rejections/fast_rotate_shake27_hulls7301/reference_1.json) | 192 | 0.0385884 [0.0271357, 0.10693] | 0.063074 [0.0338658, 0.0698989] | 0.8942 | 5/5 / 5/5 |
| [rank hull7301 ref2](../../research/shared-hull-rank-followup/results/rejections/fast_rotate_shake27_hulls7301/reference_2.json) | 123 | 0.0210724 [0.0165988, 0.0288679] | 0.0151833 [0.00884331, 0.021543] | 1.593 | 5/5 / 5/5 |
| [rank hull42 ref0](../../research/shared-hull-rank-followup/results/rejections/fast_shake8_hulls42/reference_0.json) | 48 | 0.00511119 [0.00424048, 0.00817656] | 0.00599048 [0.0035355, 0.0077759] | 1.135 | 5/5 / 5/5 |
| [rank hull42 ref1](../../research/shared-hull-rank-followup/results/rejections/fast_shake8_hulls42/reference_1.json) | 51 | 0.00509613 [0.00282508, 0.0124723] | 0.00263451 [0.0020685, 0.0137304] | 1.726 | 5/5 / 5/5 |
| [rank hull42 ref2](../../research/shared-hull-rank-followup/results/rejections/fast_shake8_hulls42/reference_2.json) | 30 | 0.00418007 [0.00294247, 0.00727766] | 0.00356962 [0.00210199, 0.0100543] | 0.8944 | 5/5 / 5/5 |
| [completion hull7301 ref0](../../research/hull-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_0.json) | 321 | 2.9878 [2.26022, 4.79155] | 1.38954 [0.820289, 1.93899] | 2.919 | 5/5 / 5/5 |
| [completion hull7301 ref2](../../research/hull-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_2.json) | 231 | 0.0843911 [0.0457744, 0.133522] | 0.0725895 [0.0514631, 0.0835868] | 1.01 | 5/5 / 5/5 |
| [completion hull42 ref0 (position)](../../research/hull-completion/results/rejections/fast_shake8_hulls42/reference_0.json) | 54 | 0.00335441 [0.00184539, 0.00622381] | 0.00324457 [0.00159012, 0.00711006] | 0.7863 | 5/5 / 5/5 |
| [completion hull42 ref1](../../research/hull-completion/results/rejections/fast_shake8_hulls42/reference_1.json) | 45 | 0.0438128 [0.0380006, 0.0837161] | 0.0253737 [0.0156865, 0.043099] | 1.93 | 5/5 / 5/5 |
| [completion hull42 ref2](../../research/hull-completion/results/rejections/fast_shake8_hulls42/reference_2.json) | 48 | 0.00711116 [0.00551222, 0.0195442] | 0.00816638 [0.00326687, 0.0164765] | 0.9282 | 5/5 / 5/5 |
| [active hull42 ref0](../../research/hull-active-completion/results/rejections/fast_shake8_hulls42/reference_0.json) | 60 | 1.5058 [1.35513, 2.94635] | 1.68277 [1.2229, 2.51028] | excluded: failed | 0/5 / 0/5 |
| [active hull42 ref1](../../research/hull-active-completion/results/rejections/fast_shake8_hulls42/reference_1.json) | 39 | 0.657061 [0.363952, 0.81625] | 0.40875 [0.325422, 0.643937] | excluded: failed | 0/5 / 0/5 |
| [active hull42 ref2](../../research/hull-active-completion/results/rejections/fast_shake8_hulls42/reference_2.json) | 51 | 0.970046 [0.713696, 1.48627] | 0.531369 [0.471611, 1.09371] | excluded: failed | 0/5 / 0/5 |
| [active hull7301 ref0 (position)](../../research/hull-active-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_0.json) | 291 | 0.151968 [0.132951, 0.211526] | 0.145099 [0.112457, 0.173634] | 1.047 | 5/5 / 5/5 |

There are 0 baseline-accepted regressions. Of the 17 eligible captures, 12 have a paired median ratio above one and 5 below one. This describes observed directions, without a confidence interval or generalization beyond these fixed captures. The three captures failed in both variants are retained, with no successful-cost ratio.

QR proposes a numerical minimum-norm Newton direction; pressure and other required nullspace SVDs remain. Count reductions do not establish improvements by themselves. Source, input and binary hashes, random pair ordering, original iteration budgets, the full original law, finite/nonnegative checks and passivity were independently re-audited. Numerical research here does not qualify a full physical trajectory.

The frozen control source is 95d224f, preceding subsequent production repairs. The measured ratios cannot be transferred to that later implementation without another prospective comparison. The earlier v1 regression remains unchanged.

Concurrency disclosure: per-attempt load averages and the 30-second process ledger are retained. A brief supplementary fixture compilation and audit execution occurred while timing was running. CPU4 affinity and single-thread settings do not establish machine isolation.
