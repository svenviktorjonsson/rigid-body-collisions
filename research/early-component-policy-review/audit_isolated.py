"""Audit all scheduled outputs without deleting failures or ranking times."""
from pathlib import Path
import hashlib
import json
import math
import sys

BASE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
summary = {"scope": "Original-law captured-system verification; no trajectory or timing ranking", "captures": [], "failures": []}
completion = json.loads((OUT / "completion.json").read_text())
summary["compiler_failures"] = [r for r in completion["compile_receipts"] if r["returncode"]]
summary["control_failures"] = [r for r in completion["controls"] if r["returncode"]]
summary["guard_differences"] = completion["guard_differences"]
summary["original_plan_unchanged"] = completion["original_plan_unchanged"]
groups = {}
for receipt in completion["replays"]:
    stem = f'capture-{receipt["index"]:02d}-{receipt["label"]}'
    path = OUT / (stem + ".json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["stdout_sha256"]
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        summary["failures"].append({"index": receipt["index"], "label": receipt["label"], "reason": "Invalid or absent JSON", "error": str(error)})
        continue
    groups.setdefault(receipt["index"], {})[receipt["label"]] = value
    if receipt["returncode"] or not value.get("accepted", False):
        summary["failures"].append({"index": receipt["index"], "label": receipt["label"], "reason": "Native or independent-gate decline", "returncode": receipt["returncode"]})


def same(a, b, key):
    return a.get(key) == b.get(key)


for index, values in sorted(groups.items()):
    if set(values) != {"frozen_default", "candidate_default", "candidate_early"}:
        summary["failures"].append({"index": index, "reason": "Missing one or more declared outputs"})
        continue
    frozen, default, early = (values[k] for k in ("frozen_default", "candidate_default", "candidate_early"))
    # The only new fields for default are schedule-receipt bookkeeping.
    default_exact = all(same(frozen, default, key) for key in frozen if key not in {"scope", "schedule"})
    starts_exact = frozen.get("initial_p") == default.get("initial_p") == early.get("initial_p")
    first_phase_exact = all(frozen.get(key) == default.get(key) == early.get(key)
                            for key in ("first256_accepted", "first256_rejected_p", "first256_sweeps"))
    receipt = early.get("early_component_receipt", {})
    item = {"index": index, "default_exact": default_exact, "starts_exact": starts_exact,
            "first_phase_exact": first_phase_exact, "all_accepted": all(v.get("accepted", False) for v in values.values()),
            "early_receipt": receipt, "default_work": default.get("stats", {}), "early_work": early.get("stats", {})}
    if frozen.get("p") and early.get("p"):
        item["max_impulse_difference"] = max(abs(a - b) for a, b in zip(frozen["p"], early["p"]))
        item["max_response_difference_m_s"] = max(abs(a - b) for a, b in zip(frozen["w"], early["w"]))
    if not default_exact or not starts_exact or not first_phase_exact:
        summary["failures"].append({"index": index, "reason": "Default or matched-seed equivalence failure"})
    # Recompute every row and independent law values from archived input,
    # without importing helper numerical merits or optimizer code.
    path = completion["replays"][3 * index]["capture"]
    data = json.loads((BASE.parents[1] / path).read_text())
    A, b, hi, dep = (data[k] for k in ("A", "b", "hi", "dependencies"))
    tol = data["tolerance_m_s"]
    for label, value in values.items():
        if not value.get("solver_accepted", False):
            continue
        p = value["p"]
        w = [sum(row[j] * p[j] for j in range(len(p))) - b[i] for i, row in enumerate(A)]
        finite = all(math.isfinite(x) for x in p + w)
        impulse_scale = max([1.] + [abs(x) for x in p])
        law = finite
        worst = {"normal_negative": 0., "normal_complement_work": 0., "cone_scaled": 0., "support_gap": 0.}
        for k, dependency in enumerate(dep):
            if dependency >= 0:
                continue
            tangent = [j for j, parent in enumerate(dep) if parent == k]
            assert len(tangent) == 2
            t, s = tangent
            eigen = .5 * (A[t][t] + A[s][s] + math.hypot(A[t][t] - A[s][s], 2 * A[t][s]))
            cap = hi[t] * max(0., p[k])
            gap = p[t] * w[t] + p[s] * w[s] + cap * math.hypot(w[t], w[s])
            cone = (math.hypot(p[t], p[s]) - cap) * eigen
            normal_bad = max(-w[k], -p[k] * A[k][k])
            normal_work = abs(p[k] * w[k])
            law &= normal_bad <= tol and normal_work <= tol * impulse_scale and p[k] <= hi[k]
            law &= cone <= tol and abs(gap) <= tol * impulse_scale
            law &= p[t] * w[t] + p[s] * w[s] <= tol * impulse_scale
            worst["normal_negative"] = max(worst["normal_negative"], normal_bad)
            worst["normal_complement_work"] = max(worst["normal_complement_work"], normal_work)
            worst["cone_scaled"] = max(worst["cone_scaled"], cone)
            worst["support_gap"] = max(worst["support_gap"], abs(gap))
        bound = sum(.5 * p[i] * (w[i] - b[i]) for i in range(len(p)))
        scale = 1. + sum(abs(p[i] * b[i]) for i in range(len(p)))
        passive = math.isfinite(bound) and math.isfinite(scale) and bound <= tol * scale
        item.setdefault("independent_checks", {})[label] = {"law": law, "passivity_bound": passive, "worst": worst}
        if not law or not passive:
            summary["failures"].append({"index": index, "label": label, "reason": "External recomputation failed"})
    summary["captures"].append(item)
summary["complete_22_captures"] = len(summary["captures"]) == 22
summary["outputs"] = len(completion["replays"])
summary["early_accepts"] = sum(c["early_receipt"].get("solves", 0) for c in summary["captures"])
summary["early_declines"] = sum(c["early_receipt"].get("declines", 0) for c in summary["captures"])
summary["passed"] = (summary["complete_22_captures"] and summary["outputs"] == 66 and not summary["failures"]
                     and not summary["compiler_failures"] and not summary["control_failures"]
                     and not summary["guard_differences"] and summary["original_plan_unchanged"])
(OUT / "independent-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "captures"}, indent=2))
