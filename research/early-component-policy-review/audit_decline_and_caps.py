"""Additive audit: decline immutability and fully retained extra helper work."""
from pathlib import Path
import hashlib
import json
import sys

out = Path(sys.argv[1])
receipt = {"auditor": Path(__file__).name, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "criterion": "All22inputs; unchanged original caps; exact baseline endpoints on every decline; account all early work in shared totals"}
with (out / "decline-budget-audit-launch.json").open("x") as stream:
    stream.write(json.dumps(receipt, indent=2) + "\n")
mapping = {"helper_calls": "support_helper_calls", "passes": "support_passes", "svd_calls": "support_svd_calls",
           "pressure_svd_calls": "support_pressure_svd_calls", "pivot_calls": "support_pivot_calls",
           "iteration_steps": "support_iteration_steps", "component_cap_rejections": "support_component_cap_rejections"}
caps = {"helper_calls": 8, "passes": 8, "svd_calls": 1024, "pressure_svd_calls": 1024, "pivot_calls": 8}
records = []
for index in range(22):
    default = json.loads((out / f"capture-{index:02d}-candidate_default.json").read_text())
    early = json.loads((out / f"capture-{index:02d}-candidate_early.json").read_text())
    local = early["early_component_receipt"]
    record = {"index": index, "early_declined": bool(local["declines"]),
              "per_call_caps_pass": all(local[k] <= v for k, v in caps.items()),
              "aggregate_two_call_caps_pass": all(early["stats"][mapping[k]] <= 2 * v for k, v in caps.items())}
    if local["declines"]:
        record["exact_baseline_endpoint_retained"] = default["p"] == early["p"] and default["w"] == early["w"]
        record["extra_work_exactly_aggregated"] = all(early["stats"][total] - default["stats"][total] == local[field]
                                                    for field, total in mapping.items())
    records.append(record)
payload = {"scope": "Supplemental original-cap/immutability audit; no new solver acceptance criterion",
           "declines": sum(r["early_declined"] for r in records), "records": records,
           "passed": all(all(v for k, v in r.items() if k not in {"index", "early_declined"}) for r in records)}
(out / "decline-budget-audit.json").write_text(json.dumps(payload, indent=2) + "\n")
print("Declines", payload["declines"], "cap/immutability/aggregate audit", payload["passed"])
