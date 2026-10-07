"""Check benchmark provenance and compare scalar physical observables.

These checks do not turn a numerical example into experimental validation.
Run from the repository root: python -m research.benchmark_tools --help
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def _number(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"Invalid {name}")
    return value


def _local_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes asset directory: {relative}")
    return path


def audit_assets(root):
    root = Path(root)
    verified = 0
    for manifest_name, folder in (
        ("source-manifest.json", "sources"),
        ("getfem-source-manifest.json", "getfem-sources"),
    ):
        manifest = json.loads((root / manifest_name).read_text())
        if not re.fullmatch(r"[0-9a-f]{40}", manifest["commit"]):
            raise ValueError("Upstream revision must be pinned to a full commit")
        for item in manifest["files"]:
            if "error" in item:
                raise ValueError(f"Incomplete download: {item}")
            expected_url = (f"https://raw.githubusercontent.com/{manifest['repository']}/"
                            f"{manifest['commit']}/{item['path']}")
            if item.get("source_url") != expected_url:
                raise ValueError(f"Source URL does not match pinned revision: {item['path']}")
            path = _local_path(root / folder, item["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Source hash mismatch: {path}")
            verified += 1
    catalog = json.loads((root / "benchmark-catalog.json").read_text())
    ids = set()
    for case in catalog["cases"]:
        if case["id"] in ids:
            raise ValueError(f"Duplicate case: {case['id']}")
        ids.add(case["id"])
        if not _local_path(root, case["local_input"]).is_file():
            raise ValueError(f"Missing input: {case['id']}")
        for key in ("parameter_provenance", "reference_status", "dimension"):
            if not case.get(key):
                raise ValueError(f"Missing {key}: {case['id']}")
        if not case.get("authenticity", {}).get("status"):
            raise ValueError(f"Missing authenticity status: {case['id']}")
        evidence = case["authenticity"]
        if evidence["status"] == "experimentally_validated":
            if not evidence.get("quantitative_experimental_validation_completed"):
                raise ValueError(f"Unsupported physical validation claim: {case['id']}")
            if not evidence.get("experimental_data_source"):
                raise ValueError(f"Experimental data source required: {case['id']}")
    return {
        "source_files_verified": verified,
        "catalog_cases": len(ids),
        "physically_validated_cases": sum(
            c["authenticity"]["status"] == "experimentally_validated"
            for c in catalog["cases"]
        ),
        "source_integrity_passed": True,
        "scope": "File integrity and provenance records; no physics accuracy claim.",
    }


def compare_results(reference, candidate, budget):
    """Compare a declared set of scalar observables in matching physical units."""
    for key in ("case_id", "physical_setup_id"):
        if not reference.get(key) or reference.get(key) != candidate.get(key):
            raise ValueError(f"Reference and candidate must share {key}")
    if any(type(r.get("schema_version")) is not int or r["schema_version"] != 1
           for r in (reference, candidate)):
        raise ValueError("Result schema_version must be 1")
    evidence_kinds = {"synthetic_fixture", "numerical_simulation", "experimental_measurement"}
    for result in (reference, candidate):
        if result.get("evidence_kind") not in evidence_kinds:
            raise ValueError("Declare evidence_kind for every result")
    specifications = budget.get("observables", {})
    if not specifications:
        raise ValueError("At least one observable budget is required")
    checks = {}
    for name, tolerance in specifications.items():
        expected_unit = tolerance.get("unit")
        if not isinstance(expected_unit, str) or not expected_unit:
            raise ValueError(f"Unit required for {name}")
        values = []
        for result in (reference, candidate):
            observable = result.get("observables", {}).get(name, {})
            if observable.get("unit") != expected_unit:
                raise ValueError(f"Missing observable or unit mismatch: {name}")
            values.append(_number(observable.get("value"), name))
        absolute = _number(tolerance.get("abs_tol"), "abs_tol", nonnegative=True)
        relative = _number(tolerance.get("rel_tol"), "rel_tol", nonnegative=True)
        allowed = absolute + relative * abs(values[0])
        if not math.isfinite(allowed):
            raise ValueError(f"Tolerance overflow: {name}")
        error = abs(values[1] - values[0])
        if not math.isfinite(error):
            raise ValueError(f"Observable difference overflow: {name}")
        checks[name] = {
            "unit": expected_unit,
            "reference": values[0],
            "candidate": values[1],
            "absolute_error": error,
            "allowed_error": allowed,
            "within_budget": error <= allowed,
        }
    timings = []
    for result in (reference, candidate):
        timings.append(_number(result.get("wall_time_s"), "wall_time_s", nonnegative=True))
    ratio = timings[0] / timings[1] if timings[1] else None
    if ratio is not None and not math.isfinite(ratio):
        raise ValueError("Runtime ratio overflow")
    return {
        "case_id": reference["case_id"],
        "within_budget": all(c["within_budget"] for c in checks.values()),
        "reference_evidence_kind": reference["evidence_kind"],
        "candidate_evidence_kind": candidate["evidence_kind"],
        "observables": checks,
        "runtime_ratio_reference_over_candidate": ratio,
        "scope": "Only the declared observables; runtime ratio is not a repeated benchmark.",
    }


def check_convergence(runs, budget, axis):
    """Require three or more refinements of one axis with others unchanged.

    A passed check means successive differences meet the supplied budget.
    It is not proof of continuum convergence or agreement with measurements.
    """
    if axis not in ("dt_s", "mesh_h_m", "solver_tolerance"):
        raise ValueError("Choose dt_s, mesh_h_m, or solver_tolerance")
    if len(runs) < 3:
        raise ValueError("At least three refinement levels are required")
    previous = None
    other_controls = None
    for run in runs:
        if run.get("evidence_kind") == "experimental_measurement":
            raise ValueError("Numerical refinement checks require simulation results, not measurements")
        controls = run.get("fidelity", {})
        for control in ("dt_s", "mesh_h_m", "solver_tolerance"):
            if _number(controls.get(control), control, nonnegative=True) == 0:
                raise ValueError(f"Positive {control} required for numerical refinement")
        level = _number(controls.get(axis), axis, nonnegative=True)
        if level == 0 or (previous is not None and level >= previous):
            raise ValueError("Supply runs from coarse to fine with strictly decreasing axis")
        others = {k: v for k, v in controls.items() if k != axis}
        if not others:
            raise ValueError("Record the other fidelity controls, held fixed during refinement")
        if other_controls is not None and others != other_controls:
            raise ValueError("Refine one axis at a time; other controls changed")
        previous, other_controls = level, others
    comparisons = [compare_results(finer, coarser, budget)
                   for coarser, finer in zip(runs[:-1], runs[1:])]
    return {
        "axis": axis,
        "levels": [r["fidelity"][axis] for r in runs],
        "successive_changes_within_budget": all(c["within_budget"] for c in comparisons[-2:]),
        "comparisons": comparisons,
        "scope": "Last two refinement differences only; other axes and physical validation remain separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Verify downloaded sources and provenance records")
    audit.add_argument("--assets", type=Path, default=Path(__file__).parent / "adaptive-benchmarks")
    compare = commands.add_parser("compare", help="Compare an engine result against a reference")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--budget", type=Path, required=True)
    convergence = commands.add_parser("convergence", help="Check one reference-refinement axis")
    convergence.add_argument("--runs", type=Path, nargs="+", required=True)
    convergence.add_argument("--budget", type=Path, required=True)
    convergence.add_argument("--axis", choices=("dt_s", "mesh_h_m", "solver_tolerance"), required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    try:
        if args.command == "audit":
            result = audit_assets(args.assets)
            passed = True
        elif args.command == "compare":
            result = compare_results(read(args.reference), read(args.candidate), read(args.budget))
            passed = result["within_budget"]
        else:
            result = check_convergence([read(p) for p in args.runs], read(args.budget), args.axis)
            passed = result["successive_changes_within_budget"]
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(2, f"Benchmark check failed: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
