import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research.benchmark_tools import audit_assets, check_convergence, compare_results


class BenchmarkToolsTests(unittest.TestCase):
    def setUp(self):
        self.result = {
            "schema_version": 1, "case_id": "impact", "physical_setup_id": "same-material-and-geometry",
            "evidence_kind": "synthetic_fixture",
            "observables": {"spin": {"value": 0.0, "unit": "rad/s"}},
            "wall_time_s": 1.0,
            "fidelity": {"dt_s": 0.1, "mesh_h_m": 0.01, "solver_tolerance": 1e-8},
        }
        self.budget = {"observables": {"spin": {"unit": "rad/s", "abs_tol": 0.01, "rel_tol": 0.02}}}

    def test_absolute_floor_at_zero_reference(self):
        candidate = copy.deepcopy(self.result)
        candidate["observables"]["spin"]["value"] = 0.009
        self.assertTrue(compare_results(self.result, candidate, self.budget)["within_budget"])
        candidate["observables"]["spin"]["value"] = 0.02
        self.assertFalse(compare_results(self.result, candidate, self.budget)["within_budget"])

    def test_mismatched_setup_or_units_are_rejected(self):
        for key in ("case_id", "physical_setup_id"):
            candidate = copy.deepcopy(self.result)
            candidate[key] = "different"
            with self.assertRaises(ValueError):
                compare_results(self.result, candidate, self.budget)
        candidate = copy.deepcopy(self.result)
        candidate["observables"]["spin"]["unit"] = "deg/s"
        with self.assertRaises(ValueError):
            compare_results(self.result, candidate, self.budget)

    def test_nonfinite_and_invalid_tolerances_are_rejected(self):
        for bad in (float("nan"), float("inf"), True):
            candidate = copy.deepcopy(self.result)
            candidate["observables"]["spin"]["value"] = bad
            with self.assertRaises(ValueError):
                compare_results(self.result, candidate, self.budget)
        budget = copy.deepcopy(self.budget)
        budget["observables"]["spin"]["abs_tol"] = -1
        with self.assertRaises(ValueError):
            compare_results(self.result, self.result, budget)

    def levels(self, values):
        runs = []
        for dt, value in zip((0.1, 0.05, 0.025), values):
            run = copy.deepcopy(self.result)
            run["fidelity"]["dt_s"] = dt
            run["observables"]["spin"]["value"] = value
            runs.append(run)
        return runs

    def test_convergence_checks_two_successive_differences(self):
        runs = self.levels((0.005, 0.002, 0.001))
        self.assertTrue(check_convergence(runs, self.budget, "dt_s")["successive_changes_within_budget"])
        runs = self.levels((1.0, 0.002, 0.001))
        self.assertFalse(check_convergence(runs, self.budget, "dt_s")["successive_changes_within_budget"])

    def test_mixed_refinement_and_missing_levels_are_rejected(self):
        runs = self.levels((0.005, 0.002, 0.001))
        with self.assertRaises(ValueError):
            check_convergence(runs[:2], self.budget, "dt_s")
        runs[-1]["fidelity"]["mesh_h_m"] = 0.005
        with self.assertRaises(ValueError):
            check_convergence(runs, self.budget, "dt_s")

    def test_refinement_order_is_checked(self):
        with self.assertRaises(ValueError):
            check_convergence(self.levels((0, 0, 0))[::-1], self.budget, "dt_s")

    def test_runtime_ratio_is_optional_for_zero_duration(self):
        candidate = copy.deepcopy(self.result)
        candidate["wall_time_s"] = 0.0
        self.assertIsNone(compare_results(self.result, candidate, self.budget)["runtime_ratio_reference_over_candidate"])

    def test_evidence_kind_is_required_and_reported(self):
        report = compare_results(self.result, self.result, self.budget)
        self.assertEqual(report["reference_evidence_kind"], "synthetic_fixture")
        candidate = copy.deepcopy(self.result)
        candidate.pop("evidence_kind")
        with self.assertRaises(ValueError):
            compare_results(self.result, candidate, self.budget)

    def test_measurements_cannot_be_numerically_refined(self):
        runs = self.levels((0, 0, 0))
        runs[0]["evidence_kind"] = "experimental_measurement"
        with self.assertRaises(ValueError):
            check_convergence(runs, self.budget, "dt_s")

    def assets(self, root):
        for manifest_name, folder in (("source-manifest.json", "sources"),
                                      ("getfem-source-manifest.json", "getfem-sources")):
            (root / folder).mkdir()
            content = b"published input\n"
            (root / folder / "input.txt").write_bytes(content)
            manifest = {"commit": "a" * 40, "repository": "example/solver", "files": [{
                "path": "input.txt", "sha256": hashlib.sha256(content).hexdigest(),
                "source_url": "https://raw.githubusercontent.com/example/solver/" + "a" * 40 + "/input.txt",
            }]}
            (root / manifest_name).write_text(json.dumps(manifest))
        case = {"id": "one", "local_input": "sources/input.txt", "dimension": "2D",
                "parameter_provenance": "numerical", "reference_status": "not computed",
                "authenticity": {"status": "not_experimentally_validated"}}
        (root / "benchmark-catalog.json").write_text(json.dumps({"cases": [case]}))

    def test_modified_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assets(root)
            self.assertEqual(audit_assets(root)["source_files_verified"], 2)
            (root / "sources/input.txt").write_text("changed material input")
            with self.assertRaises(ValueError):
                audit_assets(root)

    def test_unsupported_authenticity_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assets(root)
            path = root / "benchmark-catalog.json"
            catalog = json.loads(path.read_text())
            catalog["cases"][0]["authenticity"]["status"] = "experimentally_validated"
            path.write_text(json.dumps(catalog))
            with self.assertRaises(ValueError):
                audit_assets(root)
