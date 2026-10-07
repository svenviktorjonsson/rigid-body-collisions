"""Generate research copies and a patch; never compile or alter production files."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / "plan.json").read_text())
assert hashlib.sha256((ROOT / "plan.json").read_bytes()).hexdigest() == (ROOT / "plan.sha256").read_text().split()[0]
for path, digest in plan["source_hashes"].items():
    assert hashlib.sha256((ROOT / "frozen" / path).read_bytes()).hexdigest() == digest
shutil.copytree(ROOT / "frozen", ROOT / "candidate", dirs_exist_ok=True)
changes = []


def replace(path, before, after):
    target = ROOT / "candidate" / path
    value = target.read_text()
    assert value.count(before) == 1, (path, before)
    target.write_text(value.replace(before, after))
    changes.append({"path": path, "before": before, "after": after})


path = "spatial_backend/coulomb.h"
replace(path, "struct CoulombStats {\n", """struct CoulombStats {
 // Optional schedule receipts; these aggregate across calls, never set caps.
 int early_component_attempts=0,early_component_solves=0,early_component_declines=0;
 int early_component_helper_calls=0,early_component_skipped_components=0;
 int early_component_cap_rejections=0,early_component_passes=0,early_component_largest_rows=0;
 int early_component_expanded_contacts=0,early_component_svd_calls=0;
 int early_component_iteration_steps=0,early_component_pressure_svd_calls=0;
 int early_component_pressure_attempts=0,early_component_pivot_calls=0;
""")
replace(path,
        "int budget,double tolerance,CoulombStats& stats,std::vector<double>* rejected_impulses=nullptr,bool allow_recovery=true){",
        "int budget,double tolerance,CoulombStats& stats,std::vector<double>* rejected_impulses=nullptr,bool allow_recovery=true,bool early_component_recovery=false){")
anchor = " btVectorXu candidate=x;for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];\n if(recover){"
insertion = """ btVectorXu candidate=x;for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
#ifdef SPATIAL_LAPACK_RECOVERY
 // RESEARCH OPTION: default false preserves the frozen lane ordering.
 // Use the actual first256 rejected iterate, not a cold/captured proxy seed.
 if(early_component_recovery&&recover&&first_budget==256&&b.rows()<=4096){
  support_restart_v3::Stats early; // Fresh independent caps for THIS call.
  stats.early_component_attempts++;
  const bool accepted=support_restart_v3::solve(A,b,candidate,hi,dep,tolerance,early);
  stats.early_component_helper_calls+=early.helper_calls;
  stats.early_component_skipped_components+=early.skipped_components;
  stats.early_component_cap_rejections+=early.component_cap_rejections;
  stats.early_component_passes+=early.passes;
  stats.early_component_largest_rows=std::max(stats.early_component_largest_rows,early.largest_reduced_rows);
  stats.early_component_expanded_contacts+=early.expanded_contacts;
  stats.early_component_svd_calls+=early.svd_calls;
  stats.early_component_iteration_steps+=early.iteration_steps;
  stats.early_component_pressure_svd_calls+=early.pressure_svd_calls;
  stats.early_component_pressure_attempts+=early.pressure_attempts;
  stats.early_component_pivot_calls+=early.pivot_attempts;
  // Aggregate BOTH early and existing later tail work, including declines.
  stats.support_helper_calls+=early.helper_calls;
  stats.support_skipped_components+=early.skipped_components;
  stats.support_component_cap_rejections+=early.component_cap_rejections;
  stats.support_passes+=early.passes;
  stats.support_largest_rows=std::max(stats.support_largest_rows,early.largest_reduced_rows);
  stats.support_expanded_contacts+=early.expanded_contacts;
  stats.support_svd_calls+=early.svd_calls;
  stats.support_iteration_steps+=early.iteration_steps;
  stats.support_pressure_svd_calls+=early.pressure_svd_calls;
  stats.support_pivot_calls+=early.pivot_attempts;
  if(accepted){
   // V3 returns true only after its ORIGINAL full-row/bounds/passivity gate.
   x=candidate;stats.solves++;stats.support_solves++;stats.early_component_solves++;
   stats.last_residual=early.residual;stats.residual_max=std::max(stats.residual_max,early.residual);
   stats.sweeps_max=std::max(stats.sweeps_max,first_budget);
   double change=0;for(int i=0;i<b.rows();i++){double w=-b[i];for(int j=0;j<b.rows();j++)w+=A(i,j)*x[j];change+=.5*x[i]*(w-b[i]);}
   stats.passive_change_max=std::max(stats.passive_change_max,change);
   return true;
  }
  stats.early_component_declines++;
  // A declined trial NEVER reaches x; restore the exact original next-lane seed.
  candidate=x;for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
 }
#endif
 if(recover){"""
replace(path, anchor, insertion)
replace(path,
        "info.m_numIterations,tolerance,stats,rejection_observer?&rejected:nullptr,recovery_enabled))",
        "info.m_numIterations,tolerance,stats,rejection_observer?&rejected:nullptr,recovery_enabled,early_component_recovery))")
replace(path,
        " bool recovery_enabled=true,shared_contact_point=true;",
        " bool early_component_recovery=false; // Explicit research opt-in; velocity solve only.\n bool recovery_enabled=true,shared_contact_point=true;")

path = "spatial_backend/runner.cpp"
replace(path,
        ' coulomb_mlcp.recovery_enabled=in.value("contact_recovery",true);',
        ' coulomb_mlcp.recovery_enabled=in.value("contact_recovery",true);\n'
        ' coulomb_mlcp.early_component_recovery=in.value("early_component_recovery",false);\n'
        ' if(coulomb_mlcp.early_component_recovery&&(!coulomb_solver||!coulomb_mlcp.recovery_enabled||!coulombLapackRecoveryEnabled()))throw std::runtime_error("Early component schedule requires Coulomb and enabled compiled recovery");')
replace(path,
        ' out["coulomb_support_solves"]=coulomb_mlcp.stats.support_solves;',
        """ if(coulomb_mlcp.early_component_recovery){
  const auto& e=coulomb_mlcp.stats;
  out["early_component_policy"]={{"enabled",true},{"seed","actual first256 rejected PGS"},
   {"attempts",e.early_component_attempts},{"solves",e.early_component_solves},{"declines",e.early_component_declines},
   {"helper_calls",e.early_component_helper_calls},{"skipped_components",e.early_component_skipped_components},
   {"component_cap_rejections",e.early_component_cap_rejections},{"passes",e.early_component_passes},
   {"largest_rows",e.early_component_largest_rows},{"expanded_contacts",e.early_component_expanded_contacts},
   {"svd_calls",e.early_component_svd_calls},{"iteration_steps",e.early_component_iteration_steps},
   {"pressure_svd_calls",e.early_component_pressure_svd_calls},{"pressure_attempts",e.early_component_pressure_attempts},
   {"pivot_calls",e.early_component_pivot_calls},
   {"extra_budget","early and later tail have independent fresh caps; aggregate support counters include both"},
   {"claim","optional original-law schedule; accepted root and trajectory may differ"}};
 }
 out["coulomb_support_solves"]=coulomb_mlcp.stats.support_solves;""")

(ROOT / "transformations.json").write_text(json.dumps(changes, indent=2) + "\n")
diffs = []
for path in sorted({item["path"] for item in changes}):
    before = (ROOT / "frozen" / path).read_text()
    after = (ROOT / "candidate" / path).read_text()
    restored = after
    for change in reversed(changes):
        if change["path"] == path:
            assert restored.count(change["after"]) == 1
            restored = restored.replace(change["after"], change["before"])
    assert restored == before, path
    diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                    fromfile="a/" + path, tofile="b/" + path))
(ROOT / "optional-schedule.patch").write_text("".join(diffs))
manifest = {"frozen_source": plan["frozen_source"], "plan_sha256": hashlib.sha256((ROOT / "plan.json").read_bytes()).hexdigest(),
            "execution": "PREPARED ONLY; no compiler or numeric execution",
            "modified_copies": sorted({item["path"] for item in changes}),
            "frozen_reconstruction_exact": True,
            "candidate_hashes": {str(p.relative_to(ROOT / "candidate")): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / "candidate").rglob("*")) if p.is_file()}}
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Prepared candidate copies and reversible patch. No compiler invoked.")
