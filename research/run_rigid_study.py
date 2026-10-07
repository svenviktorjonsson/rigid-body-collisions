"""Execute predeclared planar rigid verification, refinement and adaptation tests.

Run from the repository root: python -m research.run_rigid_study --repeats 5
Both pinned backends must be built. These are numerical, not material experiments.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rigid_engine import BOX2D_COMMITS, DEFAULT_POLICY, run
from research.rigid_scenes import scenes


BUDGET = {"rms_position_m": .02, "rms_velocity_m_s": .05, "rms_spin_rad_s": .05}
REFERENCE_BUDGET = {key: value/4 for key, value in BUDGET.items()}
FIXED = {"block_fast": ("block", 1, 1), "block_standard": ("block", 1, 8),
         "block_accurate": ("block", 4, 16), "block_high": ("block", 8, 32),
         "temporal_standard": ("temporal", 1, 4), "temporal_accurate": ("temporal", 4, 16)}


def errors(reference, candidate):
    if reference["physical_setup_id"] != candidate["physical_setup_id"]:
        raise ValueError("Different physical setup")
    r, c = np.asarray(reference["states"]), np.asarray(candidate["states"])
    if r.shape != c.shape or not np.allclose(reference["times"], candidate["times"], atol=1e-12):
        raise ValueError("Compare states at identical physical sample times")
    if not np.allclose(reference["mass"], candidate["mass"], rtol=1e-6, atol=1e-7):
        raise ValueError("Mass changed across fidelity/backends")
    if not np.allclose(reference["inertia"], candidate["inertia"], rtol=1e-5, atol=1e-7):
        raise ValueError("Inertia changed across fidelity/backends")
    delta = c-r
    return {"rms_position_m": float(np.sqrt(np.mean(np.sum(delta[:, :, :2]**2, axis=2)))),
            "rms_velocity_m_s": float(np.sqrt(np.mean(np.sum(delta[:, :, 3:5]**2, axis=2)))),
            "rms_spin_rad_s": float(np.sqrt(np.mean(delta[:, :, 5]**2))),
            "max_position_m": float(np.max(np.linalg.norm(delta[:, :, :2], axis=2))),
            "final_velocity_max_m_s": float(np.max(np.linalg.norm(delta[-1, :, 3:5], axis=1))),
            "final_spin_max_rad_s": float(np.max(np.abs(delta[-1, :, 5])))}


def normalized_error(error, budget=BUDGET):
    return max(error[key]/value for key, value in budget.items())


def diagnostics(scene, result):
    states = np.asarray(result["states"]); mass = np.asarray(result["mass"]); inertia = np.asarray(result["inertia"])
    momentum = np.sum(mass[None, :, None] * states[:, :, 3:5], axis=1)
    angular = np.sum(mass[None, :] * (states[:, :, 0]*states[:, :, 4]-states[:, :, 1]*states[:, :, 3])
                     + inertia[None, :] * states[:, :, 5], axis=1)
    kinetic = .5*np.sum(mass[None, :]*np.sum(states[:, :, 3:5]**2, axis=2)+inertia[None, :]*states[:, :, 5]**2, axis=1)
    potential = -np.sum(mass[None, :] * np.einsum("tbi,i->tb", states[:, :, :2], scene["gravity"]), axis=1)
    mechanical = kinetic+potential
    return {"linear_momentum_change_max_kg_m_s": float(np.max(np.linalg.norm(momentum-momentum[0], axis=1))),
            "angular_momentum_change_max_kg_m2_s": float(np.max(np.abs(angular-angular[0]))),
            "momentum_is_internal_invariant": not any(b.get("type", "dynamic")!="dynamic" for b in scene["bodies"]) and not any(scene["gravity"]),
            "mechanical_energy_increase_above_initial_J": float(max(0, np.max(mechanical)-mechanical[0])),
            "kinetic_energy_final_J": float(kinetic[-1]),
            "late_com_height_mean_m": float(np.mean(np.sum(mass[None, :]*states[-min(240,len(states)):, :, 1], axis=1)/np.sum(mass))),
            "late_kinetic_mean_J": float(np.mean(kinetic[-min(240,len(states)):])),
            "reported_max_penetration_fraction": result["reported_max_penetration_fraction"]}


def analytic_check(scene, result):
    s = np.asarray(result["states"]); t = np.asarray(result["times"])
    id = scene["id"]
    if id == "free_flight_hexagon":
        expected = s[0, 0, :2]+t[:, None]*s[0, 0, 3:5]
        err = float(np.max(np.linalg.norm(s[:, 0, :2]-expected, axis=1)))
        return {"observable":"free_position_max_m", "error":err,"tolerance":.001,"passed":err<=.001}
    if id == "normal_rebound_boxes":
        ve = float(np.max(np.abs(s[-1, :, 3]-[-1.2, 1.2])))
        we = float(np.max(np.abs(s[-1, :, 5])))
        return {"observable":"symmetric_restitution", "velocity_error_m_s":ve,"spin_error_rad_s":we,
                "tolerances":{"velocity_m_s":.02,"spin_rad_s":.02},"passed":ve<=.02 and we<=.02}
    if id == "sliding_box":
        distance = 9/(2*.3*9.81)
        error = float(abs(s[-1, 0, 0]-s[0, 0, 0]-distance))
        return {"observable":"Coulomb_stopping_distance_m","error":error,"tolerance":.025,"passed":error<=.025}
    if id in ("incline_stick", "incline_slide"):
        law=scene["analytic"]; theta=law["theta"]; mu=law["friction"]
        tangent=np.array([np.cos(theta),-np.sin(theta)])
        measured=s[-1, 0, 3:5]@tangent
        expected=max(0,9.81*(np.sin(theta)-mu*np.cos(theta)))*t[-1]
        error=float(abs(measured-expected))
        return {"observable":"incline_final_speed_m_s","error":error,"tolerance":.025,"passed":error<=.025}
    if id == "thin_wall_ccd":
        return {"observable":"body_does_not_cross_wall","final_x_m":float(s[-1,0,0]),"passed":bool(np.max(s[:,0,0])<0)}
    return None


def execute(scene, backend, primary, solver, repeats, policy=None):
    # Warmup is excluded. Report repeated engine+controller timing, not Python/JSON startup.
    run(scene, backend=backend, primary_steps=primary, substeps=solver, policy=policy)
    results=[run(scene, backend=backend, primary_steps=primary, substeps=solver, policy=policy) for _ in range(repeats)]
    times=[r["engine_and_controller_s"] for r in results]
    result=results[0]
    result["timing"]={"samples_s":times,"median_s":statistics.median(times),
                      "min_s":min(times),"max_s":max(times),"repeats":repeats,
                      "scope":"compiled solver plus actual adaptive feature/decision work; excludes shared diagnostic/output extraction"}
    return result


def save_trace(directory, name, result):
    path=directory/(name+".npz"); directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,states=result["states"],times=result["times"],mass=result["mass"],
                        inertia=result["inertia"],selected_levels=result["selected_levels"])
    return {"path":str(path.name),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def study(output, repeats):
    output.mkdir(parents=True,exist_ok=True); cases=scenes()
    (output/"scenes.json").write_text(json.dumps(cases,indent=2)+"\n")
    references={}; qualifications={}; reference_records=[]; traces=[]
    # Independent axes, fixed physical sample rate and shared mass/geometry.
    for scene in cases:
        for backend in ("block","temporal"):
            runs={}
            for primary,solver in ((4,32),(8,32),(16,32),(16,8),(16,16)):
                runs[primary,solver]=run(scene,backend=backend,primary_steps=primary,substeps=solver)
            finest=runs[16,32]; references[scene["id"],backend]=finest
            primary_changes=[errors(runs[b,32],runs[a,32]) for a,b in ((4,8),(8,16))]
            solver_changes=[errors(runs[16,b],runs[16,a]) for a,b in ((8,16),(16,32))]
            qualified=all(normalized_error(e,REFERENCE_BUDGET)<=1 for e in primary_changes+solver_changes)
            qualifications[scene["id"],backend]=qualified
            record={"case_id":scene["id"],"backend":backend,"split":scene["split"],
                    "successive_refinement_checks_passed":qualified,"primary_changes":primary_changes,
                    "solver_changes":solver_changes,"analytic":analytic_check(scene,finest),
                    "diagnostics":diagnostics(scene,finest),
                    "scope":"three levels per axis; numerical consistency check, not convergence proof or experimental validation"}
            reference_records.append(record)
            traces.append({"case_id":scene["id"],"mode":"reference_"+backend,**save_trace(output/"traces",scene["id"]+"_reference_"+backend,finest)})
            print("reference",scene["id"],backend,"qualified",qualified,flush=True)
    (output/"reference-checks.json").write_text(json.dumps(reference_records,indent=2)+"\n")
    # Exclude unqualified training references; never use held-out cases to select thresholds.
    training=[s for s in cases if s["split"]=="train" and qualifications[s["id"],"block"]]
    policies=[]
    for travel,island,high in itertools.product((.05,.15),(4,8),((4,16),(8,32))):
        p={**DEFAULT_POLICY,"travel_threshold":travel,"island_threshold":island,
           "high_primary_steps":high[0],"high_substeps":high[1]}
        policies.append(p)
    calibration=[]
    for index,policy in enumerate(policies):
        scores=[]; elapsed=0
        for scene in training:
            r=execute(scene,"block",1,1,min(repeats,3),policy)
            scores.append(normalized_error(errors(references[scene["id"],"block"],r)))
            elapsed+=r["timing"]["median_s"]
        worst=max(scores,default=float("inf"))
        calibration.append({"index":index,"policy":policy,"worst_normalized_training_error":worst,
                            "summed_training_median_s":elapsed,"within_training_budget":bool(scores) and worst<=1})
        print("policy",index,"training_error",round(worst,4),"time_s",round(elapsed,5),flush=True)
    feasible=[c for c in calibration if c["within_training_budget"]]
    if not calibration or not training: raise RuntimeError("No qualified training cases; do not invent a calibrated policy")
    selected=min(feasible,key=lambda c:c["summed_training_median_s"]) if feasible else min(calibration,key=lambda c:c["worst_normalized_training_error"])
    frozen=selected["policy"]
    calibration_report={"training_cases":[s["id"] for s in training],
                        "excluded_unqualified_training_cases":[s["id"] for s in cases if s["split"]=="train" and s not in training],
                        "budget":BUDGET,"candidates":calibration,"selected_index":selected["index"],
                        "calibration_feasible":bool(feasible),"policy":frozen,
                        "selection_uses_held_out_results":False}
    (output/"frozen-policy.json").write_text(json.dumps(calibration_report,indent=2)+"\n")
    rows=[]; winners=[]; run_records=[]
    for scene in cases:
        reference=references[scene["id"],"block"]; by_mode={}
        for mode,(backend,primary,solver) in {**FIXED,"block_adaptive":("block",1,1)}.items():
            r=execute(scene,backend,primary,solver,repeats,frozen if mode=="block_adaptive" else None)
            run_records.append({"case_id":scene["id"],"mode":mode,"timing":r["timing"],
                                "analytic":analytic_check(scene,r),"diagnostics":diagnostics(scene,r),
                                "physical_setup_id":r["physical_setup_id"],"numerical_model":r["numerical_model"]})
            error=errors(reference,r); normalized=normalized_error(error)
            row={"case_id":scene["id"],"split":scene["split"],"mode":mode,
                 "block_reference_qualified":qualifications[scene["id"],"block"],
                 "normalized_error":normalized,"within_budget":normalized<=1,
                 "median_engine_controller_s":r["timing"]["median_s"],
                 "min_s":r["timing"]["min_s"],"max_s":r["timing"]["max_s"],
                 "solver_work_total":r["solver_work_total"],"controller_s":r["controller_s"],
                 "switches":r["switches"],**error,
                 "reported_max_penetration_fraction":r["reported_max_penetration_fraction"]}
            rows.append(row); by_mode[mode]=row
            traces.append({"case_id":scene["id"],"mode":mode,**save_trace(output/"traces",scene["id"]+"_"+mode,r)})
        fixed_pass=[r for m,r in by_mode.items() if m.startswith("block_") and m!="block_adaptive" and r["within_budget"]]
        best=min(fixed_pass,key=lambda r:r["median_engine_controller_s"]) if fixed_pass else None
        adaptive=by_mode["block_adaptive"]
        winners.append({"case_id":scene["id"],"split":scene["split"],"reference_qualified":qualifications[scene["id"],"block"],
                        "best_passing_fixed":best["mode"] if best else None,
                        "adaptive_within_budget":adaptive["within_budget"],
                        "speedup_over_best_passing_fixed":best["median_engine_controller_s"]/adaptive["median_engine_controller_s"] if best else None,
                        "adaptive_favorable_at_matched_error":bool(best) and adaptive["within_budget"] and qualifications[scene["id"],"block"]
                            and adaptive["median_engine_controller_s"]<best["median_engine_controller_s"]})
        print("evaluate",scene["id"],"adaptive_error",round(adaptive["normalized_error"],3),
              "best_fixed",best["mode"] if best else None,flush=True)
    with (output/"comparisons.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/"run-records.json").write_text(json.dumps(run_records,indent=2)+"\n")
    held=[w for w in winners if w["split"]=="test" and w["reference_qualified"]]
    summary={"evidence_kind":"numerical_simulation","material_authenticity":"no new experimental characterization",
             "backend_commits":BOX2D_COMMITS,"environment":{"python":platform.python_version(),"platform":platform.platform()},
             "repeats":repeats,"budget":BUDGET,"reference_budget":REFERENCE_BUDGET,
             "cases":len(cases),"qualified_block_references":sum(qualifications[s["id"],"block"] for s in cases),
             "held_out_qualified_cases":len(held),"held_out_adaptive_failures":sum(not w["adaptive_within_budget"] for w in held),
             "held_out_matched_error_adaptive_wins":sum(w["adaptive_favorable_at_matched_error"] for w in held),
             "calibration_feasible":bool(feasible),"case_decisions":winners,
             "interpretation":"RMS budgets only; an own-backend refined trajectory is not measured truth. Cross-backend errors include formulation differences. False-safe choices and unqualified references prevent universal guarantees."}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (output/"trace-manifest.json").write_text(json.dumps(traces,indent=2)+"\n")
    qualified=[r for r in rows if r["block_reference_qualified"] and r["split"]=="test"]
    fig,ax=plt.subplots(figsize=(9,5))
    for mode in (*FIXED,"block_adaptive"):
        group=[r for r in qualified if r["mode"]==mode]
        ax.scatter([r["median_engine_controller_s"] for r in group],[r["normalized_error"] for r in group],label=mode,s=35)
    ax.axhline(1,color="black",linestyle="--",label="declared RMS error limit")
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("Median solver + controller time per scene [s]")
    ax.set_ylabel("Largest RMS error / its unitful tolerance")
    ax.set_title("Held-out cases with qualified block references; repeated timings")
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/"accuracy-cost.png",dpi=170);plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!="case_decisions"},indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("research/rigid-benchmarks/results"))
    parser.add_argument("--repeats",type=int,default=5)
    args=parser.parse_args()
    if args.repeats<1:parser.error("Positive repeat count required")
    study(args.output,args.repeats)


if __name__=="__main__":main()
