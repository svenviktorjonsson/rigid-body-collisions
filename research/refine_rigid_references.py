"""Follow-up refinement of failed references; never changes the frozen first study."""
import hashlib
import io
import zipfile
import json
from pathlib import Path
import numpy as np
from rigid_engine import run
from research.rigid_scenes import scenes
from research.run_rigid_study import REFERENCE_BUDGET, errors, normalized_error, diagnostics, analytic_check


def compare_preserved(root):
    checks=json.loads((root/'follow-up-refinement.json').read_text());records=[]
    with zipfile.ZipFile(root/'follow-up-traces.zip') as follow, zipfile.ZipFile(root/'traces.zip') as original:
        for check in checks:
            if not check['follow_up_qualified']:continue
            id=check['case_id']
            with np.load(io.BytesIO(follow.read(f'{id}_64_128.npz')),allow_pickle=False) as arrays:
                reference={k:arrays[k].copy() for k in arrays.files}
            for mode in ['block_fast','block_standard','block_accurate','block_high','block_adaptive','reference_block']:
                with np.load(io.BytesIO(original.read(f'{id}_{mode}.npz')),allow_pickle=False) as arrays:
                    candidate={k:arrays[k].copy() for k in arrays.files}
                for k in ('mass','inertia','times'):
                    if not np.allclose(reference[k],candidate[k],rtol=1e-5,atol=1e-7):
                        raise ValueError('Preserved physical setup changed')
                d=candidate['states']-reference['states']
                e={'rms_position_m':float(np.sqrt(np.mean(np.sum(d[:,:,:2]**2,axis=2)))),
                   'rms_velocity_m_s':float(np.sqrt(np.mean(np.sum(d[:,:,3:5]**2,axis=2)))),
                   'rms_spin_rad_s':float(np.sqrt(np.mean(d[:,:,5]**2)))}
                records.append({'case_id':id,'mode':mode,'reference':'exploratory_64_primary_128_velocity_iterations',
                    'errors':e,'normalized_error':normalized_error(e),'within_budget':normalized_error(e)<=1})
    return {'scope':'Exploratory comparison of preserved first-study histories against newly qualified higher-work references. No policy retuning; original held-out evaluation unchanged.','records':records}


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-only',action='store_true',help='Recompute comparisons from preserved archives without running solvers')
    args=parser.parse_args()
    root=Path('research/rigid-benchmarks/results')
    if args.compare_only:
        (root/'follow-up-comparisons.json').write_text(json.dumps(compare_preserved(root),indent=2)+'\n')
        return
    initial=json.loads((root/'reference-checks.json').read_text())
    failed={r['case_id'] for r in initial if r['backend']=='block' and not r['successive_refinement_checks_passed']}
    # Include the isolated rebound to distinguish restitution and impact timing.
    selected=failed|{'normal_rebound_boxes'}
    records=[]; trace_manifest=[]
    archive_path=root/"follow-up-traces.zip"
    archive=zipfile.ZipFile(archive_path,"w",compression=zipfile.ZIP_STORED)
    for scene in scenes():
        if scene['id'] not in selected: continue
        runs={}
        for primary,solver in ((16,128),(32,128),(64,128),(64,32),(64,64)):
            result=run(scene,backend='block',primary_steps=primary,substeps=solver)
            runs[primary,solver]=result
            buffer=io.BytesIO()
            np.savez_compressed(buffer,states=result["states"],times=result["times"],mass=result["mass"],inertia=result["inertia"])
            data=buffer.getvalue();name=f"{scene['id']}_{primary}_{solver}.npz"
            archive.writestr(name,data)
            trace_manifest.append({"case_id":scene["id"],"primary_steps":primary,"velocity_iterations":solver,
                                   "path":name,"sha256":hashlib.sha256(data).hexdigest()})
            print(scene['id'],primary,solver,round(result['engine_and_controller_s'],4),flush=True)
        p=[errors(runs[b,128],runs[a,128]) for a,b in ((16,32),(32,64))]
        s=[errors(runs[64,b],runs[64,a]) for a,b in ((32,64),(64,128))]
        records.append({'case_id':scene['id'],'primary_levels':[16,32,64], 'primary_axis_fixed_velocity_iterations':128,
                        'velocity_iteration_levels':[32,64,128], 'solver_axis_fixed_primary_steps':64,
                        'reference_budget':REFERENCE_BUDGET,'primary_changes':p,'solver_changes':s,
                        'follow_up_qualified':all(normalized_error(e,REFERENCE_BUDGET)<=1 for e in p+s),
                        'aggregate_observables':[{'primary_steps':key[0],'velocity_iterations':key[1],**diagnostics(scene,r)} for key,r in runs.items()],
                        'analytic_finest':analytic_check(scene,runs[64,128]),
                        'interpretation':'Exploratory follow-up after observing first-study failures; not used to retune or reclassify its held-out evaluation. Position iterations remain fixed at three.'})
        (root/'follow-up-refinement.json').write_text(json.dumps(records,indent=2)+'\n')
    archive.close()
    (root/"follow-up-trace-manifest.json").write_text(json.dumps({"archive":archive_path.name,
        "archive_sha256":hashlib.sha256(archive_path.read_bytes()).hexdigest(),"traces":trace_manifest},indent=2)+"\n")
    # Analytic behavior of first-study adaptive rebound is correct even when its sampled timing differs.
    scene=next(s for s in scenes() if s['id']=='normal_rebound_boxes')
    policy=json.loads((root/'frozen-policy.json').read_text())['policy']
    r=run(scene,policy=policy)
    (root/'adaptive-rebound-check.json').write_text(json.dumps({'case_id':scene['id'],'policy':policy,'analytic':analytic_check(scene,r),
        'diagnostics':diagnostics(scene,r), 'note':'Correct outgoing velocities and zero spin do not imply accurate impact timing or full-history agreement.'},indent=2)+'\n')

    (root/'follow-up-comparisons.json').write_text(json.dumps(compare_preserved(root),indent=2)+'\n')

if __name__=='__main__':main()
