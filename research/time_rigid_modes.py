"""Interleaved follow-up timings; frozen policy and original references unchanged."""
import json
from pathlib import Path
import random
import statistics
from rigid_engine import run
from research.rigid_scenes import scenes
from research.run_rigid_study import FIXED


def main():
    root=Path('research/rigid-benchmarks/results')
    summary=json.loads((root/'summary.json').read_text())
    policy=json.loads((root/'frozen-policy.json').read_text())['policy']
    ids={c['case_id'] for c in summary['case_decisions'] if c['split']=='test' and c['reference_qualified']}
    rng=random.Random(20261004); modes={**FIXED,'block_adaptive':('block',1,1)}; records=[]
    for scene in scenes():
        if scene['id'] not in ids:continue
        samples={m:[] for m in modes};order=[]
        for m,(backend,p,s) in modes.items():run(scene,backend=backend,primary_steps=p,substeps=s,policy=policy if m=='block_adaptive' else None)
        for repeat in range(20):
            sequence=list(modes);rng.shuffle(sequence);order.append(sequence)
            for m in sequence:
                backend,p,s=modes[m]
                r=run(scene,backend=backend,primary_steps=p,substeps=s,policy=policy if m=='block_adaptive' else None)
                samples[m].append({'solver_controller_s':r['engine_and_controller_s'],'controller_s':r['controller_s'],
                                   'end_to_end_s':r['wall_time_s']})
        best_name=next(c['best_passing_fixed'] for c in summary['case_decisions'] if c['case_id']==scene['id'])
        medians={m:statistics.median(r['solver_controller_s'] for r in v) for m,v in samples.items()}
        records.append({'case_id':scene['id'],'round_order':order,'samples':samples,'median_solver_controller_s':medians,
                        'original_cheapest_passing_fixed':best_name,
                        'adaptive_over_original_best_fixed':medians['block_adaptive']/medians[best_name],
                        'physics_settings_and_policy_unchanged':True})
        print(scene['id'],'adaptive/fixed',round(medians['block_adaptive']/medians[best_name],3),flush=True)
        (root/'interleaved-timings.json').write_text(json.dumps({'seed':20261004,'repeats':20,'cases':records,
            'scope':'Follow-up after initial results. Randomized mode order within each round on one machine; no policy retuning or new accuracy claims. End-to-end includes shared output/diagnostic overhead.'},indent=2)+'\n')

if __name__=='__main__':main()
