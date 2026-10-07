"""Execute the committed, frozen prospective captured-system cost protocol."""
import argparse,hashlib,json,os,platform,random,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from research.audit_large_contact_completion import check


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main(phase,declared_plan_commit):
    plan_path=OUT/'plan.json';plan=json.loads(plan_path.read_text());build=json.loads((OUT/'build-provenance.json').read_text())
    plan_commit=subprocess.check_output(['git','rev-parse',declared_plan_commit],cwd=ROOT,text=True).strip()
    assert subprocess.check_output(['git','show',plan_commit+':research/qr-minnorm-review/plan.json'],cwd=ROOT)==plan_path.read_bytes()
    assert build['plan_sha256']==sha(plan_path.read_bytes())
    results=OUT/'results';results.mkdir(exist_ok=True)
    def guard():
        for case in plan['cases']:assert sha((ROOT/case['path']).read_bytes())==case['sha256']
        for variant in ['baseline','qr']:
            directory=OUT/variant;meta=build['variants'][variant];assert sha((directory/'replay').read_bytes())==meta['binary_sha256']
            for name,digest in meta['files'].items():assert sha((directory/name).read_bytes())==digest
    controls={key:'1' for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}
    environment=dict(os.environ,**controls);rng=random.Random(730142);records=[]
    provenance=dict(plan_commit=plan_commit,plan_sha256=sha(plan_path.read_bytes()),build_provenance_sha256=sha((OUT/'build-provenance.json').read_bytes()),phase=phase,
        runner_sha256=sha(Path(__file__).read_bytes()),thread_environment=controls,affinity_cpu=plan['controls']['affinity_cpu'],platform=platform.platform(),
        timing_scope='Descriptive paired native solver costs under observed shared collaborative workload; CPU affinity does not establish full machine isolation.')
    (results/(phase+'-provenance.json')).write_text(json.dumps(provenance,indent=2)+'\n')
    def execute(case,variant,kind,repetition,order):
        guard();load_before=os.getloadavg();started=time.perf_counter();binary=OUT/variant/'replay'
        p=subprocess.run(['taskset','-c',str(plan['controls']['affinity_cpu']),str(binary),str(ROOT/case['path'])],cwd=ROOT,env=environment,capture_output=True,text=True)
        elapsed=time.perf_counter()-started;receipt=json.loads(p.stdout) if p.stdout else dict(error=p.stderr)
        independent=check(ROOT/case['path'],receipt['p']) if 'p' in receipt else None
        if receipt.get('accepted'):assert independent['accepted']
        record=dict(capture=case['path'],capture_sha256=case['sha256'],variant=variant,kind=kind,repetition=repetition,within_pair_order=order,
            returncode=p.returncode,process_elapsed_s=elapsed,load_before=load_before,load_after=os.getloadavg(),receipt=receipt,independent_gate=independent)
        records.append(record)
        (results/(phase+'-receipts.json')).write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        print(kind,repetition,Path(case['path']).parent.parent.name,Path(case['path']).stem,variant,receipt.get('accepted'),receipt.get('solve_time_s'),flush=True)
    if phase=='validation':
        for case in plan['cases']:
            for order,variant in enumerate(['baseline','qr']):execute(case,variant,'untimed-validation',0,order)
    else:
        validation=json.loads((results/'validation-receipts.json').read_text());assert len(validation)==2*len(plan['cases'])
        validation_summary=json.loads((results/'validation-summary.json').read_text());assert validation_summary['functional_regressions']==0
        for case in plan['cases']:
            variants=['baseline','qr'];rng.shuffle(variants)
            for order,variant in enumerate(variants):execute(case,variant,'untimed-warmup',0,order)
        for repetition in range(plan['timing']['timed_pairs_per_capture']):
            cases=plan['cases'].copy();rng.shuffle(cases)
            for case in cases:
                variants=['baseline','qr'];rng.shuffle(variants)
                for order,variant in enumerate(variants):execute(case,variant,'timed',repetition,order)
    guard();summary=[]
    for case in plan['cases']:
        rows=[r for r in records if r['capture']==case['path'] and r['kind']!='untimed-warmup'];item=dict(capture=case['path'],rows=case['rows'],variants={})
        for variant in ['baseline','qr']:
            subset=[r for r in rows if r['variant']==variant];native=[r['receipt']['solve_time_s'] for r in subset if 'solve_time_s' in r['receipt']]
            item['variants'][variant]=dict(attempts=len(subset),accepted=sum(bool(r['receipt'].get('accepted') and r['independent_gate']['accepted']) for r in subset),
                median_s=float(np.median(native)) if phase=='timing' and native else None,range_s=[min(native),max(native)] if phase=='timing' and native else None,
                maximum_residual_m_s=max((r['independent_gate']['independent_full_original_residual_m_s'] for r in subset if r['independent_gate']),default=None))
        left=item['variants']['baseline'];right=item['variants']['qr'];item['functional_regression']=left['accepted']>0 and right['accepted']<left['accepted']
        if phase=='timing' and left['accepted']==right['accepted']==plan['timing']['timed_pairs_per_capture']:
            paired=[]
            for repetition in range(plan['timing']['timed_pairs_per_capture']):
                pair={r['variant']:r['receipt']['solve_time_s'] for r in rows if r['repetition']==repetition}
                paired.append(pair['baseline']/pair['qr'])
            item['paired_solve_cost_ratios_baseline_over_qr']=paired
            item['median_paired_solve_cost_ratio_baseline_over_qr']=float(np.median(paired))
            item['secondary_ratio_of_median_costs']=left['median_s']/right['median_s']
        summary.append(item)
    output=dict(**provenance,complete=True,attempt_count=len(records),cases=summary,accepted=sum(bool(r['receipt'].get('accepted')) for r in records),functional_regressions=sum(r['functional_regression'] for r in summary))
    (results/(phase+'-summary.json')).write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print('DONE',phase,len(records),'receipts',output['functional_regressions'],'functional regressions',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validation','timing'],required=True);p.add_argument('--plan-commit',required=True);args=p.parse_args();main(args.phase,args.plan_commit)
