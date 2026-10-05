"""Independent physical and paired-start audit for FIRST256 diagnostics."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.coulomb_diagnostics import System
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 p=json.loads((BASE/'plan.json').read_text())
 for f,h in p['source_hashes'].items():assert sha(ROOT/f)==h,('Frozen source changed',f)
 for f,h in p['corpus'].items():assert sha(ROOT/f)==h,('Capture changed',f)
 return p
def audit():
 plan=guard();rows=[json.loads(x)for x in(BASE/'paired-native.jsonl').read_text().splitlines()]
 expected=[(f,r,s)for f in plan['corpus']for r in range(3)for s in(['baseline','early']if r%2==0 else['early','baseline'])]
 assert [(r['capture'],r['repeat'],r['strategy'])for r in rows]==expected
 checks=[];pairs={}
 for r in rows:
  d=json.loads((ROOT/r['capture']).read_text());sys=System.from_dump(d);p=np.array(r['p']);g=sys.gate(p,d['tolerance_m_s'])
  assert r['accepted']and g['accepted'],('Original physical gate fails',r['capture'],r['strategy'],g)
  assert r['initial_p']==d['p'],'Pair start differs from exact captured p'
  assert np.allclose(sys.A@p-sys.b,r['w'],rtol=1e-11,atol=1e-10),'Archived velocity inconsistent'
  assert r['svd_calls']<=1024 and r['helper_calls']<=8 and r['passes']<=8
  if r['largest_reduced_rows']>64:
   assert r['strategy']=='early' and r['lane'].startswith('early_decline_')
   assert r['component_cap_rejections']>0 and r['helper_calls']==0 and r['svd_calls']==0, 'Over-cap component must decline before search'
   assert any(n>64 for n in r['component_sizes']), 'Missing structural rejected component'
  assert r['pressure_svd_calls']<=1024 and r['pivot_attempts']<=8
  t=r['early_fallback_tail_counters'];assert t['svd_calls']<=1024 and t['helper_calls']<=8 and t['passes']<=8 and t['pressure_svd_calls']<=1024 and t['pivot_attempts']<=8
  assert r['first256_counters']['iteration_sweeps_total']<=256
  assert np.isfinite(r['solve_seconds'])and r['solve_seconds']>=0
  if r['first256_rejected_p']:
   assert r['strategy']=='early' and len(r['first256_rejected_p'])==len(d['p'])
   seedg=sys.gate(np.array(r['first256_rejected_p']),d['tolerance_m_s'])
   assert not seedg['accepted'],'Alleged rejected FIRST256 seed physically passes'
  pairs.setdefault(r['capture'],{}).setdefault(r['repeat'],{})[r['strategy']]=r['solve_seconds']
  checks.append({'capture':r['capture'],'repeat':r['repeat'],'strategy':r['strategy'],'gate':g})
 summary=[]
 for f,repeat in pairs.items():
  bt=[v['baseline']for v in repeat.values()];et=[v['early']for v in repeat.values()]
  summary.append({'capture':f,'baseline_seconds':bt,'early_seconds':et,'median_baseline_seconds':float(np.median(bt)),'median_early_seconds':float(np.median(et)),'median_cost_ratio_baseline_over_early':float(np.median(bt)/np.median(et))})
 return {'count':len(rows),'all_accepted':True,'checks':checks,'captured_cost_only':True,'paired_costs':summary,'sum_median_baseline_seconds':sum(x['median_baseline_seconds']for x in summary),'sum_median_early_seconds':sum(x['median_early_seconds']for x in summary)}
if __name__=='__main__':
 result=audit();(BASE/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in('checks','paired_costs')},indent=2))
