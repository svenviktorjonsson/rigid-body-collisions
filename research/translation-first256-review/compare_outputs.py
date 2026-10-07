"""Descriptive comparison added after prospective source freeze; no new gates."""
from pathlib import Path
import json,sys
import numpy as np
B=Path(__file__).resolve().parent;R=B.parents[1]
def compare():
 rows=[json.loads(x)for x in(B/'paired-native.jsonl').read_text().splitlines()]
 assert len(rows)==132
 groups={}
 for index,r in enumerate(rows):groups.setdefault((r['capture'],r['repeat']),{})[r['strategy']]=(index,r)
 pause=json.loads((B/'pause-event.json').read_text());contaminated=pause['completed_rows']
 results=[]
 for (f,repeat),g in groups.items():
  i,a=g['baseline'];j,b=g['early'];d=json.loads((R/f).read_text());A=np.array(d['A']);dp=np.array(b['p'])-a['p']
  results.append({'capture':f,'repeat':repeat,'max_contact_velocity_difference_m_s':float(np.max(np.abs(A@dp))),'impulse_difference_max':float(np.max(np.abs(dp))),'energy_change_difference_J':float(b['passive_change_bound_J']-a['passive_change_bound_J']),'baseline_counters':a['baseline_counters'],'early_first256_counters':b['first256_counters'],'early_fallback_baseline_counters':b['baseline_counters'],'baseline_v3_svd':a['svd_calls'],'early_v3_svd':b['svd_calls'],'early_fallback_tail_counters':b['early_fallback_tail_counters'],'pause_contaminated_pair':contaminated in(i,j)})
 return {'schema':'first256-descriptive-output-comparison-v1','timing_ranking_qualified':False,'reason':'One real publication pause contaminates the in-flight sample; concurrent production work also affects timing. All raw times remain; medians are descriptive. Captured starts are historical final rejects, not original world warmstarts.','counter_limit':'Selected native baseline fields are retained exactly. These are not a complete total-work or total-factorization count: nested normal/pivot/projector guide fields are not all exported. V3 budgets are retained separately.','all_132_outputs_retained':True,'pairs':results,'early_strategy_counts':{lane:sum(r['lane']==lane for r in rows if r['strategy']=='early')for lane in sorted({r['lane']for r in rows if r['strategy']=='early'})}}
if __name__=='__main__':
 result=compare();(B/'output-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='pairs'},indent=2))
