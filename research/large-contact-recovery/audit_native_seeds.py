"""Audit all native seeded trials and verify accepted roots via production replay."""
import ast,hashlib,json,subprocess
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'native-seed-independent2';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent','exec'),ns);report=[]
for folder in ['results-native-seeds','results-right-cone']:
 h=H/folder;s=json.loads((h/'summary.json').read_text());assert s['complete']
 for i,r in enumerate(s['records']):
  d=json.loads((ROOT/r['input']).read_text());index=0 if '3d_hull_64/reference_0' in r['input'] else 1 if '3d_hull_64/reference_3' in r['input'] else 2;prefix=h/f'{index}-{r["seed"]}-{r["budget_per_component"]}';o=json.loads(prefix.with_suffix('.stdout.json').read_text());gate=ns['external'](d,o);assert r['accepted']==bool(gate['accepted'] and o['accepted'] and r['native_exit']==0)
  if not r['accepted']:continue
  p=D/f'{folder}-{i}.candidate.json';p.write_text(json.dumps(dict(d,p=o['p']),indent=2)+'\n');native=subprocess.run([str(ROOT/'build/spatial/spatial_coulomb_replay'),str(p),'0'],capture_output=True,text=True);out=D/f'{folder}-{i}.native.json';out.write_text(native.stdout);production_zero=bool(native.returncode==0 and json.loads(native.stdout)['accepted']);report.append({'variant':folder,'input':r['input'],'seed':r['seed'],'budget_per_component':r['budget_per_component'],'original_input_sha256':sha(ROOT/r['input']),'candidate_sha256':sha(p),'independent':gate,'production_budget_zero_accepted':production_zero,'production_residual_m_s':json.loads(native.stdout)['stats']['residual_m_s']})
(H/'native-seed-independent-audit.json').write_text(json.dumps({'passed':True,'records':report,'production_replay_sha256':sha(ROOT/'build/spatial/spatial_coulomb_replay'),'scope':'Original full-law instantaneous roots verified, not full trajectory or performance qualification.'},indent=2)+'\n');print('All seeded trials audited; production zero-budget accepted',sum(x['production_budget_zero_accepted'] for x in report),'of',len(report),'independently accepted candidates.')
