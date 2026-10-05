"""Recompute original full laws for every retained face-search candidate."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original-law-audit','exec'),ns)
report=[]
for folder,planname in [('results-sliding','sliding-plan.json'),('results-sliding-many','sliding-many-plan.json'),('results-reduced-faces','reduced-face-plan.json'),('results-reduced-supports','reduced-support-plan.json')]:
 D=H/folder
 if not (D/'summary.json').exists():continue
 plan=json.loads((H/planname).read_text());path,sha=next(iter(plan['inputs'].items()));assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha;d=json.loads((ROOT/path).read_text());A=np.array(d['A']);b=np.array(d['b']);dep=np.array(d['dependencies']);hi=np.array(d['hi']);s=json.loads((D/'summary.json').read_text());assert s['complete']
 for a in s['attempts']:
  ids=a.get('component_rows',plan.get('component_rows'));inv={k:i for i,k in enumerate(ids)};M=A[np.ix_(ids,ids)];rhs=b[ids];q=np.array(a['candidate_p']);local=dict(A=M.tolist(),b=rhs.tolist(),dependencies=[inv[k] if k>=0 else -1 for k in dep[ids]],hi=hi[ids].tolist(),tolerance_m_s=d['tolerance_m_s']);gate=ns['external'](local,dict(p=q.tolist(),w=(M@q-rhs).tolist()))
  for k,v in gate.items():
   old=a['original_gate'][k]
   if isinstance(v,bool):assert v==old
   else:assert np.isclose(v,old,rtol=1e-10,atol=1e-12),(folder,k,v,old)
 candidate=json.loads((D/'candidate.json').read_text());assert candidate['A']==d['A'] and candidate['b']==d['b'] and candidate['dependencies']==d['dependencies'] and candidate['hi']==d['hi'] and candidate['tolerance_m_s']==d['tolerance_m_s'];q=np.array(candidate['p']);gate=ns['external'](d,dict(p=q.tolist(),w=(A@q-b).tolist()));assert s['accepted']==bool(gate['accepted'] and s['native_exit']==0)
 native=json.loads((D/'native.stdout.json').read_text());assert native['accepted']==(s['native_exit']==0)
 report.append({'directory':folder,'attempts':len(s['attempts']),'accepted':s['accepted'],'native_exit':s['native_exit'],'all_original_trial_laws_recomputed':True})
(H/'face-independent-audit.json').write_text(json.dumps({'passed':True,'records':report,'scope':'Candidate archive verification; no full-world or performance qualification.'},indent=2)+'\n');print(json.dumps(report))
