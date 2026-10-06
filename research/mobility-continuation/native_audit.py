"""Independent actual bounded native candidate and source/runtime records."""
import ast,hashlib,json,subprocess
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=H/'native-results';load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();provenance=load(D/'provenance.json');summary=load(D/'summary.json');plan=load(H/'plan.json')
for path,digest in dict(provenance['guards'],**provenance['runtime']).items():assert sha(path)==digest
assert sha(provenance['binary_path'])==provenance['binary_sha256'];fn=next(n for n in ast.parse((ROOT/'research/new-combined-contact-review/run-20261005T194211Z/run23.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='external');ns={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'independent','exec'),ns);records=[]
for i,(path,digest) in enumerate(plan['inputs'].items()):
 assert sha(ROOT/path)==digest;data=load(ROOT/path);out=load(D/f'{i}.stdout.json');stored=summary['records'][i];gate=ns['external'](data,out);assert gate==stored['independent'];assert stored['accepted']==bool(gate['accepted'] and out['accepted'] and stored['exit']==0);policy=stored['policy'];assert policy['svds']<=policy['stage_attempts']*2048 and policy['iterations']<=policy['stage_attempts']*2048;assert policy['component_cap']==192 and policy['stage_cap']==17;records.append({'input':path,'original_gates_passed':stored['accepted'],'original_residual_m_s':gate['projection_m_s'],'max_abs_impulse':float(np.max(np.abs(out['p']))),'reported_budget_counts_verified':True,'scope':'Instantaneous original-matrix acceptance only; no production/world/performance qualification.'})
(H/'native-independent-audit.json').write_text(json.dumps({'passed':True,'source_binary_runtime_verified':True,'records':records},indent=2,allow_nan=False)+'\n');print('Bounded native mobility final gates/provenance audit PASS')
