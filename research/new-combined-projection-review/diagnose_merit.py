"""Read-only analytic residual/Jacobian comparison, no optimizer calls."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1];sys.path.insert(0,str(ROOT))
from research.coulomb_diagnostics import System
plan=json.loads((P/'merit-diagnosis-plan.json').read_text())
for path,h in plan['inputs_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
capture=ROOT/'research/hull-combined-completion/results/rejections/fast_shake8_hulls42/reference_1.json';data=json.loads(capture.read_text());system=System.from_dump(data)
def describe(p):
 F,J=system.equations(p,True);fb=F.copy();JB=J.copy();w=system.A@p-system.b;contacts=[]
 for k,rows,mu,rn,rt in system.contacts:
  u=p[k]/rn;v=w[k];length=np.hypot(u,v);fb[k]=u+v-length;du=1-u/length if length else 1.;dv=1-v/length if length else 1.;JB[k]=dv*system.A[k];JB[k,k]+=du/rn
  contacts.append(dict(normal=k,pn=float(p[k]),normal_velocity=float(v),scaled_pressure=float(u),tangent_rows=list(rows),slip_velocity=w[list(rows)].tolist(),normal_projection=float(F[k]),normal_FB=float(fb[k]),tangent_residual_norm=float(np.linalg.norm(F[list(rows)]))))
 singular=np.linalg.svd(J,compute_uv=False);return dict(projection_merit=float(F@F),FB_merit=float(fb@fb),max_residual_difference=float(np.max(abs(F-fb))),max_J_difference=float(np.max(abs(J-JB))),projection_gradient_inf=float(np.max(abs(J.T@F))),FB_gradient_inf=float(np.max(abs(JB.T@fb))),projection_J_singular_values=singular.tolist(),projection_J_rank_relative1e12=int(np.sum(singular>singular[0]*1e-12)),contacts=contacts,full_original_gate=system.gate(p,data['tolerance_m_s']))
out={'captured':describe(np.array(data['p']))}
for name in ['projection-jac-scaled','FB-jac-scaled']:
 trial=json.loads((P/'results'/f'{name}.json').read_text());out[name]=describe(np.array(trial['native']['candidate_impulse']))
(P/'merit-diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
for name,row in out.items():print(name,{k:v for k,v in row.items()if k not in ['contacts','projection_J_singular_values','full_original_gate']})
