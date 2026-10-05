"""Guarded unchanged position target: geometry/LP/native active-face controls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.optimize import linprog
H=Path(__file__).resolve().parent;ROOT=H.parents[1];plan=json.loads((H/'plan.json').read_text());D=H/'results';D.mkdir(exist_ok=False)
def save(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
assert os.environ['OMP_NUM_THREADS']=='1' and os.environ['OPENBLAS_NUM_THREADS']=='1'
paths=[ROOT/p for p in plan['inputs']];assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in zip(paths,plan['inputs'].values()))
d=json.loads(paths[0].read_text());g=json.loads(paths[1].read_text());A=np.array(d['A']);b=np.array(d['b']);bodies=[q for q in g['bodies'] if q['inverse_mass']>0];index={q['solver_body_id']:i for i,q in enumerate(bodies)};J=np.zeros((len(b),3*len(bodies)));inverse=np.repeat([q['inverse_mass'] for q in bodies],3)
for i,row in enumerate(g['rows']):
    for side in ['a','b']:
        if row['solver_body_id_'+side] in index:
            k=index[row['solver_body_id_'+side]];J[i,3*k:3*k+3]+=np.array(row['linear_jacobian_'+side])
error=float(np.max(np.abs(A-(J*inverse)@J.T)));assert error<1e-12
lp=linprog(np.zeros(J.shape[1]),A_ub=-J,b_ub=-b,bounds=[(None,None)]*J.shape[1],method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
certificate={'gram_max_difference':error,'primal_lp_success':lp.success,'lp_status':lp.status,'lp_message':lp.message}
if lp.success:certificate.update(primal_velocity=lp.x.tolist(),minimum_original_rate_slack_m_s=float(np.min(J@lp.x-b)))
save(D/'geometry-primal.json',certificate)
BUILD=Path('/home/viktor/.cache/physics-large-position-20261006');BUILD.mkdir(exist_ok=False);base=ROOT/'build/spatial/_deps/bullet-build/src';libs=[base/'BulletDynamics/libBulletDynamics.a',base/'BulletCollision/libBulletCollision.a',base/'LinearMath/libLinearMath.a'];records=[]
for cap in plan['native_variants']:
    exe=BUILD/f'normal_{cap}';cmd=['c++','-O3','-DNDEBUG','-DBT_USE_DOUBLE_PRECISION',f'-DPOSITION_MAX_ROWS={cap}','-I/home/viktor/.cache/physics-relation-20261005/deps/_deps/bullet-src/src','-I/home/viktor/.cache/physics-relation-20261005/deps/_deps/json-src/include',str(H/'prototype/replay.cpp'),'-o',str(exe),*[str(p) for p in libs]]
    compile_result=subprocess.run(cmd,capture_output=True,text=True);(D/f'{cap}-compile.stderr').write_text(compile_result.stderr);save(D/f'{cap}-compile.json',{'command':cmd,'exit':compile_result.returncode});compile_result.check_returncode()
    start=time.perf_counter();native=subprocess.run([str(exe),str(paths[0])],capture_output=True,text=True);elapsed=time.perf_counter()-start;(D/f'{cap}.stdout.json').write_text(native.stdout);(D/f'{cap}.stderr').write_text(native.stderr);output=json.loads(native.stdout);p=np.array(output['p']);w=A@p-b;res=float(np.max(np.abs(p-np.maximum(0,p-w/np.diag(A)))*np.diag(A)));energy=float(.5*p@(w-b));scale=1+np.sum(abs(p*b));accepted=np.isfinite(p).all() and (p>=0).all() and (p<=np.array(d['hi'])).all() and res<=d['tolerance_m_s'] and energy<=d['tolerance_m_s']*scale
    assert bool(output['accepted'])==bool(output['found'] and accepted);records.append({'cap':cap,'native_exit':native.returncode,'accepted':output['accepted'],'independent_residual_m_s':res,'elapsed_s_descriptive':elapsed,'stats':output['stats']});save(D/'progress.json',records);print(records[-1],flush=True)
save(D/'summary.json',{'complete':True,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'certificate':certificate,'native_records':records,'scope':'Unchanged instantaneous position-system evidence; no trajectory/geometry correction is applied.'})
