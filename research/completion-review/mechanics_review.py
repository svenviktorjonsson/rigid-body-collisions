"""Independent physical-structure and event-regression evidence, not ranking."""
from dataclasses import asdict
import hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parent
BASE='2f5ffc38dd69db72442d6c9972703092bfa91d3b'


def structure():
    records=[]
    for path in sorted(Path('research/shared-hull-rank-followup/results/rejections').glob('*/*.json')):
        data=json.loads(path.read_text());A=np.array(data['A']);b=np.array(data['b']);p=np.array(data['p'])
        normals=np.flatnonzero(np.array(data['dependencies'])<0);active=normals[p[normals]>1e-9]
        eigen=np.linalg.eigvalsh(A);An=A[np.ix_(normals,normals)]
        feasible=linprog(np.zeros(len(normals)),A_ub=-An,b_ub=-b[normals],bounds=[(0,None)]*len(normals),method='highs')
        singular=np.linalg.svd(A[active,:],compute_uv=False)
        rank=int(np.sum(singular>singular[0]*1e-12)) if len(singular) else 0
        record=dict(capture=str(path),capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(b),
                    symmetry_error=float(np.max(abs(A-A.T))),minimum_eigenvalue=float(eigen[0]),maximum_eigenvalue=float(eigen[-1]),
                    normal_only_inequality_feasible=bool(feasible.success),active_normal_rows=active.tolist(),active_normal_target_rank=rank)
        assert record['symmetry_error']==0 and eigen[0]>=-1e-12*max(1.,eigen[-1])
        assert feasible.success and rank==len(active)
        records.append(record)
    return dict(cases=records,interpretation='No mobility asymmetry, non-PSD physics, or inconsistent warm active-normal equalities found. Normal-only LP is not a certificate of a complete frictional solution. Mode-search failure is not an infeasibility certificate.')


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def events():
    source=subprocess.check_output(['git','show',BASE+':research/elastic_patch.py'])
    frozen=ROOT/'elastic_patch_before.py';frozen.write_bytes(source);old=load_module('elastic_before_review',frozen)
    current=Path('research/elastic_patch.py').read_bytes();new=load_module('elastic_after_review','research/elastic_patch.py')
    setups=[('rest',(0,0,.1),(0,0,0),False),('grazing',(0,0,.1),(1,0,0),False),
            ('side_with_floor_grazing',(.1,0,.1),(-1,0,0),True),('corner_inward',(.1,0,.1),(-1,0,-1),True)]
    records=[]
    for version,module in [('before',old),('after',new)]:
        for exponent in (0,2):
            for name,x,v,side in setups:
                material=module.Material(compression_exponent=exponent,tangent_stiffness=0,twist_stiffness=0)
                planes=(module.Plane(),module.Plane((1,0,0),name='side')) if side else (module.Plane(),)
                row=dict(version=version,name=name,material=asdict(material),position=x,velocity=v,duration_s=.002,rhs_budget=3000)
                try:
                    result=module.simulate(material,position=x,velocity=v,planes=planes,duration=.002,max_rhs_evaluations=3000)
                    row.update(accepted=True,rhs_evaluations=result['rhs_evaluations'],contact_events=len(result['events']),
                               energy_residual_max_J=float(max(abs(result['energy_residual_J']))),final_state=result['states'][-1].tolist())
                except RuntimeError as e:row.update(accepted=False,error=str(e))
                if version=='after':assert row['accepted'] and row['energy_residual_max_J']<1e-11
                if version=='before' and name!='corner_inward':assert not row['accepted']
                records.append(row)
    assert current==Path('research/elastic_patch.py').read_bytes(),'Elastic source changed during evidence collection'
    return dict(before_commit=BASE,before_source_sha256=hashlib.sha256(source).hexdigest(),
                after_source_sha256=hashlib.sha256(current).hexdigest(),cases=records)


if __name__=='__main__':
    result=dict(schema='independent-completion-review-v1',structure=structure(),elastic_events=events())
    (ROOT/'independent-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Independent review PASS: six mobility structures and sixteen frozen/current elastic event cases')
