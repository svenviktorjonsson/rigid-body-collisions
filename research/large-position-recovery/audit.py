"""Independent absolute gates and reconstructed translational witnesses."""
import json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
plan=json.loads((H/'plan.json').read_text());paths=[ROOT/p for p in plan['inputs']];d=json.loads(paths[0].read_text());g=json.loads(paths[1].read_text());A=np.array(d['A']);b=np.array(d['b']);bodies=[q for q in g['bodies'] if q['inverse_mass']>0];index={q['solver_body_id']:i for i,q in enumerate(bodies)};inverse=np.repeat([q['inverse_mass'] for q in bodies],3);J=np.zeros((len(b),len(inverse)))
for i,row in enumerate(g['rows']):
    for side in ['a','b']:
        if row['solver_body_id_'+side] in index:
            k=index[row['solver_body_id_'+side]];J[i,3*k:3*k+3]+=np.array(row['linear_jacobian_'+side])
assert np.max(abs(A-(J*inverse)@J.T))<1e-12
records=[];lp_records=[]
for folder in ['results','results-pipeline']:
    lp=json.loads((H/folder/'geometry-primal.json').read_text());slack=float(np.min(J@np.array(lp['primal_velocity'])-b)) if lp.get('primal_lp_success') else None
    # HiGHS status is not a physical feasibility gate; verify actual original rates.
    lp_records.append({'folder':folder,'solver_reported_success':lp['primal_lp_success'],'original_minimum_slack_m_s':slack,'verified_within_original_tolerance':bool(slack is not None and slack>=-d['tolerance_m_s'])})
    for cap in [384,512]:
        o=json.loads((H/folder/f'{cap}.stdout.json').read_text());p=np.array(o['p']);w=A@p-b;res=float(np.max(abs(p-np.maximum(0,p-w/np.diag(A)))*np.diag(A)));energy=float(.5*p@(w-b));scale=1+np.sum(abs(p*b));v=inverse*(J.T@p);minimum=float(np.min(J@v-b));accepted=np.isfinite(v).all() and (p>=0).all() and (p<=np.array(d['hi'])).all() and res<=d['tolerance_m_s'] and energy<=d['tolerance_m_s']*scale
        assert o['accepted']==bool(o['found'] and accepted)
        if cap==512:assert accepted and minimum>=-d['tolerance_m_s']
        records.append({'folder':folder,'cap':cap,'accepted':o['accepted'],'original_projection_m_s':res,'minimum_original_geometry_rate_slack_m_s':minimum,'maximum_translation_speed_m_s':float(np.max(np.linalg.norm(v.reshape(-1,3),axis=1))),'primal_velocity':v.tolist()})
report={'passed':True,'original_target_feasibility_verified_by_native_primal_witness':True,'lp_status_requires_direct_slack_check':lp_records,'records':records,'scope':'No displacement is applied; these prove only the saved unchanged numerical position system.'}
(H/'independent-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
