"""Independent yield-tail refinement on immutable hybrid elastic source.

Retain the failed coarse force-capacity check as well as tighter same-material
numerical controls. No state is accepted merely because integration succeeded.
"""
import hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np

SOURCE_COMMIT='9aa38ed2d9c7118394e3fd1ee744f4ee439b7dac'
ROOT=Path(__file__).resolve().parent


def run():
    source=subprocess.check_output(['git','show',SOURCE_COMMIT+':research/elastic_patch.py'])
    frozen=ROOT/'elastic_patch_after.py';frozen.write_bytes(source)
    spec=importlib.util.spec_from_file_location('elastic_tail_frozen',frozen);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    material=module.Material(normal_stiffness=1e8,tangent_stiffness=1e8*2/7,twist_stiffness=4e5,friction=10,compression_exponent=0)
    records=[]
    for atol in (1e-13,1e-14,1e-15,1e-16):
        result=module.simulate(material,position=[0,0,.1001],velocity=[.2,0,-1],omega=[0,-5,10],gravity=[0,0,-9.81],
                               duration=.0007,sample_dt=2e-7,max_step=2e-6,rtol=1e-11,atol=atol)
        state=result['states'];strain=result['strain'];delta=np.maximum(0,material.radius-state[:,2])
        independent_K=.5*material.mass*np.sum(state[:,3:6]**2,axis=1)+.5*material.inertia*np.sum(state[:,6:9]**2,axis=1)
        independent_U=.5*material.normal_stiffness*delta**2+.5*np.einsum('ti,i,ti->t',strain,material.stiffness,strain)
        # Contact release has zero stored history; incoming/outgoing free-flight
        # histories are identically zero. Gravity potential is +m*g*z.
        accounted=independent_K+material.mass*9.81*state[:,2]+independent_U+result['dissipated_J']
        residual=float(np.max(abs(accounted-accounted[0])))
        sep=float(sum(e['separation_loss_J'] for e in result['events']))
        peak_normal=float(np.max(result['normal_force_N']));capacity_budget=1e-5+1e-8*peak_normal
        independent_impulse=material.mass*(state[-1,3:6]-state[0,3:6]-np.array([0,0,-9.81])*.0007)
        torque_impulse=np.cross([0,0,-material.radius],result['linear_impulse_N_s'][-1])+result['couple_impulse_N_m_s'][-1]
        assert np.max(abs(independent_impulse-result['linear_impulse_N_s'][-1]))<1e-9
        assert np.max(abs(material.inertia*(state[-1,6:9]-state[0,6:9])-torque_impulse))<1e-9
        assert residual<1e-9 and np.min(np.diff(result['dissipated_J']))>=-1e-12
        record=dict(atol=atol,rhs_evaluations=result['rhs_evaluations'],energy_residual_max_J=residual,
                    max_yield_excess_N=float(result['max_yield_excess_N']),capacity_budget_N=capacity_budget,
                    capacity_pass=bool(result['max_yield_excess_N']<=capacity_budget),
                    dissipated_J=float(result['dissipated_J'][-1]),separation_loss_J=sep,
                    plastic_loss_J=float(result['dissipated_J'][-1])-sep,material_events=result['material_events'],contact_events=result['events'])
        records.append(record)
    assert not records[0]['capacity_pass'] and all(r['capacity_pass'] for r in records[1:])
    assert records[-1]['max_yield_excess_N']<1e-5
    assert abs(records[-1]['plastic_loss_J']-records[-2]['plastic_loss_J'])<1e-11
    package=dict(schema='independent-elastic-tail-refinement-v1',source_commit=SOURCE_COMMIT,source_sha256=hashlib.sha256(source).hexdigest(),
                 cases=records,interpretation='Force-capacity failure of the coarse numerical profile is retained. Tighter numerical profiles converge on the same material and show the finite-gravity tail loss is plastic flow, not deleted separation energy.')
    (ROOT/'elastic-tail-refinement.json').write_text(json.dumps(package,indent=2)+'\n')
    print('Independent tail review PASS: coarse capacity rejection retained; tighter same-material profiles qualify')


if __name__=='__main__':run()
